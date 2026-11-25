"""Stage 2 multi-asset chair reconstruction (cls 7): part-level backrest/seat scaling, dual-mode anchor & 0-penetration solver."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points

from ipcr.architecture.frame_floors import FloorCeilingResult
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import FurnitureConfig
from ipcr.furniture.library import CanonicalLibraryAsset, FurnitureLibrary, make_fallback_parametric_chair
from ipcr.furniture.tables_cabinets_sofas import TableSpec, _local_floor_z
from ipcr.types import MeshPart
from ipcr.validator.mesh_relations import meshes_penetrate
from ipcr.working_labels import _voxel_connected_components_3d


def _segment_individual_chairs(
    chair_uvz: np.ndarray,
    fc: FloorCeilingResult,
    cfg: FurnitureConfig,
) -> list[np.ndarray]:
    """Split merged chair rows/blobs into individual chair point clusters using backrest separation & slot spacing."""
    if len(chair_uvz) < cfg.chair_min_points:
        return []

    raw_comps = _voxel_connected_components_3d(chair_uvz, voxel_size=0.08)
    singles: list[np.ndarray] = []

    def _split_strip(sub: np.ndarray, depth: int = 0) -> None:
        if len(sub) < cfg.chair_min_points:
            return
        u0, u1 = float(np.percentile(sub[:, 0], 2)), float(np.percentile(sub[:, 0], 98))
        v0, v1 = float(np.percentile(sub[:, 1], 2)), float(np.percentile(sub[:, 1], 98))
        du, dv = u1 - u0, v1 - v0
        z_lo = float(np.percentile(sub[:, 2], 2))
        z_hi = float(np.percentile(sub[:, 2], 98))
        z_floor = _local_floor_z(fc, 0.5 * (u0 + u1), 0.5 * (v0 + v1), z_lo)
        h = z_hi - z_floor
        if h < cfg.chair_min_height_m or h > cfg.chair_max_height_m:
            return

        if max(du, dv) <= 1.05:
            singles.append(sub)
            return

        if depth == 0:
            z_cut = z_floor + max(0.55 * h, h - 0.42)
            upper = sub[sub[:, 2] >= z_cut]
            if len(upper) >= cfg.chair_min_points:
                u_comps = _voxel_connected_components_3d(upper, voxel_size=0.055)
                valid_u = [upper[c] for c in u_comps if len(c) >= max(80, cfg.chair_min_points // 3)]
                if len(valid_u) >= 2:
                    cents = np.array([[float(np.median(g[:, 0])), float(np.median(g[:, 1]))] for g in valid_u])
                    d2 = ((sub[:, None, :2] - cents[None, :, :]) ** 2).sum(axis=2)
                    assign = np.argmin(d2, axis=1)
                    for k in range(len(valid_u)):
                        grp = sub[assign == k]
                        if len(grp) >= cfg.chair_min_points:
                            _split_strip(grp, depth=depth + 1)
                    return

        axis = 0 if du >= dv else 1
        lo, hi = (u0, u1) if axis == 0 else (v0, v1)
        span = hi - lo
        n_slots = max(2, int(round(span / cfg.chair_row_slot_m)))
        edges = np.linspace(lo - 0.02, hi + 0.02, n_slots + 1)
        for k in range(n_slots):
            m = (sub[:, axis] >= edges[k]) & (sub[:, axis] < edges[k + 1])
            slot_pts = sub[m]
            if len(slot_pts) < cfg.chair_min_points:
                continue
            other = 1 - axis
            o_span = float(np.percentile(slot_pts[:, other], 98) - np.percentile(slot_pts[:, other], 2))
            if o_span > 1.10 and depth < 2:
                _split_strip(slot_pts, depth=depth + 1)
            else:
                singles.append(slot_pts)

    for comp in raw_comps:
        if len(comp) < cfg.chair_min_points:
            continue
        _split_strip(chair_uvz[comp], depth=0)

    return singles


def _snap_dir_uv(vec: np.ndarray, tol_deg: float = 12.0) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    u = vec / n
    cos_t = float(np.cos(np.radians(tol_deg)))
    for axis in (
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    ):
        if float(np.dot(u, axis)) >= cos_t:
            return axis
    return u


def _detect_seat_z(sub_z: np.ndarray, z_floor: float) -> float:
    """Detect dominant horizontal seat cushion height in [z_floor + 0.30, z_floor + 0.58]."""
    seat_band = sub_z[(sub_z >= z_floor + 0.30) & (sub_z <= z_floor + 0.58)]
    if len(seat_band) < 20:
        return z_floor + 0.45
    bins = np.arange(z_floor + 0.29, z_floor + 0.60, 0.025)
    hist, edges = np.histogram(seat_band, bins=bins)
    pk = int(np.argmax(hist))
    zc = 0.5 * (edges[pk] + edges[pk + 1])
    near = seat_band[np.abs(seat_band - zc) <= 0.04]
    return float(np.median(near)) if len(near) >= 10 else zc


def _estimate_chair_front_dir(
    sub: np.ndarray,
    z_floor: float,
    z_seat: float,
    h: float,
    has_backrest: bool,
    table_specs: list[TableSpec],
    shell: ExteriorShellPlan,
) -> tuple[np.ndarray, TableSpec | None, np.ndarray]:
    """Infer chair facing direction (+front_dir in UV) for backrest-visible OR seat-only chairs."""
    cx, cy = float(np.median(sub[:, 0])), float(np.median(sub[:, 1]))
    pt = Point(cx, cy)

    hi = sub[sub[:, 2] >= z_seat + 0.18]
    lo = sub[sub[:, 2] <= z_seat + 0.08]

    asym_vec = None
    if has_backrest and len(hi) >= 20 and len(lo) >= 20:
        v_raw = lo[:, :2].mean(axis=0) - hi[:, :2].mean(axis=0)
        if float(np.linalg.norm(v_raw)) >= 0.04:
            asym_vec = v_raw / float(np.linalg.norm(v_raw))

    best_table: TableSpec | None = None
    best_dist = 1.25 if not has_backrest else 0.85
    to_table_raw = None
    for ts in table_specs:
        if abs(ts.z_floor - z_floor) > 0.50:
            continue
        d = float(ts.poly_uv.distance(pt))
        if d < best_dist:
            _, np_t = nearest_points(pt, ts.poly_uv)
            vec = np.array([np_t.x - cx, np_t.y - cy], dtype=np.float64)
            if float(np.linalg.norm(vec)) < 0.02:
                tc = ts.poly_uv.centroid
                vec = np.array([tc.x - cx, tc.y - cy], dtype=np.float64)
            if float(np.linalg.norm(vec)) >= 0.02:
                best_dist = d
                best_table = ts
                to_table_raw = vec / float(np.linalg.norm(vec))

    # Mode A (Backrest visible): continuous PCA normal of upper backrest plane!
    if has_backrest and len(hi) >= 35:
        h_mean = hi[:, :2].mean(axis=0)
        hc = hi[:, :2] - h_mean
        cov = (hc.T @ hc) / len(hi)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals[1] > max(eigvals[0] * 3.0, 0.012):
            n_back = eigvecs[:, 0].copy()
            n_back /= max(float(np.linalg.norm(n_back)), 1e-9)
            sign_ref = asym_vec
            if sign_ref is None or abs(float(np.dot(n_back, sign_ref))) < 0.25:
                if to_table_raw is not None and abs(float(np.dot(n_back, to_table_raw))) >= 0.20:
                    sign_ref = to_table_raw
                else:
                    np_w, _ = nearest_points(shell.inner_uv.exterior, pt)
                    sign_ref = np.array([cx - np_w.x, cy - np_w.y], dtype=np.float64)
            if float(np.dot(n_back, sign_ref)) < 0:
                n_back = -n_back
            return _snap_dir_uv(n_back, tol_deg=12.0), best_table, hi[:, :2]

    # Mode B (Seat-only / missing backrest): face TOWARD nearest table (or away from nearest wall!)
    if best_table is not None and to_table_raw is not None:
        if asym_vec is None or float(np.dot(asym_vec, to_table_raw)) > -0.45:
            return _snap_dir_uv(to_table_raw, tol_deg=18.0), best_table, (hi[:, :2] if len(hi) >= 15 else sub[:, :2])
    if asym_vec is not None:
        return _snap_dir_uv(asym_vec, tol_deg=12.0), best_table, (hi[:, :2] if len(hi) >= 15 else sub[:, :2])

    np_w, _ = nearest_points(shell.inner_uv.exterior, pt)
    away = np.array([cx - np_w.x, cy - np_w.y], dtype=np.float64)
    return _snap_dir_uv(away, tol_deg=15.0), best_table, (hi[:, :2] if len(hi) >= 15 else sub[:, :2])


def _build_candidate_chair_meshes(
    asset: CanonicalLibraryAsset | None,
    fc: FloorCeilingResult,
    center_uv: np.ndarray,
    z_floor: float,
    front_dir: np.ndarray,
    scale_x: float,
    scale_y: float,
    scale_z: float,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    if asset is not None:
        v_uvz, t = asset.transform_to_uvz(center_uv, z_floor + 0.002, front_dir, scale_x, scale_y, scale_z)
        return [(fc.frame.uvz_to_xyz(v_uvz), t, "chair")]

    fb_parts, _ext = make_fallback_parametric_chair()
    fy = front_dir / max(float(np.linalg.norm(front_dir)), 1e-9)
    rx = np.array([fy[1], -fy[0]], dtype=np.float64)
    out: list[tuple[np.ndarray, np.ndarray, str]] = []
    for bv, bt, brole in fb_parts:
        v = bv.copy()
        v[:, 0] *= scale_x
        v[:, 1] *= scale_y
        v[:, 2] *= scale_z
        uvz = np.empty_like(v)
        uvz[:, 0] = center_uv[0] + v[:, 0] * rx[0] + v[:, 1] * fy[0]
        uvz[:, 1] = center_uv[1] + v[:, 0] * rx[1] + v[:, 1] * fy[1]
        uvz[:, 2] = z_floor + 0.002 + v[:, 2]
        out.append((fc.frame.uvz_to_xyz(uvz), bt.copy(), brole))
    return out


def build_chair_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    existing_parts: list[MeshPart],
    table_specs: list[TableSpec],
    library: FurnitureLibrary,
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: FurnitureConfig,
) -> tuple[list[MeshPart], dict[str, Any]]:
    """Place multi-asset library chairs with part-level backrest/seat scaling, dual-mode anchor & 0-penetration solver."""
    uvz = fc.frame.pts_to_uvz(points_local)
    cpts = uvz[work_labels == 7]
    if len(cpts) < cfg.chair_min_points:
        return [], {"emitted_chairs": 0, "emitted_parts": 0, "assets_used": [], "chairs": [], "rejected": []}

    singles = _segment_individual_chairs(cpts, fc, cfg)
    singles.sort(key=len, reverse=True)

    collider_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for p in existing_parts:
        collider_list.append((p.vertices.min(axis=0), p.vertices.max(axis=0), p.vertices, p.triangles))
    collider_list.sort(key=lambda item: len(item[3]))

    inner_safe = shell.inner_uv.buffer(-0.006)
    parts: list[MeshPart] = []
    logs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    placed_chair_centers: list[tuple[float, float, float, float]] = []

    def _pose_is_collision_free(
        cand_center: np.ndarray,
        z_fl: float,
        radius_approx: float,
        cand_meshes: list[tuple[np.ndarray, np.ndarray, str]],
    ) -> bool:
        for pu, pv, pz, pr in placed_chair_centers:
            if abs(pz - z_fl) < 0.60:
                if np.hypot(float(cand_center[0]) - pu, float(cand_center[1]) - pv) < (radius_approx + pr) * 0.70:
                    return False

        for v_xyz, tri, _role in cand_meshes:
            v_uv = fc.frame.xy_to_uv(v_xyz[:, :2])
            u_lo, v_lo = float(v_uv[:, 0].min()), float(v_uv[:, 1].min())
            u_hi, v_hi = float(v_uv[:, 0].max()), float(v_uv[:, 1].max())
            for cu, cv in ((u_lo, v_lo), (u_hi, v_lo), (u_hi, v_hi), (u_lo, v_hi)):
                if not inner_safe.contains(Point(cu, cv)):
                    return False

            c_min = v_xyz.min(axis=0)
            c_max = v_xyz.max(axis=0)
            for e_min, e_max, e_v, e_t in collider_list:
                if np.any(c_max <= e_min + 1e-4) or np.any(e_max <= c_min + 1e-4):
                    continue
                res = meshes_penetrate(v_xyz, tri, e_v, e_t, eps=5e-4)
                if res["penetrates"]:
                    return False
        return True

    for idx, sub in enumerate(singles):
        u0, u1 = float(np.percentile(sub[:, 0], 3)), float(np.percentile(sub[:, 0], 97))
        v0, v1 = float(np.percentile(sub[:, 1], 3)), float(np.percentile(sub[:, 1], 97))
        z_lo = float(np.percentile(sub[:, 2], 2))
        z_hi = float(np.percentile(sub[:, 2], 98))
        cx, cy = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        z_floor = _local_floor_z(fc, cx, cy, z_lo)
        z_seat = _detect_seat_z(sub[:, 2], z_floor)
        h_raw = z_hi - z_floor

        # Dual-mode detection: does this chair cluster have a genuine upper backrest OR is it seat-only / low-back?
        upper_mask = sub[:, 2] >= (z_seat + 0.18)
        upper_cnt = int(np.sum(upper_mask))
        upper_ratio = upper_cnt / max(len(sub), 1)
        has_backrest = (h_raw >= 0.78) and (upper_cnt >= 35) and (upper_ratio >= 0.12)
        anchor_mode = "backrest_anchored" if has_backrest else "seat_anchored"

        h_meas = float(np.clip(h_raw, 0.76, 1.15)) if has_backrest else float(np.clip(max(h_raw, (z_seat - z_floor) * 1.95), 0.72, 0.92))
        front_dir, near_table, back_uv = _estimate_chair_front_dir(
            sub, z_floor, z_seat, h_meas, has_backrest, table_specs, shell
        )
        lat_dir = np.array([front_dir[1], -front_dir[0]], dtype=np.float64)

        s_front = sub[:, :2] @ front_dir
        s_lat = sub[:, :2] @ lat_dir
        depth_meas = float(np.percentile(s_front, 96) - np.percentile(s_front, 4))
        width_meas = float(np.percentile(s_lat, 96) - np.percentile(s_lat, 4))

        # Measure functional reference width: backrest width in Mode A, seat width/depth in Mode B!
        seat_slice = sub[(sub[:, 2] >= z_seat - 0.10) & (sub[:, 2] <= z_seat + 0.12)]
        if len(seat_slice) < 25:
            seat_slice = sub
        seat_w_meas = float(np.percentile(seat_slice[:, :2] @ lat_dir, 95) - np.percentile(seat_slice[:, :2] @ lat_dir, 5))
        seat_d_meas = float(np.percentile(seat_slice[:, :2] @ front_dir, 95) - np.percentile(seat_slice[:, :2] @ front_dir, 5))
        back_w_meas = (
            float(np.percentile(back_uv @ lat_dir, 96) - np.percentile(back_uv @ lat_dir, 4))
            if (has_backrest and len(back_uv) >= 20)
            else None
        )

        # Rank all verified chair models in 3.模型库 (lowpoly_office_chair vs office_chair_modern) for THIS chair!
        ranked_assets = library.rank_assets(
            7,
            (seat_w_meas if not has_backrest else width_meas, depth_meas, h_meas),
            back_width_m=back_w_meas,
            seat_only=(not has_backrest),
        )
        candidate_assets: list[CanonicalLibraryAsset | None] = ranked_assets if ranked_assets else [None]

        cid_str = f"chair_{len(logs):02d}"
        placed_meshes = None
        chosen_asset: CanonicalLibraryAsset | None = None
        used_shift = (0.0, 0.0)
        used_sxyz = (1.0, 1.0, 1.0)
        final_center = np.array([cx, cy], dtype=np.float64)

        # Try top-ranked library asset first; if tight-row wheel collision prevents clean fit, also try alternate verified chair!
        for asset in candidate_assets:
            base_ext = asset.extents_xyz if asset is not None else np.array([0.52, 0.54, 0.95], dtype=np.float64)
            asset_back_w = float(asset.backrest_width_m) if asset is not None else 0.50
            asset_back_yc = float(asset.backrest_y_center) if asset is not None else -0.255
            asset_seat_w = float(asset.seat_width_m) if asset is not None else 0.52
            asset_seat_d = float(asset.seat_depth_m) if asset is not None else 0.50
            asset_seat_yc = float(asset.seat_y_center) if asset is not None else 0.0
            asset_seat_zc = float(asset.seat_z_center) if asset is not None else 0.45

            if has_backrest:
                # Part-level backrest width scaling (eliminates skinny backrest when wheel base is wider!)
                s_z = h_meas / float(base_ext[2])
                if back_w_meas is not None and back_w_meas >= 0.34:
                    raw_sx = back_w_meas / max(asset_back_w, 0.35)
                else:
                    raw_sx = width_meas / max(asset_seat_w, 0.40)
                s_x = float(np.clip(raw_sx, 0.72, 1.08))
                if depth_meas < 0.42:
                    s_y = float(np.clip(s_x * 0.86, 0.66, 0.90))
                else:
                    raw_sy = depth_meas / float(base_ext[1])
                    s_y = float(np.clip(raw_sy, s_x * 0.75, s_x * 1.08))
                    s_y = float(np.clip(s_y, 0.66, 0.98))

                back_s_center = float(np.median(back_uv @ front_dir))
                lat_mid = float(np.median(back_uv @ lat_dir))
                init_front_s = back_s_center - asset_back_yc * s_y
                base_center_uv = front_dir * init_front_s + lat_dir * lat_mid

                search_candidates: list[tuple[float, float, float, float]] = [
                    # (df, dl, sxy_mult, sz_mult) — exact position first, then armrest height trim, then mild wheel/pullback trim
                    (0.0, 0.0, 1.0, 1.0),
                    (0.0, 0.0, 1.0, 0.93),
                    (0.0, 0.0, 0.92, 0.90),
                    (0.0, 0.0, 0.85, 0.86),
                    (-0.03, 0.0, 0.95, 0.93),
                    (-0.03, 0.0, 0.86, 0.86),
                    (0.03, 0.0, 0.95, 0.93),
                    (-0.06, 0.0, 0.92, 0.90),
                    (-0.10, 0.0, 0.90, 0.90),
                    (-0.15, 0.0, 0.88, 0.88),
                ]
                for lat_d in (-0.04, 0.04, -0.08, 0.08):
                    for back_d in (0.0, -0.04, -0.10, -0.16):
                        search_candidates.append((back_d, lat_d, 0.88, 0.88))
                for back_d in (0.0, -0.06, -0.14, -0.22):
                    for lat_d in (0.0, -0.06, 0.06):
                        search_candidates.append((back_d, lat_d, 0.80, 0.84))
            else:
                # Mode B (Seat-Only / Missing Backrest): anchor seat center directly onto measured seat centroid!
                # User explicitly approved small collision-avoidance shift (e.g. forward away from wall) to guarantee 0 penetration.
                s_z = float(np.clip((z_seat - z_floor) / max(asset_seat_zc, 0.35), 0.72, 0.92))
                s_x = float(np.clip(seat_w_meas / max(asset_seat_w, 0.40), 0.68, 0.95))
                s_y = float(np.clip(seat_d_meas / max(asset_seat_d, 0.40), s_x * 0.80, s_x * 1.10))
                s_y = float(np.clip(s_y, 0.66, 0.92))

                seat_front_c = float(np.median(seat_slice[:, :2] @ front_dir))
                seat_lat_c = float(np.median(seat_slice[:, :2] @ lat_dir))
                init_front_s = seat_front_c - asset_seat_yc * s_y
                base_center_uv = front_dir * init_front_s + lat_dir * seat_lat_c

                # Prioritize exact seat centroid first, then forward shift (+df away from back wall!) & lateral nudges
                search_candidates = [
                    (0.0, 0.0, 1.0, 1.0),
                    (0.04, 0.0, 1.0, 0.95),
                    (0.08, 0.0, 0.96, 0.92),
                    (0.12, 0.0, 0.94, 0.90),
                    (0.16, 0.0, 0.90, 0.88),
                    (0.20, 0.0, 0.86, 0.85),
                    (-0.04, 0.0, 0.95, 0.92),
                    (-0.08, 0.0, 0.90, 0.90),
                ]
                for lat_d in (-0.05, 0.05, -0.10, 0.10):
                    for fd in (0.0, 0.06, 0.12, 0.18, -0.05):
                        search_candidates.append((fd, lat_d, 0.88, 0.88))
                for fd in (0.04, 0.10, 0.16, 0.22):
                    for lat_d in (0.0, -0.06, 0.06):
                        search_candidates.append((fd, lat_d, 0.80, 0.82))

            base_rad = 0.5 * float(min(base_ext[0], base_ext[1]))
            for df, dl, sxy_m, sz_m in search_candidates:
                cand_center = base_center_uv + front_dir * df + lat_dir * dl
                cand_sx, cand_sy, cand_sz = s_x * sxy_m, s_y * sxy_m, s_z * sz_m
                cand_rad = base_rad * min(cand_sx, cand_sy)
                cand_meshes = _build_candidate_chair_meshes(
                    asset, fc, cand_center, z_floor, front_dir, cand_sx, cand_sy, cand_sz
                )
                if _pose_is_collision_free(cand_center, z_floor, cand_rad, cand_meshes):
                    placed_meshes = cand_meshes
                    chosen_asset = asset
                    used_shift = (df, dl)
                    used_sxyz = (cand_sx, cand_sy, cand_sz)
                    final_center = cand_center
                    placed_chair_centers.append((float(cand_center[0]), float(cand_center[1]), float(z_floor), cand_rad))
                    break
            if placed_meshes is not None:
                break

        if placed_meshes is None:
            rejected.append({
                "candidate_idx": idx,
                "anchor_mode": anchor_mode,
                "point_count": int(len(sub)),
                "center_uv": [round(cx, 3), round(cy, 3)],
                "reason": "no_collision_free_pose_within_bounded_search",
            })
            continue

        for p_idx, (v_xyz, tri, brole) in enumerate(placed_meshes):
            pid = cid_str if len(placed_meshes) == 1 else f"{cid_str}_{brole}_{p_idx:02d}"
            parts.append(
                MeshPart(
                    part_id=pid,
                    class_id=7,
                    role=brole,
                    vertices=v_xyz,
                    triangles=tri,
                    evidence_kind="library_matched" if chosen_asset is not None else "parametric_rule_fallback",
                    metadata={
                        "chair_id": cid_str,
                        "anchor_mode": anchor_mode,
                        "asset_id": chosen_asset.asset_id if chosen_asset is not None else "parametric_fallback_chair",
                        "asset_sha256": chosen_asset.sha256 if chosen_asset is not None else None,
                        "scale_xyz": [round(used_sxyz[0], 4), round(used_sxyz[1], 4), round(used_sxyz[2], 4)],
                        "front_dir_uv": [round(float(front_dir[0]), 4), round(float(front_dir[1]), 4)],
                        "collision_shift_fl_m": [round(used_shift[0], 3), round(used_shift[1], 3)],
                        "point_count": int(len(sub)),
                    },
                )
            )
            collider_list.append((v_xyz.min(axis=0), v_xyz.max(axis=0), v_xyz, tri))

        logs.append({
            "chair_id": cid_str,
            "anchor_mode": anchor_mode,
            "library_matched": chosen_asset is not None,
            "asset_id": chosen_asset.asset_id if chosen_asset is not None else "parametric_fallback_chair",
            "center_uv": [round(float(final_center[0]), 3), round(float(final_center[1]), 3)],
            "height_m": round(float((chosen_asset.extents_xyz[2] if chosen_asset else 0.95) * used_sxyz[2]), 3),
            "front_dir_uv": [round(float(front_dir[0]), 4), round(float(front_dir[1]), 4)],
            "near_table_id": near_table.table_id if near_table is not None else None,
            "collision_shift_fl_m": [round(used_shift[0], 3), round(used_shift[1], 3)],
            "point_count": int(len(sub)),
        })

    return parts, {
        "emitted_chairs": len(logs),
        "emitted_parts": len(parts),
        "assets_used": sorted({str(c["asset_id"]) for c in logs if c.get("library_matched")}),
        "seat_anchored_count": sum(1 for c in logs if c["anchor_mode"] == "seat_anchored"),
        "backrest_anchored_count": sum(1 for c in logs if c["anchor_mode"] == "backrest_anchored"),
        "chairs": logs,
        "rejected": rejected,
    }
