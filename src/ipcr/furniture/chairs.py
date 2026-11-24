"""Stage 2 library-matched chair reconstruction (cls 7): backrest-PCA continuous yaw, ergonomic scale & zero-pushback solver."""

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
    """Snap ONLY when already nearly axis-aligned (within 12 deg), preserving genuinely angled chairs!"""
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


def _estimate_chair_front_dir(
    sub: np.ndarray,
    z_floor: float,
    h: float,
    table_specs: list[TableSpec],
    shell: ExteriorShellPlan,
) -> tuple[np.ndarray, TableSpec | None, np.ndarray]:
    """Infer continuous chair facing direction (+front_dir in UV) from upper backrest PCA + table/seat sign.

    Returns (front_dir_uv, nearest_table, upper_backrest_uv_pts).
    """
    cx, cy = float(np.median(sub[:, 0])), float(np.median(sub[:, 1]))
    pt = Point(cx, cy)

    # 1. Extract upper backrest slice (top 38% of chair height) — planar backrest gives accurate continuous yaw!
    hi = sub[sub[:, 2] >= z_floor + 0.62 * h]
    lo = sub[sub[:, 2] <= z_floor + 0.50 * h]

    asym_vec = None
    if len(hi) >= 20 and len(lo) >= 20:
        v_raw = lo[:, :2].mean(axis=0) - hi[:, :2].mean(axis=0)
        if float(np.linalg.norm(v_raw)) >= 0.04:
            asym_vec = v_raw / float(np.linalg.norm(v_raw))

    # Nearest table cue (for 180-deg sign resolution & fallback)
    best_table: TableSpec | None = None
    best_dist = 0.85
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

    # 2. Primary continuous yaw from planar upper backrest PCA (minor eigenvector = exact backrest normal!)
    if len(hi) >= 40:
        h_mean = hi[:, :2].mean(axis=0)
        hc = hi[:, :2] - h_mean
        cov = (hc.T @ hc) / len(hi)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Backrest is strongly elongated along chair width (eigvals[1] >> eigvals[0])
        if eigvals[1] > max(eigvals[0] * 3.0, 0.012):
            n_back = eigvecs[:, 0].copy()
            n_back /= max(float(np.linalg.norm(n_back)), 1e-9)
            # Resolve 180-deg sign using seat asymmetry first, then nearest table, then away from wall!
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

    # 3. Fallback: nearest table direction (snapped gently at 15 deg) or backrest asymmetry
    if best_table is not None and to_table_raw is not None:
        if asym_vec is None or float(np.dot(asym_vec, to_table_raw)) > -0.45:
            return _snap_dir_uv(to_table_raw, tol_deg=15.0), best_table, (hi[:, :2] if len(hi) >= 15 else sub[:, :2])
    if asym_vec is not None:
        return _snap_dir_uv(asym_vec, tol_deg=12.0), best_table, (hi[:, :2] if len(hi) >= 15 else sub[:, :2])

    np_w, _ = nearest_points(shell.inner_uv.exterior, pt)
    away = np.array([cx - np_w.x, cy - np_w.y], dtype=np.float64)
    return _snap_dir_uv(away, tol_deg=12.0), best_table, (hi[:, :2] if len(hi) >= 15 else sub[:, :2])


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
    """Return list of (verts_xyz, tris, role_suffix) for a candidate chair pose."""
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
    """Place library-matched watertight chairs with backrest-PCA yaw, ergonomic scale & zero-pushback solver."""
    uvz = fc.frame.pts_to_uvz(points_local)
    cpts = uvz[work_labels == 7]
    if len(cpts) < cfg.chair_min_points:
        return [], {"emitted_chairs": 0, "emitted_parts": 0, "chairs": [], "rejected": []}

    asset = library.get_best_asset(7)
    base_ext = asset.extents_xyz if asset is not None else np.array([0.52, 0.54, 0.95], dtype=np.float64)
    back_y_center = float(asset.backrest_y_center) if asset is not None else -0.255

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
                if np.hypot(float(cand_center[0]) - pu, float(cand_center[1]) - pv) < (radius_approx + pr) * 0.72:
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
        h_meas = float(np.clip(z_hi - z_floor, 0.72, 1.15))

        front_dir, near_table, back_uv = _estimate_chair_front_dir(sub, z_floor, h_meas, table_specs, shell)
        lat_dir = np.array([front_dir[1], -front_dir[0]], dtype=np.float64)

        s_front = sub[:, :2] @ front_dir
        s_lat = sub[:, :2] @ lat_dir
        depth_meas = float(np.percentile(s_front, 96) - np.percentile(s_front, 4))
        width_meas = float(np.percentile(s_lat, 96) - np.percentile(s_lat, 4))

        # Ergonomic bounded scaling: match actual conference/office chair width & depth cleanly (avoid oversized executive bulk!)
        s_z = h_meas / float(base_ext[2])
        raw_sx = width_meas / float(base_ext[0])
        s_x = float(np.clip(raw_sx, 0.65, 0.96))
        if depth_meas < 0.42:
            # Seat-occluded backrest-only scan -> match natural conference chair depth (~0.56-0.62m)
            s_y = float(np.clip(s_x * 0.90, 0.62, 0.80))
        else:
            raw_sy = depth_meas / float(base_ext[1])
            s_y = float(np.clip(raw_sy, s_x * 0.78, s_x * 1.12))
            s_y = float(np.clip(s_y, 0.62, 0.92))

        # Anchor canonical backrest center directly onto measured upper-backrest center plane!
        back_s_center = float(np.median(back_uv @ front_dir))
        lat_mid = float(np.median(back_uv @ lat_dir))
        init_front_s = back_s_center - back_y_center * s_y
        base_center_uv = front_dir * init_front_s + lat_dir * lat_mid

        # Priority search:
        # 1. Exact backrest position (df=0, dl=0) at full height, then gentle armrest-clearance height trim (0.93 / 0.87 / 0.82)
        #    so chairs tucked under tables stay DEAD-ON their real backrest instead of getting pushed 25cm backward!
        # 2. Tiny pullback/lateral nudges (2-14cm) only if colliding with legs/walls/neighbor chairs
        # 3. Mild overall trim (0.88x) for tightly packed conference rows
        search_candidates: list[tuple[float, float, float, float]] = [
            # (df, dl, sxy_mult, sz_mult)
            (0.0, 0.0, 1.0, 1.0),
            (0.0, 0.0, 1.0, 0.93),
            (0.0, 0.0, 1.0, 0.87),
            (0.0, 0.0, 0.94, 0.84),
            (-0.03, 0.0, 1.0, 0.95),
            (-0.03, 0.0, 1.0, 0.87),
            (0.03, 0.0, 1.0, 0.95),
            (-0.06, 0.0, 1.0, 0.93),
            (-0.06, 0.0, 1.0, 0.86),
            (-0.10, 0.0, 1.0, 0.92),
            (-0.14, 0.0, 1.0, 0.92),
            (-0.20, 0.0, 1.0, 0.92),
        ]
        for lat_d in (-0.04, 0.04, -0.08, 0.08):
            for back_d in (0.0, -0.04, -0.10, -0.16):
                search_candidates.append((back_d, lat_d, 0.96, 0.90))
        for back_d in (0.0, -0.06, -0.14, -0.22):
            for lat_d in (0.0, -0.06, 0.06):
                search_candidates.append((back_d, lat_d, 0.86, 0.84))

        cid_str = f"chair_{len(logs):02d}"
        base_rad = 0.5 * float(min(base_ext[0], base_ext[1]))
        placed_meshes = None
        used_shift = (0.0, 0.0)
        used_sm = (1.0, 1.0)

        for df, dl, sxy_m, sz_m in search_candidates:
            cand_center = base_center_uv + front_dir * df + lat_dir * dl
            cand_sx, cand_sy, cand_sz = s_x * sxy_m, s_y * sxy_m, s_z * sz_m
            cand_rad = base_rad * min(cand_sx, cand_sy)
            cand_meshes = _build_candidate_chair_meshes(
                asset, fc, cand_center, z_floor, front_dir, cand_sx, cand_sy, cand_sz
            )
            if _pose_is_collision_free(cand_center, z_floor, cand_rad, cand_meshes):
                placed_meshes = cand_meshes
                used_shift = (df, dl)
                used_sm = (sxy_m, sz_m)
                placed_chair_centers.append((float(cand_center[0]), float(cand_center[1]), float(z_floor), cand_rad))
                break

        if placed_meshes is None:
            rejected.append({
                "candidate_idx": idx,
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
                    evidence_kind="library_matched" if asset is not None else "parametric_rule_fallback",
                    metadata={
                        "chair_id": cid_str,
                        "asset_id": asset.asset_id if asset is not None else "parametric_fallback_chair",
                        "asset_sha256": asset.sha256 if asset is not None else None,
                        "scale_xyz": [round(s_x * used_sm[0], 4), round(s_y * used_sm[0], 4), round(s_z * used_sm[1], 4)],
                        "front_dir_uv": [round(float(front_dir[0]), 4), round(float(front_dir[1]), 4)],
                        "collision_shift_fl_m": [round(used_shift[0], 3), round(used_shift[1], 3)],
                        "point_count": int(len(sub)),
                    },
                )
            )
            collider_list.append((v_xyz.min(axis=0), v_xyz.max(axis=0), v_xyz, tri))

        logs.append({
            "chair_id": cid_str,
            "library_matched": asset is not None,
            "asset_id": asset.asset_id if asset is not None else "parametric_fallback_chair",
            "center_uv": [
                round(float(base_center_uv[0] + front_dir[0] * used_shift[0] + lat_dir[0] * used_shift[1]), 3),
                round(float(base_center_uv[1] + front_dir[1] * used_shift[0] + lat_dir[1] * used_shift[1]), 3),
            ],
            "height_m": round(h_meas * used_sm[1], 3),
            "front_dir_uv": [round(float(front_dir[0]), 4), round(float(front_dir[1]), 4)],
            "near_table_id": near_table.table_id if near_table is not None else None,
            "collision_shift_fl_m": [round(used_shift[0], 3), round(used_shift[1], 3)],
            "point_count": int(len(sub)),
        })

    return parts, {
        "emitted_chairs": len(logs),
        "emitted_parts": len(parts),
        "library_asset_used": asset.asset_id if asset is not None else None,
        "chairs": logs,
        "rejected": rejected,
    }
