"""B16 Evidence-gated watertight stepped stairs, stair-well cutouts, and open handrail/baluster railings.

Only triggers when genuine multi-step stair (class 13) or continuous railing (class 14)
evidence survives working_labels; isolated specks quarantined by WL-B produce 0 parts with N/A audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ipcr.architecture.frame_floors import FloorCeilingResult
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import ArchitectureConfig
from ipcr.geometry_primitives import extrude_polygon_solid, make_oriented_box, signed_mesh_volume
from ipcr.types import MeshPart
from ipcr.working_labels import _voxel_connected_components_3d


@dataclass
class _FlightGeom:
    part_id: str
    poly_uv: Polygon
    run_dir: np.ndarray
    lat_dir: np.ndarray
    s_lo: float
    s_hi: float
    w_lo: float
    w_hi: float
    z_lo: float
    z_hi: float
    n_steps: int

    def step_top_z_at(self, u: float, v: float) -> float:
        """Return exact top tread Z of this flight at UV point (u, v)."""
        s = float(u * self.run_dir[0] + v * self.run_dir[1])
        if s <= self.s_lo:
            return self.z_lo
        if s >= self.s_hi:
            return self.z_hi
        frac = (s - self.s_lo) / max(self.s_hi - self.s_lo, 1e-6)
        step_idx = min(self.n_steps - 1, max(0, int(np.floor(frac * self.n_steps))))
        dz = (self.z_hi - self.z_lo) / self.n_steps
        return self.z_lo + (step_idx + 1) * dz


def _sw_rect_poly(
    s_lo: float, s_hi: float, w_lo: float, w_hi: float, run_dir: np.ndarray, lat_dir: np.ndarray
) -> Polygon:
    c0 = run_dir * s_lo + lat_dir * w_lo
    c1 = run_dir * s_hi + lat_dir * w_lo
    c2 = run_dir * s_hi + lat_dir * w_hi
    c3 = run_dir * s_lo + lat_dir * w_hi
    return orient(Polygon([c0, c1, c2, c3]), sign=1.0)


def _inscribe_sw_rect(
    s_lo: float,
    s_hi: float,
    w_lo: float,
    w_hi: float,
    run_dir: np.ndarray,
    lat_dir: np.ndarray,
    allowed_xy: Polygon | MultiPolygon,
    *,
    min_run_m: float = 0.75,
    min_width_m: float = 0.55,
) -> tuple[float, float, float, float] | None:
    """Shrink any colliding side of a stair flight rectangle so it lies 100% inside allowed_xy."""
    step = 0.025
    for _ in range(80):
        if s_hi - s_lo < min_run_m or w_hi - w_lo < min_width_m:
            return None
        rect = _sw_rect_poly(s_lo, s_hi, w_lo, w_hi, run_dir, lat_dir)
        if allowed_xy.contains(rect):
            return s_lo, s_hi, w_lo, w_hi

        # Test 12 samples along each of the 4 sides to find which side(s) poke outside
        ts = np.linspace(0.0, 1.0, 13)
        sides_out = []
        # s_lo side
        pts_slo = [run_dir * s_lo + lat_dir * (w_lo + t * (w_hi - w_lo)) for t in ts]
        out_slo = sum(0 if allowed_xy.contains(Point(p[0], p[1])) else 1 for p in pts_slo)
        sides_out.append((out_slo, "s_lo"))
        # s_hi side
        pts_shi = [run_dir * s_hi + lat_dir * (w_lo + t * (w_hi - w_lo)) for t in ts]
        out_shi = sum(0 if allowed_xy.contains(Point(p[0], p[1])) else 1 for p in pts_shi)
        sides_out.append((out_shi, "s_hi"))
        # w_lo side
        pts_wlo = [run_dir * (s_lo + t * (s_hi - s_lo)) + lat_dir * w_lo for t in ts]
        out_wlo = sum(0 if allowed_xy.contains(Point(p[0], p[1])) else 1 for p in pts_wlo)
        sides_out.append((out_wlo, "w_lo"))
        # w_hi side
        pts_whi = [run_dir * (s_lo + t * (s_hi - s_lo)) + lat_dir * w_hi for t in ts]
        out_whi = sum(0 if allowed_xy.contains(Point(p[0], p[1])) else 1 for p in pts_whi)
        sides_out.append((out_whi, "w_hi"))

        sides_out.sort(reverse=True)
        if sides_out[0][0] == 0:
            # Interior obstacle (like a pillar inside the rectangle): shrink all sides slightly
            w_lo += step
            w_hi -= step
            continue
        worst = sides_out[0][1]
        if worst == "s_lo":
            s_lo += step
        elif worst == "s_hi":
            s_hi -= step
        elif worst == "w_lo":
            w_lo += step
        else:
            w_hi -= step
    return None


def _snap_dir_to_room_axis(vec: np.ndarray, tol_deg: float = 18.0) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=np.float64)
    u = vec / norm
    candidates = [
        np.array([1.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, -1.0]),
    ]
    cos_tol = float(np.cos(np.radians(tol_deg)))
    for c in candidates:
        if float(np.dot(u, c)) >= cos_tol:
            return c
    return u


def _build_stepped_flight_mesh(
    fc: FloorCeilingResult,
    run_dir: np.ndarray,
    lat_dir: np.ndarray,
    s_lo: float,
    s_hi: float,
    w_lo: float,
    w_hi: float,
    z_lo: float,
    z_hi: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one closed manifold watertight stepped-profile staircase solid with sloped soffit."""
    dz = (z_hi - z_lo) / n_steps
    ds = (s_hi - s_lo) / n_steps
    waist = min(0.24, max(0.15, 1.25 * dz))

    # Build simple 2D side-elevation profile polygon in (s, z) plane
    pts_sz: list[tuple[float, float]] = [(s_lo, z_lo)]
    for k in range(n_steps):
        sk0 = s_lo + k * ds
        sk1 = s_lo + (k + 1) * ds
        zk1 = z_lo + (k + 1) * dz
        pts_sz.append((sk0, zk1))
        pts_sz.append((sk1, zk1))

    # Sloped soffit underneath: strictly below every inner step corner (s_lo + k*ds, z_lo + k*dz)
    z_back_bot = max(z_lo, z_hi - waist)
    if z_back_bot > z_lo + 0.02:
        pts_sz.append((s_hi, z_back_bot))
        s_foot = float(np.clip(s_lo + waist * (ds / max(dz, 1e-6)), s_lo + 0.05, s_hi - 0.05))
        pts_sz.append((s_foot, z_lo))
    else:
        pts_sz.append((s_hi, z_lo))

    profile_poly = orient(Polygon(pts_sz).buffer(0), sign=1.0)
    if isinstance(profile_poly, MultiPolygon):
        profile_poly = max(profile_poly.geoms, key=lambda g: g.area)

    # Extrude 2D (s, z) profile along width w in [w_lo, w_hi]
    v_szw, tri = extrude_polygon_solid(profile_poly, w_lo, w_hi)
    s_val = v_szw[:, 0]
    z_val = v_szw[:, 1]
    w_val = v_szw[:, 2]

    uv = s_val[:, None] * run_dir[None, :] + w_val[:, None] * lat_dir[None, :]
    uvz = np.column_stack([uv, z_val])
    v_xyz = fc.frame.uvz_to_xyz(uvz)
    if signed_mesh_volume(v_xyz, tri) < 0:
        tri = tri[:, [0, 2, 1]]
    return v_xyz, tri


def _fit_and_build_flight(
    sub_uvz: np.ndarray,
    fc: FloorCeilingResult,
    allowed_xy: Polygon | MultiPolygon,
    flight_idx: int,
) -> tuple[MeshPart, _FlightGeom] | None:
    """Fit a single straight stepped flight to sub_uvz and return watertight MeshPart + geometry."""
    if len(sub_uvz) < 400:
        return None
    z = sub_uvz[:, 2]
    z_lo_raw, z_hi_raw = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    if z_hi_raw - z_lo_raw < 0.45:
        return None

    # Regression of horizontal (u, v) vs z gives exact uphill run direction
    zc = z - float(np.mean(z))
    zz = float(np.dot(zc, zc))
    if zz < 1e-6:
        return None
    uc = sub_uvz[:, 0] - float(np.mean(sub_uvz[:, 0]))
    vc = sub_uvz[:, 1] - float(np.mean(sub_uvz[:, 1]))
    raw_dir = np.array([float(np.dot(uc, zc)) / zz, float(np.dot(vc, zc)) / zz], dtype=np.float64)
    if float(np.linalg.norm(raw_dir)) < 0.15:
        return None

    run_dir = _snap_dir_to_room_axis(raw_dir, tol_deg=18.0)
    lat_dir = np.array([-run_dir[1], run_dir[0]], dtype=np.float64)

    s_vals = sub_uvz[:, :2] @ run_dir
    w_vals = sub_uvz[:, :2] @ lat_dir

    # Verify monotonicity along run_dir
    s_std = float(np.std(s_vals))
    z_std = float(np.std(z))
    if s_std < 0.15 or z_std < 0.12:
        return None
    corr = float(np.corrcoef(s_vals, z)[0, 1])
    if corr < 0.55:
        return None

    s_lo, s_hi = float(np.percentile(s_vals, 2)), float(np.percentile(s_vals, 98))
    w_lo, w_hi = float(np.percentile(w_vals, 3)), float(np.percentile(w_vals, 97))

    # Snap z_lo / z_hi to detected floor levels when close (leaving 4mm vertical clearance!)
    z_lo = z_lo_raw
    z_hi = z_hi_raw
    for flv in fc.floor_levels:
        if abs(z_lo - flv.z_top) <= 0.30:
            z_lo = flv.z_top + 0.004
        if abs(z_hi - flv.z_top) <= 0.30:
            z_hi = flv.z_top - 0.004
    z_lo = max(z_lo, fc.floor_z + 0.004)
    if fc.ceiling_z is not None:
        z_hi = min(z_hi, fc.ceiling_z - 0.05)
    if z_hi - z_lo < 0.40:
        return None

    inscribed = _inscribe_sw_rect(s_lo, s_hi, w_lo, w_hi, run_dir, lat_dir, allowed_xy)
    if inscribed is None:
        return None
    s_lo, s_hi, w_lo, w_hi = inscribed

    n_steps = int(np.clip(round((z_hi - z_lo) / 0.175), 3, 30))
    try:
        v_xyz, tri = _build_stepped_flight_mesh(fc, run_dir, lat_dir, s_lo, s_hi, w_lo, w_hi, z_lo, z_hi, n_steps)
    except ValueError:
        return None

    poly_uv = _sw_rect_poly(s_lo, s_hi, w_lo, w_hi, run_dir, lat_dir)
    pid = f"stair_flight_{flight_idx:02d}"
    part = MeshPart(
        part_id=pid,
        class_id=13,
        role="stair_flight",
        vertices=v_xyz,
        triangles=tri,
        evidence_kind="measured",
        metadata={
            "point_count": int(len(sub_uvz)),
            "n_steps": n_steps,
            "run_m": round(s_hi - s_lo, 3),
            "width_m": round(w_hi - w_lo, 3),
            "rise_m": round(z_hi - z_lo, 3),
            "z_span": [round(z_lo, 4), round(z_hi, 4)],
            "run_dir_uv": [round(float(run_dir[0]), 4), round(float(run_dir[1]), 4)],
            "monotonic_corr": round(corr, 4),
        },
    )
    geom = _FlightGeom(
        part_id=pid,
        poly_uv=poly_uv,
        run_dir=run_dir,
        lat_dir=lat_dir,
        s_lo=s_lo,
        s_hi=s_hi,
        w_lo=w_lo,
        w_hi=w_hi,
        z_lo=z_lo,
        z_hi=z_hi,
        n_steps=n_steps,
    )
    return part, geom


def build_stairs_and_railings(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    interior_wall_polys_uv: list[Polygon],
    column_polys_uv: list[Polygon],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
) -> tuple[list[MeshPart], list[tuple[Polygon, float, float]], list[tuple[Polygon, float, float]], dict[str, Any]]:
    """Build watertight stepped stairs (cls 13) + open handrail/baluster railings (cls 14) per B16.

    Returns:
      - parts: list of closed manifold MeshPart (class 13 and 14)
      - stair_well_cutouts: list of (poly_uv, z_lo, z_hi) to cut upper floor slabs cleanly
      - beam_blocker_bands: list of (poly_uv, z_lo, z_hi) so overhead beams never pierce stairs/railings
      - audit: structured report dictionary
    """
    uvz = fc.frame.pts_to_uvz(points_local)
    stair_pts = uvz[work_labels == 13]
    rail_pts = uvz[work_labels == 14]

    if len(stair_pts) < 500 and len(rail_pts) < 500:
        return [], [], [], {
            "status": "N/A",
            "stair_points_after_working_labels": int(len(stair_pts)),
            "railing_points_after_working_labels": int(len(rail_pts)),
            "emitted_stair_flights": 0,
            "emitted_railing_parts": 0,
            "note": "No qualifying stair or railing evidence in scene (any tiny floating specks were quarantined in WL-B).",
        }

    # Base horizontal clearance domain inside room, away from walls & columns
    allowed_xy = shell.inner_uv.buffer(-0.008)
    static_blockers = [*interior_wall_polys_uv, *column_polys_uv]
    if static_blockers:
        allowed_xy = allowed_xy.difference(unary_union([p.buffer(0.010) for p in static_blockers])).buffer(0)

    parts: list[MeshPart] = []
    flights: list[_FlightGeom] = []
    stair_well_cutouts: list[tuple[Polygon, float, float]] = []
    beam_blocker_bands: list[tuple[Polygon, float, float]] = []
    flight_logs: list[dict[str, Any]] = []
    railing_logs: list[dict[str, Any]] = []

    # 1. Build stepped stair flights (class 13)
    placed_flight_union: Polygon | MultiPolygon | None = None
    if len(stair_pts) >= 500:
        comps = _voxel_connected_components_3d(stair_pts, voxel_size=0.22)
        for comp in comps:
            if len(comp) < 500:
                continue
            sub = stair_pts[comp]
            cur_allowed = allowed_xy
            if placed_flight_union is not None and not placed_flight_union.is_empty:
                cur_allowed = cur_allowed.difference(placed_flight_union.buffer(0.015)).buffer(0)
            res = _fit_and_build_flight(sub, fc, cur_allowed, len(flights))
            if res is None:
                # Could be a 2-flight L/U turn: try splitting into lower & upper half-flights at median Z
                z_med = float(np.median(sub[:, 2]))
                for half in (sub[sub[:, 2] <= z_med], sub[sub[:, 2] > z_med]):
                    cur_allowed = allowed_xy
                    if placed_flight_union is not None and not placed_flight_union.is_empty:
                        cur_allowed = cur_allowed.difference(placed_flight_union.buffer(0.015)).buffer(0)
                    hres = _fit_and_build_flight(half, fc, cur_allowed, len(flights))
                    if hres is not None:
                        fpart, fgeom = hres
                        parts.append(fpart)
                        flights.append(fgeom)
                        placed_flight_union = (
                            fgeom.poly_uv
                            if placed_flight_union is None
                            else unary_union([placed_flight_union, fgeom.poly_uv])
                        )
                        stair_well_cutouts.append((fgeom.poly_uv.buffer(0.04), fgeom.z_lo - 0.05, fgeom.z_hi + 0.15))
                        beam_blocker_bands.append((fgeom.poly_uv.buffer(0.03), fgeom.z_lo - 0.05, fgeom.z_hi + 0.15))
                        flight_logs.append(fpart.metadata | {"part_id": fpart.part_id})
                continue

            fpart, fgeom = res
            parts.append(fpart)
            flights.append(fgeom)
            placed_flight_union = (
                fgeom.poly_uv
                if placed_flight_union is None
                else unary_union([placed_flight_union, fgeom.poly_uv])
            )
            stair_well_cutouts.append((fgeom.poly_uv.buffer(0.04), fgeom.z_lo - 0.05, fgeom.z_hi + 0.15))
            beam_blocker_bands.append((fgeom.poly_uv.buffer(0.03), fgeom.z_lo - 0.05, fgeom.z_hi + 0.15))
            flight_logs.append(fpart.metadata | {"part_id": fpart.part_id})

    # 2. Build open handrail + vertical baluster railings (class 14, strictly B16: NO solid sheets!)
    rail_allowed_base = shell.inner_uv.buffer(-0.045)
    if static_blockers:
        rail_allowed_base = rail_allowed_base.difference(unary_union([p.buffer(0.045) for p in static_blockers])).buffer(0)

    placed_rail_bands: list[tuple[Polygon, float, float]] = []
    rail_group_idx = 0

    if len(rail_pts) >= 500:
        r_comps = _voxel_connected_components_3d(rail_pts, voxel_size=0.18)
        for comp in r_comps:
            if len(comp) < 400:
                continue
            sub = rail_pts[comp]
            mean_uv = sub[:, :2].mean(axis=0)
            centered = sub[:, :2] - mean_uv
            cov = centered.T @ centered / max(len(sub), 1)
            eigvals, eigvecs = np.linalg.eigh(cov)
            r_dir = _snap_dir_to_room_axis(eigvecs[:, 1], tol_deg=15.0)
            r_lat = np.array([-r_dir[1], r_dir[0]], dtype=np.float64)

            s_all = sub[:, :2] @ r_dir
            w_all = sub[:, :2] @ r_lat
            s_lo, s_hi = float(np.percentile(s_all, 2)), float(np.percentile(s_all, 98))
            if s_hi - s_lo < 0.60:
                continue
            w_mid = float(np.median(w_all))

            # Fit top handrail & bottom baluster envelope along 0.40m bins
            n_bins = max(2, int(np.ceil((s_hi - s_lo) / 0.40)))
            edges = np.linspace(s_lo, s_hi, n_bins + 1)
            bin_s: list[float] = []
            bin_zt: list[float] = []
            bin_zb: list[float] = []
            for bi in range(n_bins):
                m = (s_all >= edges[bi]) & (s_all <= edges[bi + 1])
                if int(np.sum(m)) >= 6:
                    bin_s.append(0.5 * (edges[bi] + edges[bi + 1]))
                    bin_zt.append(float(np.percentile(sub[m, 2], 92)))
                    bin_zb.append(float(np.percentile(sub[m, 2], 8)))
            if len(bin_s) < 2:
                continue

            bs = np.asarray(bin_s, dtype=np.float64)
            bzt = np.asarray(bin_zt, dtype=np.float64)
            bzb = np.asarray(bin_zb, dtype=np.float64)
            slope_t, inter_t = np.polyfit(bs, bzt, 1)
            if abs(float(slope_t) * (s_hi - s_lo)) < 0.30:
                slope_t = 0.0
                inter_t = float(np.median(bzt))
            slope_b, inter_b = np.polyfit(bs, bzb, 1)
            if abs(float(slope_t)) < 1e-6:
                slope_b = 0.0
                inter_b = float(np.median(bzb))

            def z_top_fn(sv: float, st: float = float(slope_t), it: float = float(inter_t)) -> float:
                val = st * sv + it
                if fc.ceiling_z is not None:
                    val = min(val, fc.ceiling_z - 0.06)
                return val

            def z_bot_fn(sv: float, sb_: float = float(slope_b), ib_: float = float(inter_b)) -> float:
                return max(fc.floor_z + 0.008, sb_ * sv + ib_)

            z_cluster_lo = min(z_bot_fn(s_lo), z_bot_fn(s_hi))
            z_cluster_hi = max(z_top_fn(s_lo), z_top_fn(s_hi))
            if z_cluster_hi - z_cluster_lo < 0.35:
                continue

            # Subtract any already-placed railings overlapping in Z
            cur_rail_allowed = rail_allowed_base
            ov_rails = [
                p.buffer(0.045)
                for p, zb, zt in placed_rail_bands
                if zb <= z_cluster_hi + 0.05 and zt >= z_cluster_lo - 0.05
            ]
            if ov_rails:
                cur_rail_allowed = cur_rail_allowed.difference(unary_union(ov_rails)).buffer(0)

            p_start = r_dir * s_lo + r_lat * w_mid
            p_end = r_dir * s_hi + r_lat * w_mid
            line = LineString([p_start, p_end]).intersection(cur_rail_allowed)
            segs = (
                list(line.geoms)
                if isinstance(line, MultiLineString)
                else ([line] if isinstance(line, LineString) else [])
            )

            for seg in segs:
                if seg.length < 0.50:
                    continue
                c_pts = np.asarray(list(seg.coords), dtype=np.float64)
                sa_raw = float(c_pts[0] @ r_dir)
                sb_raw = float(c_pts[-1] @ r_dir)
                sa, sb = min(sa_raw, sb_raw) + 0.02, max(sa_raw, sb_raw) - 0.02
                if sb - sa < 0.45:
                    continue

                # Compute safe handrail endpoints strictly above stairs & clear of upper floor/ceiling slabs
                hr_w = 0.055
                hr_h = 0.055
                zt_a = z_top_fn(sa)
                zt_b = z_top_fn(sb)

                # Ensure handrail stays strictly above any stair flight it crosses in XY
                for t_frac in np.linspace(0.0, 1.0, 9):
                    sc = sa + t_frac * (sb - sa)
                    uv_c = r_dir * sc + r_lat * w_mid
                    pt_c = Point(float(uv_c[0]), float(uv_c[1]))
                    for fl in flights:
                        if fl.poly_uv.buffer(0.04).contains(pt_c):
                            min_hr_z = fl.step_top_z_at(float(uv_c[0]), float(uv_c[1])) + 0.12
                            cur_z = zt_a + t_frac * (zt_b - zt_a)
                            if cur_z < min_hr_z:
                                lift = min_hr_z - cur_z
                                zt_a += lift
                                zt_b += lift

                # Check if handrail box collides vertically with any upper floor slab or secondary ceiling
                seg_uv_poly = LineString([r_dir * sa + r_lat * w_mid, r_dir * sb + r_lat * w_mid]).buffer(0.045)
                hr_z_lo = min(zt_a, zt_b) - hr_h
                hr_z_hi = max(zt_a, zt_b)
                is_sloped = abs(zt_b - zt_a) >= 0.30
                slab_clash = False
                for clv in fc.ceiling_levels:
                    if clv.polygon_uv.intersects(seg_uv_poly):
                        if hr_z_lo < clv.z_bottom + cfg.ceiling_thickness_m + 0.005 and hr_z_hi > clv.z_bottom - 0.005:
                            slab_clash = True
                if not is_sloped:
                    for flv in fc.floor_levels:
                        if flv.is_primary:
                            continue
                        if flv.polygon_uv.intersects(seg_uv_poly):
                            if hr_z_lo < flv.z_top + 0.005 and hr_z_hi > flv.z_top - cfg.floor_thickness_m - 0.005:
                                slab_clash = True
                if slab_clash:
                    continue

                # Build top handrail oriented box
                p0_uvz = np.array([*(r_dir * sa + r_lat * w_mid), zt_a - 0.5 * hr_h], dtype=np.float64)
                p1_uvz = np.array([*(r_dir * sb + r_lat * w_mid), zt_b - 0.5 * hr_h], dtype=np.float64)
                p0_xyz = fc.frame.uvz_to_xyz(p0_uvz[None, :])[0]
                p1_xyz = fc.frame.uvz_to_xyz(p1_uvz[None, :])[0]
                axis_l = p1_xyz - p0_xyz
                hr_len = float(np.linalg.norm(axis_l))
                if hr_len < 0.45:
                    continue
                axis_l /= hr_len
                lat_xyz = fc.frame.uvz_to_xyz(np.array([[r_lat[0], r_lat[1], 0.0]]))[0]
                lat_xyz /= max(float(np.linalg.norm(lat_xyz)), 1e-9)
                up_perp = np.cross(axis_l, lat_xyz)
                up_perp /= max(float(np.linalg.norm(up_perp)), 1e-9)
                if up_perp[2] < 0:
                    up_perp = -up_perp

                hr_v, hr_t = make_oriented_box(
                    0.5 * (p0_xyz + p1_xyz),
                    axis_l,
                    lat_xyz,
                    up_perp,
                    (hr_len, hr_w, hr_h),
                )
                hr_pid = f"railing_{rail_group_idx:02d}_handrail"
                parts.append(
                    MeshPart(
                        part_id=hr_pid,
                        class_id=14,
                        role="railing_handrail",
                        vertices=hr_v,
                        triangles=hr_t,
                        evidence_kind="measured",
                        metadata={
                            "length_m": round(float(hr_len), 3),
                            "z_ends": [round(float(zt_a), 4), round(float(zt_b), 4)],
                            "sloped": bool(abs(float(zt_b) - float(zt_a)) >= 0.30),
                        },
                    )
                )

                # Build open vertical balusters strictly below the handrail bar (zero penetration!)
                bal_w = 0.04
                slope_eff = abs(zt_b - zt_a) / max(sb - sa, 1e-6)
                n_bal = max(2, int(np.floor((sb - sa) / 0.45)) + 1)
                s_posts = np.linspace(sa + 0.04, sb - 0.04, n_bal)
                bal_count = 0
                for b_idx, sp in enumerate(s_posts):
                    t_f = (sp - sa) / max(sb - sa, 1e-6)
                    uv_p = r_dir * sp + r_lat * w_mid
                    # Generous top clearance below sloped handrail bottom
                    hr_bot_local = (zt_a + t_f * (zt_b - zt_a)) - hr_h * np.hypot(1.0, slope_eff) - 0.015 - 0.5 * bal_w * slope_eff
                    zb = z_bot_fn(float(sp))

                    # Lift baluster bottom strictly above any overlapping stair step tread!
                    b_corners = [
                        uv_p + r_dir * ds_c + r_lat * dw_c
                        for ds_c in (-0.5 * bal_w, 0.5 * bal_w)
                        for dw_c in (-0.5 * bal_w, 0.5 * bal_w)
                    ]
                    b_poly = Polygon(b_corners)
                    for fl in flights:
                        if fl.poly_uv.buffer(0.015).intersects(b_poly):
                            max_step_z = max(fl.step_top_z_at(float(c[0]), float(c[1])) for c in b_corners)
                            zb = max(zb, max_step_z + 0.010)

                    # Keep baluster clear of upper floor slabs & secondary ceilings
                    for flv in fc.floor_levels:
                        if flv.is_primary:
                            continue
                        if flv.polygon_uv.buffer(0.01).intersects(b_poly):
                            fbot = flv.z_top - cfg.floor_thickness_m
                            if zb < flv.z_top + 0.005 and hr_bot_local > fbot - 0.005:
                                if hr_bot_local - flv.z_top > 0.25:
                                    zb = max(zb, flv.z_top + 0.006)
                                else:
                                    hr_bot_local = min(hr_bot_local, fbot - 0.006)
                    for clv in fc.ceiling_levels:
                        if clv.polygon_uv.buffer(0.01).intersects(b_poly):
                            ctop = clv.z_bottom + cfg.ceiling_thickness_m
                            if zb < ctop + 0.005 and hr_bot_local > clv.z_bottom - 0.005:
                                hr_bot_local = min(hr_bot_local, clv.z_bottom - 0.006)

                    bh = hr_bot_local - zb
                    if bh < 0.18:
                        continue
                    c_xyz = fc.frame.uvz_to_xyz(np.array([[uv_p[0], uv_p[1], 0.5 * (zb + hr_bot_local)]]))[0]
                    dir_xyz = fc.frame.uvz_to_xyz(np.array([[r_dir[0], r_dir[1], 0.0]]))[0]
                    dir_xyz /= max(float(np.linalg.norm(dir_xyz)), 1e-9)
                    bv, bt = make_oriented_box(
                        c_xyz,
                        dir_xyz,
                        lat_xyz,
                        np.array([0.0, 0.0, 1.0]),
                        (bal_w, bal_w, bh),
                    )
                    parts.append(
                        MeshPart(
                            part_id=f"railing_{rail_group_idx:02d}_post_{b_idx:02d}",
                            class_id=14,
                            role="railing_baluster",
                            vertices=bv,
                            triangles=bt,
                            evidence_kind="measured",
                            metadata={"z_span": [round(zb, 4), round(hr_bot_local, 4)]},
                        )
                    )
                    bal_count += 1

                placed_rail_bands.append((seg_uv_poly, z_cluster_lo, max(zt_a, zt_b)))
                beam_blocker_bands.append((seg_uv_poly.buffer(0.02), z_cluster_lo, max(zt_a, zt_b) + 0.05))
                if is_sloped:
                    # A sloped stair railing passes through the stairwell opening — include it in stairwell cutout!
                    stair_well_cutouts.append((seg_uv_poly.buffer(0.03), z_cluster_lo - 0.05, max(zt_a, zt_b) + 0.05))
                railing_logs.append({
                    "group_id": f"railing_{rail_group_idx:02d}",
                    "length_m": round(float(hr_len), 3),
                    "baluster_count": int(bal_count),
                    "sloped": bool(abs(float(zt_b) - float(zt_a)) >= 0.30),
                    "z_ends": [round(float(zt_a), 4), round(float(zt_b), 4)],
                })
                rail_group_idx += 1

    emitted_stairs = len(flights)
    emitted_rails = sum(1 for p in parts if p.class_id == 14)
    return parts, stair_well_cutouts, beam_blocker_bands, {
        "status": "PASS" if (emitted_stairs > 0 or emitted_rails > 0) else "N/A",
        "stair_points_after_working_labels": int(len(stair_pts)),
        "railing_points_after_working_labels": int(len(rail_pts)),
        "emitted_stair_flights": emitted_stairs,
        "emitted_railing_groups": rail_group_idx,
        "emitted_railing_parts": emitted_rails,
        "flights": flight_logs,
        "railings": railing_logs,
    }


def evaluate_stairs_and_railings(
    points_local: np.ndarray,
    work_labels: np.ndarray,
) -> tuple[list[MeshPart], dict[str, Any]]:
    """Backward-compatible lightweight evidence check (used only by legacy unit tests without shell)."""
    stair_pts = points_local[work_labels == 13]
    rail_pts = points_local[work_labels == 14]
    if len(stair_pts) == 0 and len(rail_pts) == 0:
        return [], {
            "status": "N/A",
            "stair_points_after_working_labels": 0,
            "railing_points_after_working_labels": 0,
            "emitted_parts": 0,
            "note": "No valid stair or railing evidence in scene (any tiny floating specks were quarantined in WL-B).",
        }
    return [], {
        "status": "NOT_CHECKED",
        "stair_points_after_working_labels": int(len(stair_pts)),
        "railing_points_after_working_labels": int(len(rail_pts)),
        "emitted_parts": 0,
        "note": "Use build_stairs_and_railings() with full shell context to build stairs & railings.",
    }
