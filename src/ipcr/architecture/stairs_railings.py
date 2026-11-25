"""B16 Evidence-gated watertight stepped stairs, stair-well cutouts, and open handrail/baluster railings.

Only triggers when genuine multi-step stair (class 13) or continuous railing (class 14)
evidence survives working_labels; isolated specks quarantined by WL-B produce 0 parts with N/A audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon, box
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
    """Shrink any colliding side of a stair flight rectangle so it lies 100% inside allowed_xy.

    Uses greedy outside-area reduction so angled/notched stairwell corners trim the minimal offending side
    instead of oscillating or collapsing the flight.
    """
    step = 0.03
    for _ in range(160):
        if s_hi - s_lo < min_run_m or w_hi - w_lo < min_width_m:
            return None
        rect = _sw_rect_poly(s_lo, s_hi, w_lo, w_hi, run_dir, lat_dir)
        if allowed_xy.contains(rect):
            return s_lo, s_hi, w_lo, w_hi

        out_area = max(0.0, float(rect.area) - float(rect.intersection(allowed_xy).area))
        if out_area <= 1e-5:
            return s_lo, s_hi, w_lo, w_hi

        # Try trimming each of the 4 sides and pick the move that removes the highest fraction of outside area
        best_move = None
        best_score = -1.0
        candidates = [
            (s_lo + step, s_hi, w_lo, w_hi),
            (s_lo, s_hi - step, w_lo, w_hi),
            (s_lo, s_hi, w_lo + step, w_hi),
            (s_lo, s_hi, w_lo, w_hi - step),
        ]
        for cs0, cs1, cw0, cw1 in candidates:
            if cs1 - cs0 < min_run_m or cw1 - cw0 < min_width_m:
                continue
            crect = _sw_rect_poly(cs0, cs1, cw0, cw1, run_dir, lat_dir)
            c_in = float(crect.intersection(allowed_xy).area)
            c_out = max(0.0, float(crect.area) - c_in)
            removed_out = out_area - c_out
            if removed_out > 1e-6:
                lost_total = max(float(rect.area) - float(crect.area), 1e-6)
                # Ratio of removed outside area to total removed area (1.0 = pure overhang trim!)
                score = (removed_out / lost_total) + 0.15 * removed_out
                if score > best_score:
                    best_score = score
                    best_move = (cs0, cs1, cw0, cw1)

        if best_move is not None:
            s_lo, s_hi, w_lo, w_hi = best_move
        else:
            # Interior obstacle: gently trim the longer span
            if (s_hi - s_lo) >= (w_hi - w_lo):
                s_lo += step
                s_hi -= step
            else:
                w_lo += step
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


def _estimate_flight_dir_and_corr(
    sub_uvz: np.ndarray,
    *,
    snap_tol_deg: float = 8.0,
) -> tuple[np.ndarray | None, float]:
    """Estimate uphill run direction (unit 2D UV vector) and linear (s, z) correlation."""
    if len(sub_uvz) < 200:
        return None, -1.0
    z = sub_uvz[:, 2]
    zc = z - float(np.mean(z))
    zz = float(np.dot(zc, zc))
    if zz < 1e-6:
        return None, -1.0
    uc = sub_uvz[:, 0] - float(np.mean(sub_uvz[:, 0]))
    vc = sub_uvz[:, 1] - float(np.mean(sub_uvz[:, 1]))
    raw_dir = np.array([float(np.dot(uc, zc)) / zz, float(np.dot(vc, zc)) / zz], dtype=np.float64)
    if float(np.linalg.norm(raw_dir)) < 0.15:
        return None, -1.0
    run_dir = _snap_dir_to_room_axis(raw_dir, tol_deg=snap_tol_deg)
    s_vals = sub_uvz[:, :2] @ run_dir
    if float(np.std(s_vals)) < 0.15 or float(np.std(z)) < 0.12:
        return None, -1.0
    corr = float(np.corrcoef(s_vals, z)[0, 1])
    return run_dir, corr


def _detect_turn_split(sub_uvz: np.ndarray, *, min_half_corr: float = 0.78) -> tuple[float, float] | None:
    """Detect a turn (>=63 deg L-turn or U-turn) inside a stair component and return (z_turn_split, z_land_top).

    Straight single flights have dot(dir_lo, dir_hi) ~ +1.0 and NEVER split.
    L-turn (dot ~ 0.0) and U-turn (dot ~ -1.0) staircases split cleanly at the turn elevation.
    """
    if len(sub_uvz) < 1500:
        return None
    z = sub_uvz[:, 2]
    z_span = float(np.percentile(z, 98) - np.percentile(z, 2))
    if z_span < 1.15:
        return None
    _, g_corr = _estimate_flight_dir_and_corr(sub_uvz, snap_tol_deg=8.0)

    best_score = -1.0
    best_zt: float | None = None
    for q in (25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75):
        zt = float(np.percentile(z, q))
        lo = sub_uvz[z < zt - 0.10]
        hi = sub_uvz[z > zt + 0.10]
        if len(lo) < 400 or len(hi) < 400:
            continue
        if float(np.percentile(lo[:, 2], 98) - np.percentile(lo[:, 2], 2)) < 0.50:
            continue
        if float(np.percentile(hi[:, 2], 98) - np.percentile(hi[:, 2], 2)) < 0.50:
            continue
        d_lo, c_lo = _estimate_flight_dir_and_corr(lo, snap_tol_deg=8.0)
        d_hi, c_hi = _estimate_flight_dir_and_corr(hi, snap_tol_deg=8.0)
        if d_lo is None or d_hi is None:
            continue
        if c_lo < min_half_corr or c_hi < min_half_corr:
            continue
        dot_lh = float(np.dot(d_lo, d_hi))
        if dot_lh > 0.45:
            continue
        # Only compare against global corr for mild turns; true >=75-deg L/U-turns often have a
        # spurious high lateral g_corr across the two flights!
        if dot_lh > 0.15 and 0.5 * (c_lo + c_hi) <= max(g_corr, 0.0) + 0.03:
            continue
        score = c_lo + c_hi
        if score > best_score:
            best_score = score
            best_zt = zt

    if best_zt is None:
        return None

    # Refine z_land_top from the horizontal turn plateau near best_zt
    mid_z = z[(z >= best_zt - 0.25) & (z <= best_zt + 0.25)]
    z_land_top = best_zt
    if len(mid_z) >= 80:
        lo_m, hi_m = float(mid_z.min()), float(mid_z.max())
        if hi_m - lo_m >= 0.05:
            bins = np.arange(lo_m - 0.02, hi_m + 0.06, 0.04)
            hist, edges = np.histogram(mid_z, bins=bins)
            pk = int(np.argmax(hist))
            zc = 0.5 * (edges[pk] + edges[pk + 1])
            band = mid_z[np.abs(mid_z - zc) <= 0.06]
            z_land_top = float(np.median(band)) if len(band) else zc
        else:
            z_land_top = float(np.median(mid_z))
    return best_zt, z_land_top


def _split_stair_turns_recursive(
    sub_uvz: np.ndarray,
    *,
    z_lo_bound: float | None = None,
    z_hi_bound: float | None = None,
    depth: int = 0,
    max_depth: int = 2,
) -> list[tuple[np.ndarray, float | None, float | None]]:
    """Recursively decompose a multi-turn stair component (L-turn, 2-flight U-turn, or 3-flight U-turn)
    into an ordered sequence of straight flight segments [(pts, z_lo_override, z_hi_override), ...].
    """
    if depth >= max_depth:
        return [(sub_uvz, z_lo_bound, z_hi_bound)]
    min_corr = 0.78 if depth == 0 else 0.84
    turn = _detect_turn_split(sub_uvz, min_half_corr=min_corr)
    if turn is None:
        return [(sub_uvz, z_lo_bound, z_hi_bound)]

    z_turn_split, z_land_top = turn
    lo_pts = sub_uvz[sub_uvz[:, 2] < z_land_top - 0.08]
    if len(lo_pts) < 400:
        lo_pts = sub_uvz[sub_uvz[:, 2] < z_turn_split - 0.06]
    hi_pts = sub_uvz[sub_uvz[:, 2] > z_land_top + 0.08]
    if len(hi_pts) < 400:
        hi_pts = sub_uvz[sub_uvz[:, 2] > z_turn_split + 0.06]

    left = _split_stair_turns_recursive(
        lo_pts, z_lo_bound=z_lo_bound, z_hi_bound=z_land_top, depth=depth + 1, max_depth=max_depth
    )
    right = _split_stair_turns_recursive(
        hi_pts, z_lo_bound=z_land_top + 0.004, z_hi_bound=z_hi_bound, depth=depth + 1, max_depth=max_depth
    )
    return left + right


def _local_ceiling_above_poly(poly_uv: Polygon, fc: FloorCeilingResult) -> float | None:
    """Return the local ceiling bottom elevation directly covering poly_uv (handles multi-height stairwells)."""
    for clv in fc.ceiling_levels:
        if (not clv.is_primary) and fc.ceiling_z is not None and clv.z_bottom > fc.ceiling_z + 0.65:
            if clv.polygon_uv.buffer(0.25).intersects(poly_uv):
                return clv.z_bottom
    return fc.ceiling_z


def _build_cloud_footprint_uv(
    pts_uv: np.ndarray,
    *,
    cell: float = 0.10,
    pad_m: float = 0.08,
    close_iters: int = 1,
) -> Polygon | MultiPolygon | None:
    """Build exact 2D point-cloud occupancy polygon in UV space (G10b strict boundary containment).

    Guarantees reconstructed stairs and landings NEVER protrude more than `pad_m` outside real class-13 points!
    """
    if len(pts_uv) < 40:
        return None
    u0 = float(pts_uv[:, 0].min()) - 2.0 * cell
    v0 = float(pts_uv[:, 1].min()) - 2.0 * cell
    u1 = float(pts_uv[:, 0].max()) + 2.0 * cell
    v1 = float(pts_uv[:, 1].max()) + 2.0 * cell
    nu = int(np.ceil((u1 - u0) / cell)) + 2
    nv = int(np.ceil((v1 - v0) / cell)) + 2
    if nu > 400 or nv > 400:
        return None
    counts = np.zeros((nu, nv), dtype=np.int32)
    iu = np.clip(np.floor((pts_uv[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
    iv = np.clip(np.floor((pts_uv[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
    np.add.at(counts, (iu, iv), 1)
    mask = counts >= 2
    if close_iters > 0:
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=close_iters)
    mask = ndimage.binary_fill_holes(mask)
    ys, xs = np.nonzero(mask.T)  # xs=iu, ys=iv
    if len(xs) == 0:
        return None
    rects = [
        box(u0 + int(ix) * cell, v0 + int(iy) * cell, u0 + (int(ix) + 1) * cell, v0 + (int(iy) + 1) * cell)
        for ix, iy in zip(xs, ys)
    ]
    poly = unary_union(rects).buffer(pad_m).buffer(0)
    if poly.is_empty:
        return None
    return poly


def _fit_and_build_flight(
    sub_uvz: np.ndarray,
    fc: FloorCeilingResult,
    allowed_xy: Polygon | MultiPolygon,
    flight_idx: int,
    *,
    z_lo_override: float | None = None,
    z_hi_override: float | None = None,
) -> tuple[MeshPart, _FlightGeom] | None:
    """Fit a single straight stepped flight to sub_uvz and return watertight MeshPart + geometry."""
    if len(sub_uvz) < 400:
        return None
    z = sub_uvz[:, 2]
    z_lo_raw, z_hi_raw = float(np.percentile(z, 2)), float(np.percentile(z, 98))
    if z_hi_raw - z_lo_raw < 0.45:
        return None

    run_dir, corr = _estimate_flight_dir_and_corr(sub_uvz, snap_tol_deg=8.0)
    if run_dir is None or corr < 0.55:
        return None
    lat_dir = np.array([-run_dir[1], run_dir[0]], dtype=np.float64)

    s_vals = sub_uvz[:, :2] @ run_dir
    w_vals = sub_uvz[:, :2] @ lat_dir

    s_lo, s_hi = float(np.percentile(s_vals, 2)), float(np.percentile(s_vals, 98))
    w_lo, w_hi = float(np.percentile(w_vals, 3)), float(np.percentile(w_vals, 97))

    # Snap z_lo / z_hi to detected floor levels when close (leaving 4mm vertical clearance!)
    z_lo = z_lo_override if z_lo_override is not None else z_lo_raw
    z_hi = z_hi_override if z_hi_override is not None else z_hi_raw
    for flv in fc.floor_levels:
        if z_lo_override is None and abs(z_lo - flv.z_top) <= 0.30:
            z_lo = flv.z_top + 0.004
        if z_hi_override is None and abs(z_hi - flv.z_top) <= 0.30:
            z_hi = flv.z_top - 0.004
    z_lo = max(z_lo, fc.floor_z + 0.004)

    raw_poly = _sw_rect_poly(s_lo, s_hi, w_lo, w_hi, run_dir, lat_dir)
    local_ceil = _local_ceiling_above_poly(raw_poly, fc)
    if local_ceil is not None and z_hi_raw <= local_ceil + 0.25:
        z_hi = min(z_hi, local_ceil - 0.05)
    if z_hi - z_lo < 0.40:
        return None

    # G10b: Strictly constrain flight rectangle inside BOTH room clearance AND actual class-13 point cloud footprint!
    cloud_dom = _build_cloud_footprint_uv(sub_uvz[:, :2], cell=0.10, pad_m=0.08, close_iters=1)
    effective_allowed = allowed_xy.intersection(cloud_dom).buffer(0) if cloud_dom is not None else allowed_xy
    if effective_allowed.is_empty:
        return None

    inscribed = _inscribe_sw_rect(s_lo, s_hi, w_lo, w_hi, run_dir, lat_dir, effective_allowed)
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


def _build_stair_landing(
    land_pts: np.ndarray,
    fl1: _FlightGeom,
    fl2: _FlightGeom,
    z_land_top: float,
    fc: FloorCeilingResult,
    allowed_xy: Polygon | MultiPolygon,
    landing_idx: int,
) -> tuple[MeshPart, Polygon, float, float] | None:
    """Build a closed watertight horizontal mid-landing connecting fl1 and fl2 with 6mm clearance."""
    if len(land_pts) < 120:
        return None
    r_dir = fl1.run_dir
    l_dir = fl1.lat_dir
    s_l = land_pts[:, :2] @ r_dir
    w_l = land_pts[:, :2] @ l_dir
    s_lo_raw = float(np.percentile(s_l, 2))
    s_hi_raw = float(np.percentile(s_l, 98))
    w_lo_raw = float(np.percentile(w_l, 2))
    w_hi_raw = float(np.percentile(w_l, 98))

    # Bridge across the combined lateral span of fl1 and fl2 at the turn end of fl1
    fl2_corners = np.asarray(fl2.poly_uv.exterior.coords, dtype=np.float64)[:, :2]
    fl2_w = fl2_corners @ l_dir
    fl2_s = fl2_corners @ r_dir
    w_lo = min(w_lo_raw, fl1.w_lo, float(fl2_w.min()))
    w_hi = max(w_hi_raw, fl1.w_hi, float(fl2_w.max()))
    # Start landing right at the top of fl1 (or measured landing start) and extend to back of landing
    s_lo = min(s_lo_raw, fl1.s_hi - 0.05)
    s_hi = max(s_hi_raw, fl1.s_hi + 0.65, float(fl2_s.max()) if float(np.dot(fl1.run_dir, fl2.run_dir)) < -0.5 else s_hi_raw)
    if s_hi - s_lo < 0.45 or w_hi - w_lo < 0.65:
        return None

    rect = _sw_rect_poly(s_lo, s_hi, w_lo, w_hi, r_dir, l_dir)
    land_cloud_dom = _build_cloud_footprint_uv(land_pts[:, :2], cell=0.10, pad_m=0.10, close_iters=1)
    clipped = rect.intersection(allowed_xy)
    if land_cloud_dom is not None:
        clipped = clipped.intersection(land_cloud_dom)
    clipped = clipped.difference(unary_union([fl1.poly_uv.buffer(0.006), fl2.poly_uv.buffer(0.006)])).buffer(0)
    geoms = (
        list(clipped.geoms)
        if isinstance(clipped, MultiPolygon)
        else ([clipped] if isinstance(clipped, Polygon) else [])
    )
    valid = [g for g in geoms if (not g.is_empty) and g.area >= 0.35]
    if not valid:
        return None
    poly = orient(max(valid, key=lambda g: g.area).buffer(0), sign=1.0)
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)

    thickness = 0.18
    z_bot = max(fc.floor_z + 0.004, z_land_top - thickness)
    z_top = z_land_top
    if z_top - z_bot < 0.08:
        return None
    try:
        v_uv, tri = extrude_polygon_solid(poly, z_bot, z_top)
    except ValueError:
        return None

    pid = f"stair_landing_{landing_idx:02d}"
    part = MeshPart(
        part_id=pid,
        class_id=13,
        role="stair_landing",
        vertices=fc.frame.uvz_to_xyz(v_uv),
        triangles=tri,
        evidence_kind="measured",
        metadata={
            "point_count": int(len(land_pts)),
            "area_m2": round(float(poly.area), 3),
            "z_span": [round(z_bot, 4), round(z_top, 4)],
            "connects": [fl1.part_id, fl2.part_id],
        },
    )
    return part, poly, z_bot, z_top


def build_stairs_and_railings(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    interior_wall_polys_uv: list[Polygon],
    column_polys_uv: list[Polygon],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
    *,
    column_z_bands: list[tuple[Polygon, float, float]] | None = None,
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

    # Base horizontal clearance domain inside room, away from walls & overlapping columns
    # (Z-aware for columns: a short under-stairs support post ending below a flight's soffit does NOT block it!)
    allowed_wall_xy = shell.inner_uv.buffer(-0.008)
    if interior_wall_polys_uv:
        allowed_wall_xy = allowed_wall_xy.difference(
            unary_union([p.buffer(0.010) for p in interior_wall_polys_uv])
        ).buffer(0)

    def _allowed_xy_for_flight(sub_pts: np.ndarray, z_lo_hint: float | None, z_hi_hint: float | None) -> Polygon | MultiPolygon:
        dom = allowed_wall_xy
        if column_z_bands:
            d_est, _ = _estimate_flight_dir_and_corr(sub_pts, snap_tol_deg=8.0)
            z0 = z_lo_hint if z_lo_hint is not None else float(np.percentile(sub_pts[:, 2], 2))
            z1 = z_hi_hint if z_hi_hint is not None else float(np.percentile(sub_pts[:, 2], 98))
            blocking_cols: list[Polygon] = []
            for cpoly, cz0, cz1 in column_z_bands:
                if cz1 < z0 - 0.05 or cz0 > z1 + 0.05:
                    continue
                # If we know flight slope, check if the column sits strictly beneath the flight soffit at the column XY!
                if d_est is not None:
                    s_pts = sub_pts[:, :2] @ d_est
                    s_lo_p, s_hi_p = float(np.percentile(s_pts, 2)), float(np.percentile(s_pts, 98))
                    c_xy = np.asarray(cpoly.exterior.coords, dtype=np.float64)[:, :2]
                    s_c_min = float((c_xy @ d_est).min())
                    if s_hi_p > s_lo_p + 0.20:
                        frac = np.clip((s_c_min - s_lo_p) / (s_hi_p - s_lo_p), 0.0, 1.0)
                        soffit_z_at_col = z0 + frac * (z1 - z0) - 0.28
                        if cz1 <= soffit_z_at_col - 0.03:
                            continue
                blocking_cols.append(cpoly.buffer(0.010))
            if blocking_cols:
                dom = dom.difference(unary_union(blocking_cols)).buffer(0)
        elif column_polys_uv:
            dom = dom.difference(unary_union([p.buffer(0.010) for p in column_polys_uv])).buffer(0)
        return dom

    allowed_xy = allowed_wall_xy
    static_blockers = [*interior_wall_polys_uv, *column_polys_uv]
    if column_polys_uv:
        allowed_xy = allowed_xy.difference(unary_union([p.buffer(0.010) for p in column_polys_uv])).buffer(0)

    parts: list[MeshPart] = []
    flights: list[_FlightGeom] = []
    landings: list[tuple[Polygon, float, float]] = []
    stair_well_cutouts: list[tuple[Polygon, float, float]] = []
    beam_blocker_bands: list[tuple[Polygon, float, float]] = []
    flight_logs: list[dict[str, Any]] = []
    landing_logs: list[dict[str, Any]] = []
    railing_logs: list[dict[str, Any]] = []

    # 1. Build stepped stair flights + U/L-turn mid-landings (class 13)
    placed_flight_union: Polygon | MultiPolygon | None = None
    if len(stair_pts) >= 500:
        comps = _voxel_connected_components_3d(stair_pts, voxel_size=0.22)
        for comp in comps:
            if len(comp) < 500:
                continue
            sub = stair_pts[comp]

            # Recursively decompose any L-turn, 2-flight U-turn, or 3-flight U-turn staircase!
            segments = _split_stair_turns_recursive(sub, max_depth=2)
            if len(segments) > 1:
                comp_flights: list[_FlightGeom] = []
                for seg_pts, z_lo_ov, z_hi_ov in segments:
                    seg_clean = seg_pts
                    # For direct 180-deg U-turn pairs, trim lateral bleed from the preceding flight
                    if comp_flights and len(seg_pts) >= 400:
                        prev_fl = comp_flights[-1]
                        d_cur, _ = _estimate_flight_dir_and_corr(seg_pts, snap_tol_deg=8.0)
                        if d_cur is not None and float(np.dot(prev_fl.run_dir, d_cur)) < -0.50:
                            w_cur = seg_pts[:, :2] @ prev_fl.lat_dir
                            n_above = int(np.sum(w_cur > prev_fl.w_hi + 0.05))
                            n_below = int(np.sum(w_cur < prev_fl.w_lo - 0.05))
                            if n_above >= max(400, n_below):
                                seg_clean = seg_pts[w_cur > prev_fl.w_hi + 0.05]
                            elif n_below >= 400:
                                seg_clean = seg_pts[w_cur < prev_fl.w_lo - 0.05]

                    cur_allowed_s = _allowed_xy_for_flight(seg_clean, z_lo_ov, z_hi_ov)
                    if placed_flight_union is not None and not placed_flight_union.is_empty:
                        cur_allowed_s = cur_allowed_s.difference(placed_flight_union.buffer(0.015)).buffer(0)
                    fres = _fit_and_build_flight(
                        seg_clean,
                        fc,
                        cur_allowed_s,
                        len(flights),
                        z_lo_override=z_lo_ov,
                        z_hi_override=z_hi_ov,
                    )
                    if fres is not None:
                        fpart_s, fgeom_s = fres
                        parts.append(fpart_s)
                        flights.append(fgeom_s)
                        comp_flights.append(fgeom_s)
                        placed_flight_union = (
                            fgeom_s.poly_uv
                            if placed_flight_union is None
                            else unary_union([placed_flight_union, fgeom_s.poly_uv])
                        )
                        stair_well_cutouts.append((fgeom_s.poly_uv.buffer(0.04), fgeom_s.z_lo - 0.05, fgeom_s.z_hi + 0.15))
                        beam_blocker_bands.append((fgeom_s.poly_uv.buffer(0.03), fgeom_s.z_lo - 0.05, fgeom_s.z_hi + 0.15))
                        flight_logs.append(fpart_s.metadata | {"part_id": fpart_s.part_id})

                # Connect each consecutive flight pair with a crisp horizontal turn landing when evidence exists
                for k in range(len(comp_flights) - 1):
                    fl_a, fl_b = comp_flights[k], comp_flights[k + 1]
                    z_turn_k = 0.5 * (fl_a.z_hi + fl_b.z_lo)
                    land_pts_k = sub[(sub[:, 2] >= z_turn_k - 0.25) & (sub[:, 2] <= z_turn_k + 0.18)]
                    cur_allowed_l = allowed_xy
                    if placed_flight_union is not None and not placed_flight_union.is_empty:
                        cur_allowed_l = cur_allowed_l.difference(placed_flight_union.buffer(0.006)).buffer(0)
                    lres = _build_stair_landing(land_pts_k, fl_a, fl_b, fl_a.z_hi, fc, cur_allowed_l, len(landings))
                    if lres is not None:
                        lpart, lpoly, lz0, lz1 = lres
                        parts.append(lpart)
                        landings.append((lpoly, lz0, lz1))
                        placed_flight_union = (
                            lpoly if placed_flight_union is None else unary_union([placed_flight_union, lpoly])
                        )
                        stair_well_cutouts.append((lpoly.buffer(0.04), lz0 - 0.05, lz1 + 0.15))
                        beam_blocker_bands.append((lpoly.buffer(0.03), lz0 - 0.05, lz1 + 0.15))
                        landing_logs.append(lpart.metadata | {"part_id": lpart.part_id})

                if comp_flights:
                    continue

            cur_allowed = allowed_xy
            if placed_flight_union is not None and not placed_flight_union.is_empty:
                cur_allowed = cur_allowed.difference(placed_flight_union.buffer(0.015)).buffer(0)
            res = _fit_and_build_flight(sub, fc, cur_allowed, len(flights))
            if res is None:
                # Fallback split at median Z
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

    # 2. Build B16 open handrail + vertical baluster railings (class 14, NO solid sheets, NO broken floating stubs!):
    #    Phase 2a: Flight-guided dual-side continuous sloped stair handrails (anchored strictly onto stair treads!)
    #    Phase 2b: Remaining non-stair horizontal guardrails (anchored strictly onto supporting floor slabs!)
    rail_allowed_base = shell.inner_uv.buffer(-0.045)
    if static_blockers:
        rail_allowed_base = rail_allowed_base.difference(unary_union([p.buffer(0.045) for p in static_blockers])).buffer(0)

    placed_rail_bands: list[tuple[Polygon, float, float]] = []
    rail_group_idx = 0
    consumed_rail_mask = np.zeros(len(rail_pts), dtype=bool)

    def _emit_railing_run(
        r_dir: np.ndarray,
        r_lat: np.ndarray,
        w_mid: float,
        sa: float,
        sb: float,
        zt_a: float,
        zt_b: float,
        z_bot_fn,
        *,
        host_flight: _FlightGeom | None = None,
    ) -> None:
        nonlocal rail_group_idx
        if sb - sa < 0.55:
            return
        hr_w = 0.055
        hr_h = 0.055
        is_sloped = abs(float(zt_b) - float(zt_a)) >= 0.30

        # Ensure handrail stays strictly above any stair flight it crosses in XY
        for t_frac in np.linspace(0.0, 1.0, 11):
            sc = sa + t_frac * (sb - sa)
            uv_c = r_dir * sc + r_lat * w_mid
            pt_c = Point(float(uv_c[0]), float(uv_c[1]))
            for fl in flights:
                if fl.poly_uv.buffer(0.04).contains(pt_c):
                    min_hr_z = fl.step_top_z_at(float(uv_c[0]), float(uv_c[1])) + 0.16
                    cur_z = zt_a + t_frac * (zt_b - zt_a)
                    if cur_z < min_hr_z:
                        lift = min_hr_z - cur_z
                        zt_a += lift
                        zt_b += lift

        seg_uv_poly = LineString([r_dir * sa + r_lat * w_mid, r_dir * sb + r_lat * w_mid]).buffer(0.045)
        hr_z_lo = min(zt_a, zt_b) - hr_h
        hr_z_hi = max(zt_a, zt_b)
        for clv in fc.ceiling_levels:
            if clv.polygon_uv.intersects(seg_uv_poly):
                if hr_z_lo < clv.z_bottom + cfg.ceiling_thickness_m + 0.005 and hr_z_hi > clv.z_bottom - 0.005:
                    return
        if not is_sloped:
            for flv in fc.floor_levels:
                if flv.is_primary:
                    continue
                if flv.polygon_uv.intersects(seg_uv_poly):
                    if hr_z_lo < flv.z_top + 0.005 and hr_z_hi > flv.z_top - cfg.floor_thickness_m - 0.005:
                        return

        p0_uvz = np.array([*(r_dir * sa + r_lat * w_mid), zt_a - 0.5 * hr_h], dtype=np.float64)
        p1_uvz = np.array([*(r_dir * sb + r_lat * w_mid), zt_b - 0.5 * hr_h], dtype=np.float64)
        p0_xyz = fc.frame.uvz_to_xyz(p0_uvz[None, :])[0]
        p1_xyz = fc.frame.uvz_to_xyz(p1_uvz[None, :])[0]
        axis_l = p1_xyz - p0_xyz
        hr_len = float(np.linalg.norm(axis_l))
        if hr_len < 0.55:
            return
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
                    "sloped": bool(is_sloped),
                    "host_flight_id": host_flight.part_id if host_flight is not None else None,
                },
            )
        )

        # Build open vertical balusters strictly anchored down onto stair treads or floor slabs (NEVER floating!)
        bal_w = 0.04
        slope_eff = abs(zt_b - zt_a) / max(sb - sa, 1e-6)
        n_bal = max(2, int(np.floor((sb - sa) / 0.50)) + 1)
        s_posts = np.linspace(sa + 0.05, sb - 0.05, n_bal)
        bal_count = 0
        z_min_post = min(zt_a, zt_b)
        for b_idx, sp in enumerate(s_posts):
            t_f = (sp - sa) / max(sb - sa, 1e-6)
            uv_p = r_dir * sp + r_lat * w_mid
            hr_bot_local = (
                (zt_a + t_f * (zt_b - zt_a))
                - hr_h * np.hypot(1.0, slope_eff)
                - 0.015
                - 0.5 * bal_w * slope_eff
            )
            b_corners = [
                uv_p + r_dir * ds_c + r_lat * dw_c
                for ds_c in (-0.5 * bal_w, 0.5 * bal_w)
                for dw_c in (-0.5 * bal_w, 0.5 * bal_w)
            ]
            b_poly = Polygon(b_corners)

            if host_flight is not None:
                # Strictly anchor every stair baluster directly onto its supporting stair tread!
                zb = max(host_flight.step_top_z_at(float(c[0]), float(c[1])) for c in b_corners) + 0.008
            else:
                zb = float(z_bot_fn(float(sp)))
                # Snap horizontal guardrail posts down onto supporting floor level (within 0.45m)
                for flv in fc.floor_levels:
                    if flv.is_primary or flv.polygon_uv.buffer(0.08).intersects(b_poly):
                        if abs(zb - flv.z_top) <= 0.45:
                            zb = flv.z_top + 0.006
                            break

            for fl in flights:
                if fl.poly_uv.buffer(0.015).intersects(b_poly):
                    max_step_z = max(fl.step_top_z_at(float(c[0]), float(c[1])) for c in b_corners)
                    zb = max(zb, max_step_z + 0.008)

            for lpoly, lz0, lz1 in landings:
                if lpoly.buffer(0.015).intersects(b_poly):
                    if zb < lz1 + 0.006 and hr_bot_local > lz0 - 0.006:
                        if hr_bot_local - lz1 > 0.20:
                            zb = max(zb, lz1 + 0.008)
                        else:
                            hr_bot_local = min(hr_bot_local, lz0 - 0.006)

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
            z_min_post = min(z_min_post, zb)
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
                    metadata={"z_span": [round(float(zb), 4), round(float(hr_bot_local), 4)]},
                )
            )
            bal_count += 1

        placed_rail_bands.append((seg_uv_poly, z_min_post, max(zt_a, zt_b)))
        # Register upper-level / stair railings as ceiling cutouts so no lower-storey ceiling ever slices through them!
        stair_well_cutouts.append((seg_uv_poly.buffer(0.03), z_min_post - 0.02, max(zt_a, zt_b) + 0.05))
        beam_blocker_bands.append((seg_uv_poly.buffer(0.02), z_min_post, max(zt_a, zt_b) + 0.05))
        if is_sloped:
            stair_well_cutouts.append((seg_uv_poly.buffer(0.03), z_min_post - 0.05, max(zt_a, zt_b) + 0.05))
        railing_logs.append({
            "group_id": f"railing_{rail_group_idx:02d}",
            "length_m": round(float(hr_len), 3),
            "baluster_count": int(bal_count),
            "sloped": bool(is_sloped),
            "host_flight_id": host_flight.part_id if host_flight is not None else None,
            "z_ends": [round(float(zt_a), 4), round(float(zt_b), 4)],
        })
        rail_group_idx += 1

    # Phase 2a: Unified dual-side continuous sloped stair handrails along every reconstructed flight
    if len(rail_pts) >= 250 and flights:
        for fl in flights:
            s_r = rail_pts[:, :2] @ fl.run_dir
            w_r = rail_pts[:, :2] @ fl.lat_dir
            pitch_slope = (fl.z_hi - fl.z_lo) / max(fl.s_hi - fl.s_lo, 1e-6)
            pitch_z_all = fl.z_lo + np.clip(s_r - fl.s_lo, -0.45, (fl.s_hi - fl.s_lo) + 0.45) * pitch_slope
            near_fl = (
                (~consumed_rail_mask)
                & (s_r >= fl.s_lo - 0.45)
                & (s_r <= fl.s_hi + 0.45)
                & (w_r >= fl.w_lo - 0.40)
                & (w_r <= fl.w_hi + 0.40)
                & (rail_pts[:, 2] >= pitch_z_all - 0.25)
                & (rail_pts[:, 2] <= pitch_z_all + 1.55)
            )
            if int(np.sum(near_fl)) < 250:
                continue
            w_center = 0.5 * (fl.w_lo + fl.w_hi)

            for side_idx, side_mask in (
                (0, near_fl & (w_r <= w_center)),
                (1, near_fl & (w_r > w_center)),
            ):
                if int(np.sum(side_mask)) < 250:
                    continue
                consumed_rail_mask |= side_mask
                sub_s = s_r[side_mask]
                sub_w = w_r[side_mask]
                sub_z = rail_pts[side_mask, 2]
                s_lo_r = max(fl.s_lo + 0.04, float(np.percentile(sub_s, 2)))
                s_hi_r = min(fl.s_hi - 0.04, float(np.percentile(sub_s, 98)))
                # Require meaningful span along flight (>= 30% of flight run) to avoid tiny stray clusters
                if s_hi_r - s_lo_r < max(0.85, 0.30 * (fl.s_hi - fl.s_lo)):
                    continue
                # Only bridge full flight run when railing points ALREADY span >= 70% of the flight (never extrapolate into open air!)
                if s_hi_r - s_lo_r >= 0.70 * (fl.s_hi - fl.s_lo):
                    s_lo_r = fl.s_lo + 0.06
                    s_hi_r = fl.s_hi - 0.06

                # Place rail strictly at the measured class-14 lateral coordinate (NEVER teleport away from real railing points!)
                w_target = fl.w_lo + 0.045 if side_idx == 0 else fl.w_hi - 0.045
                w_meas = float(np.median(sub_w))
                if abs(w_meas - w_target) > 0.35:
                    continue
                w_rail = float(np.clip(w_meas, fl.w_lo + 0.035, fl.w_hi - 0.035))

                # Verify real class-14 point-cloud bin coverage along [s_lo_r, s_hi_r] (>=55% of 0.35m bins occupied)
                n_rbins = max(3, int(round((s_hi_r - s_lo_r) / 0.35)))
                rb_edges = np.linspace(s_lo_r, s_hi_r, n_rbins + 1)
                near_lat = np.abs(sub_w - w_rail) <= 0.28
                occ_bins = sum(
                    1
                    for k in range(n_rbins)
                    if int(np.sum(near_lat & (sub_s >= rb_edges[k]) & (sub_s <= rb_edges[k + 1]))) >= 5
                )
                if occ_bins / n_rbins < 0.55:
                    continue

                # Measure constant handrail height above the flight pitch line (using 88th percentile)
                pitch_z_pts = fl.z_lo + (sub_s - fl.s_lo) * pitch_slope
                dz_above = sub_z - pitch_z_pts
                h_rail = float(np.clip(np.percentile(dz_above, 88), 0.78, 1.12))

                # Clip centerline inside room shell minus walls/columns and existing overlapping rails
                p_start = fl.run_dir * s_lo_r + fl.lat_dir * w_rail
                p_end = fl.run_dir * s_hi_r + fl.lat_dir * w_rail
                cur_fl_allowed = rail_allowed_base
                ov_r = [
                    p.buffer(0.045)
                    for p, zb, zt in placed_rail_bands
                    if zb <= fl.z_hi + h_rail + 0.05 and zt >= fl.z_lo - 0.05
                ]
                if ov_r:
                    cur_fl_allowed = cur_fl_allowed.difference(unary_union(ov_r)).buffer(0)
                line = LineString([p_start, p_end]).intersection(cur_fl_allowed)
                segs = (
                    list(line.geoms)
                    if isinstance(line, MultiLineString)
                    else ([line] if isinstance(line, LineString) else [])
                )
                if not segs:
                    continue
                # Keep the longest continuous clipped run (avoids fragmented stubs!)
                best_seg = max(segs, key=lambda g: g.length)
                if best_seg.length < 0.80:
                    continue
                c_pts = np.asarray(list(best_seg.coords), dtype=np.float64)
                sa_raw = float(c_pts[0] @ fl.run_dir)
                sb_raw = float(c_pts[-1] @ fl.run_dir)
                sa, sb = min(sa_raw, sb_raw) + 0.02, max(sa_raw, sb_raw) - 0.02
                zt_a = fl.z_lo + (sa - fl.s_lo) * pitch_slope + h_rail
                zt_b = fl.z_lo + (sb - fl.s_lo) * pitch_slope + h_rail
                # Never extrapolate handrail higher than measured railing points + 0.20m (prevents phantom overhang!)
                max_meas_rail_z = float(np.percentile(sub_z, 98)) + 0.20
                if zt_b > max_meas_rail_z and pitch_slope > 0.15:
                    s_cap = fl.s_lo + (max_meas_rail_z - h_rail - fl.z_lo) / pitch_slope
                    sb = min(sb, s_cap)
                    if sb - sa < 0.65:
                        continue
                    zt_b = fl.z_lo + (sb - fl.s_lo) * pitch_slope + h_rail
                local_ceil = _local_ceiling_above_poly(fl.poly_uv, fc)
                if local_ceil is not None and fl.z_hi <= local_ceil + 0.10:
                    zt_a = min(zt_a, local_ceil - 0.06)
                    zt_b = min(zt_b, local_ceil - 0.06)

                _emit_railing_run(
                    fl.run_dir,
                    fl.lat_dir,
                    w_rail,
                    sa,
                    sb,
                    zt_a,
                    zt_b,
                    lambda sv, _fl=fl: _fl.z_lo + (sv - _fl.s_lo) * pitch_slope,
                    host_flight=fl,
                )

    # Phase 2b: Remaining non-flight horizontal guardrails (e.g. 2F mezzanine edge guardrails)
    rem_rail = rail_pts[~consumed_rail_mask]
    if len(rem_rail) >= 500:
        r_comps = _voxel_connected_components_3d(rem_rail, voxel_size=0.18)
        for comp in r_comps:
            if len(comp) < 500:
                continue
            sub = rem_rail[comp]
            mean_uv = sub[:, :2].mean(axis=0)
            centered = sub[:, :2] - mean_uv
            cov = centered.T @ centered / max(len(sub), 1)
            _, eigvecs = np.linalg.eigh(cov)
            r_dir = _snap_dir_to_room_axis(eigvecs[:, 1], tol_deg=15.0)
            r_lat = np.array([-r_dir[1], r_dir[0]], dtype=np.float64)

            s_all = sub[:, :2] @ r_dir
            w_all = sub[:, :2] @ r_lat
            s_lo, s_hi = float(np.percentile(s_all, 2)), float(np.percentile(s_all, 98))
            if s_hi - s_lo < 0.80:
                continue
            w_mid = float(np.median(w_all))
            zt_med = float(np.percentile(sub[:, 2], 92))
            zb_med = max(fc.floor_z + 0.008, float(np.percentile(sub[:, 2], 8)))
            if zt_med - zb_med < 0.45:
                continue

            cur_rail_allowed = rail_allowed_base
            ov_rails = [
                p.buffer(0.045)
                for p, zb, zt in placed_rail_bands
                if zb <= zt_med + 0.05 and zt >= zb_med - 0.05
            ]
            if ov_rails:
                cur_rail_allowed = cur_rail_allowed.difference(unary_union(ov_rails)).buffer(0)

            line = LineString([r_dir * s_lo + r_lat * w_mid, r_dir * s_hi + r_lat * w_mid]).intersection(cur_rail_allowed)
            segs = (
                list(line.geoms)
                if isinstance(line, MultiLineString)
                else ([line] if isinstance(line, LineString) else [])
            )
            for seg in segs:
                if seg.length < 0.70:
                    continue
                c_pts = np.asarray(list(seg.coords), dtype=np.float64)
                sa_raw = float(c_pts[0] @ r_dir)
                sb_raw = float(c_pts[-1] @ r_dir)
                sa, sb = min(sa_raw, sb_raw) + 0.02, max(sa_raw, sb_raw) - 0.02
                _emit_railing_run(
                    r_dir,
                    r_lat,
                    w_mid,
                    sa,
                    sb,
                    zt_med,
                    zt_med,
                    lambda _sv, _zb=zb_med: _zb,
                    host_flight=None,
                )

    emitted_stairs = len(flights)
    emitted_rails = sum(1 for p in parts if p.class_id == 14)
    return parts, stair_well_cutouts, beam_blocker_bands, {
        "status": "PASS" if (emitted_stairs > 0 or emitted_rails > 0) else "N/A",
        "stair_points_after_working_labels": int(len(stair_pts)),
        "railing_points_after_working_labels": int(len(rail_pts)),
        "emitted_stair_flights": emitted_stairs,
        "emitted_stair_landings": len(landings),
        "emitted_railing_groups": rail_group_idx,
        "emitted_railing_parts": emitted_rails,
        "flights": flight_logs,
        "landings": landing_logs,
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
