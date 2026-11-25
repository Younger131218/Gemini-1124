"""Stage 3 B18 regular clutter reconstruction: boxes, vertical/oblique cylinders, cones/pyramids & semantic exclusions."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import Point

from ipcr.architecture.frame_floors import FloorCeilingResult
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import ClutterConfig
from ipcr.furniture.chairs import _find_1d_density_valleys
from ipcr.furniture.tables_cabinets_sofas import TableSpec, _fit_obb_2d, _local_floor_z
from ipcr.geometry_primitives import (
    make_oriented_box,
    make_oriented_cone_or_pyramid,
    make_oriented_cylinder,
)
from ipcr.types import MeshPart
from ipcr.validator.mesh_relations import meshes_penetrate
from ipcr.working_labels import _voxel_connected_components_3d


def _voxel_uniform_3d(pts: np.ndarray, cell: float = 0.03) -> np.ndarray:
    """3D voxel-grid uniformization so one-sided LiDAR density spikes never bias primitive fitting (G4)."""
    if len(pts) < 20:
        return pts
    q = np.floor(pts / cell).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    n_u = int(inv.max()) + 1
    if n_u < 10:
        return pts
    out = np.zeros((n_u, 3), dtype=np.float64)
    cnts = np.bincount(inv)
    for ax in range(3):
        np.add.at(out[:, ax], inv, pts[:, ax])
    out /= cnts[:, None]
    return out


def _local_ceiling_z(fc: FloorCeilingResult, u: float, v: float, z_fl: float) -> float:
    """Resolve local overhead ceiling Z above (u, v) in single or multi-storey scenes."""
    pt = Point(float(u), float(v))
    cands: list[float] = []
    for clv in fc.ceiling_levels or []:
        if clv.z_bottom > z_fl + 1.20:
            if clv.is_primary or clv.polygon_uv.buffer(0.05).contains(pt):
                cands.append(float(clv.z_bottom))
    if cands:
        return min(cands)
    return float(fc.ceiling_z) if fc.ceiling_z is not None else z_fl + 4.0


def _circle_fit_2d(xy: np.ndarray) -> tuple[np.ndarray, float, float, float, float] | None:
    """Algebraic Kasa circle fit returning (center_2d, radius, p50_err, p90_err, arc_deg)."""
    if len(xy) < 12:
        return None
    xm, ym = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    x = xy[:, 0] - xm
    y = xy[:, 1] - ym
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(xy), dtype=np.float64)])
    b = x * x + y * y
    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c_val = float(sol[0]), float(sol[1]), float(sol[2])
    r2 = cx * cx + cy * cy + c_val
    if r2 <= 1e-6:
        return None
    r = float(np.sqrt(r2))
    span = float(max(xy[:, 0].max() - xy[:, 0].min(), xy[:, 1].max() - xy[:, 1].min()))
    # Reject huge flat-wall arcs masquerading as giant circles (B18)
    if r < 0.03 or r > max(1.2 * span, 0.80):
        return None
    center = np.array([xm + cx, ym + cy], dtype=np.float64)
    dists = np.linalg.norm(xy - center[None, :], axis=1)
    errs = np.abs(dists - r)
    p50 = float(np.percentile(errs, 50))
    p90 = float(np.percentile(errs, 90))

    # Measure continuous angular arc coverage (360 - largest angular gap)
    ang = np.sort(np.arctan2(xy[:, 1] - center[1], xy[:, 0] - center[0]))
    gaps = np.diff(ang)
    wrap_gap = float((ang[0] + 2.0 * np.pi) - ang[-1])
    max_gap = max(float(gaps.max()) if len(gaps) else 2.0 * np.pi, wrap_gap)
    arc_deg = float(np.degrees(max(0.0, 2.0 * np.pi - max_gap)))
    return center, r, p50, p90, arc_deg


def _box_surface_residuals(pts_uvz: np.ndarray, center_uv: np.ndarray, l_dir: np.ndarray, d_dir: np.ndarray, L: float, D: float, z0: float, z1: float) -> tuple[float, float, int]:
    """Compute distance from points to closest face of an oriented 3D box + count of supported faces."""
    rel = pts_uvz[:, :2] - center_uv[None, :]
    sl = rel @ l_dir
    sd = rel @ d_dir
    sz = pts_uvz[:, 2] - 0.5 * (z0 + z1)
    hl, hd, hz = max(0.5 * L, 1e-4), max(0.5 * D, 1e-4), max(0.5 * (z1 - z0), 1e-4)

    # Signed distance to each axis slab boundary
    dl = np.abs(sl) - hl
    dd = np.abs(sd) - hd
    dz = np.abs(sz) - hz
    # Outside distance (exact Euclidean to box) vs inside distance to nearest face
    out_d = np.sqrt(np.maximum(dl, 0.0) ** 2 + np.maximum(dd, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    in_d = np.minimum(np.minimum(np.abs(dl), np.abs(dd)), np.abs(dz))
    surf_d = np.where((dl > 0) | (dd > 0) | (dz > 0), out_d, in_d)

    # Count distinct faces with genuine point support (within 3.5cm of that face, >= 6% of points or >= 8 uniform voxels)
    tol = 0.035
    min_f = max(6, int(0.05 * len(pts_uvz)))
    face_hits = [
        int(np.sum((np.abs(sl - hl) <= tol) & (np.abs(sd) <= hd + tol) & (np.abs(sz) <= hz + tol))) >= min_f,
        int(np.sum((np.abs(sl + hl) <= tol) & (np.abs(sd) <= hd + tol) & (np.abs(sz) <= hz + tol))) >= min_f,
        int(np.sum((np.abs(sd - hd) <= tol) & (np.abs(sl) <= hl + tol) & (np.abs(sz) <= hz + tol))) >= min_f,
        int(np.sum((np.abs(sd + hd) <= tol) & (np.abs(sl) <= hl + tol) & (np.abs(sz) <= hz + tol))) >= min_f,
        int(np.sum((np.abs(sz - hz) <= tol) & (np.abs(sl) <= hl + tol) & (np.abs(sd) <= hd + tol))) >= min_f,
        int(np.sum((np.abs(sz + hz) <= tol) & (np.abs(sl) <= hl + tol) & (np.abs(sd) <= hd + tol))) >= min_f,
    ]
    return float(np.percentile(surf_d, 50)), float(np.percentile(surf_d, 90)), int(sum(face_hits))


def _fit_vertical_cylinder(pts_u: np.ndarray, z0: float, z1: float, min_arc_deg: float) -> dict[str, Any] | None:
    """Fit a vertical cylinder and return residual metrics if arc coverage meets B18 minimum."""
    h = z1 - z0
    if h < 0.06:
        return None
    # Fit circle on side wall points (exclude flat top/bottom caps if tall enough)
    side = pts_u[(pts_u[:, 2] >= z0 + 0.08 * h) & (pts_u[:, 2] <= z1 - 0.08 * h)] if h >= 0.18 else pts_u
    if len(side) < 12:
        side = pts_u
    cf = _circle_fit_2d(side[:, :2])
    if cf is None:
        return None
    center_2d, r, _sp50, _sp90, arc_deg = cf
    if arc_deg < min_arc_deg:
        return None

    # Full 3D distance to closed vertical cylinder surface (side wall + top/bottom caps)
    rad = np.linalg.norm(pts_u[:, :2] - center_2d[None, :], axis=1)
    dr = rad - r
    hz = 0.5 * h
    sz = pts_u[:, 2] - 0.5 * (z0 + z1)
    dz = np.abs(sz) - hz
    out_d = np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dz, 0.0) ** 2)
    in_d = np.minimum(np.abs(dr), np.abs(dz))
    surf_d = np.where((dr > 0) | (dz > 0), out_d, in_d)
    return {
        "kind": "cylinder_vertical",
        "p0_uvz": np.array([center_2d[0], center_2d[1], z0], dtype=np.float64),
        "p1_uvz": np.array([center_2d[0], center_2d[1], z1], dtype=np.float64),
        "radius": r,
        "arc_deg": arc_deg,
        "p50": float(np.percentile(surf_d, 50)),
        "p90": float(np.percentile(surf_d, 90)),
    }


def _fit_oblique_cylinder(pts_u: np.ndarray, min_arc_deg: float) -> dict[str, Any] | None:
    """Fit a general 3D slanted/horizontal cylinder along PCA major axis (B18: 圆柱不局限竖直方向)."""
    if len(pts_u) < 20:
        return None
    mean = pts_u.mean(axis=0)
    centered = pts_u - mean
    cov = (centered.T @ centered) / len(pts_u)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Require strong elongated cylindrical aspect (lambda_max >= 3.2 * lambda_mid) AND non-vertical (> 12 deg from Z)
    if eigvals[2] < max(eigvals[1] * 3.2, 0.008):
        return None
    axis = eigvecs[:, 2].copy()
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    if abs(axis[2]) > float(np.cos(np.radians(12.0))):
        return None  # nearly vertical — handled by _fit_vertical_cylinder
    if axis[2] < 0:
        axis = -axis

    # Orthonormal cross-section basis (b1, b2) perpendicular to axis
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.85 else np.array([1.0, 0.0, 0.0])
    b1 = np.cross(axis, ref)
    b1 /= max(float(np.linalg.norm(b1)), 1e-12)
    b2 = np.cross(axis, b1)
    b2 /= max(float(np.linalg.norm(b2)), 1e-12)

    t = centered @ axis
    t0, t1 = float(np.percentile(t, 2)), float(np.percentile(t, 98))
    length = t1 - t0
    if length < 0.22:
        return None
    px = centered @ b1
    py = centered @ b2
    side_m = (t >= t0 + 0.08 * length) & (t <= t1 - 0.08 * length)
    pxy = np.column_stack([px[side_m], py[side_m]]) if int(np.sum(side_m)) >= 12 else np.column_stack([px, py])
    cf = _circle_fit_2d(pxy)
    if cf is None:
        return None
    c2, r, _sp50, _sp90, arc_deg = cf
    if arc_deg < min_arc_deg or length < 1.8 * r:
        return None

    rad = np.hypot(px - c2[0], py - c2[1])
    dr = rad - r
    ht = 0.5 * length
    dt = np.abs(t - 0.5 * (t0 + t1)) - ht
    out_d = np.sqrt(np.maximum(dr, 0.0) ** 2 + np.maximum(dt, 0.0) ** 2)
    in_d = np.minimum(np.abs(dr), np.abs(dt))
    surf_d = np.where((dr > 0) | (dt > 0), out_d, in_d)

    c_axis = mean + b1 * c2[0] + b2 * c2[1]
    return {
        "kind": "cylinder_oblique",
        "p0_uvz": c_axis + axis * t0,
        "p1_uvz": c_axis + axis * t1,
        "radius": r,
        "arc_deg": arc_deg,
        "p50": float(np.percentile(surf_d, 50)),
        "p90": float(np.percentile(surf_d, 90)),
    }


def _fit_cone_or_pyramid(pts_u: np.ndarray, z0: float, z1: float) -> dict[str, Any] | None:
    """Detect linearly tapering cone or triangular pyramid (B18: 三角锥 / 圆锥)."""
    h = z1 - z0
    if h < 0.20 or len(pts_u) < 25:
        return None
    # Split into 5 vertical slices and check monotonic linear radial taper from base to apex
    edges = np.linspace(z0, z1, 6)
    z_mids = []
    r_vals = []
    # Use mean (not median!) because equilateral triangles have median != centroid!
    top_sl = pts_u[pts_u[:, 2] >= z0 + 0.75 * h]
    bot_sl = pts_u[pts_u[:, 2] <= z0 + 0.35 * h]
    if len(top_sl) >= 3:
        c_xy = top_sl[:, :2].mean(axis=0)
    elif len(bot_sl) >= 3:
        c_xy = bot_sl[:, :2].mean(axis=0)
    else:
        c_xy = pts_u[:, :2].mean(axis=0)

    for k in range(5):
        sl = pts_u[(pts_u[:, 2] >= edges[k]) & (pts_u[:, 2] <= edges[k + 1])]
        if len(sl) < 3:
            continue
        rm = float(np.percentile(np.linalg.norm(sl[:, :2] - c_xy[None, :], axis=1), 85))
        z_mids.append(0.5 * (edges[k] + edges[k + 1]) - z0)
        r_vals.append(rm)
    if len(z_mids) < 4:
        return None
    zm = np.asarray(z_mids, dtype=np.float64)
    rv = np.asarray(r_vals, dtype=np.float64)
    if rv[0] < 0.08 or rv[-1] > 0.45 * rv[0]:
        return None  # not sharply tapering to an apex!

    # Estimate true circumradius r_base at z=z0 (98th percentile so triangular corners set circumradius accurately!)
    low_m = (pts_u[:, 2] >= z0) & (pts_u[:, 2] <= z0 + 0.70 * h)
    low_pts = pts_u[low_m] if int(np.sum(low_m)) >= 12 else pts_u
    low_frac = np.maximum(1.0 - (low_pts[:, 2] - z0) / h, 0.25)
    r_est_all = np.linalg.norm(low_pts[:, :2] - c_xy[None, :], axis=1) / low_frac
    r_base_tri = float(np.clip(np.percentile(r_est_all, 98), 0.08, 1.5))
    r_base_cone = float(np.clip(np.percentile(r_est_all, 65), 0.08, 1.5))
    actual_r = np.linalg.norm(pts_u[:, :2] - c_xy[None, :], axis=1)
    taper = np.clip(1.0 - (pts_u[:, 2] - z0) / h, 0.0, 1.0)

    # Decide triangular pyramid (sides=3) vs circular cone (sides=18) by comparing radial residual to 3-gon vs circle
    rel_xy = pts_u[:, :2] - c_xy[None, :]
    ang = np.arctan2(rel_xy[:, 1], rel_xy[:, 0])
    # Require genuine multi-side angular wrap (>= 220 deg) so one-sided doorway corner noise NEVER forms a pyramid!
    ang_s = np.sort(ang)
    gaps = np.diff(ang_s)
    wrap_gap = float((ang_s[0] + 2.0 * np.pi) - ang_s[-1])
    max_gap = max(float(gaps.max()) if len(gaps) else 2.0 * np.pi, wrap_gap)
    arc_deg = float(np.degrees(max(0.0, 2.0 * np.pi - max_gap)))
    if arc_deg < 220.0:
        return None
    # Circular cone surface error
    cone_err = np.abs(actual_r - r_base_cone * taper) * (h / float(np.hypot(h, r_base_cone)))
    # Test 3-sided triangular pyramid across 24 phase angles (0..120 deg)
    best_tri_p90 = 1e9
    best_tri_p50 = 1e9
    best_phase = 0.0
    cos_60 = 0.5
    pred_tri_r = r_base_tri * taper
    for deg in range(0, 120, 5):
        ph = float(np.radians(deg))
        # Regular triangle with vertices at ph + k*(2pi/3): edge midpoints are at ph + pi/3 + k*(2pi/3)
        rel_a = ((ang - ph) % (2.0 * np.pi / 3.0)) - (np.pi / 3.0)
        tri_r = pred_tri_r * cos_60 / np.maximum(np.cos(rel_a), 0.25)
        terr = np.abs(actual_r - tri_r) * (h / float(np.hypot(h, r_base_tri)))
        tp90 = float(np.percentile(terr, 90))
        if tp90 < best_tri_p90:
            best_tri_p90 = tp90
            best_tri_p50 = float(np.percentile(terr, 50))
            best_phase = ph

    cone_p50 = float(np.percentile(cone_err, 50))
    cone_p90 = float(np.percentile(cone_err, 90))
    if best_tri_p90 < cone_p90 - 0.005:
        return {
            "kind": "triangular_pyramid",
            "base_uvz": np.array([c_xy[0], c_xy[1], z0], dtype=np.float64),
            "apex_uvz": np.array([c_xy[0], c_xy[1], z1], dtype=np.float64),
            "radius": r_base_tri,
            "sides": 3,
            "phase_rad": best_phase,
            "p50": best_tri_p50,
            "p90": best_tri_p90,
        }
    return {
        "kind": "cone",
        "base_uvz": np.array([c_xy[0], c_xy[1], z0], dtype=np.float64),
        "apex_uvz": np.array([c_xy[0], c_xy[1], z1], dtype=np.float64),
        "radius": r_base_cone,
        "sides": 18,
        "phase_rad": 0.0,
        "p50": cone_p50,
        "p90": cone_p90,
    }


def _check_semantic_exclusion(
    sub_uvz: np.ndarray,
    sub_rgb: np.ndarray | None,
    pts_u: np.ndarray,
    z_floor: float,
    best_p90: float,
    cfg: ClutterConfig,
) -> str | None:
    """B18 semantic exclusions: plants/flowerpots and pedestrians take priority over local geometry."""
    # 1. RGB green foliage dominance -> plant / flowerpot
    if sub_rgb is not None and len(sub_rgb) > 0:
        r = sub_rgb[:, 0].astype(np.int32)
        g = sub_rgb[:, 1].astype(np.int32)
        b = sub_rgb[:, 2].astype(np.int32)
        green_mask = (g > r + 15) & (g > b + 15) & (g > 45)
        if float(green_mask.mean()) >= cfg.plant_green_ratio_threshold:
            return "excluded_semantic_plant_rgb"

    u0, u1 = float(np.percentile(pts_u[:, 0], 2)), float(np.percentile(pts_u[:, 0], 98))
    v0, v1 = float(np.percentile(pts_u[:, 1], 2)), float(np.percentile(pts_u[:, 1], 98))
    z0, z1 = float(np.percentile(pts_u[:, 2], 2)), float(np.percentile(pts_u[:, 2], 98))
    du, dv, dz = u1 - u0, v1 - v0, z1 - z0
    max_xy = max(du, dv)
    min_xy = min(du, dv)

    # 2. Geometric potted-plant / bush crown fallback (works even without RGB!):
    #    wider irregular upper foliage canopy above narrow pot stem + poor regular surface fit
    if dz >= 0.45 and max_xy >= 0.35 and best_p90 > 0.042:
        low_sl = pts_u[pts_u[:, 2] <= z0 + 0.35 * dz]
        hi_sl = pts_u[pts_u[:, 2] >= z0 + 0.55 * dz]
        if len(low_sl) >= 6 and len(hi_sl) >= 10:
            low_span = max(float(low_sl[:, 0].max() - low_sl[:, 0].min()), float(low_sl[:, 1].max() - low_sl[:, 1].min()))
            hi_span = max(float(hi_sl[:, 0].max() - hi_sl[:, 0].min()), float(hi_sl[:, 1].max() - hi_sl[:, 1].min()))
            if hi_span > max(low_span * 1.45, 0.38):
                return "excluded_semantic_plant_geometry"

    # 3. Pedestrian / human silhouette exclusion (floor-standing 1.45–1.98m irregular human figure)
    if (z0 - z_floor) <= 0.15 and 1.45 <= dz <= 1.98 and 0.36 <= max_xy <= 0.88 and 0.20 <= min_xy <= 0.58:
        if best_p90 > 0.048:
            return "excluded_semantic_pedestrian"
    return None


def _split_oversized_clutter_strip(
    sub_uvz: np.ndarray,
    sub_idx: np.ndarray,
    cfg: ClutterConfig,
    depth: int = 0,
) -> list[np.ndarray]:
    """Decompose oversized low-fill connected strips via 1D density valleys so we never emit giant noise AABBs (B18/G4)."""
    if len(sub_idx) < cfg.min_points:
        return []
    u0, u1 = float(np.percentile(sub_uvz[:, 0], 2)), float(np.percentile(sub_uvz[:, 0], 98))
    v0, v1 = float(np.percentile(sub_uvz[:, 1], 2)), float(np.percentile(sub_uvz[:, 1], 98))
    du, dv = u1 - u0, v1 - v0
    if max(du, dv) <= 2.20 or depth >= 2:
        return [sub_idx]

    axis = 0 if du >= dv else 1
    lo, hi = (u0, u1) if axis == 0 else (v0, v1)
    cuts = _find_1d_density_valleys(sub_uvz[:, axis], bin_size=0.06, min_peak_sep_m=0.35, valley_ratio=0.20)
    if not cuts:
        return [sub_idx]
    edges = [lo - 0.05] + cuts + [hi + 0.05]
    out: list[np.ndarray] = []
    for k in range(len(edges) - 1):
        m = (sub_uvz[:, axis] >= edges[k]) & (sub_uvz[:, axis] < edges[k + 1])
        if int(np.sum(m)) >= cfg.min_points:
            out.extend(_split_oversized_clutter_strip(sub_uvz[m], sub_idx[m], cfg, depth=depth + 1))
    return out


def build_clutter_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    existing_parts: list[MeshPart],
    table_specs: list[TableSpec],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ClutterConfig,
    *,
    rgb_uint8: np.ndarray | None = None,
) -> tuple[list[MeshPart], dict[str, Any]]:
    """Stage 3 B18 regular clutter builder: closed boxes, vertical/oblique cylinders, cones/pyramids + 0-penetration check."""
    if not cfg.enabled:
        return [], {"emitted_clutter": 0, "emitted_parts": 0, "clutter": [], "rejected": []}

    idx12 = np.flatnonzero(work_labels == 12)
    if len(idx12) < cfg.min_points:
        return [], {"emitted_clutter": 0, "emitted_parts": 0, "clutter": [], "rejected": []}

    pts12_local = points_local[idx12]
    pts12_uvz = fc.frame.pts_to_uvz(pts12_local)
    rgb12 = rgb_uint8[idx12] if rgb_uint8 is not None else None

    # 1. Strip exterior-shell-outside clutter points (window frames/curtains/facade bleed outside inner wall!)
    inner_poly = shell.inner_uv.buffer(-0.005)
    min_u, min_v, max_u, max_v = inner_poly.bounds
    in_bb = (
        (pts12_uvz[:, 0] >= min_u)
        & (pts12_uvz[:, 0] <= max_u)
        & (pts12_uvz[:, 1] >= min_v)
        & (pts12_uvz[:, 1] <= max_v)
    )
    # Fast vectorized interior check via shapely.contains_xy (shapely 2.x)
    from shapely import contains_xy

    in_shell = np.zeros(len(pts12_uvz), dtype=bool)
    if np.any(in_bb):
        in_shell[in_bb] = contains_xy(inner_poly, pts12_uvz[in_bb, 0], pts12_uvz[in_bb, 1])

    # Also drop points far above primary ceiling or far below primary floor
    z_top_lim = (fc.ceiling_z + 0.02) if fc.ceiling_z is not None else float(pts12_uvz[:, 2].max() + 1.0)
    in_shell &= (pts12_uvz[:, 2] >= fc.floor_z - 0.05) & (pts12_uvz[:, 2] <= z_top_lim)

    # Strip clutter points already inside existing interior solids using ROOM UV AABBs!
    # CRITICAL: NEVER use world-XY AABBs of rotated exterior walls (yaw=-32 deg swells a wall's world AABB across half the room!).
    # Exterior walls are already excluded by shell.inner_uv; here we mask only interior walls/columns/beams/cabinets/sofas/stairs.
    solid_mask_parts = [
        p for p in existing_parts
        if (p.class_id in (3, 4, 9, 10, 13, 14)) or (p.class_id == 2 and p.role == "interior_wall")
    ]
    keep_mask = in_shell.copy()
    if np.any(keep_mask) and solid_mask_parts:
        k_idx = np.flatnonzero(keep_mask)
        kp_uvz = pts12_uvz[k_idx]
        buried = np.zeros(len(k_idx), dtype=bool)
        for p in solid_mask_parts:
            p_uvz = fc.frame.pts_to_uvz(p.vertices)
            pmin = p_uvz.min(axis=0) - 0.015
            pmax = p_uvz.max(axis=0) + 0.015
            in_p = (
                (kp_uvz[:, 0] >= pmin[0]) & (kp_uvz[:, 0] <= pmax[0])
                & (kp_uvz[:, 1] >= pmin[1]) & (kp_uvz[:, 1] <= pmax[1])
                & (kp_uvz[:, 2] >= pmin[2]) & (kp_uvz[:, 2] <= pmax[2])
            )
            buried |= in_p
        keep_mask[k_idx[buried]] = False

    kept_rel = np.flatnonzero(keep_mask)
    if len(kept_rel) < cfg.min_points:
        return [], {
            "emitted_clutter": 0,
            "emitted_parts": 0,
            "clutter": [],
            "rejected": [{"reason": "all_clutter_outside_shell_or_inside_existing_solids", "raw_points": int(len(idx12))}],
        }

    # 2. Cluster interior class-12 points and split oversized low-fill strips
    sub_uvz_all = pts12_uvz[kept_rel]
    raw_comps = _voxel_connected_components_3d(sub_uvz_all, voxel_size=cfg.voxel_cluster_size_m)
    clusters: list[np.ndarray] = []
    for comp in raw_comps:
        if len(comp) < cfg.min_points:
            continue
        rel_comp = kept_rel[comp]
        clusters.extend(_split_oversized_clutter_strip(pts12_uvz[rel_comp], rel_comp, cfg, depth=0))
    clusters.sort(key=len, reverse=True)

    # Pre-index all existing solid colliders for exact real-mesh 0-penetration verification (B12)
    collider_list: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
        (p.vertices.min(axis=0), p.vertices.max(axis=0), p.vertices, p.triangles) for p in existing_parts
    ]
    collider_list.sort(key=lambda item: len(item[3]))

    inner_safe = shell.inner_uv.buffer(-0.004)
    parts: list[MeshPart] = []
    emitted_logs: list[dict[str, Any]] = []
    rejected_logs: list[dict[str, Any]] = []

    def _mesh_is_collision_free(v_xyz: np.ndarray, tri: np.ndarray) -> bool:
        v_uv = fc.frame.xy_to_uv(v_xyz[:, :2])
        # Check convex hull / corners stay inside interior shell
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
            if meshes_penetrate(v_xyz, tri, e_v, e_t, eps=5e-4)["penetrates"]:
                return False
        return True

    for ci, rel_idx in enumerate(clusters):
        sub_uvz = pts12_uvz[rel_idx]
        sub_rgb = rgb12[rel_idx] if rgb12 is not None else None
        pts_u = _voxel_uniform_3d(sub_uvz, cell=0.03)
        if len(pts_u) < 15:
            rejected_logs.append({"cluster_idx": ci, "point_count": int(len(rel_idx)), "reason": "too_few_unique_voxels"})
            continue

        z0_raw = float(np.percentile(pts_u[:, 2], 2))
        z1_raw = float(np.percentile(pts_u[:, 2], 98))
        u_mid, v_mid = float(np.median(pts_u[:, 0])), float(np.median(pts_u[:, 1]))
        z_floor = _local_floor_z(fc, u_mid, v_mid, z0_raw)
        z_ceil = _local_ceiling_z(fc, u_mid, v_mid, z_floor)

        # Reject anything outside vertical storey clear space
        if z1_raw > z_ceil + 0.04 or z0_raw < z_floor - 0.08:
            rejected_logs.append({"cluster_idx": ci, "point_count": int(len(rel_idx)), "reason": "outside_storey_z_band"})
            continue

        # Support snapping: check physical support (floor, tabletop, ceiling-suspended, or wall-mounted)!
        support_kind = "unsupported_floating"
        z0 = max(z0_raw, z_floor + 0.002)
        z1 = min(z1_raw, z_ceil - 0.002)
        if z0_raw <= z_floor + 0.06:
            z0 = z_floor + 0.002
            support_kind = "floor_resting"
        else:
            pt_uv = Point(u_mid, v_mid)
            for ts in table_specs:
                if abs(ts.z_floor - z_floor) <= 0.40 and ts.poly_uv.buffer(0.06).contains(pt_uv):
                    if -0.09 <= (z0_raw - ts.z_top) <= 0.12 and z1_raw > ts.z_top + 0.04:
                        z0 = ts.z_top + 0.004
                        support_kind = f"tabletop_{ts.table_id}"
                        break
            if support_kind == "unsupported_floating":
                if (z_ceil - z1_raw) <= 0.75:
                    support_kind = "ceiling_suspended"
                else:
                    # Check if mounted flush against interior shell wall (dist <= 0.12m)
                    dist_wall = float(shell.inner_uv.exterior.distance(pt_uv))
                    # Account for half-depth of object touching the wall
                    min_xy_raw = min(
                        float(np.percentile(pts_u[:, 0], 98) - np.percentile(pts_u[:, 0], 2)),
                        float(np.percentile(pts_u[:, 1], 98) - np.percentile(pts_u[:, 1], 2)),
                    )
                    if dist_wall <= 0.5 * min_xy_raw + 0.12:
                        support_kind = "wall_mounted"

        # Gravity / Anti-Floating-Noise Gate: reject unsupported mid-air noise clusters!
        if support_kind == "unsupported_floating":
            rejected_logs.append({
                "cluster_idx": ci,
                "point_count": int(len(rel_idx)),
                "z_above_floor_m": round(z0_raw - z_floor, 3),
                "reason": "rejected_unsupported_floating_noise",
            })
            continue
        if support_kind == "wall_mounted" and len(rel_idx) < 2000:
            rejected_logs.append({
                "cluster_idx": ci,
                "point_count": int(len(rel_idx)),
                "reason": "rejected_sparse_wall_bleed_noise",
            })
            continue
        if support_kind == "ceiling_suspended" and len(rel_idx) < 1000:
            rejected_logs.append({
                "cluster_idx": ci,
                "point_count": int(len(rel_idx)),
                "reason": "rejected_sparse_ceiling_noise",
            })
            continue
        if z1 <= z0 + 0.012:
            rejected_logs.append({"cluster_idx": ci, "point_count": int(len(rel_idx)), "reason": "zero_or_paper_thin_height"})
            continue
        # Ensure thin horizontal fixtures (like suspended LED light bars) have valid >= 2.5cm thickness
        if (z1 - z0) < 0.025:
            z_mid = 0.5 * (z0 + z1)
            z0 = max(z_floor + 0.002, z_mid - 0.014)
            z1 = min(z_ceil - 0.002, z_mid + 0.014)

        # Fit candidate 1: Min-area convex-hull OBB Box (immune to 1:1 square PCA tilt — G2!)
        c_uv, l_dir, d_dir, L, D = _fit_obb_2d(pts_u[:, :2], snap_deg=12.0)
        H = z1 - z0
        L = max(L, 0.035)
        D = max(D, 0.035)
        max_span = max(L, D, H)
        vol_box = L * D * H
        if max_span < cfg.min_major_span_m or vol_box < cfg.min_volume_m3:
            rejected_logs.append({
                "cluster_idx": ci,
                "point_count": int(len(rel_idx)),
                "dims_m": [round(L, 3), round(D, 3), round(H, 3)],
                "reason": "below_min_clutter_scale",
            })
            continue

        box_p50, box_p90, box_faces = _box_surface_residuals(pts_u, c_uv, l_dir, d_dir, L, D, z0, z1)
        candidates: list[dict[str, Any]] = [
            {
                "kind": "box",
                "center_uv": c_uv,
                "l_dir": l_dir,
                "d_dir": d_dir,
                "dims": (L, D, H),
                "z0": z0,
                "z1": z1,
                "p50": box_p50,
                "p90": box_p90,
                "supported_faces": box_faces,
            }
        ]

        # Fit candidate 2: Vertical Cylinder
        vcyl = _fit_vertical_cylinder(pts_u, z0, z1, min_arc_deg=cfg.min_cylinder_arc_deg)
        if vcyl is not None and 0.76 <= (D / max(L, 1e-4)) <= 1.32:
            candidates.append(vcyl)

        # Fit candidate 3: Oblique / Horizontal Cylinder (B18: 圆柱不局限竖直方向)
        ocyl = _fit_oblique_cylinder(pts_u, min_arc_deg=cfg.min_cylinder_arc_deg)
        if ocyl is not None:
            candidates.append(ocyl)

        # Fit candidate 4: Triangular Pyramid / Cone (B18: 三角锥 — MUST be floor/tabletop supported, never floating!)
        if support_kind == "floor_resting" or support_kind.startswith("tabletop_"):
            cone = _fit_cone_or_pyramid(pts_u, z0, z1)
            if cone is not None:
                candidates.append(cone)

        # Wall-mounted fixtures must be rectangular boxes (never floating cones/oblique cylinders!)
        if support_kind == "wall_mounted":
            candidates = [c for c in candidates if c["kind"] == "box"]

        # Pick best-fitting primitive (small 8mm parsimony bonus for box when tied)
        def _rank_key(cand: dict[str, Any]) -> float:
            bonus = -0.008 if cand["kind"] == "box" else 0.0
            return float(cand["p90"]) + 0.5 * float(cand["p50"]) + bonus

        best = min(candidates, key=_rank_key)

        # Semantic exclusion gate FIRST (B18: plants/flowerpots/pedestrians excluded even if locally regular!)
        excl = _check_semantic_exclusion(sub_uvz, sub_rgb, pts_u, z_floor, float(best["p90"]), cfg)
        if excl is not None:
            rejected_logs.append({
                "cluster_idx": ci,
                "point_count": int(len(rel_idx)),
                "dims_m": [round(L, 3), round(D, 3), round(H, 3)],
                "best_p90_m": round(float(best["p90"]), 4),
                "reason": excl,
            })
            continue

        # B18 regularity & anti-over-envelope gate!
        is_thin_fixture = (min(L, D, H) <= 0.14) and (float(best["p90"]) <= 0.032) and (len(pts_u) >= 25)
        # Allow slightly wider p90 (<= 7.8cm) ONLY for dense, grounded 3+-face boxes (e.g., recessed fire-hydrant box
        # whose door panel or wall-mixed corner adds a 7cm tail while p50 <= 3.0cm stays razor-tight!)
        is_dense_grounded_box = (
            best["kind"] == "box"
            and support_kind == "floor_resting"
            and len(rel_idx) >= 3000
            and best["supported_faces"] >= 3
            and vol_box <= 0.65
            and float(best["p50"]) <= 0.030
        )
        max_p90_allow = 0.078 if is_dense_grounded_box else cfg.regularity_max_surface_p90_m
        if float(best["p90"]) > max_p90_allow or float(best["p50"]) > cfg.regularity_max_surface_p50_m:
            rejected_logs.append({
                "cluster_idx": ci,
                "point_count": int(len(rel_idx)),
                "dims_m": [round(L, 3), round(D, 3), round(H, 3)],
                "best_kind": best["kind"],
                "best_p50_m": round(float(best["p50"]), 4),
                "best_p90_m": round(float(best["p90"]), 4),
                "reason": "irregular_or_sparse_fit_failed",
            })
            continue

        # Reject single-face hollow giant boxes (B18: 单面、稀疏弧段或噪点大包围盒不足以证明完整实体)
        if best["kind"] == "box" and (not is_thin_fixture):
            if best["supported_faces"] < 2:
                rejected_logs.append({
                    "cluster_idx": ci,
                    "point_count": int(len(rel_idx)),
                    "dims_m": [round(L, 3), round(D, 3), round(H, 3)],
                    "supported_faces": best["supported_faces"],
                    "reason": "single_face_insufficient_for_solid_box",
                })
                continue
            # Reject huge bloated low-density clouds (volume > 1.2 m³ with very sparse surface occupancy)
            approx_surf_area = 2.0 * (L * D + L * H + D * H)
            vox_surf_area = len(pts_u) * (0.03 * 0.03)
            if vol_box > 1.2 and (vox_surf_area / max(approx_surf_area, 1e-4)) < 0.14:
                rejected_logs.append({
                    "cluster_idx": ci,
                    "point_count": int(len(rel_idx)),
                    "dims_m": [round(L, 3), round(D, 3), round(H, 3)],
                    "reason": "giant_sparse_envelope_rejected",
                })
                continue

        # Build watertight 3D mesh with bounded gentle shrink/lift attempts to resolve <=3cm contact grazing
        placed_mesh: tuple[np.ndarray, np.ndarray] | None = None
        for shrink, lift_z in ((1.0, 0.0), (0.98, 0.004), (0.95, 0.010), (0.92, 0.020)):
            try:
                if best["kind"] == "box":
                    z_lo_try = min(z0 + lift_z, z1 - 0.02)
                    c_try = np.array([c_uv[0], c_uv[1], 0.5 * (z_lo_try + z1)], dtype=np.float64)
                    au = np.array([l_dir[0], l_dir[1], 0.0], dtype=np.float64)
                    av = np.array([d_dir[0], d_dir[1], 0.0], dtype=np.float64)
                    aw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    v_uvz, tri = make_oriented_box(
                        c_try, au, av, aw, (max(L * shrink, 0.03), max(D * shrink, 0.03), max(z1 - z_lo_try, 0.02))
                    )
                elif best["kind"] in ("cylinder_vertical", "cylinder_oblique"):
                    p0 = best["p0_uvz"].copy()
                    p1 = best["p1_uvz"].copy()
                    if best["kind"] == "cylinder_vertical":
                        p0[2] = min(p0[2] + lift_z, p1[2] - 0.02)
                    else:
                        mid = 0.5 * (p0 + p1)
                        p0 = mid + (p0 - mid) * shrink
                        p1 = mid + (p1 - mid) * shrink
                        p0[2] += lift_z
                        p1[2] += lift_z
                    v_uvz, tri = make_oriented_cylinder(
                        p0, p1, max(float(best["radius"]) * shrink, 0.025), segments=cfg.cylinder_segments
                    )
                    if best["kind"] == "cylinder_oblique" and float(v_uvz[:, 2].min()) < z_floor + 0.002:
                        v_uvz[:, 2] += (z_floor + 0.002) - float(v_uvz[:, 2].min())
                else:  # triangular_pyramid or cone
                    bp = best["base_uvz"].copy()
                    ap = best["apex_uvz"].copy()
                    bp[2] = min(bp[2] + lift_z, ap[2] - 0.03)
                    v_uvz, tri = make_oriented_cone_or_pyramid(
                        bp,
                        ap,
                        max(float(best["radius"]) * shrink, 0.03),
                        sides=int(best["sides"]),
                        phase_rad=float(best["phase_rad"]),
                    )
            except ValueError:
                continue

            v_xyz = fc.frame.uvz_to_xyz(v_uvz)
            if _mesh_is_collision_free(v_xyz, tri):
                placed_mesh = (v_xyz, tri)
                break

        if placed_mesh is None:
            rejected_logs.append({
                "cluster_idx": ci,
                "point_count": int(len(rel_idx)),
                "dims_m": [round(L, 3), round(D, 3), round(H, 3)],
                "best_kind": best["kind"],
                "reason": "penetrates_existing_solid_or_shell",
            })
            continue

        v_xyz, tri = placed_mesh
        pid = f"clutter_{len(parts):02d}"
        parts.append(
            MeshPart(
                part_id=pid,
                class_id=12,
                role=f"clutter_{best['kind']}",
                vertices=v_xyz,
                triangles=tri,
                evidence_kind="measured",
                metadata={
                    "clutter_id": pid,
                    "primitive_type": best["kind"],
                    "support_kind": support_kind,
                    "p50_residual_m": round(float(best["p50"]), 4),
                    "p90_residual_m": round(float(best["p90"]), 4),
                    "point_count": int(len(rel_idx)),
                },
            )
        )
        collider_list.append((v_xyz.min(axis=0), v_xyz.max(axis=0), v_xyz, tri))
        emitted_logs.append({
            "clutter_id": pid,
            "primitive_type": best["kind"],
            "support_kind": support_kind,
            "center_uv": [round(u_mid, 3), round(v_mid, 3)],
            "dims_ldh_m": [round(L, 3), round(D, 3), round(H, 3)],
            "z_span": [round(z0, 4), round(z1, 4)],
            "p50_residual_m": round(float(best["p50"]), 4),
            "p90_residual_m": round(float(best["p90"]), 4),
            "point_count": int(len(rel_idx)),
        })

    return parts, {
        "emitted_clutter": len(emitted_logs),
        "emitted_parts": len(parts),
        "by_primitive_type": {
            k: sum(1 for e in emitted_logs if e["primitive_type"] == k)
            for k in sorted({e["primitive_type"] for e in emitted_logs})
        },
        "clutter": emitted_logs,
        "rejected": rejected_logs,
    }
