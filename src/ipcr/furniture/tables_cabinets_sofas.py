"""Stage 2 B15 structural reconstruction for tables (cls 8), cabinets (cls 9), and sofas (cls 10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ipcr.architecture.frame_floors import FloorCeilingResult, _raster_mask_to_polygon, _snap_orthogonal_polygon
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import FurnitureConfig
from ipcr.geometry_primitives import extrude_polygon_solid
from ipcr.types import MeshPart
from ipcr.working_labels import _voxel_connected_components_3d


@dataclass
class TableSpec:
    table_id: str
    poly_uv: Polygon
    z_floor: float
    z_top: float
    tabletop_z_bot: float
    leg_polys_uv: list[Polygon] = field(default_factory=list)
    is_rectangular: bool = True


def _local_floor_z(fc: FloorCeilingResult, u: float, v: float, z_lo_meas: float) -> float:
    """Resolve supporting storey floor elevation under UV point (u, v) in single or multi-storey scenes."""
    best = fc.floor_z
    pt = Point(float(u), float(v))
    for flv in fc.floor_levels:
        if flv.z_top <= z_lo_meas + 0.35 and flv.z_top > best:
            if flv.is_primary or flv.polygon_uv.buffer(0.05).contains(pt):
                best = flv.z_top
    return best


def _extract_contour_uv(pts_uv: np.ndarray, *, cell: float = 0.05) -> Polygon | None:
    """Extract clean hole-free functional 2D contour from top-surface points (preserves L/U shapes per B15!)."""
    if len(pts_uv) < 30:
        return None
    pad = 0.25
    u0, v0 = float(pts_uv[:, 0].min()) - pad, float(pts_uv[:, 1].min()) - pad
    u1, v1 = float(pts_uv[:, 0].max()) + pad, float(pts_uv[:, 1].max()) + pad
    nu = int(np.ceil((u1 - u0) / cell)) + 2
    nv = int(np.ceil((v1 - v0) / cell)) + 2
    if nu * nv > 2_000_000:
        cell = 0.08
        nu = int(np.ceil((u1 - u0) / cell)) + 2
        nv = int(np.ceil((v1 - v0) / cell)) + 2
    grid = np.zeros((nu, nv), dtype=bool)
    iu = np.clip(np.floor((pts_uv[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
    iv = np.clip(np.floor((pts_uv[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
    grid[iu, iv] = True
    closed = ndimage.binary_closing(grid, structure=np.ones((3, 3), dtype=bool), iterations=2)
    try:
        raw = _raster_mask_to_polygon(closed, u0, v0, cell)
    except ValueError:
        return None
    if raw.is_empty or raw.area < 0.05:
        return None
    return _snap_orthogonal_polygon(raw, simplify_tol=0.06, min_edge_len=0.15)


def build_table_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    static_blockers_uv: list[Polygon],
    z_blocker_bands: list[tuple[Polygon, float, float]],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: FurnitureConfig,
) -> tuple[list[MeshPart], list[TableSpec], dict[str, Any]]:
    """Build B15 open-frame structural tables (exact functional tabletop contour + open under-desk legs)."""
    uvz = fc.frame.pts_to_uvz(points_local)
    tpts = uvz[work_labels == 8]
    if len(tpts) < cfg.table_min_points:
        return [], [], {"emitted_tables": 0, "emitted_parts": 0, "tables": [], "rejected": []}

    allowed_base = shell.inner_uv.buffer(-cfg.clearance_m)
    if static_blockers_uv:
        allowed_base = allowed_base.difference(unary_union([p.buffer(cfg.clearance_m) for p in static_blockers_uv])).buffer(0)

    comps = _voxel_connected_components_3d(tpts, voxel_size=0.09)
    parts: list[MeshPart] = []
    specs: list[TableSpec] = []
    logs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    placed_union: Polygon | MultiPolygon | None = None

    for ci, comp in enumerate(comps):
        if len(comp) < cfg.table_min_points:
            continue
        sub = tpts[comp]
        u_mid, v_mid = float(np.median(sub[:, 0])), float(np.median(sub[:, 1]))
        z_lo_raw = float(np.percentile(sub[:, 2], 3))
        z_floor = _local_floor_z(fc, u_mid, v_mid, z_lo_raw)
        z_top = float(np.percentile(sub[:, 2], 94))
        h = z_top - z_floor

        if h < cfg.table_min_height_m or h > cfg.table_max_height_m:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "height_above_floor_m": round(h, 3),
                "reason": "height_out_of_table_range",
            })
            continue

        # Extract tabletop surface points (top 14cm slice) to preserve exact functional contour (B15)
        top_slice = sub[sub[:, 2] >= z_top - 0.14]
        if len(top_slice) < 80:
            top_slice = sub
        contour = _extract_contour_uv(top_slice[:, :2], cell=0.05)
        if contour is None:
            continue

        # Decide rectangular vs genuine L/U-shaped contour via fill ratio against tight axis-aligned box
        u0, v0, u1, v1 = contour.bounds
        aabb_poly = box(u0, v0, u1, v1)
        fill_ratio = float(contour.area / max(aabb_poly.area, 1e-6))
        is_rect = fill_ratio >= 0.80
        if is_rect:
            # Use crisp 2-98% measured UV rectangle for clean straight tabletop edges
            ru0, ru1 = float(np.percentile(top_slice[:, 0], 2)), float(np.percentile(top_slice[:, 0], 98))
            rv0, rv1 = float(np.percentile(top_slice[:, 1], 2)), float(np.percentile(top_slice[:, 1], 98))
            poly_uv = box(ru0, rv0, ru1, rv1)
        else:
            poly_uv = contour

        # Clip strictly inside allowed room shell minus stairs/railings overlapping vertically & prior tables
        cur_allowed = allowed_base
        ov_bands = [p.buffer(cfg.clearance_m) for p, bz0, bz1 in z_blocker_bands if bz0 <= z_top + 0.02 and bz1 >= z_floor - 0.02]
        if ov_bands:
            cur_allowed = cur_allowed.difference(unary_union(ov_bands))
        if placed_union is not None and not placed_union.is_empty:
            cur_allowed = cur_allowed.difference(placed_union.buffer(cfg.clearance_m))

        clipped = poly_uv.intersection(cur_allowed).buffer(0)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if not isinstance(clipped, Polygon) or clipped.is_empty or clipped.area < cfg.table_min_area_m2:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "reason": "below_min_area_after_shell_clip",
            })
            continue

        poly_clean = orient(clipped.buffer(0), sign=1.0)
        if isinstance(poly_clean, MultiPolygon):
            poly_clean = max(poly_clean.geoms, key=lambda g: g.area)
        minx, miny, maxx, maxy = poly_clean.bounds
        du, dv = maxx - minx, maxy - miny
        if min(du, dv) < 0.28:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "extent_uv_m": [round(du, 3), round(dv, 3)],
                "reason": "too_narrow_for_table",
            })
            continue

        top_bot = max(z_floor + 0.15, z_top - cfg.table_top_thickness_m)
        if z_top - top_bot < 0.02:
            continue

        try:
            tv_uv, tt = extrude_polygon_solid(poly_clean, top_bot, z_top)
        except ValueError:
            continue

        tid = f"table_{len(specs):02d}"
        parts.append(
            MeshPart(
                part_id=f"{tid}_top",
                class_id=8,
                role="table_top",
                vertices=fc.frame.uvz_to_xyz(tv_uv),
                triangles=tt,
                evidence_kind="parametric_rule_fallback",
                metadata={
                    "table_id": tid,
                    "shape": "rectangular" if is_rect else "shaped_contour",
                    "area_m2": round(float(poly_clean.area), 3),
                    "extent_uv_m": [round(du, 3), round(dv, 3)],
                    "z_top_above_floor_m": round(z_top - z_floor, 3),
                    "point_count": int(len(comp)),
                    "library_fallback_reason": "raw_library_table_glbs_non_watertight_b21",
                },
            )
        )

        # Build open structural legs strictly beneath tabletop (leaving open knee/chair tuck space per B15!)
        leg_sz = cfg.table_leg_size_m
        leg_z0 = z_floor + 0.002
        leg_z1 = top_bot - 0.002
        leg_polys: list[Polygon] = []

        if leg_z1 - leg_z0 >= 0.25:
            leg_centers: list[tuple[float, float]] = []
            if is_rect and len(list(poly_clean.exterior.coords)) <= 6:
                # Place legs slightly inset so tucked chairs along long sides never hit legs
                long_u = du >= dv
                in_u = min(0.18 if (not long_u and du >= 1.1) else 0.09, du * 0.25)
                in_v = min(0.18 if (long_u and dv >= 1.1) else 0.09, dv * 0.25)
                u_len = du - 2 * in_u
                v_len = dv - 2 * in_v
                nu_legs = max(2, int(np.ceil(u_len / 2.2)) + 1) if long_u else 2
                nv_legs = max(2, int(np.ceil(v_len / 2.2)) + 1) if not long_u else 2
                us = np.linspace(minx + in_u, maxx - in_u, nu_legs)
                vs = np.linspace(miny + in_v, maxy - in_v, nv_legs)
                for lu in us:
                    for lv in vs:
                        leg_centers.append((float(lu), float(lv)))
            else:
                # Shaped/L-table: place legs at vertices of inset polygon
                inset = poly_clean.buffer(-0.10, join_style=2)
                if isinstance(inset, MultiPolygon):
                    inset = max(inset.geoms, key=lambda g: g.area)
                if isinstance(inset, Polygon) and not inset.is_empty:
                    for cx, cy in list(inset.exterior.coords)[:-1]:
                        if all(np.hypot(cx - ox, cy - oy) >= 0.35 for ox, oy in leg_centers):
                            leg_centers.append((float(cx), float(cy)))

            for lidx, (lu, lv) in enumerate(leg_centers):
                lbox = box(lu - 0.5 * leg_sz, lv - 0.5 * leg_sz, lu + 0.5 * leg_sz, lv + 0.5 * leg_sz)
                if not poly_clean.contains(lbox) or not cur_allowed.contains(lbox):
                    continue
                if any(lbox.intersects(prev.buffer(0.01)) for prev in leg_polys):
                    continue
                try:
                    lv_uv, lt = extrude_polygon_solid(lbox, leg_z0, leg_z1)
                except ValueError:
                    continue
                parts.append(
                    MeshPart(
                        part_id=f"{tid}_leg_{lidx:02d}",
                        class_id=8,
                        role="table_leg",
                        vertices=fc.frame.uvz_to_xyz(lv_uv),
                        triangles=lt,
                        evidence_kind="parametric_rule_fallback",
                        metadata={"table_id": tid},
                    )
                )
                leg_polys.append(lbox)

        specs.append(
            TableSpec(
                table_id=tid,
                poly_uv=poly_clean,
                z_floor=z_floor,
                z_top=z_top,
                tabletop_z_bot=top_bot,
                leg_polys_uv=leg_polys,
                is_rectangular=is_rect,
            )
        )
        placed_union = poly_clean if placed_union is None else unary_union([placed_union, poly_clean])
        logs.append({
            "table_id": tid,
            "shape": "rectangular" if is_rect else "shaped_contour",
            "extent_uv_m": [round(du, 3), round(dv, 3)],
            "area_m2": round(float(poly_clean.area), 3),
            "z_span": [round(z_floor, 4), round(z_top, 4)],
            "leg_count": len(leg_polys),
            "point_count": int(len(comp)),
        })

    return parts, specs, {
        "emitted_tables": len(specs),
        "emitted_parts": len(parts),
        "tables": logs,
        "rejected": rejected,
    }


def build_cabinet_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    static_blockers_uv: list[Polygon],
    z_blocker_bands: list[tuple[Polygon, float, float]],
    table_specs: list[TableSpec],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: FurnitureConfig,
) -> tuple[list[MeshPart], list[tuple[Polygon, float, float]], dict[str, Any]]:
    """Build B15/B12 cabinets (freestanding/wall-side cabinets + under-table pedestals, never swallowing tables)."""
    uvz = fc.frame.pts_to_uvz(points_local)
    cpts = uvz[work_labels == 9]
    if len(cpts) < cfg.cabinet_min_points:
        return [], [], {"emitted_cabinets": 0, "cabinets": [], "rejected": []}

    allowed_base = shell.inner_uv.buffer(-cfg.clearance_m)
    if static_blockers_uv:
        allowed_base = allowed_base.difference(unary_union([p.buffer(cfg.clearance_m) for p in static_blockers_uv])).buffer(0)

    comps = _voxel_connected_components_3d(cpts, voxel_size=0.10)
    parts: list[MeshPart] = []
    cab_bands: list[tuple[Polygon, float, float]] = []
    logs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    placed_union: Polygon | MultiPolygon | None = None

    for ci, comp in enumerate(comps):
        if len(comp) < cfg.cabinet_min_points:
            continue
        sub = cpts[comp]
        u0, u1 = float(np.percentile(sub[:, 0], 2)), float(np.percentile(sub[:, 0], 98))
        v0, v1 = float(np.percentile(sub[:, 1], 2)), float(np.percentile(sub[:, 1], 98))
        du, dv = u1 - u0, v1 - v0
        z_lo_raw = float(np.percentile(sub[:, 2], 2))
        z_hi_raw = float(np.percentile(sub[:, 2], 98))
        z_floor = _local_floor_z(fc, 0.5 * (u0 + u1), 0.5 * (v0 + v1), z_lo_raw)

        if min(du, dv) < cfg.cabinet_min_depth_m or max(du, dv) < cfg.cabinet_min_width_m:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "extent_uv_m": [round(du, 3), round(dv, 3)],
                "reason": "paper_thin_or_too_small_for_cabinet",
            })
            continue
        if z_hi_raw - z_lo_raw < cfg.cabinet_min_height_m:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "height_m": round(z_hi_raw - z_lo_raw, 3),
                "reason": "below_min_cabinet_height",
            })
            continue

        # Ground near-floor cabinets cleanly; reject tiny high floating specks
        if z_lo_raw - z_floor <= 0.45:
            z_bot = z_floor + 0.002
        elif len(comp) >= 2000 and min(du, dv) >= 0.22:
            z_bot = z_lo_raw
        else:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "z_above_floor_m": round(z_lo_raw - z_floor, 3),
                "reason": "floating_wall_bleed_rejected",
            })
            continue

        z_top = z_hi_raw
        if fc.ceiling_z is not None:
            z_top = min(z_top, fc.ceiling_z - 0.005)
        # Cap beneath any overhead upper floor or secondary ceiling
        cand_rect = box(u0, v0, u1, v1)
        for flv in fc.floor_levels:
            if not flv.is_primary and flv.polygon_uv.intersects(cand_rect):
                fbot = flv.z_top - 0.12
                if z_bot < fbot - 0.20 and z_top > fbot - 0.005:
                    z_top = fbot - 0.005
        for clv in fc.ceiling_levels:
            if not clv.is_primary and clv.polygon_uv.intersects(cand_rect):
                if z_bot < clv.z_bottom - 0.20 and z_top > clv.z_bottom - 0.005:
                    z_top = clv.z_bottom - 0.005

        # B12/B15 Table Clearance: under-table pedestal stays strictly below tabletop; tall cabinet never swallows table!
        extra_blockers: list[Polygon] = [
            p.buffer(cfg.clearance_m) for p, bz0, bz1 in z_blocker_bands if bz0 <= z_top + 0.02 and bz1 >= z_bot - 0.02
        ]
        role = "cabinet"
        for ts in table_specs:
            if not ts.poly_uv.intersects(cand_rect):
                continue
            if z_hi_raw <= ts.z_top + 0.05:
                # Genuine under-table pedestal / modesty panel: cap strictly below tabletop bottom & avoid legs!
                z_top = min(z_top, ts.tabletop_z_bot - 0.006)
                extra_blockers.extend(lp.buffer(cfg.clearance_m) for lp in ts.leg_polys_uv)
                role = "cabinet_under_table"
            else:
                # Freestanding tall cabinet adjacent to table: subtract table XY footprint so it NEVER swallows table!
                extra_blockers.append(ts.poly_uv.buffer(cfg.clearance_m))

        if z_top - z_bot < cfg.cabinet_min_height_m:
            rejected.append({"cluster_idx": ci, "point_count": int(len(comp)), "reason": "insufficient_height_after_clearance"})
            continue

        cur_allowed = allowed_base
        if extra_blockers:
            cur_allowed = cur_allowed.difference(unary_union(extra_blockers))
        if placed_union is not None and not placed_union.is_empty:
            cur_allowed = cur_allowed.difference(placed_union.buffer(cfg.clearance_m))

        clipped = cand_rect.intersection(cur_allowed).buffer(0)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if not isinstance(clipped, Polygon) or clipped.is_empty or clipped.area < 0.05:
            rejected.append({"cluster_idx": ci, "point_count": int(len(comp)), "reason": "eliminated_by_clearance"})
            continue

        poly_clean = orient(clipped.buffer(0), sign=1.0)
        if isinstance(poly_clean, MultiPolygon):
            poly_clean = max(poly_clean.geoms, key=lambda g: g.area)
        try:
            cv_uv, ct = extrude_polygon_solid(poly_clean, z_bot, z_top)
        except ValueError:
            continue

        pid = f"cabinet_{len(parts):02d}"
        parts.append(
            MeshPart(
                part_id=pid,
                class_id=9,
                role=role,
                vertices=fc.frame.uvz_to_xyz(cv_uv),
                triangles=ct,
                evidence_kind="parametric_rule_fallback",
                metadata={
                    "extent_uv_m": [round(du, 3), round(dv, 3)],
                    "area_m2": round(float(poly_clean.area), 3),
                    "z_span": [round(z_bot, 4), round(z_top, 4)],
                    "point_count": int(len(comp)),
                    "library_fallback_reason": "raw_library_cabinet_glbs_non_watertight_b21",
                },
            )
        )
        cab_bands.append((poly_clean, z_bot, z_top))
        placed_union = poly_clean if placed_union is None else unary_union([placed_union, poly_clean])
        logs.append(parts[-1].metadata | {"part_id": pid, "role": role})

    return parts, cab_bands, {
        "emitted_cabinets": len(parts),
        "cabinets": logs,
        "rejected": rejected,
    }


def build_sofa_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    static_blockers_uv: list[Polygon],
    z_blocker_bands: list[tuple[Polygon, float, float]],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: FurnitureConfig,
) -> tuple[list[MeshPart], list[tuple[Polygon, float, float]], dict[str, Any]]:
    """Build B15 multi-part structural sofas (seat + backrest + armrests) while rejecting thin wall-bleed."""
    uvz = fc.frame.pts_to_uvz(points_local)
    spts = uvz[work_labels == 10]
    if len(spts) < cfg.sofa_min_points:
        return [], [], {"emitted_sofas": 0, "sofas": [], "rejected": []}

    allowed_base = shell.inner_uv.buffer(-cfg.clearance_m)
    if static_blockers_uv:
        allowed_base = allowed_base.difference(unary_union([p.buffer(cfg.clearance_m) for p in static_blockers_uv])).buffer(0)

    comps = _voxel_connected_components_3d(spts, voxel_size=0.10)
    parts: list[MeshPart] = []
    sofa_bands: list[tuple[Polygon, float, float]] = []
    logs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    placed_union: Polygon | MultiPolygon | None = None

    for ci, comp in enumerate(comps):
        if len(comp) < cfg.sofa_min_points:
            continue
        sub = spts[comp]
        u0, u1 = float(np.percentile(sub[:, 0], 2)), float(np.percentile(sub[:, 0], 98))
        v0, v1 = float(np.percentile(sub[:, 1], 2)), float(np.percentile(sub[:, 1], 98))
        du, dv = u1 - u0, v1 - v0
        z_lo = float(np.percentile(sub[:, 2], 2))
        z_hi = float(np.percentile(sub[:, 2], 98))
        z_floor = _local_floor_z(fc, 0.5 * (u0 + u1), 0.5 * (v0 + v1), z_lo)
        h = z_hi - z_floor

        if min(du, dv) < cfg.sofa_min_depth_m or max(du, dv) < cfg.sofa_min_width_m or h < cfg.sofa_min_height_m:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "extent_uv_m": [round(du, 3), round(dv, 3)],
                "height_m": round(h, 3),
                "reason": "below_min_sofa_dimensions",
            })
            continue
        if z_lo - z_floor > 0.35:
            rejected.append({"cluster_idx": ci, "point_count": int(len(comp)), "reason": "floating_sofa_rejected"})
            continue

        cand_rect = box(u0, v0, u1, v1)
        cur_allowed = allowed_base
        ov = [p.buffer(cfg.clearance_m) for p, bz0, bz1 in z_blocker_bands if bz0 <= z_hi + 0.02 and bz1 >= z_floor - 0.02]
        if ov:
            cur_allowed = cur_allowed.difference(unary_union(ov))
        if placed_union is not None and not placed_union.is_empty:
            cur_allowed = cur_allowed.difference(placed_union.buffer(cfg.clearance_m))

        clipped = cand_rect.intersection(cur_allowed).buffer(0)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if not isinstance(clipped, Polygon) or clipped.is_empty or clipped.area < 0.35:
            rejected.append({"cluster_idx": ci, "point_count": int(len(comp)), "reason": "eliminated_by_clearance"})
            continue

        minx, miny, maxx, maxy = clipped.bounds
        w_u, w_v = maxx - minx, maxy - miny
        if min(w_u, w_v) < cfg.sofa_min_depth_m * 0.85:
            continue

        # Determine backrest side (high-Z centroid vs low-Z centroid along shorter depth axis)
        high_pts = sub[sub[:, 2] >= z_floor + 0.65 * h]
        low_pts = sub[sub[:, 2] <= z_floor + 0.45 * h]
        z_seat = z_floor + min(0.44, max(0.32, 0.50 * h))
        z_arm = z_floor + min(0.64, max(z_seat + 0.14, 0.72 * h))
        back_t = min(0.22, min(w_u, w_v) * 0.28)
        arm_t = min(0.14, max(w_u, w_v) * 0.12)

        sid = f"sofa_{len(logs):02d}"
        if w_u >= w_v:
            # Sofa spans U, depth along V
            back_at_max_v = (
                float(high_pts[:, 1].mean()) > float(low_pts[:, 1].mean())
                if len(high_pts) and len(low_pts)
                else True
            )
            if back_at_max_v:
                back_box = box(minx, maxy - back_t, maxx, maxy)
                seat_box = box(minx + arm_t + 0.004, miny, maxx - arm_t - 0.004, maxy - back_t - 0.004)
                arm_l = box(minx, miny, minx + arm_t, maxy - back_t - 0.004)
                arm_r = box(maxx - arm_t, miny, maxx, maxy - back_t - 0.004)
            else:
                back_box = box(minx, miny, maxx, miny + back_t)
                seat_box = box(minx + arm_t + 0.004, miny + back_t + 0.004, maxx - arm_t - 0.004, maxy)
                arm_l = box(minx, miny + back_t + 0.004, minx + arm_t, maxy)
                arm_r = box(maxx - arm_t, miny + back_t + 0.004, maxx, maxy)
        else:
            # Sofa spans V, depth along U
            back_at_max_u = (
                float(high_pts[:, 0].mean()) > float(low_pts[:, 0].mean())
                if len(high_pts) and len(low_pts)
                else True
            )
            if back_at_max_u:
                back_box = box(maxx - back_t, miny, maxx, maxy)
                seat_box = box(minx, miny + arm_t + 0.004, maxx - back_t - 0.004, maxy - arm_t - 0.004)
                arm_l = box(minx, miny, maxx - back_t - 0.004, miny + arm_t)
                arm_r = box(minx, maxy - arm_t, maxx - back_t - 0.004, maxy)
            else:
                back_box = box(minx, miny, minx + back_t, maxy)
                seat_box = box(minx + back_t + 0.004, miny + arm_t + 0.004, maxx, maxy - arm_t - 0.004)
                arm_l = box(minx + back_t + 0.004, miny, maxx, miny + arm_t)
                arm_r = box(minx + back_t + 0.004, maxy - arm_t, maxx, maxy)

        emitted_sub = 0
        for sub_name, sub_poly, z1 in (
            ("back", back_box, z_hi),
            ("seat", seat_box, z_seat),
            ("arm_l", arm_l, z_arm),
            ("arm_r", arm_r, z_arm),
        ):
            sp = sub_poly.intersection(cur_allowed).buffer(0)
            if isinstance(sp, MultiPolygon):
                sp = max(sp.geoms, key=lambda g: g.area)
            if not isinstance(sp, Polygon) or sp.is_empty or sp.area < 0.015:
                continue
            try:
                sv_uv, st = extrude_polygon_solid(orient(sp, sign=1.0), z_floor + 0.002, z1)
            except ValueError:
                continue
            parts.append(
                MeshPart(
                    part_id=f"{sid}_{sub_name}",
                    class_id=10,
                    role=f"sofa_{sub_name}",
                    vertices=fc.frame.uvz_to_xyz(sv_uv),
                    triangles=st,
                    evidence_kind="parametric_rule_fallback",
                    metadata={"sofa_id": sid, "library_fallback_reason": "raw_library_sofa_glbs_non_watertight_b21"},
                )
            )
            emitted_sub += 1

        if emitted_sub > 0:
            sofa_bands.append((clipped, z_floor, z_hi))
            placed_union = clipped if placed_union is None else unary_union([placed_union, clipped])
            logs.append({
                "sofa_id": sid,
                "extent_uv_m": [round(w_u, 3), round(w_v, 3)],
                "height_m": round(h, 3),
                "parts": emitted_sub,
                "point_count": int(len(comp)),
            })

    return parts, sofa_bands, {
        "emitted_sofas": len(logs),
        "emitted_parts": len(parts),
        "sofas": logs,
        "rejected": rejected,
    }
