"""Columns (class 4) and clean orthogonal-strip overhead beams (class 3) clipped inside room shell."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ipcr.architecture.frame_floors import FloorCeilingResult
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import ArchitectureConfig
from ipcr.geometry_primitives import extrude_polygon_solid
from ipcr.types import MeshPart
from ipcr.working_labels import _voxel_connected_components_3d


def _lowest_overhead_slab_bottom(
    poly_uv: Polygon,
    z_ref_bot: float,
    fc: FloorCeilingResult,
    cfg: ArchitectureConfig,
    fallback_top: float,
) -> float:
    """Find the lowest overhead slab bottom (upper floor bottom or ceiling bottom) above poly_uv."""
    candidates: list[float] = []
    if fc.ceiling_z is not None and fc.ceiling_z > z_ref_bot + 0.15:
        candidates.append(fc.ceiling_z)
    for clv in fc.ceiling_levels:
        if clv.z_bottom > z_ref_bot + 0.15 and clv.polygon_uv.intersects(poly_uv):
            inter = clv.polygon_uv.intersection(poly_uv)
            if inter.area >= min(poly_uv.area * 0.25, 0.15):
                candidates.append(clv.z_bottom)
    for flv in fc.floor_levels:
        if flv.is_primary:
            continue
        slab_bot = flv.z_top - cfg.floor_thickness_m
        if slab_bot > z_ref_bot + 0.15 and flv.polygon_uv.intersects(poly_uv):
            inter = flv.polygon_uv.intersection(poly_uv)
            if inter.area >= min(poly_uv.area * 0.20, 0.10):
                candidates.append(slab_bot)
    return min(candidates) if candidates else fallback_top


def _supporting_floor_top(
    poly_uv: Polygon,
    z_meas_lo: float,
    fc: FloorCeilingResult,
) -> float:
    """Snap a structural base to the closest floor level beneath it (within 0.45m), else floor_z."""
    best = fc.floor_z
    for flv in fc.floor_levels:
        if flv.z_top <= z_meas_lo + 0.30:
            if flv.is_primary or flv.polygon_uv.intersects(poly_uv):
                if abs(z_meas_lo - flv.z_top) <= 0.45 and flv.z_top > best:
                    best = flv.z_top
    return best


def build_column_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    interior_wall_polys_uv: list[Polygon],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
) -> tuple[list[MeshPart], list[Polygon], list[tuple[Polygon, float, float]], dict[str, Any]]:
    """Build closed column solids clipped strictly inside inner_uv minus interior walls.

    Returns (parts, col_polys_uv, slab_cutouts, audit).
    """
    uvz = fc.frame.pts_to_uvz(points_local)
    col_pts = uvz[work_labels == 4]
    if len(col_pts) < cfg.column_min_points:
        return [], [], [], {"emitted_count": 0, "rejected": []}

    z_top_limit = fc.ceiling_z if fc.ceiling_z is not None else float(np.percentile(col_pts[:, 2], 98))
    comps = _voxel_connected_components_3d(col_pts, voxel_size=0.12)
    parts: list[MeshPart] = []
    col_polys_uv: list[Polygon] = []
    slab_cutouts: list[tuple[Polygon, float, float]] = []
    rejected: list[dict[str, Any]] = []
    accepted_log: list[dict[str, Any]] = []

    allowed = shell.inner_uv.buffer(-0.004)
    if interior_wall_polys_uv:
        allowed = allowed.difference(unary_union([p.buffer(0.004) for p in interior_wall_polys_uv]))
    placed_union: Polygon | MultiPolygon | None = None

    for idx, comp in enumerate(comps):
        if len(comp) < cfg.column_min_points:
            continue
        sub = col_pts[comp]
        u_lo, u_hi = float(np.percentile(sub[:, 0], 3)), float(np.percentile(sub[:, 0], 97))
        v_lo, v_hi = float(np.percentile(sub[:, 1], 3)), float(np.percentile(sub[:, 1], 97))
        z_lo = float(np.percentile(sub[:, 2], 2))
        z_hi = float(np.percentile(sub[:, 2], 98))

        if z_hi - z_lo < cfg.column_min_height_m:
            rejected.append({
                "component_idx": idx,
                "point_count": int(len(comp)),
                "height_m": round(z_hi - z_lo, 3),
                "reason": "below_min_column_height",
            })
            continue

        raw_rect = box(u_lo, v_lo, u_hi, v_hi)
        clipped = raw_rect.intersection(allowed)
        if placed_union is not None and not placed_union.is_empty:
            clipped = clipped.difference(placed_union.buffer(0.004))
        clipped = clipped.buffer(0)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if clipped.is_empty or not isinstance(clipped, Polygon):
            rejected.append({"component_idx": idx, "point_count": int(len(comp)), "reason": "outside_inner_room"})
            continue

        minx, miny, maxx, maxy = clipped.bounds
        sec_u, sec_v = maxx - minx, maxy - miny
        if min(sec_u, sec_v) < cfg.column_min_section_m or clipped.area < cfg.column_min_section_m ** 2:
            rejected.append({
                "component_idx": idx,
                "point_count": int(len(comp)),
                "section_m": [round(sec_u, 3), round(sec_v, 3)],
                "reason": "section_too_thin_after_wall_clip",
            })
            continue

        base_z = _supporting_floor_top(clipped, z_lo, fc)
        col_z_min = base_z + 0.002

        # Determine local overhead slab (upper mezzanine or lower wing ceiling)
        local_overhead = _lowest_overhead_slab_bottom(clipped, col_z_min, fc, cfg, z_top_limit)
        # Check if the column genuinely rises through the local overhead slab (e.g. full-height column)
        frac_above_local = float(np.mean(sub[:, 2] > local_overhead + cfg.floor_thickness_m + 0.25))
        if local_overhead < z_top_limit - 0.30 and frac_above_local < 0.08:
            # Lower-storey column supporting the mezzanine/lower ceiling: cap cleanly beneath it!
            col_z_max = local_overhead - 0.002 if (local_overhead - z_hi) <= 0.65 else min(z_hi, local_overhead - 0.002)
        else:
            # Full-height column rising to main ceiling: register as slab cutout for any upper floor it pierces
            col_z_max = z_top_limit - 0.002 if (z_top_limit - z_hi) <= 0.65 else min(z_hi, z_top_limit - 0.002)

        if col_z_max <= col_z_min + 0.20:
            continue

        poly_clean = orient(clipped.buffer(0), sign=1.0)
        if isinstance(poly_clean, MultiPolygon):
            poly_clean = max(poly_clean.geoms, key=lambda g: g.area)
        v_uv, tri = extrude_polygon_solid(poly_clean, col_z_min, col_z_max)
        pid = f"column_{len(parts):02d}"
        parts.append(
            MeshPart(
                part_id=pid,
                class_id=4,
                role="column",
                vertices=fc.frame.uvz_to_xyz(v_uv),
                triangles=tri,
                evidence_kind="measured",
                metadata={
                    "point_count": int(len(comp)),
                    "section_uv_m": [round(sec_u, 3), round(sec_v, 3)],
                    "z_span": [round(col_z_min, 4), round(col_z_max, 4)],
                },
            )
        )
        col_polys_uv.append(poly_clean)
        slab_cutouts.append((poly_clean, col_z_min, col_z_max))
        placed_union = poly_clean if placed_union is None else unary_union([placed_union, poly_clean])
        accepted_log.append(parts[-1].metadata | {"part_id": pid})

    return parts, col_polys_uv, slab_cutouts, {
        "emitted_count": len(parts),
        "columns": accepted_log,
        "rejected": rejected,
    }


def _greedy_maximal_rectangles(mask: np.ndarray, min_cells: int = 4) -> list[tuple[int, int, int, int]]:
    """Decompose an arbitrary 2D binary mask (even complex cross/grid beams!) into disjoint axis-aligned rectangles."""
    rem = mask.copy()
    rects: list[tuple[int, int, int, int]] = []
    nrows, ncols = rem.shape

    while True:
        heights = np.zeros(ncols, dtype=np.int32)
        best_area = 0
        best_rect = (0, 0, 0, 0)
        for r in range(nrows):
            row = rem[r]
            heights = np.where(row, heights + 1, 0)
            stack: list[tuple[int, int]] = []
            for c in range(ncols + 1):
                h = int(heights[c]) if c < ncols else 0
                start = c
                while stack and stack[-1][1] > h:
                    idx, sh = stack.pop()
                    area = sh * (c - idx)
                    if area > best_area:
                        best_area = area
                        best_rect = (r - sh + 1, r + 1, idx, c)
                    start = idx
                stack.append((start, h))

        if best_area < min_cells:
            break
        r0, r1, c0, c1 = best_rect
        rects.append(best_rect)
        rem[r0:r1, c0:c1] = False

    return rects


def build_beam_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    column_polys_uv: list[Polygon],
    interior_wall_polys_uv: list[Polygon],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
    *,
    extra_blocker_bands: list[tuple[Polygon, float, float]] | None = None,
) -> tuple[list[MeshPart], dict[str, Any]]:
    """Build crisp orthogonal overhead beam solids with local-storey z_top and hole-preserving clipping."""
    uvz = fc.frame.pts_to_uvz(points_local)
    beam_pts = uvz[work_labels == 3]
    if len(beam_pts) < cfg.beam_min_points:
        return [], {"emitted_count": 0, "reason": "insufficient_beam_points"}

    z_ceil = fc.ceiling_z if fc.ceiling_z is not None else float(np.percentile(beam_pts[:, 2], 98))
    overhead = beam_pts[(beam_pts[:, 2] <= z_ceil + 0.08) & (beam_pts[:, 2] >= fc.floor_z + 1.5)]
    if len(overhead) < cfg.beam_min_points:
        return [], {"emitted_count": 0, "reason": "insufficient_overhead_beam_points"}

    cell = 0.10
    minx, miny, maxx, maxy = shell.inner_uv.bounds
    nu = int(np.ceil((maxx - minx) / cell)) + 2
    nv = int(np.ceil((maxy - miny) / cell)) + 2

    iu = np.floor((overhead[:, 0] - minx) / cell).astype(np.int64)
    iv = np.floor((overhead[:, 1] - miny) / cell).astype(np.int64)
    valid = (iu >= 0) & (iu < nu) & (iv >= 0) & (iv < nv)
    overhead = overhead[valid]
    iu, iv = iu[valid], iv[valid]

    counts = np.zeros((nu, nv), dtype=np.int32)
    np.add.at(counts, (iu, iv), 1)
    nz = counts[counts > 0]
    min_cell_pts = int(np.clip(np.percentile(nz, 25), 3, 12)) if len(nz) else 4
    mask = counts >= min_cell_pts
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    mask = ndimage.binary_opening(mask, structure=np.ones((2, 2), dtype=bool), iterations=1)

    rects_idx = _greedy_maximal_rectangles(mask, min_cells=4)
    if not rects_idx:
        return [], {"emitted_count": 0, "reason": "no_beam_rectangles"}

    allowed_domain = shell.inner_uv.buffer(-0.005)
    blockers = [*column_polys_uv, *interior_wall_polys_uv]
    if blockers:
        allowed_domain = allowed_domain.difference(unary_union([p.buffer(0.006) for p in blockers])).buffer(0)
    z_bands = extra_blocker_bands or []

    u_edges = minx + np.arange(nu + 1, dtype=np.float64) * cell
    v_edges = miny + np.arange(nv + 1, dtype=np.float64) * cell
    parts: list[MeshPart] = []
    beam_log: list[dict[str, Any]] = []
    placed_union: Polygon | MultiPolygon | None = None

    for r0, r1, c0, c1 in rects_idx:
        in_rect = (iu >= r0) & (iu < r1) & (iv >= c0) & (iv < c1)
        rpts = overhead[in_rect]
        if len(rpts) < 80:
            continue

        rect_poly = box(float(u_edges[r0]), float(v_edges[c0]), float(u_edges[r1]), float(v_edges[c1]))
        z_bot = float(np.percentile(rpts[:, 2], 8))
        z_meas_top = float(np.percentile(rpts[:, 2], 96))

        # Snap beam top to the LOCAL overhead slab (mezzanine floor bottom or lower/main ceiling bottom)!
        overhead_bot = _lowest_overhead_slab_bottom(rect_poly, z_bot, fc, cfg, z_ceil)
        if overhead_bot - z_meas_top <= 0.55:
            z_top = overhead_bot - 0.003
        else:
            # Beam sits well below the overhead slab (e.g. exposed lower-storey beam without mezzanine points directly above)
            z_top = min(z_meas_top, overhead_bot - 0.003)

        # Also avoid colliding with any upper floor slab or secondary ceiling vertically!
        for flv in fc.floor_levels:
            if flv.is_primary:
                continue
            slab_bot = flv.z_top - cfg.floor_thickness_m
            if flv.polygon_uv.intersects(rect_poly) and z_bot < flv.z_top and z_top > slab_bot - 0.003:
                if z_bot < slab_bot - 0.08:
                    z_top = min(z_top, slab_bot - 0.003)
                else:
                    z_bot = max(z_bot, flv.z_top + 0.003)

        for clv in fc.ceiling_levels:
            if clv.is_primary:
                continue
            c_top = clv.z_bottom + cfg.ceiling_thickness_m
            if clv.polygon_uv.intersects(rect_poly) and z_bot < c_top and z_top > clv.z_bottom - 0.003:
                if z_bot < clv.z_bottom - 0.08:
                    z_top = min(z_top, clv.z_bottom - 0.003)
                else:
                    z_bot = max(z_bot, c_top + 0.003)

        if z_top - z_bot < cfg.beam_min_height_m:
            continue

        clipped = rect_poly.intersection(allowed_domain)
        ov_bands = [p.buffer(0.008) for p, bz0, bz1 in z_bands if bz0 <= z_top + 0.02 and bz1 >= z_bot - 0.02]
        if ov_bands:
            clipped = clipped.difference(unary_union(ov_bands))
        if placed_union is not None and not placed_union.is_empty:
            clipped = clipped.difference(placed_union.buffer(0.002))
        clipped = clipped.buffer(0)

        geoms = (
            list(clipped.geoms)
            if isinstance(clipped, MultiPolygon)
            else ([clipped] if isinstance(clipped, Polygon) else [])
        )
        for g in geoms:
            if g.is_empty or g.area < 0.04:
                continue
            # CRITICAL: Preserve g.interiors (holes cut out around interior columns!)
            g_clean = orient(g.buffer(0), sign=1.0)
            if isinstance(g_clean, MultiPolygon):
                g_clean = max(g_clean.geoms, key=lambda x: x.area)
            if g_clean.is_empty or g_clean.area < 0.04:
                continue
            try:
                v_uv, tri = extrude_polygon_solid(g_clean, z_bot, z_top)
            except ValueError:
                continue
            pid = f"beam_{len(parts):02d}"
            parts.append(
                MeshPart(
                    part_id=pid,
                    class_id=3,
                    role="beam",
                    vertices=fc.frame.uvz_to_xyz(v_uv),
                    triangles=tri,
                    evidence_kind="measured",
                    metadata={
                        "area_m2": round(float(g_clean.area), 3),
                        "z_span": [round(z_bot, 4), round(z_top, 4)],
                        "point_count": int(len(rpts)),
                    },
                )
            )
            placed_union = g_clean if placed_union is None else unary_union([placed_union, g_clean])
            beam_log.append(parts[-1].metadata | {"part_id": pid})

    return parts, {
        "emitted_count": len(parts),
        "beams": beam_log,
    }
