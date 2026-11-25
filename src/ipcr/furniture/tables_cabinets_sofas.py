"""Stage 2 B10/B15 structural & multi-asset library reconstruction for tables (8), cabinets (9), and sofas (10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, unary_union

from ipcr.architecture.frame_floors import FloorCeilingResult, _raster_mask_to_polygon, _snap_orthogonal_polygon
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import FurnitureConfig
from ipcr.furniture.library import FurnitureLibrary
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


def _dominant_tabletop_z(sub_z: np.ndarray) -> float:
    """Find the dominant horizontal tabletop plane Z peak (rejecting tall monitor/chair-back bleed!)."""
    z_lo = float(np.percentile(sub_z, 55))
    z_hi = float(np.percentile(sub_z, 99))
    upper = sub_z[(sub_z >= z_lo) & (sub_z <= z_hi)]
    if len(upper) < 30 or (z_hi - z_lo) < 0.04:
        return float(np.percentile(sub_z, 92))
    bins = np.arange(z_lo - 0.01, z_hi + 0.03, 0.02)
    hist, edges = np.histogram(upper, bins=bins)
    pk = int(np.argmax(hist))
    zc = 0.5 * (edges[pk] + edges[pk + 1])
    band = upper[np.abs(upper - zc) <= 0.035]
    return float(np.median(band)) if len(band) >= 15 else zc


def _extract_denoised_tabletop_contour(pts_uv: np.ndarray, *, cell: float = 0.05) -> tuple[Polygon | None, np.ndarray]:
    """Density-denoised 2D tabletop contour: strips edge noise wisps while faithfully preserving L/U/curved desks (B15)."""
    if len(pts_uv) < 40:
        return None, pts_uv
    pad = 0.25
    u0, v0 = float(pts_uv[:, 0].min()) - pad, float(pts_uv[:, 1].min()) - pad
    u1, v1 = float(pts_uv[:, 0].max()) + pad, float(pts_uv[:, 1].max()) + pad
    nu = int(np.ceil((u1 - u0) / cell)) + 2
    nv = int(np.ceil((v1 - v0) / cell)) + 2
    if nu * nv > 2_000_000:
        cell = 0.08
        nu = int(np.ceil((u1 - u0) / cell)) + 2
        nv = int(np.ceil((v1 - v0) / cell)) + 2

    iu = np.clip(np.floor((pts_uv[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
    iv = np.clip(np.floor((pts_uv[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
    counts = np.zeros((nu, nv), dtype=np.int32)
    np.add.at(counts, (iu, iv), 1)
    nz = counts[counts > 0]
    if len(nz) == 0:
        return None, pts_uv

    # Strip sparse stray noise wisps around tabletop edges (< 12% of median tabletop cell density)
    min_cell = max(2, int(float(np.median(nz)) * 0.12))
    mask = counts >= min_cell
    # Morphological opening (2x2 or 3x3 cross) shears 1-cell wisps while preserving thin desk wings
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    opened = ndimage.binary_opening(mask, structure=cross, iterations=1)
    if np.any(opened):
        mask = opened
    closed = ndimage.binary_closing(mask, structure=np.ones((3, 3), dtype=bool), iterations=2)

    # Keep the dominant connected tabletop component so disconnected scatter fragments are dropped!
    filled = ndimage.binary_fill_holes(closed)
    labeled, ncomp = ndimage.label(filled)
    if ncomp == 0:
        return None, pts_uv
    comp_pts = [int(counts[labeled == cid].sum()) for cid in range(1, ncomp + 1)]
    best_cid = 1 + int(np.argmax(comp_pts))
    clean_mask = labeled == best_cid
    core_uv = pts_uv[clean_mask[iu, iv]]
    if len(core_uv) < 30:
        core_uv = pts_uv

    try:
        raw = _raster_mask_to_polygon(clean_mask, u0, v0, cell)
    except ValueError:
        return None, core_uv
    if raw.is_empty or raw.area < 0.08:
        return None, core_uv

    # Smooth softly: keep curved arcs smooth (tol=0.05) while snapping only near-axis straight edges (<=12 deg)
    simp = raw.simplify(0.05, preserve_topology=True).buffer(0)
    if isinstance(simp, MultiPolygon):
        simp = max(simp.geoms, key=lambda g: g.area)
    return orient(Polygon(simp.exterior.coords), sign=1.0), core_uv


def _extract_dense_core_uv(pts_uv: np.ndarray, *, cell: float = 0.08) -> np.ndarray:
    """Filter out sparse disconnected wall-bleed slivers behind a cabinet via 2D density-connected core."""
    if len(pts_uv) < 200:
        return pts_uv
    u0, v0 = float(pts_uv[:, 0].min()) - 0.16, float(pts_uv[:, 1].min()) - 0.16
    u1, v1 = float(pts_uv[:, 0].max()) + 0.16, float(pts_uv[:, 1].max()) + 0.16
    nu = int(np.ceil((u1 - u0) / cell)) + 2
    nv = int(np.ceil((v1 - v0) / cell)) + 2
    if nu * nv > 500_000:
        return pts_uv
    iu = np.clip(np.floor((pts_uv[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
    iv = np.clip(np.floor((pts_uv[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
    counts = np.zeros((nu, nv), dtype=np.int32)
    np.add.at(counts, (iu, iv), 1)
    nz = counts[counts > 0]
    if len(nz) == 0:
        return pts_uv
    min_cell = max(4, int(float(np.median(nz)) * 0.12))
    mask = counts >= min_cell
    labeled, ncomp = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if ncomp <= 1:
        return pts_uv
    comp_pts = [int(counts[labeled == cid].sum()) for cid in range(1, ncomp + 1)]
    best_cid = 1 + int(np.argmax(comp_pts))
    keep_mask = labeled[iu, iv] == best_cid
    if int(keep_mask.sum()) >= len(pts_uv) * 0.45:
        return pts_uv[keep_mask]
    return pts_uv


def _fit_obb_2d(pts_uv: np.ndarray, snap_deg: float = 12.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit clean 2D OBB returning (center_uv, long_dir, depth_dir, length, depth)."""
    mean_uv = pts_uv.mean(axis=0)
    centered = pts_uv - mean_uv
    cov = (centered.T @ centered) / max(len(pts_uv), 1)
    _, eigvecs = np.linalg.eigh(cov)
    long_dir = eigvecs[:, 1].copy()
    long_dir /= max(float(np.linalg.norm(long_dir)), 1e-9)

    cos_s = float(np.cos(np.radians(snap_deg)))
    for ax in (np.array([1.0, 0.0]), np.array([-1.0, 0.0]), np.array([0.0, 1.0]), np.array([0.0, -1.0])):
        if float(np.dot(long_dir, ax)) >= cos_s:
            long_dir = ax.copy()
            break
    depth_dir = np.array([-long_dir[1], long_dir[0]], dtype=np.float64)

    sl = centered @ long_dir
    sd = centered @ depth_dir
    l0, l1 = float(np.percentile(sl, 2)), float(np.percentile(sl, 98))
    d0, d1 = float(np.percentile(sd, 2)), float(np.percentile(sd, 98))
    center = mean_uv + long_dir * (0.5 * (l0 + l1)) + depth_dir * (0.5 * (d0 + d1))
    return center, long_dir, depth_dir, l1 - l0, d1 - d0


def _obb_poly(center: np.ndarray, long_dir: np.ndarray, depth_dir: np.ndarray, length: float, depth: float) -> Polygon:
    hl, hd = 0.5 * length, 0.5 * depth
    corners = [
        center - long_dir * hl - depth_dir * hd,
        center + long_dir * hl - depth_dir * hd,
        center + long_dir * hl + depth_dir * hd,
        center - long_dir * hl + depth_dir * hd,
    ]
    return orient(Polygon(corners), sign=1.0)


def _has_real_concave_cutout(contour: Polygon, obb: Polygon, long_dir: np.ndarray, depth_dir: np.ndarray) -> bool:
    """Distinguish true L/U/curved desks (recessed cutout >= 0.38m in BOTH dimensions) from tiny rectangular edge noise!"""
    fill_ratio = float(contour.area / max(obb.area, 1e-6))
    if fill_ratio >= 0.83:
        return False
    missing = obb.difference(contour).buffer(0)
    geoms = (
        list(missing.geoms)
        if isinstance(missing, MultiPolygon)
        else ([missing] if isinstance(missing, Polygon) else [])
    )
    for g in geoms:
        if g.is_empty or g.area < 0.16:
            continue
        pts = np.asarray(list(g.exterior.coords)[:-1], dtype=np.float64)
        sl = pts @ long_dir
        sd = pts @ depth_dir
        cut_l = float(sl.max() - sl.min())
        cut_d = float(sd.max() - sd.min())
        if min(cut_l, cut_d) >= 0.38:
            return True
    return False


def build_table_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    static_blockers_uv: list[Polygon],
    z_blocker_bands: list[tuple[Polygon, float, float]],
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: FurnitureConfig,
    *,
    library: FurnitureLibrary | None = None,
) -> tuple[list[MeshPart], list[TableSpec], dict[str, Any]]:
    """Build B10/B15 tables: denoised true-shape gate (library folding_table for compatible rects + exact L/curved contour)."""
    uvz = fc.frame.pts_to_uvz(points_local)
    tpts = uvz[work_labels == 8]
    if len(tpts) < cfg.table_min_points:
        return [], [], {"emitted_tables": 0, "emitted_parts": 0, "tables": [], "rejected": []}

    table_asset = library.get_best_asset(8) if library is not None else None
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
        z_top = _dominant_tabletop_z(sub[:, 2])
        h = z_top - z_floor

        if h < cfg.table_min_height_m or h > cfg.table_max_height_m:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "height_above_floor_m": round(h, 3),
                "reason": "height_out_of_table_range",
            })
            continue

        # Tight tabletop horizontal plane slice (+-4.5cm) rejects chair-back & under-desk bleed!
        top_slice = sub[np.abs(sub[:, 2] - z_top) <= 0.048]
        if len(top_slice) < 80:
            top_slice = sub[sub[:, 2] >= z_top - 0.10]
        if len(top_slice) < 50:
            top_slice = sub

        contour, core_uv = _extract_denoised_tabletop_contour(top_slice[:, :2], cell=0.05)
        if contour is None:
            continue

        center_uv, long_dir, depth_dir, length, depth = _fit_obb_2d(core_uv, snap_deg=12.0)
        obb_rect = _obb_poly(center_uv, long_dir, depth_dir, length, depth)

        # True L/U/curved desk vs rectangular desk with tiny noise wisps!
        is_irregular = _has_real_concave_cutout(contour, obb_rect, long_dir, depth_dir)
        is_rect = not is_irregular
        if is_rect:
            poly_uv = obb_rect
        else:
            # Keep exact denoised L/U/curved contour (if straight L-desk, gently orthogonalize; preserve smooth arcs!)
            poly_uv = contour

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
        if min(length, depth) < 0.28:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "obb_ld_m": [round(length, 3), round(depth, 3)],
                "reason": "too_narrow_for_table",
            })
            continue

        tid = f"table_{len(specs):02d}"
        top_bot = max(z_floor + 0.15, z_top - cfg.table_top_thickness_m)

        # Case A: Rectangular table within library folding_table proportions -> use verified watertight folding_table.glb!
        aspect = length / max(depth, 1e-6)
        use_lib_table = (
            is_rect
            and table_asset is not None
            and 1.05 <= length <= 2.75
            and 0.40 <= depth <= 1.15
            and 1.45 <= aspect <= 4.30
            and cur_allowed.buffer(0.002).contains(obb_rect)
        )
        if use_lib_table and table_asset is not None:
            # Canonical folding_table has long axis along X (width), depth along Y -> front_dir = depth_dir, right_dir = long_dir
            front_for_table = np.array([-long_dir[1], long_dir[0]], dtype=np.float64)
            sx = length / float(table_asset.extents_xyz[0])
            sy = depth / float(table_asset.extents_xyz[1])
            sz = h / float(table_asset.extents_xyz[2])
            tv_uv, tt = table_asset.transform_to_uvz(center_uv, z_floor + 0.002, front_for_table, sx, sy, sz)
            parts.append(
                MeshPart(
                    part_id=tid,
                    class_id=8,
                    role="table_library",
                    vertices=fc.frame.uvz_to_xyz(tv_uv),
                    triangles=tt,
                    evidence_kind="library_matched",
                    metadata={
                        "table_id": tid,
                        "asset_id": table_asset.asset_id,
                        "asset_sha256": table_asset.sha256,
                        "shape": "rectangular_library",
                        "obb_ldh_m": [round(length, 3), round(depth, 3), round(h, 3)],
                        "point_count": int(len(comp)),
                    },
                )
            )
            specs.append(
                TableSpec(
                    table_id=tid,
                    poly_uv=obb_rect,
                    z_floor=z_floor,
                    z_top=z_top,
                    tabletop_z_bot=top_bot,
                    leg_polys_uv=[],
                    is_rectangular=True,
                )
            )
            placed_union = obb_rect if placed_union is None else unary_union([placed_union, obb_rect])
            logs.append({
                "table_id": tid,
                "library_matched": True,
                "asset_id": table_asset.asset_id,
                "shape": "rectangular_library",
                "obb_ldh_m": [round(length, 3), round(depth, 3), round(h, 3)],
                "area_m2": round(float(obb_rect.area), 3),
                "z_span": [round(z_floor, 4), round(z_top, 4)],
                "point_count": int(len(comp)),
            })
            continue

        # Case B/C: Giant conference table OR true L-shaped / U-shaped / curved desk -> B15 open-frame structural table!
        if z_top - top_bot < 0.02:
            continue
        try:
            tv_uv, tt = extrude_polygon_solid(poly_clean, top_bot, z_top)
        except ValueError:
            continue

        shape_kind = "rectangular_conference" if is_rect else "irregular_shaped_contour"
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
                    "shape": shape_kind,
                    "area_m2": round(float(poly_clean.area), 3),
                    "obb_ldh_m": [round(length, 3), round(depth, 3), round(h, 3)],
                    "z_top_above_floor_m": round(h, 3),
                    "point_count": int(len(comp)),
                    "library_fallback_reason": "irregular_or_oversized_conference_table_b10_b15",
                },
            )
        )

        leg_sz = cfg.table_leg_size_m
        leg_z0 = z_floor + 0.002
        leg_z1 = top_bot - 0.002
        leg_polys: list[Polygon] = []

        if leg_z1 - leg_z0 >= 0.25:
            leg_centers: list[tuple[float, float]] = []
            if is_rect:
                in_l = min(0.14, length * 0.20)
                in_d = min(0.18 if depth >= 1.1 else 0.09, depth * 0.25)
                nl = max(2, int(np.ceil((length - 2 * in_l) / 2.2)) + 1)
                for sl_pos in np.linspace(-0.5 * length + in_l, 0.5 * length - in_l, nl):
                    for sd_pos in (-0.5 * depth + in_d, 0.5 * depth - in_d):
                        p_uv = center_uv + long_dir * sl_pos + depth_dir * sd_pos
                        leg_centers.append((float(p_uv[0]), float(p_uv[1])))
            else:
                inset = poly_clean.buffer(-0.10, join_style=2)
                if isinstance(inset, MultiPolygon):
                    inset = max(inset.geoms, key=lambda g: g.area)
                if isinstance(inset, Polygon) and not inset.is_empty:
                    # Sample corners / perimeter vertices every >=0.45m for stable support under L/curved wings
                    for cx_v, cy_v in list(inset.exterior.coords)[:-1]:
                        if all(np.hypot(cx_v - ox, cy_v - oy) >= 0.45 for ox, oy in leg_centers):
                            leg_centers.append((float(cx_v), float(cy_v)))

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
            "library_matched": False,
            "shape": shape_kind,
            "obb_ldh_m": [round(length, 3), round(depth, 3), round(h, 3)],
            "area_m2": round(float(poly_clean.area), 3),
            "z_span": [round(z_floor, 4), round(z_top, 4)],
            "leg_count": len(leg_polys),
            "point_count": int(len(comp)),
        })

    return parts, specs, {
        "emitted_tables": len(specs),
        "emitted_parts": len(parts),
        "library_matched_tables": sum(1 for x in logs if x.get("library_matched")),
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
    *,
    library: FurnitureLibrary | None = None,
) -> tuple[list[MeshPart], list[tuple[Polygon, float, float]], dict[str, Any]]:
    """Build B10/B15 cabinets with dense-core OBB depth and multi-asset selection (armored_cabinet2 / metal_filing_cabinet)."""
    uvz = fc.frame.pts_to_uvz(points_local)
    cpts = uvz[work_labels == 9]
    if len(cpts) < cfg.cabinet_min_points:
        return [], [], {"emitted_cabinets": 0, "emitted_parts": 0, "cabinets": [], "rejected": []}

    allowed_base = shell.inner_uv.buffer(-cfg.clearance_m)
    if static_blockers_uv:
        allowed_base = allowed_base.difference(unary_union([p.buffer(cfg.clearance_m) for p in static_blockers_uv])).buffer(0)

    comps = _voxel_connected_components_3d(cpts, voxel_size=0.10)
    parts: list[MeshPart] = []
    cab_bands: list[tuple[Polygon, float, float]] = []
    logs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    placed_union: Polygon | MultiPolygon | None = None
    cab_group_idx = 0

    for ci, comp in enumerate(comps):
        if len(comp) < cfg.cabinet_min_points:
            continue
        sub = cpts[comp]
        core_uv = _extract_dense_core_uv(sub[:, :2], cell=0.08)
        center_uv, long_dir, depth_dir, length, depth = _fit_obb_2d(core_uv, snap_deg=12.0)

        z_lo_raw = float(np.percentile(sub[:, 2], 2))
        z_hi_raw = float(np.percentile(sub[:, 2], 98))
        z_floor = _local_floor_z(fc, float(center_uv[0]), float(center_uv[1]), z_lo_raw)

        if depth < cfg.cabinet_min_depth_m or length < cfg.cabinet_min_width_m:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "obb_ld_m": [round(length, 3), round(depth, 3)],
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

        if z_lo_raw - z_floor <= 0.45:
            z_bot = z_floor + 0.002
        elif len(comp) >= 2000 and depth >= 0.22:
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
        cand_poly = _obb_poly(center_uv, long_dir, depth_dir, length, depth)

        for flv in fc.floor_levels:
            if not flv.is_primary and flv.polygon_uv.intersects(cand_poly):
                fbot = flv.z_top - 0.12
                if z_bot < fbot - 0.20 and z_top > fbot - 0.005:
                    z_top = fbot - 0.005
        for clv in fc.ceiling_levels:
            if not clv.is_primary and clv.polygon_uv.intersects(cand_poly):
                if z_bot < clv.z_bottom - 0.20 and z_top > clv.z_bottom - 0.005:
                    z_top = clv.z_bottom - 0.005

        extra_blockers: list[Polygon] = [
            p.buffer(cfg.clearance_m) for p, bz0, bz1 in z_blocker_bands if bz0 <= z_top + 0.02 and bz1 >= z_bot - 0.02
        ]
        is_under_table = False
        for ts in table_specs:
            if not ts.poly_uv.intersects(cand_poly):
                continue
            if z_hi_raw <= ts.z_top + 0.05:
                z_top = min(z_top, ts.tabletop_z_bot - 0.006)
                extra_blockers.extend(lp.buffer(cfg.clearance_m) for lp in ts.leg_polys_uv)
                is_under_table = True
            else:
                extra_blockers.append(ts.poly_uv.buffer(cfg.clearance_m))

        height = z_top - z_bot
        if height < cfg.cabinet_min_height_m:
            rejected.append({"cluster_idx": ci, "point_count": int(len(comp)), "reason": "insufficient_height_after_clearance"})
            continue

        cur_allowed = allowed_base
        if extra_blockers:
            cur_allowed = cur_allowed.difference(unary_union(extra_blockers))
        if placed_union is not None and not placed_union.is_empty:
            cur_allowed = cur_allowed.difference(placed_union.buffer(cfg.clearance_m))

        clipped = cand_poly.intersection(cur_allowed).buffer(0)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if not isinstance(clipped, Polygon) or clipped.is_empty or clipped.area < 0.05:
            rejected.append({"cluster_idx": ci, "point_count": int(len(comp)), "reason": "eliminated_by_clearance"})
            continue

        sd_core = (core_uv - center_uv[None, :]) @ depth_dir
        pos_cnt = int(np.sum(sd_core > 0.15 * depth))
        neg_cnt = int(np.sum(sd_core < -0.15 * depth))
        np_wall, _ = nearest_points(shell.inner_uv.exterior, Point(float(center_uv[0]), float(center_uv[1])))
        away_wall = np.array([center_uv[0] - np_wall.x, center_uv[1] - np_wall.y], dtype=np.float64)
        if abs(pos_cnt - neg_cnt) >= max(pos_cnt, neg_cnt) * 0.20:
            front_dir = depth_dir if pos_cnt >= neg_cnt else -depth_dir
        else:
            front_dir = depth_dir if float(np.dot(away_wall, depth_dir)) >= 0 else -depth_dir
        right_dir = np.array([front_dir[1], -front_dir[0]], dtype=np.float64)

        # Rank all verified cabinet assets (armored_cabinet2 vs metal_filing_cabinet) by measured (depth, height)!
        ranked_cabs = (
            library.rank_assets(9, (min(length, 1.1), depth, height)) if library is not None else []
        )
        cab_asset = ranked_cabs[0] if ranked_cabs else None

        used_library = False
        if cab_asset is not None and not is_under_table and height >= 0.80 and depth >= 0.18:
            c_pts = np.asarray(list(clipped.exterior.coords)[:-1], dtype=np.float64)
            sr = (c_pts - center_uv[None, :]) @ right_dir
            sf = (c_pts - center_uv[None, :]) @ front_dir
            r_lo, r_hi = float(sr.min()), float(sr.max())
            f_lo, f_hi = float(sf.min()), float(sf.max())
            eff_w = r_hi - r_lo
            eff_d = f_hi - f_lo
            unit_gap = 0.008
            asset_w, asset_d, asset_h = float(cab_asset.extents_xyz[0]), float(cab_asset.extents_xyz[1]), float(cab_asset.extents_xyz[2])
            n_units = max(1, int(round(eff_w / max(asset_w, 0.50))))
            unit_w = (eff_w - (n_units - 1) * unit_gap) / n_units
            if unit_w >= 0.24 and eff_d >= 0.16:
                emitted_units = 0
                unit_polys: list[Polygon] = []
                for uk in range(n_units):
                    ur_lo = r_lo + uk * (unit_w + unit_gap)
                    ur_hi = ur_lo + unit_w
                    u_center = center_uv + right_dir * (0.5 * (ur_lo + ur_hi)) + front_dir * (0.5 * (f_lo + f_hi))
                    upoly = _obb_poly(u_center, right_dir, front_dir, unit_w, eff_d)
                    if not cur_allowed.buffer(0.002).contains(upoly):
                        upoly_in = _obb_poly(u_center, right_dir, front_dir, unit_w * 0.96, eff_d * 0.96)
                        if not cur_allowed.buffer(0.002).contains(upoly_in):
                            continue
                        sx = (unit_w * 0.96) / asset_w
                        sy = (eff_d * 0.96) / asset_d
                        upoly = upoly_in
                    else:
                        sx = unit_w / asset_w
                        sy = eff_d / asset_d
                    sz = height / asset_h
                    v_uvz, tri = cab_asset.transform_to_uvz(u_center, z_bot, front_dir, sx, sy, sz)
                    pid = f"cabinet_{cab_group_idx:02d}" if n_units == 1 else f"cabinet_{cab_group_idx:02d}_unit_{uk:02d}"
                    parts.append(
                        MeshPart(
                            part_id=pid,
                            class_id=9,
                            role="cabinet",
                            vertices=fc.frame.uvz_to_xyz(v_uvz),
                            triangles=tri,
                            evidence_kind="library_matched",
                            metadata={
                                "cabinet_group": f"cabinet_{cab_group_idx:02d}",
                                "asset_id": cab_asset.asset_id,
                                "asset_sha256": cab_asset.sha256,
                                "unit_index": uk,
                                "unit_count": n_units,
                                "unit_wdh_m": [round(unit_w, 3), round(eff_d, 3), round(height, 3)],
                                "front_dir_uv": [round(float(front_dir[0]), 4), round(float(front_dir[1]), 4)],
                                "z_span": [round(z_bot, 4), round(z_top, 4)],
                                "point_count": int(len(comp)),
                            },
                        )
                    )
                    unit_polys.append(upoly)
                    cab_bands.append((upoly, z_bot, z_top))
                    emitted_units += 1

                if emitted_units > 0:
                    used_library = True
                    u_union = unary_union(unit_polys)
                    placed_union = u_union if placed_union is None else unary_union([placed_union, u_union])
                    logs.append({
                        "cabinet_id": f"cabinet_{cab_group_idx:02d}",
                        "library_matched": True,
                        "asset_id": cab_asset.asset_id,
                        "unit_count": emitted_units,
                        "obb_ldh_m": [round(eff_w, 3), round(eff_d, 3), round(height, 3)],
                        "front_dir_uv": [round(float(front_dir[0]), 4), round(float(front_dir[1]), 4)],
                        "z_span": [round(z_bot, 4), round(z_top, 4)],
                        "point_count": int(len(comp)),
                    })
                    cab_group_idx += 1

        if not used_library:
            poly_clean = orient(clipped.buffer(0), sign=1.0)
            if isinstance(poly_clean, MultiPolygon):
                poly_clean = max(poly_clean.geoms, key=lambda g: g.area)
            try:
                cv_uv, ct = extrude_polygon_solid(poly_clean, z_bot, z_top)
            except ValueError:
                continue
            pid = f"cabinet_{cab_group_idx:02d}"
            role = "cabinet_under_table" if is_under_table else "cabinet"
            parts.append(
                MeshPart(
                    part_id=pid,
                    class_id=9,
                    role=role,
                    vertices=fc.frame.uvz_to_xyz(cv_uv),
                    triangles=ct,
                    evidence_kind="parametric_rule_fallback",
                    metadata={
                        "obb_ldh_m": [round(length, 3), round(depth, 3), round(height, 3)],
                        "area_m2": round(float(poly_clean.area), 3),
                        "z_span": [round(z_bot, 4), round(z_top, 4)],
                        "point_count": int(len(comp)),
                        "library_fallback_reason": "under_table_pedestal_or_library_unavailable",
                    },
                )
            )
            cab_bands.append((poly_clean, z_bot, z_top))
            placed_union = poly_clean if placed_union is None else unary_union([placed_union, poly_clean])
            logs.append(parts[-1].metadata | {"cabinet_id": pid, "library_matched": False, "role": role})
            cab_group_idx += 1

    return parts, cab_bands, {
        "emitted_cabinets": len(logs),
        "emitted_parts": len(parts),
        "assets_used": sorted({str(c["asset_id"]) for c in logs if c.get("library_matched")}),
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
    *,
    library: FurnitureLibrary | None = None,
) -> tuple[list[MeshPart], list[tuple[Polygon, float, float]], dict[str, Any]]:
    """Build library-matched watertight sofa (sofa.glb) or multi-part fallback while rejecting thin wall-bleed."""
    uvz = fc.frame.pts_to_uvz(points_local)
    spts = uvz[work_labels == 10]
    if len(spts) < cfg.sofa_min_points:
        return [], [], {"emitted_sofas": 0, "emitted_parts": 0, "sofas": [], "rejected": []}

    sofa_asset = library.get_best_asset(10) if library is not None else None
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
        center_uv, long_dir, depth_dir, length, depth = _fit_obb_2d(sub[:, :2], snap_deg=12.0)
        z_lo = float(np.percentile(sub[:, 2], 2))
        z_hi = float(np.percentile(sub[:, 2], 98))
        z_floor = _local_floor_z(fc, float(center_uv[0]), float(center_uv[1]), z_lo)
        h = z_hi - z_floor

        if depth < cfg.sofa_min_depth_m or length < cfg.sofa_min_width_m or h < cfg.sofa_min_height_m:
            rejected.append({
                "cluster_idx": ci,
                "point_count": int(len(comp)),
                "obb_ld_m": [round(length, 3), round(depth, 3)],
                "height_m": round(h, 3),
                "reason": "below_min_sofa_dimensions",
            })
            continue
        if z_lo - z_floor > 0.35:
            rejected.append({"cluster_idx": ci, "point_count": int(len(comp)), "reason": "floating_sofa_rejected"})
            continue

        cand_poly = _obb_poly(center_uv, long_dir, depth_dir, length, depth)
        cur_allowed = allowed_base
        ov = [p.buffer(cfg.clearance_m) for p, bz0, bz1 in z_blocker_bands if bz0 <= z_hi + 0.02 and bz1 >= z_floor - 0.02]
        if ov:
            cur_allowed = cur_allowed.difference(unary_union(ov))
        if placed_union is not None and not placed_union.is_empty:
            cur_allowed = cur_allowed.difference(placed_union.buffer(cfg.clearance_m))

        clipped = cand_poly.intersection(cur_allowed).buffer(0)
        if isinstance(clipped, MultiPolygon):
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if not isinstance(clipped, Polygon) or clipped.is_empty or clipped.area < 0.35:
            rejected.append({"cluster_idx": ci, "point_count": int(len(comp)), "reason": "eliminated_by_clearance"})
            continue

        high_pts = sub[sub[:, 2] >= z_floor + 0.65 * h]
        low_pts = sub[sub[:, 2] <= z_floor + 0.45 * h]
        if len(high_pts) >= 20 and len(low_pts) >= 20:
            v_front = low_pts[:, :2].mean(axis=0) - high_pts[:, :2].mean(axis=0)
            front_dir = depth_dir if float(np.dot(v_front, depth_dir)) >= 0 else -depth_dir
        else:
            np_w, _ = nearest_points(shell.inner_uv.exterior, Point(float(center_uv[0]), float(center_uv[1])))
            away = np.array([center_uv[0] - np_w.x, center_uv[1] - np_w.y], dtype=np.float64)
            front_dir = depth_dir if float(np.dot(away, depth_dir)) >= 0 else -depth_dir

        sid = f"sofa_{len(logs):02d}"
        if sofa_asset is not None and cur_allowed.buffer(0.002).contains(cand_poly):
            sx = length / float(sofa_asset.extents_xyz[0])
            sy = depth / float(sofa_asset.extents_xyz[1])
            sz = h / float(sofa_asset.extents_xyz[2])
            v_uvz, tri = sofa_asset.transform_to_uvz(center_uv, z_floor + 0.002, front_dir, sx, sy, sz)
            parts.append(
                MeshPart(
                    part_id=sid,
                    class_id=10,
                    role="sofa",
                    vertices=fc.frame.uvz_to_xyz(v_uvz),
                    triangles=tri,
                    evidence_kind="library_matched",
                    metadata={"asset_id": sofa_asset.asset_id, "asset_sha256": sofa_asset.sha256},
                )
            )
            sofa_bands.append((cand_poly, z_floor, z_hi))
            placed_union = cand_poly if placed_union is None else unary_union([placed_union, cand_poly])
            logs.append({"sofa_id": sid, "library_matched": True, "asset_id": sofa_asset.asset_id, "obb_ldh_m": [round(length, 3), round(depth, 3), round(h, 3)]})
            continue

        minx, miny, maxx, maxy = clipped.bounds
        w_u, w_v = maxx - minx, maxy - miny
        z_seat = z_floor + min(0.44, max(0.32, 0.50 * h))
        z_arm = z_floor + min(0.64, max(z_seat + 0.14, 0.72 * h))
        back_t = min(0.22, min(w_u, w_v) * 0.28)
        arm_t = min(0.14, max(w_u, w_v) * 0.12)
        if w_u >= w_v:
            back_at_max_v = float(front_dir[1]) < 0
            if back_at_max_v:
                sub_specs = [
                    ("back", box(minx, maxy - back_t, maxx, maxy), z_hi),
                    ("seat", box(minx + arm_t + 0.004, miny, maxx - arm_t - 0.004, maxy - back_t - 0.004), z_seat),
                    ("arm_l", box(minx, miny, minx + arm_t, maxy - back_t - 0.004), z_arm),
                    ("arm_r", box(maxx - arm_t, miny, maxx, maxy - back_t - 0.004), z_arm),
                ]
            else:
                sub_specs = [
                    ("back", box(minx, miny, maxx, miny + back_t), z_hi),
                    ("seat", box(minx + arm_t + 0.004, miny + back_t + 0.004, maxx - arm_t - 0.004, maxy), z_seat),
                    ("arm_l", box(minx, miny + back_t + 0.004, minx + arm_t, maxy), z_arm),
                    ("arm_r", box(maxx - arm_t, miny + back_t + 0.004, maxx, maxy), z_arm),
                ]
        else:
            back_at_max_u = float(front_dir[0]) < 0
            if back_at_max_u:
                sub_specs = [
                    ("back", box(maxx - back_t, miny, maxx, maxy), z_hi),
                    ("seat", box(minx, miny + arm_t + 0.004, maxx - back_t - 0.004, maxy - arm_t - 0.004), z_seat),
                    ("arm_l", box(minx, miny, maxx - back_t - 0.004, miny + arm_t), z_arm),
                    ("arm_r", box(minx, maxy - arm_t, maxx - back_t - 0.004, maxy), z_arm),
                ]
            else:
                sub_specs = [
                    ("back", box(minx, miny, minx + back_t, maxy), z_hi),
                    ("seat", box(minx + back_t + 0.004, miny + arm_t + 0.004, maxx, maxy - arm_t - 0.004), z_seat),
                    ("arm_l", box(minx + back_t + 0.004, miny, maxx, miny + arm_t), z_arm),
                    ("arm_r", box(minx + back_t + 0.004, maxy - arm_t, maxx, maxy), z_arm),
                ]

        emitted_sub = 0
        for sub_name, sub_poly, z1 in sub_specs:
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
                    metadata={"sofa_id": sid},
                )
            )
            emitted_sub += 1

        if emitted_sub > 0:
            sofa_bands.append((clipped, z_floor, z_hi))
            placed_union = clipped if placed_union is None else unary_union([placed_union, clipped])
            logs.append({"sofa_id": sid, "library_matched": False, "obb_ldh_m": [round(length, 3), round(depth, 3), round(h, 3)], "parts": emitted_sub})

    return parts, sofa_bands, {
        "emitted_sofas": len(logs),
        "emitted_parts": len(parts),
        "library_asset_used": sofa_asset.asset_id if sofa_asset is not None else None,
        "sofas": logs,
        "rejected": rejected,
    }
