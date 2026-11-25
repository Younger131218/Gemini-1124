"""B13a closed exterior wall shell (guaranteed disjoint mitered quads) + interior partition walls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ipcr.architecture.frame_floors import FloorCeilingResult
from ipcr.config import ArchitectureConfig
from ipcr.geometry_primitives import extrude_polygon_solid, make_oriented_box, signed_mesh_volume
from ipcr.types import MeshPart, WallSegment
from ipcr.working_labels import _extract_dominant_vertical_wall_planes, _voxel_connected_components_3d


@dataclass
class ExteriorShellPlan:
    inner_uv: Polygon
    outer_uv: Polygon
    walls: list[WallSegment]
    # Per-wall exact 2D polygon in UV (guaranteed pairwise disjoint & strictly outside inner_uv!)
    wall_polys_uv: dict[str, Polygon]
    audit: dict[str, Any]


def _measure_wall_edge_evidence(
    p0: np.ndarray,
    p1: np.ndarray,
    out_n: np.ndarray,
    wall_uvz: np.ndarray,
    floor_z: float,
    ceiling_z: float,
    cfg: ArchitectureConfig,
) -> dict[str, Any]:
    """Measure wall support points, thickness (dual vs outward-anchored), and z bounds."""
    seg = p1 - p0
    length = float(np.linalg.norm(seg))
    t = seg / max(length, 1e-9)

    rel = wall_uvz[:, :2] - p0[None, :]
    s = rel @ t
    d = rel @ out_n
    mask = (s >= -0.05) & (s <= length + 0.05) & (d >= -0.10) & (d <= cfg.max_wall_thickness_m + 0.10)
    pts = wall_uvz[mask]
    cnt = int(len(pts))

    thickness = cfg.default_wall_thickness_m
    anchor_mode = "single_face_outward"
    if cnt >= 200:
        d_sub = d[mask]
        d_lo, d_hi = float(np.percentile(d_sub, 5)), float(np.percentile(d_sub, 95))
        span_d = d_hi - d_lo
        if cfg.min_wall_thickness_m <= span_d <= cfg.max_wall_thickness_m:
            low_frac = float(np.mean(d_sub <= d_lo + 0.04))
            high_frac = float(np.mean(d_sub >= d_hi - 0.04))
            if low_frac >= 0.20 and high_frac >= 0.20 and span_d >= 0.08:
                thickness = float(np.clip(span_d, cfg.min_wall_thickness_m, cfg.max_wall_thickness_m))
                anchor_mode = "dual_face"
        meas_z_min = float(np.percentile(pts[:, 2], 2))
        meas_z_max = float(np.percentile(pts[:, 2], 98))
    else:
        anchor_mode = "inferred_closure"
        meas_z_min = floor_z
        meas_z_max = ceiling_z

    return {
        "point_count": cnt,
        "thickness_m": round(thickness, 4),
        "anchor_mode": anchor_mode,
        "measured_z_min": round(meas_z_min, 4),
        "measured_z_max": round(meas_z_max, 4),
    }


def plan_exterior_wall_shell(
    fc: FloorCeilingResult,
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
) -> ExteriorShellPlan:
    """Build exact pairwise-disjoint closed exterior walls around the room footprint (B13a)."""
    inner_poly = orient(fc.inner_footprint_uv, sign=1.0)
    in_coords = np.asarray(list(inner_poly.exterior.coords)[:-1], dtype=np.float64)
    n = len(in_coords)

    uvz = fc.frame.pts_to_uvz(points_local)
    wall_uvz = uvz[work_labels == 2]
    z_top = fc.ceiling_z if fc.ceiling_z is not None else float(np.percentile(wall_uvz[:, 2], 98))

    normals = np.zeros((n, 2), dtype=np.float64)
    tangents = np.zeros((n, 2), dtype=np.float64)
    thicknesses = np.zeros(n, dtype=np.float64)
    evidences: list[dict[str, Any]] = []

    for i in range(n):
        p0 = in_coords[i]
        p1 = in_coords[(i + 1) % n]
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        t = seg / max(length, 1e-9)
        out_n = np.array([t[1], -t[0]], dtype=np.float64)
        normals[i] = out_n
        tangents[i] = t
        ev = _measure_wall_edge_evidence(p0, p1, out_n, wall_uvz, fc.floor_z, z_top, cfg)
        thicknesses[i] = float(ev["thickness_m"])
        evidences.append(ev)

    # Use clean mitered buffer of inner_poly for the true outer shell boundary (handles arbitrary concavities cleanly!)
    med_thick = float(np.median(thicknesses)) if len(thicknesses) else cfg.default_wall_thickness_m
    outer_poly = inner_poly.buffer(med_thick, join_style=2, mitre_limit=2.5)
    if isinstance(outer_poly, MultiPolygon):
        outer_poly = max(outer_poly.geoms, key=lambda g: g.area)
    outer_poly = orient(Polygon(outer_poly.exterior.coords).buffer(0), sign=1.0)
    shell_band = outer_poly.difference(inner_poly).buffer(0)

    # Partition shell_band cleanly into pairwise-disjoint wall polygons (longest walls claim corners first!)
    edge_order = sorted(range(n), key=lambda idx: float(np.linalg.norm(in_coords[(idx + 1) % n] - in_coords[idx])), reverse=True)
    claimed_union: Polygon | MultiPolygon | None = None
    wall_polys_by_idx: dict[int, Polygon] = {}

    for i in edge_order:
        p0 = in_coords[i]
        p1 = in_coords[(i + 1) % n]
        t = tangents[i]
        out_n = normals[i]
        th = max(float(thicknesses[i]), med_thick)
        # Extended outward strip covers corner miters cleanly, then clipped to unclaimed shell_band
        ext = th * 1.5
        a_in = p0 - t * ext
        b_in = p1 + t * ext
        b_out = b_in + out_n * (th + 0.05)
        a_out = a_in + out_n * (th + 0.05)
        raw_strip = Polygon([a_in, b_in, b_out, a_out]).buffer(0)
        piece = raw_strip.intersection(shell_band)
        if claimed_union is not None and not claimed_union.is_empty:
            piece = piece.difference(claimed_union)
        piece = piece.buffer(0)
        if isinstance(piece, MultiPolygon):
            piece = max(piece.geoms, key=lambda g: g.area)
        if isinstance(piece, Polygon) and not piece.is_empty and piece.area > 1e-4:
            piece_clean = orient(Polygon(piece.exterior.coords).buffer(0), sign=1.0)
            if isinstance(piece_clean, MultiPolygon):
                piece_clean = max(piece_clean.geoms, key=lambda g: g.area)
            wall_polys_by_idx[i] = piece_clean
            claimed_union = piece_clean if claimed_union is None else unary_union([claimed_union, piece_clean])

    walls: list[WallSegment] = []
    wall_polys: dict[str, Polygon] = {}
    wall_audit: list[dict[str, Any]] = []
    hi_ceils = [c for c in fc.ceiling_levels if (not c.is_primary) and c.z_bottom > z_top + 0.45]
    max_hi_ceil = max((float(c.z_bottom) for c in hi_ceils), default=z_top)
    wall_uvz_all = fc.frame.pts_to_uvz(points_local[work_labels == 2]) if hi_ceils else np.empty((0, 3), dtype=np.float64)

    for i in range(n):
        if i not in wall_polys_by_idx:
            continue
        p0_in = in_coords[i]
        p1_in = in_coords[(i + 1) % n]
        wid = f"ext_wall_{i:02d}"
        ev = evidences[i]
        # Per-wall local ceiling height: walls enclosing a higher stairwell/wing ceiling rise to max_hi_ceil!
        # For a long shared wall (e.g. 1F lobby + 2F stairwell), keep 1F base at z_top and attach an upper_extension
        # along the measured high-wall segment [s_lo, s_hi] so the 1F lobby wall stays flush under ceiling_primary.
        w_z_top = z_top
        upper_ext: list[float] | None = None
        seg_len = float(np.linalg.norm(p1_in - p0_in))
        if hi_ceils and len(wall_uvz_all) > 0 and seg_len > 0.05:
            rel_w = wall_uvz_all[:, :2] - p0_in[None, :]
            s_w = rel_w @ tangents[i]
            d_w = np.abs(rel_w @ normals[i])
            hi_mask = (
                (s_w >= -0.08)
                & (s_w <= seg_len + 0.08)
                & (d_w <= max(0.28, med_thick + 0.10))
                & (wall_uvz_all[:, 2] >= z_top + 0.25)
            )
            if int(np.sum(hi_mask)) >= 300:
                s_hi_vals = s_w[hi_mask]
                s_hi_lo = max(0.0, float(np.percentile(s_hi_vals, 2)) - 0.10)
                s_hi_hi = min(seg_len, float(np.percentile(s_hi_vals, 98)) + 0.10)
                if (s_hi_hi - s_hi_lo) >= 0.60 * seg_len or seg_len <= 3.8:
                    w_z_top = max_hi_ceil
                elif (s_hi_hi - s_hi_lo) >= 0.50:
                    upper_ext = [round(s_hi_lo, 4), round(s_hi_hi, 4), round(max_hi_ceil, 4)]

        meta: dict[str, Any] = {
            "foot_completed_m": round(max(0.0, float(ev["measured_z_min"]) - fc.floor_z), 4),
            "head_completed_m": round(max(0.0, w_z_top - float(ev["measured_z_max"])), 4),
        }
        if upper_ext is not None:
            meta["upper_extension"] = upper_ext

        wseg = WallSegment(
            wall_id=wid,
            p0_xy=p0_in.copy(),
            p1_xy=p1_in.copy(),
            normal_xy=normals[i].copy(),
            tangent_xy=tangents[i].copy(),
            thickness_m=med_thick,
            z_min=fc.floor_z,
            z_max=w_z_top,
            is_exterior=True,
            point_count=int(ev["point_count"]),
            measured_z_min=float(ev["measured_z_min"]),
            measured_z_max=float(ev["measured_z_max"]),
            anchor_mode=str(ev["anchor_mode"]),
            metadata=meta,
        )
        walls.append(wseg)
        wall_polys[wid] = wall_polys_by_idx[i]
        wall_audit.append({
            "wall_id": wid,
            "length_m": round(wseg.length_m, 3),
            "thickness_m": round(wseg.thickness_m, 4),
            "point_count": wseg.point_count,
            "anchor_mode": wseg.anchor_mode,
            "z_span": [round(wseg.z_min, 4), round(wseg.z_max, 4)],
            "measured_z_span": [ev["measured_z_min"], ev["measured_z_max"]],
            **wseg.metadata,
        })

    return ExteriorShellPlan(
        inner_uv=inner_poly,
        outer_uv=outer_poly,
        walls=walls,
        wall_polys_uv=wall_polys,
        audit={
            "exterior_wall_count": len(walls),
            "inner_area_m2": round(float(inner_poly.area), 3),
            "outer_area_m2": round(float(outer_poly.area), 3),
            "walls": wall_audit,
        },
    )


def _slice_wall_poly_along_s(
    wall_poly: Polygon,
    w: WallSegment,
    sa: float,
    sb: float,
    is_first: bool,
    is_last: bool,
) -> list[Polygon]:
    """Intersect a wall's exact 2D polygon with the s-slab [sa, sb] (extended at wall ends to keep corner miters)."""
    s_lo = -w.thickness_m * 3.0 if is_first else sa
    s_hi = w.length_m + w.thickness_m * 3.0 if is_last else sb
    d_pad = w.thickness_m * 3.0 + 0.5
    p0 = w.p0_xy
    t = w.tangent_xy
    n = w.normal_xy
    slab = Polygon([
        p0 + t * s_lo - n * d_pad,
        p0 + t * s_hi - n * d_pad,
        p0 + t * s_hi + n * d_pad,
        p0 + t * s_lo + n * d_pad,
    ])
    clipped = wall_poly.intersection(slab).buffer(0)
    geoms = list(clipped.geoms) if isinstance(clipped, MultiPolygon) else ([clipped] if isinstance(clipped, Polygon) else [])
    out: list[Polygon] = []
    for g in geoms:
        if not g.is_empty and g.area > 1e-4:
            out.append(orient(Polygon(g.exterior.coords), sign=1.0))
    return out


def collect_raised_exterior_wall_cutouts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
) -> list[tuple[Polygon, float, float]]:
    """Return (poly_uv, z_min, z_max) cutouts for raised stairwell exterior walls so primary ceiling stops flush."""
    if fc.ceiling_z is None:
        return []
    cutouts: list[tuple[Polygon, float, float]] = []
    for w in shell.walls:
        wpoly = shell.wall_polys_uv[w.wall_id]
        if w.z_max > fc.ceiling_z + 0.05:
            cutouts.append((wpoly, w.z_min, w.z_max))
        uext = w.metadata.get("upper_extension")
        if uext is not None:
            sa_hi, sb_hi, z_hi_top = float(uext[0]), float(uext[1]), float(uext[2])
            if z_hi_top > w.z_max + 0.05:
                is_first = sa_hi <= 0.15
                is_last = sb_hi >= w.length_m - 0.15
                for sp in _slice_wall_poly_along_s(wpoly, w, sa_hi, sb_hi, is_first=is_first, is_last=is_last):
                    cutouts.append((sp, w.z_max + 0.002, z_hi_top))
    return cutouts


def build_exterior_wall_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
) -> list[MeshPart]:
    """Extrude every exterior wall into pairwise-disjoint watertight solids around all openings (B13a/B14)."""
    parts: list[MeshPart] = []

    for w in shell.walls:
        wpoly = shell.wall_polys_uv[w.wall_id]
        length = w.length_m
        if length < 0.05 or w.z_max <= w.z_min + 0.02:
            continue

        # Emit upper stairwell extension on long shared walls (disjoint 2mm above 1F wall top)
        uext = w.metadata.get("upper_extension")
        if uext is not None:
            sa_hi, sb_hi, z_hi_top = float(uext[0]), float(uext[1]), float(uext[2])
            if z_hi_top > w.z_max + 0.05:
                is_first = sa_hi <= 0.15
                is_last = sb_hi >= length - 0.15
                for u_idx, sp in enumerate(_slice_wall_poly_along_s(wpoly, w, sa_hi, sb_hi, is_first=is_first, is_last=is_last)):
                    try:
                        uv_v, uv_t = extrude_polygon_solid(sp, w.z_max + 0.002, z_hi_top)
                    except ValueError:
                        continue
                    parts.append(
                        MeshPart(
                            part_id=f"{w.wall_id}_upper_{u_idx:02d}",
                            class_id=2,
                            role="exterior_wall",
                            vertices=fc.frame.uvz_to_xyz(uv_v),
                            triangles=uv_t,
                            evidence_kind="measured",
                            metadata={
                                "wall_id": w.wall_id,
                                "s_range": [round(sa_hi, 4), round(sb_hi, 4)],
                                "z_range": [round(w.z_max + 0.002, 4), round(z_hi_top, 4)],
                                "thickness_m": w.thickness_m,
                            },
                        )
                    )

        openings = sorted(w.openings, key=lambda o: o.s_min)
        if not openings:
            v_uv, tri = extrude_polygon_solid(wpoly, w.z_min, w.z_max)
            parts.append(
                MeshPart(
                    part_id=w.wall_id,
                    class_id=2,
                    role="exterior_wall",
                    vertices=fc.frame.uvz_to_xyz(v_uv),
                    triangles=tri,
                    evidence_kind="measured" if w.point_count >= 200 else "inferred_completion",
                    metadata={"wall_id": w.wall_id, "openings_cut": 0, "thickness_m": w.thickness_m},
                )
            )
            continue

        s_cuts = sorted(
            {0.0, length}
            | {float(np.clip(o.s_min, 0.0, length)) for o in openings}
            | {float(np.clip(o.s_max, 0.0, length)) for o in openings}
        )
        s_clean: list[float] = []
        for sc in s_cuts:
            if not s_clean or sc - s_clean[-1] > 0.01:
                s_clean.append(sc)
            else:
                s_clean[-1] = sc if sc == length else s_clean[-1]

        sub_idx = 0
        for i in range(len(s_clean) - 1):
            sa, sb = s_clean[i], s_clean[i + 1]
            if sb - sa < 0.015:
                continue
            s_mid = 0.5 * (sa + sb)
            strip_ops = [
                (float(np.clip(o.z_min, w.z_min, w.z_max)), float(np.clip(o.z_max, w.z_min, w.z_max)))
                for o in openings
                if o.s_min < s_mid < o.s_max
            ]
            strip_ops.sort()
            merged_ops: list[tuple[float, float]] = []
            for za, zb in strip_ops:
                if zb <= za + 0.02:
                    continue
                if merged_ops and za <= merged_ops[-1][1] + 0.01:
                    merged_ops[-1] = (merged_ops[-1][0], max(merged_ops[-1][1], zb))
                else:
                    merged_ops.append((za, zb))

            solid_z: list[tuple[float, float]] = []
            cur_z = w.z_min
            for za, zb in merged_ops:
                if za > cur_z + 0.015:
                    solid_z.append((cur_z, za))
                cur_z = max(cur_z, zb)
            if w.z_max > cur_z + 0.015:
                solid_z.append((cur_z, w.z_max))

            strip_polys = _slice_wall_poly_along_s(wpoly, w, sa, sb, is_first=(i == 0), is_last=(i == len(s_clean) - 2))
            for sp in strip_polys:
                for za, zb in solid_z:
                    try:
                        v_uv, tri = extrude_polygon_solid(sp, za, zb)
                    except ValueError:
                        continue
                    role = "exterior_wall"
                    if merged_ops:
                        if zb <= merged_ops[0][0] + 0.02:
                            role = "exterior_wall_sill"
                        elif za >= merged_ops[-1][1] - 0.02:
                            role = "exterior_wall_lintel"
                    parts.append(
                        MeshPart(
                            part_id=f"{w.wall_id}_seg_{sub_idx:02d}",
                            class_id=2,
                            role=role,
                            vertices=fc.frame.uvz_to_xyz(v_uv),
                            triangles=tri,
                            evidence_kind="measured" if w.point_count >= 200 else "inferred_completion",
                            metadata={
                                "wall_id": w.wall_id,
                                "s_range": [round(sa, 4), round(sb, 4)],
                                "z_range": [round(za, 4), round(zb, 4)],
                            },
                        )
                    )
                    sub_idx += 1

    return parts


def _extrude_wall_sz_profile(
    fc: FloorCeilingResult,
    origin_uv: np.ndarray,
    t_dir: np.ndarray,
    n_dir: np.ndarray,
    d_lo: float,
    d_hi: float,
    profile_sz: Polygon,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a 2D (s, z) wall side-elevation polygon across wall thickness [d_lo, d_hi]."""
    prof = orient(profile_sz.buffer(0), sign=1.0)
    if isinstance(prof, MultiPolygon):
        prof = max(prof.geoms, key=lambda g: g.area)
    v_szd, tri = extrude_polygon_solid(prof, d_lo, d_hi)
    s_val = v_szd[:, 0]
    z_val = v_szd[:, 1]
    d_val = v_szd[:, 2]
    uv = origin_uv[None, :] + s_val[:, None] * t_dir[None, :] + d_val[:, None] * n_dir[None, :]
    v_xyz = fc.frame.uvz_to_xyz(np.column_stack([uv, z_val]))
    if signed_mesh_volume(v_xyz, tri) < 0:
        tri = tri[:, [0, 2, 1]]
    return v_xyz, tri


def _build_stepped_wall_profile_sz(
    s_vals: np.ndarray,
    z_vals: np.ndarray,
    s_lo: float,
    s_hi: float,
    z_floor_snap: float,
    z_ceil_snap: float,
    *,
    bin_w: float = 0.28,
    min_bin_pts: int = 35,
    snap_floor_tol: float = 0.35,
    snap_ceil_tol: float = 0.30,
) -> tuple[Polygon | None, float, float, bool]:
    """Build a crisp simplified 2D (s, z) side-elevation profile polygon for an irregular/stepped/sloped wall.

    Returns (profile_poly_sz, z_min_eff, z_max_eff, is_irregular).
    - If the wall has a nearly uniform rectangular profile across its span, returns is_irregular=False.
    - If the wall is an overhead header/clerestory wall (lifted bottom) or has a sloped/stepped top/bottom
      (e.g. under-stairs triangular spandrel wall or stairwell step-wall), returns a clean simplified polygon!
    """
    span = s_hi - s_lo
    if span < 0.35 or len(s_vals) < 100:
        return None, 0.0, 0.0, False
    n_bins = max(3, int(round(span / bin_w)))
    edges = np.linspace(s_lo, s_hi, n_bins + 1)
    bin_s: list[float] = []
    bin_zlo: list[float] = []
    bin_zhi: list[float] = []
    for k in range(n_bins):
        m = (s_vals >= edges[k]) & (s_vals <= edges[k + 1])
        if int(np.sum(m)) < min_bin_pts:
            continue
        zb = z_vals[m]
        zl = float(np.percentile(zb, 4))
        zh = float(np.percentile(zb, 96))
        if zh - zl < 0.22:
            continue
        bin_s.append(0.5 * float(edges[k] + edges[k + 1]))
        bin_zlo.append(zl)
        bin_zhi.append(zh)

    if len(bin_s) < 2:
        return None, 0.0, 0.0, False

    zlo_arr = np.asarray(bin_zlo, dtype=np.float64)
    zhi_arr = np.asarray(bin_zhi, dtype=np.float64)

    # Guard against a single narrow end-jamb/post dragging an overhead header wall down to the floor:
    # If >=70% of bins share a lifted bottom (> median_zlo - 0.30) while <=20% end bins dip sharply lower (>0.60m),
    # clip those end-jamb outliers to the header's true bottom so we don't block an open doorway/walkway!
    med_zlo = float(np.median(zlo_arr))
    med_zhi = float(np.median(zhi_arr))
    high_bot_frac = float(np.mean(zlo_arr >= med_zlo - 0.25))
    if high_bot_frac >= 0.70 and med_zlo > z_floor_snap + 0.55:
        zlo_arr = np.where(zlo_arr < med_zlo - 0.55, med_zlo, zlo_arr)

    # Snap bin bottoms/tops cleanly to supporting floor / overhead ceiling when close
    zlo_arr = np.where(zlo_arr <= z_floor_snap + snap_floor_tol, z_floor_snap + 0.002, zlo_arr)
    zhi_arr = np.where(zhi_arr >= z_ceil_snap - snap_ceil_tol, z_ceil_snap - 0.002, zhi_arr)
    zhi_arr = np.minimum(zhi_arr, z_ceil_snap - 0.002)
    zlo_arr = np.maximum(zlo_arr, z_floor_snap + 0.002)

    valid = (zhi_arr - zlo_arr) >= 0.22
    if int(np.sum(valid)) < 2:
        return None, 0.0, 0.0, False
    bs = np.asarray(bin_s, dtype=np.float64)[valid]
    zlo_arr = zlo_arr[valid]
    zhi_arr = zhi_arr[valid]

    z_min_eff = float(np.min(zlo_arr))
    z_max_eff = float(np.max(zhi_arr))
    bot_var = float(np.max(zlo_arr) - np.min(zlo_arr))
    top_var = float(np.max(zhi_arr) - np.min(zhi_arr))
    is_irregular = (bot_var >= 0.35) or (top_var >= 0.35)

    # Extend outermost valid bins cleanly to [s_lo, s_hi]
    top_pts: list[tuple[float, float]] = [(s_lo, float(zhi_arr[0]))]
    for sv, zv in zip(bs, zhi_arr):
        top_pts.append((float(sv), float(zv)))
    top_pts.append((s_hi, float(zhi_arr[-1])))

    bot_pts: list[tuple[float, float]] = [(s_hi, float(zlo_arr[-1]))]
    for sv, zv in zip(bs[::-1], zlo_arr[::-1]):
        bot_pts.append((float(sv), float(zv)))
    bot_pts.append((s_lo, float(zlo_arr[0])))

    raw_prof = Polygon(top_pts + bot_pts).buffer(0)
    if isinstance(raw_prof, MultiPolygon):
        raw_prof = max(raw_prof.geoms, key=lambda g: g.area)
    if not isinstance(raw_prof, Polygon) or raw_prof.is_empty or raw_prof.area < 0.15:
        return None, z_min_eff, z_max_eff, False
    simp = raw_prof.simplify(0.08, preserve_topology=True).buffer(0)
    if isinstance(simp, MultiPolygon):
        simp = max(simp.geoms, key=lambda g: g.area)
    if not isinstance(simp, Polygon) or simp.is_empty or simp.area < 0.15:
        simp = raw_prof
    return orient(simp, sign=1.0), z_min_eff, z_max_eff, is_irregular


def build_interior_wall_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
) -> tuple[list[MeshPart], list[Polygon], list[tuple[Polygon, float, float]], dict[str, Any]]:
    """Detect and build standalone interior walls (including overhead headers & sloped under-stairs walls)."""
    uvz = fc.frame.pts_to_uvz(points_local)
    wpts = uvz[work_labels == 2]
    if len(wpts) < cfg.wall_min_points:
        return [], [], [], {"interior_wall_count": 0, "walls": []}

    # Exclude wall points within 0.45m of the exterior wall band (avoid duplicating exterior wall scans!)
    from shapely import contains_xy

    inner_core = shell.inner_uv.buffer(-0.45)
    if inner_core.is_empty:
        return [], [], [], {"interior_wall_count": 0, "walls": []}

    in_mask = contains_xy(inner_core, wpts[:, 0], wpts[:, 1])
    int_pts = wpts[in_mask]
    if len(int_pts) < cfg.wall_min_points:
        return [], [], [], {"interior_wall_count": 0, "walls": []}

    z_primary_ceil = fc.ceiling_z if fc.ceiling_z is not None else float(np.percentile(wpts[:, 2], 98))
    z_max_ceil = max((float(c.z_bottom) for c in fc.ceiling_levels), default=z_primary_ceil)
    comps = _voxel_connected_components_3d(int_pts, voxel_size=0.12)
    parts: list[MeshPart] = []
    polys_uv: list[Polygon] = []
    slab_cutouts: list[tuple[Polygon, float, float]] = []
    logs: list[dict[str, Any]] = []
    allowed = shell.inner_uv.buffer(-0.005)
    placed_union: Polygon | MultiPolygon | None = None

    for idx, comp in enumerate(comps):
        if len(comp) < cfg.wall_min_points:
            continue
        sub = int_pts[comp]
        mean_uv = sub[:, :2].mean(axis=0)
        centered = sub[:, :2] - mean_uv
        cov = centered.T @ centered / max(len(sub), 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        t_dir = eigvecs[:, 1]
        n_dir = eigvecs[:, 0]
        ang = abs(float(np.degrees(np.arctan2(abs(t_dir[1]), abs(t_dir[0])))))
        if ang < 7.0:
            t_dir = np.array([1.0, 0.0])
            n_dir = np.array([0.0, 1.0])
        elif ang > 83.0:
            t_dir = np.array([0.0, 1.0])
            n_dir = np.array([1.0, 0.0])

        s_proj = centered @ t_dir
        d_proj = centered @ n_dir
        s_lo, s_hi = float(np.percentile(s_proj, 2)), float(np.percentile(s_proj, 98))
        d_lo, d_hi = float(np.percentile(d_proj, 5)), float(np.percentile(d_proj, 95))
        length = s_hi - s_lo
        meas_thick = d_hi - d_lo
        z_lo_raw = float(np.percentile(sub[:, 2], 2))
        z_hi_raw = float(np.percentile(sub[:, 2], 98))

        if length < cfg.wall_min_length_m or (z_hi_raw - z_lo_raw) < cfg.wall_min_height_m:
            continue
        if meas_thick > 0.65:
            continue

        thick = float(np.clip(max(meas_thick, cfg.min_wall_thickness_m), cfg.min_wall_thickness_m, cfg.max_wall_thickness_m))
        d_mid = 0.5 * (d_lo + d_hi)
        p0 = mean_uv + t_dir * s_lo + n_dir * (d_mid - 0.5 * thick)
        p1 = mean_uv + t_dir * s_hi + n_dir * (d_mid - 0.5 * thick)
        p2 = mean_uv + t_dir * s_hi + n_dir * (d_mid + 0.5 * thick)
        p3 = mean_uv + t_dir * s_lo + n_dir * (d_mid + 0.5 * thick)
        poly = Polygon([p0, p1, p2, p3]).intersection(allowed).buffer(0)
        if placed_union is not None and not placed_union.is_empty:
            poly = poly.difference(placed_union.buffer(0.004)).buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        if not isinstance(poly, Polygon) or poly.is_empty or poly.area < 0.03:
            continue

        # Determine local floor base & local ceiling top (supports 2F clerestory/step walls above 1F ceiling!)
        base_z = fc.floor_z
        for flv in fc.floor_levels:
            if not flv.is_primary and flv.polygon_uv.intersects(poly) and abs(z_lo_raw - flv.z_top) <= 0.40:
                base_z = max(base_z, flv.z_top)

        local_top = z_primary_ceil
        # Check if this wall genuinely sits in/borders a higher ceiling wing (e.g. 2F stairwell step-wall!)
        for clv in fc.ceiling_levels:
            if not clv.is_primary and clv.z_bottom > z_primary_ceil + 0.45:
                if clv.polygon_uv.buffer(0.60).intersects(poly) and z_hi_raw > z_primary_ceil + 0.30:
                    local_top = max(local_top, float(clv.z_bottom))
            elif not clv.is_primary and clv.z_bottom > base_z + 0.30 and clv.polygon_uv.intersects(poly):
                local_top = min(local_top, float(clv.z_bottom))
        for flv in fc.floor_levels:
            if not flv.is_primary and flv.z_top - cfg.floor_thickness_m > base_z + 0.30 and flv.polygon_uv.intersects(poly):
                if float(np.mean(sub[:, 2] > flv.z_top + 0.25)) < 0.08:
                    local_top = min(local_top, flv.z_top - cfg.floor_thickness_m)

        # Build true 2D (s, z) profile so overhead headers & sloped under-stairs walls NEVER drop fake curtains to the floor!
        poly_coords = np.asarray(poly.exterior.coords, dtype=np.float64)[:, :2]
        rel_poly = poly_coords - mean_uv[None, :]
        s_clip_lo = float(np.min(rel_poly @ t_dir))
        s_clip_hi = float(np.max(rel_poly @ t_dir))
        prof_sz, z0, z1, is_irreg = _build_stepped_wall_profile_sz(
            s_proj,
            sub[:, 2],
            max(s_lo, s_clip_lo),
            min(s_hi, s_clip_hi),
            base_z,
            local_top,
        )
        if prof_sz is None or z1 <= z0 + 0.25:
            continue

        # Also avoid colliding with primary ceiling slab when a 2F clerestory wall starts just below/inside ceiling_primary
        if fc.ceiling_z is not None and z0 >= fc.ceiling_z - 0.12 and z1 > fc.ceiling_z + cfg.ceiling_thickness_m + 0.15:
            # Start cleanly flush at fc.ceiling_z + 0.002 and register in slab_cutouts so ceiling_primary trims cleanly around it!
            z0_snap = max(z0, fc.ceiling_z + 0.002)
            prof_sz = prof_sz.intersection(Polygon([
                (s_clip_lo - 1.0, z0_snap),
                (s_clip_hi + 1.0, z0_snap),
                (s_clip_hi + 1.0, local_top - 0.002),
                (s_clip_lo - 1.0, local_top - 0.002),
            ])).buffer(0)
            if isinstance(prof_sz, MultiPolygon):
                prof_sz = max(prof_sz.geoms, key=lambda g: g.area)
            if not isinstance(prof_sz, Polygon) or prof_sz.is_empty or prof_sz.area < 0.15:
                continue
            z0 = z0_snap

        poly_clean = orient(poly.buffer(0), sign=1.0)
        if isinstance(poly_clean, MultiPolygon):
            poly_clean = max(poly_clean.geoms, key=lambda g: g.area)

        if is_irreg:
            try:
                v_xyz, tri = _extrude_wall_sz_profile(
                    fc,
                    mean_uv,
                    t_dir,
                    n_dir,
                    d_mid - 0.5 * thick,
                    d_mid + 0.5 * thick,
                    prof_sz,
                )
            except ValueError:
                continue
        else:
            v_uv, tri = extrude_polygon_solid(poly_clean, z0, z1)
            v_xyz = fc.frame.uvz_to_xyz(v_uv)

        pid = f"int_wall_{len(parts):02d}"
        role = "interior_wall_header" if z0 > base_z + 0.55 else ("interior_wall_irregular" if is_irreg else "interior_wall")
        parts.append(
            MeshPart(
                part_id=pid,
                class_id=2,
                role=role,
                vertices=v_xyz,
                triangles=tri,
                evidence_kind="measured",
                metadata={
                    "length_m": round(length, 3),
                    "thickness_m": round(thick, 4),
                    "z_span": [round(z0, 4), round(z1, 4)],
                    "irregular_profile": bool(is_irreg),
                    "point_count": int(len(comp)),
                },
            )
        )
        # Only floor-standing interior walls act as full-height 2D XY blockers;
        # overhead header / 2F clerestory walls only block objects overlapping their [z0, z1] band!
        if z0 <= fc.floor_z + 0.55:
            polys_uv.append(poly_clean)
        slab_cutouts.append((poly_clean, z0, z1))
        placed_union = poly_clean if placed_union is None else unary_union([placed_union, poly_clean])
        logs.append(parts[-1].metadata | {"part_id": pid})

    return parts, polys_uv, slab_cutouts, {"interior_wall_count": len(parts), "walls": logs}
