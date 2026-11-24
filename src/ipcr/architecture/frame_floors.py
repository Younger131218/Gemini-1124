"""Principal room-axis estimation, multi-storey elevation detection, and hole-free floor/ceiling domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ipcr.config import ArchitectureConfig
from ipcr.geometry_primitives import extrude_polygon_solid
from ipcr.types import MeshPart
from ipcr.working_labels import _extract_dominant_vertical_wall_planes


@dataclass
class RoomFrame:
    """Rigid 2D rotation aligning dominant room walls with (u, v) axes."""

    angle_rad: float
    cos_a: float
    sin_a: float

    def xy_to_uv(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=np.float64)
        u = xy[..., 0] * self.cos_a + xy[..., 1] * self.sin_a
        v = -xy[..., 0] * self.sin_a + xy[..., 1] * self.cos_a
        return np.stack([u, v], axis=-1)

    def uv_to_xy(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float64)
        x = uv[..., 0] * self.cos_a - uv[..., 1] * self.sin_a
        y = uv[..., 0] * self.sin_a + uv[..., 1] * self.cos_a
        return np.stack([x, y], axis=-1)

    def pts_to_uvz(self, pts: np.ndarray) -> np.ndarray:
        out = np.empty_like(pts, dtype=np.float64)
        out[:, :2] = self.xy_to_uv(pts[:, :2])
        out[:, 2] = pts[:, 2]
        return out

    def uvz_to_xyz(self, uvz: np.ndarray) -> np.ndarray:
        out = np.empty_like(uvz, dtype=np.float64)
        out[:, :2] = self.uv_to_xy(uvz[:, :2])
        out[:, 2] = uvz[:, 2]
        return out


@dataclass
class FloorCeilingResult:
    frame: RoomFrame
    floor_z: float
    ceiling_z: float | None
    inner_footprint_uv: Polygon
    audit: dict[str, Any]


def estimate_room_frame(points_local: np.ndarray, work_labels: np.ndarray) -> RoomFrame:
    """Estimate dominant room wall orientation via 4-fold circular voting on vertical wall planes."""
    wall_pts = points_local[work_labels == 2]
    planes = _extract_dominant_vertical_wall_planes(
        wall_pts, voxel_size=0.06, plane_dist=0.05, min_plane_voxels=60, max_planes=16
    )
    if not planes:
        return RoomFrame(angle_rad=0.0, cos_a=1.0, sin_a=0.0)

    sx, sy = 0.0, 0.0
    for _c, n, bmin, bmax in planes:
        ang = float(np.arctan2(n[1], n[0]))
        span = bmax - bmin
        weight = float(np.hypot(span[0], span[1])) * max(float(span[2]), 0.5)
        sx += weight * float(np.cos(4.0 * ang))
        sy += weight * float(np.sin(4.0 * ang))

    angle = 0.25 * float(np.arctan2(sy, sx))
    # Normalize to [-pi/4, pi/4) so axes stay close to world X/Y when unrotated
    while angle >= np.pi / 4.0:
        angle -= np.pi / 2.0
    while angle < -np.pi / 4.0:
        angle += np.pi / 2.0
    return RoomFrame(angle_rad=angle, cos_a=float(np.cos(angle)), sin_a=float(np.sin(angle)))


def _dominant_z_peak(z_vals: np.ndarray, bin_size: float = 0.025) -> float:
    zmin, zmax = float(z_vals.min()), float(z_vals.max())
    if zmax - zmin < bin_size * 2:
        return float(np.median(z_vals))
    bins = np.arange(zmin - bin_size * 0.5, zmax + bin_size * 1.5, bin_size)
    hist, edges = np.histogram(z_vals, bins=bins)
    idx = int(np.argmax(hist))
    zc = 0.5 * (edges[idx] + edges[idx + 1])
    band = z_vals[np.abs(z_vals - zc) <= 0.06]
    return float(np.median(band)) if len(band) else zc


def _raster_mask_to_polygon(mask: np.ndarray, u0: float, v0: float, cell: float) -> Polygon:
    """Convert a 2D boolean raster mask (no interior holes) into a clean Shapely Polygon."""
    # Fill all interior scan-shadow holes (Q09/Q10/Q11)
    filled = ndimage.binary_fill_holes(mask)
    # Keep largest connected component (or components >= 5% of largest)
    labeled, ncomp = ndimage.label(filled)
    if ncomp == 0:
        raise ValueError("Empty floor raster mask")
    sizes = ndimage.sum(filled, labeled, range(1, ncomp + 1))
    max_size = float(np.max(sizes))
    keep = np.zeros_like(filled, dtype=bool)
    for cid, sz in enumerate(sizes, start=1):
        if sz >= max(max_size * 0.05, 16):
            keep |= labeled == cid

    # Build exact union of horizontal runs using shared edge arrays (avoids 1e-16 float seam gaps!)
    from shapely.geometry import box

    rects: list[Polygon] = []
    nrows, ncols = keep.shape
    u_edges = u0 + np.arange(nrows + 1, dtype=np.float64) * cell
    v_edges = v0 + np.arange(ncols + 1, dtype=np.float64) * cell
    for r in range(nrows):
        row = keep[r]
        if not np.any(row):
            continue
        padded = np.concatenate(([False], row, [False]))
        diffs = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(diffs == 1)
        ends = np.flatnonzero(diffs == -1)
        u_lo = float(u_edges[r])
        u_hi = float(u_edges[r + 1])
        for c0, c1 in zip(starts, ends):
            rects.append(box(u_lo, float(v_edges[c0]), u_hi, float(v_edges[c1])))

    merged = unary_union(rects)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda g: g.area)
    # Remove any accidental tiny interior rings (non-design holes)
    merged = orient(Polygon(merged.exterior.coords).buffer(0), sign=1.0)
    if isinstance(merged, MultiPolygon):
        merged = max(merged.geoms, key=lambda g: g.area)
    return merged


def _remove_small_raster_jogs(pts: np.ndarray, *, max_jog_m: float = 0.45, min_neighbor_m: float = 1.2) -> np.ndarray:
    """Eliminate spurious 1-4 cell raster staircase notches along straight walls while preserving true alcoves."""
    arr = pts.copy()
    for _ in range(4):
        n = len(arr)
        if n <= 4:
            break
        changed = False
        for i in range(n):
            p_prev = arr[(i - 1) % n]
            p0 = arr[i]
            p1 = arr[(i + 1) % n]
            p_next = arr[(i + 2) % n]
            e_prev = p0 - p_prev
            e_cur = p1 - p0
            e_next = p_next - p1
            l_prev = float(np.linalg.norm(e_prev))
            l_cur = float(np.linalg.norm(e_cur))
            l_next = float(np.linalg.norm(e_next))
            if l_cur > max_jog_m or l_prev < 1e-4 or l_next < 1e-4:
                continue
            # Check if previous and next edges are nearly parallel (same wall direction with a small lateral jog)
            u_prev = e_prev / l_prev
            u_next = e_next / l_next
            if float(np.dot(u_prev, u_next)) > 0.96 and max(l_prev, l_next) >= min_neighbor_m:
                # Project jog onto the longer neighbor's line!
                if l_prev >= l_next:
                    n_perp = np.array([-u_prev[1], u_prev[0]])
                    # Shift p1 and p_next laterally onto p_prev->p0 line
                    lat_p1 = float(np.dot(p1 - p0, n_perp))
                    lat_pn = float(np.dot(p_next - p0, n_perp))
                    if abs(lat_p1) <= max_jog_m:
                        arr[(i + 1) % n] = p1 - n_perp * lat_p1
                        arr[(i + 2) % n] = p_next - n_perp * lat_pn
                        changed = True
                        break
                else:
                    n_perp = np.array([-u_next[1], u_next[0]])
                    lat_p0 = float(np.dot(p0 - p1, n_perp))
                    lat_pp = float(np.dot(p_prev - p1, n_perp))
                    if abs(lat_p0) <= max_jog_m:
                        arr[i] = p0 - n_perp * lat_p0
                        arr[(i - 1) % n] = p_prev - n_perp * lat_pp
                        changed = True
                        break
        # Collapse newly collinear / zero-length vertices
        cleaned: list[np.ndarray] = []
        m = len(arr)
        for k in range(m):
            prev = arr[(k - 1) % m]
            cur = arr[k]
            nxt = arr[(k + 1) % m]
            if np.linalg.norm(cur - prev) < 0.05:
                continue
            v1 = cur - prev
            v2 = nxt - cur
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 > 1e-6 and n2 > 1e-6:
                cross = abs(v1[0] * v2[1] - v1[1] * v2[0]) / (n1 * n2)
                dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
                if cross < 0.04 and dot > 0.0:
                    continue
            cleaned.append(cur)
        if len(cleaned) >= 3:
            arr = np.asarray(cleaned, dtype=np.float64)
        if not changed:
            break
    return arr


def _snap_orthogonal_polygon(poly: Polygon, *, simplify_tol: float = 0.14, min_edge_len: float = 0.25) -> Polygon:
    """Simplify raster staircase edges, remove tiny wall kinks, and snap near-axis wall edges."""
    smoothed = poly.buffer(0.10, join_style=2).buffer(-0.10, join_style=2)
    if isinstance(smoothed, MultiPolygon):
        smoothed = max(smoothed.geoms, key=lambda g: g.area)
    simp = smoothed.simplify(simplify_tol, preserve_topology=True)
    if isinstance(simp, MultiPolygon):
        simp = max(simp.geoms, key=lambda g: g.area)
    coords = list(orient(simp, sign=1.0).exterior.coords)[:-1]
    if len(coords) < 3:
        return orient(poly, sign=1.0)

    pts = np.asarray(coords, dtype=np.float64)
    for _ in range(3):
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            dx = pts[j, 0] - pts[i, 0]
            dy = pts[j, 1] - pts[i, 1]
            length = float(np.hypot(dx, dy))
            if length < 1e-4:
                continue
            ang = abs(float(np.degrees(np.arctan2(abs(dy), abs(dx)))))
            if ang < 12.0:
                mid_v = 0.5 * (pts[i, 1] + pts[j, 1])
                pts[i, 1] = mid_v
                pts[j, 1] = mid_v
            elif ang > 78.0:
                mid_u = 0.5 * (pts[i, 0] + pts[j, 0])
                pts[i, 0] = mid_u
                pts[j, 0] = mid_u

        pts = _remove_small_raster_jogs(pts, max_jog_m=0.45, min_neighbor_m=1.2)

    cand = Polygon(pts).buffer(0)
    if isinstance(cand, MultiPolygon):
        cand = max(cand.geoms, key=lambda g: g.area)
    if cand.is_empty or cand.area < poly.area * 0.70:
        return orient(simp, sign=1.0)
    return orient(Polygon(cand.exterior.coords), sign=1.0)


def _refine_footprint_edges_to_walls(
    poly_uv: Polygon,
    wall_uvz: np.ndarray,
    *,
    search_band_m: float = 0.35,
    min_support_pts: int = 80,
) -> tuple[Polygon, list[dict[str, Any]]]:
    """Snap each outer footprint boundary edge to the inner surface of real wall points."""
    coords = np.asarray(list(orient(poly_uv, sign=1.0).exterior.coords)[:-1], dtype=np.float64)
    n = len(coords)
    if n < 3 or len(wall_uvz) < 200:
        return poly_uv, []

    # Build half-plane line equations for each edge: dot(p, outward_normal) = c
    normals = np.zeros((n, 2), dtype=np.float64)
    offsets = np.zeros(n, dtype=np.float64)
    edge_logs: list[dict[str, Any]] = []

    wall_uv = wall_uvz[:, :2]
    for i in range(n):
        p0 = coords[i]
        p1 = coords[(i + 1) % n]
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-6:
            normals[i] = np.array([1.0, 0.0])
            offsets[i] = float(np.dot(p0, normals[i]))
            continue
        t = seg / length
        # For CCW polygon, outward normal is (t_y, -t_x)
        out_n = np.array([t[1], -t[0]], dtype=np.float64)
        c0 = float(np.dot(p0, out_n))

        rel = wall_uv - p0[None, :]
        s_proj = rel @ t
        d_perp = rel @ out_n
        # Match wall points along this edge and within search band
        mask = (s_proj >= -0.10) & (s_proj <= length + 0.10) & (np.abs(d_perp) <= search_band_m)
        cnt = int(np.sum(mask))
        snapped = False
        c_new = c0
        if cnt >= min_support_pts:
            # Find the DOMINANT vertical wall plane peak (argmax of 2cm histogram) rather than
            # a weak inner tail, preventing furniture/baseboard bleed from pulling the wall inward!
            d_vals = d_perp[mask]
            d_lo, d_hi = float(d_vals.min()), float(d_vals.max())
            if d_hi - d_lo > 0.04:
                bins = np.arange(d_lo - 0.01, d_hi + 0.03, 0.02)
                hist, edges = np.histogram(d_vals, bins=bins)
                dom_bin = int(np.argmax(hist))
                peak_d = 0.5 * (edges[dom_bin] + edges[dom_bin + 1])
                in_band = d_vals[np.abs(d_vals - peak_d) <= 0.035]
                shift = float(np.median(in_band)) if len(in_band) else peak_d
            else:
                shift = float(np.median(d_vals))
            c_new = c0 + shift
            snapped = True

        normals[i] = out_n
        offsets[i] = c_new
        edge_logs.append({
            "edge_index": i,
            "length_m": round(length, 3),
            "wall_support_points": cnt,
            "snapped": snapped,
            "shift_m": round(c_new - c0, 4),
        })

    # Recompute exact corner intersections of consecutive snapped edge lines
    new_coords = coords.copy()
    for i in range(n):
        prev_i = (i - 1) % n
        a_mat = np.vstack([normals[prev_i], normals[i]])
        det = float(np.linalg.det(a_mat))
        if abs(det) > 0.15:
            pt = np.linalg.solve(a_mat, np.array([offsets[prev_i], offsets[i]], dtype=np.float64))
            if np.linalg.norm(pt - coords[i]) <= 0.80:
                new_coords[i] = pt

    refined = Polygon(new_coords).buffer(0)
    if isinstance(refined, MultiPolygon):
        refined = max(refined.geoms, key=lambda g: g.area)
    if refined.is_empty or refined.area < poly_uv.area * 0.75:
        return orient(poly_uv, sign=1.0), edge_logs
    return orient(Polygon(refined.exterior.coords), sign=1.0), edge_logs


def analyze_floor_and_ceiling(
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
) -> FloorCeilingResult:
    """Estimate principal frame, floor/ceiling elevations, and hole-free snapped room footprint."""
    frame = estimate_room_frame(points_local, work_labels)
    uvz = frame.pts_to_uvz(points_local)

    floor_pts = uvz[work_labels == 1]
    if len(floor_pts) == 0:
        raise ValueError("Cannot build room domain: work_labels has 0 floor (class 1) points")
    floor_z = _dominant_z_peak(floor_pts[:, 2])

    ceil_pts = uvz[work_labels == 11]
    ceiling_z = _dominant_z_peak(ceil_pts[:, 2]) if len(ceil_pts) >= 200 else None

    # Build 2D occupancy grid combining floor + low/mid walls + doors/windows
    # Generous padding (0.80m >> closing radius) prevents border erosion in binary_closing!
    struct_mask = (
        (work_labels == 1)
        | ((work_labels == 2) & (uvz[:, 2] <= floor_z + 2.5))
        | (work_labels == 6)
    )
    occ_uv = uvz[struct_mask, :2]
    cell = 0.10
    pad = 0.80
    u0 = float(occ_uv[:, 0].min()) - pad
    v0 = float(occ_uv[:, 1].min()) - pad
    u1 = float(occ_uv[:, 0].max()) + pad
    v1 = float(occ_uv[:, 1].max()) + pad
    nu = int(np.ceil((u1 - u0) / cell)) + 2
    nv = int(np.ceil((v1 - v0) / cell)) + 2

    # Include floor points + attached column footprints (so exterior walls run straight behind pilasters!)
    floor_band = floor_pts[np.abs(floor_pts[:, 2] - floor_z) <= 0.15]
    col_pts = uvz[(work_labels == 4) & (uvz[:, 2] >= floor_z - 0.10) & (uvz[:, 2] <= floor_z + 2.5)]
    grid = np.zeros((nu, nv), dtype=bool)
    iu = np.clip(np.floor((floor_band[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
    iv = np.clip(np.floor((floor_band[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
    grid[iu, iv] = True
    if len(col_pts) > 0:
        ciu = np.clip(np.floor((col_pts[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
        civ = np.clip(np.floor((col_pts[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
        grid[ciu, civ] = True

    # Morphological closing bridges furniture scan shadows & sparse scan gaps safely inside padding
    struct = np.ones((5, 5), dtype=bool)
    closed = ndimage.binary_closing(grid, structure=struct, iterations=2)
    raw_poly = _raster_mask_to_polygon(closed, u0, v0, cell)
    ortho_poly = _snap_orthogonal_polygon(raw_poly)

    wall_uvz = uvz[work_labels == 2]
    snapped_poly, edge_logs = _refine_footprint_edges_to_walls(ortho_poly, wall_uvz)

    return FloorCeilingResult(
        frame=frame,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        inner_footprint_uv=snapped_poly,
        audit={
            "principal_angle_deg": round(float(np.degrees(frame.angle_rad)), 3),
            "floor_z_local": round(floor_z, 4),
            "ceiling_z_local": round(ceiling_z, 4) if ceiling_z is not None else None,
            "clear_height_m": round(ceiling_z - floor_z, 4) if ceiling_z is not None else None,
            "raw_floor_area_m2": round(float(raw_poly.area), 3),
            "snapped_inner_area_m2": round(float(snapped_poly.area), 3),
            "footprint_vertex_count": len(list(snapped_poly.exterior.coords)) - 1,
            "edge_snapping": edge_logs,
        },
    )


def build_floor_and_ceiling_parts(
    fc: FloorCeilingResult,
    outer_shell_uv: Polygon,
    cfg: ArchitectureConfig,
) -> list[MeshPart]:
    """Extrude watertight hole-free floor and ceiling slabs spanning the exact outer shell (B13/B13a)."""
    parts: list[MeshPart] = []

    # Floor slab extends from [floor_z - floor_thickness, floor_z], flush beneath exterior walls!
    fv_uv, ft = extrude_polygon_solid(outer_shell_uv, fc.floor_z - cfg.floor_thickness_m, fc.floor_z)
    fv_xyz = fc.frame.uvz_to_xyz(fv_uv)
    parts.append(
        MeshPart(
            part_id="floor_primary",
            class_id=1,
            role="floor",
            vertices=fv_xyz,
            triangles=ft,
            evidence_kind="measured",
            metadata={
                "z_top_local": round(fc.floor_z, 4),
                "thickness_m": cfg.floor_thickness_m,
                "outer_area_m2": round(float(outer_shell_uv.area), 3),
            },
        )
    )

    if fc.ceiling_z is not None:
        # Ceiling slab extends from [ceiling_z, ceiling_z + ceiling_thickness], flush atop exterior walls!
        cv_uv, ct = extrude_polygon_solid(outer_shell_uv, fc.ceiling_z, fc.ceiling_z + cfg.ceiling_thickness_m)
        cv_xyz = fc.frame.uvz_to_xyz(cv_uv)
        parts.append(
            MeshPart(
                part_id="ceiling_primary",
                class_id=11,
                role="ceiling",
                vertices=cv_xyz,
                triangles=ct,
                evidence_kind="measured",
                metadata={
                    "z_bottom_local": round(fc.ceiling_z, 4),
                    "thickness_m": cfg.ceiling_thickness_m,
                    "outer_area_m2": round(float(outer_shell_uv.area), 3),
                },
            )
        )

    return parts
