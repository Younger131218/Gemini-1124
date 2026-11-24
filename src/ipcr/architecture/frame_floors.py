"""Principal room-axis estimation, multi-storey elevation detection, and hole-free floor/ceiling domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon, box
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
class FloorLevel:
    level_id: int
    z_top: float
    polygon_uv: Polygon
    point_count: int
    is_primary: bool = False


@dataclass
class CeilingLevel:
    level_id: int
    z_bottom: float
    polygon_uv: Polygon
    point_count: int
    is_primary: bool = False


@dataclass
class FloorCeilingResult:
    frame: RoomFrame
    floor_z: float
    ceiling_z: float | None
    inner_footprint_uv: Polygon
    audit: dict[str, Any]
    floor_levels: list[FloorLevel] = field(default_factory=list)
    ceiling_levels: list[CeilingLevel] = field(default_factory=list)


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


def _detect_z_peaks(
    pts_uvz: np.ndarray,
    *,
    bin_size: float = 0.04,
    min_sep_z: float = 1.5,
    min_pts: int = 4000,
    min_area_m2: float = 3.0,
    band_half_m: float = 0.14,
    max_levels: int = 4,
) -> list[tuple[float, int]]:
    """Detect distinct horizontal slab elevation peaks separated by at least min_sep_z."""
    if len(pts_uvz) == 0:
        return []
    z_vals = pts_uvz[:, 2]
    zmin, zmax = float(z_vals.min()), float(z_vals.max())
    if zmax - zmin < min_sep_z:
        zc = _dominant_z_peak(z_vals)
        cnt = int(np.sum(np.abs(z_vals - zc) <= band_half_m))
        return [(zc, cnt)]

    bins = np.arange(zmin - bin_size * 0.5, zmax + bin_size * 1.5, bin_size)
    hist, edges = np.histogram(z_vals, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    work_hist = hist.astype(np.int64).copy()

    peaks: list[tuple[float, int]] = []
    for _ in range(max_levels):
        idx = int(np.argmax(work_hist))
        if work_hist[idx] < max(20, min_pts // 25):
            break
        zc_raw = float(centers[idx])
        # Suppress neighborhood before validation so we never stall on rejected sub-peaks
        work_hist[np.abs(centers - zc_raw) < min_sep_z] = 0

        band_mask = np.abs(z_vals - zc_raw) <= band_half_m
        cnt = int(np.sum(band_mask))
        if cnt < min_pts and len(peaks) > 0:
            continue
        band_pts = pts_uvz[band_mask]
        zc = float(np.median(band_pts[:, 2]))

        # Check 2D occupied area at 0.20m cells
        cell = 0.20
        iu = np.floor(band_pts[:, 0] / cell).astype(np.int64)
        iv = np.floor(band_pts[:, 1] / cell).astype(np.int64)
        keys = iu * 1_000_003 + iv
        area_m2 = float(len(np.unique(keys))) * (cell * cell)
        if area_m2 < min_area_m2 and len(peaks) > 0:
            continue
        peaks.append((zc, cnt))

    if not peaks:
        zc = _dominant_z_peak(z_vals)
        peaks.append((zc, int(np.sum(np.abs(z_vals - zc) <= band_half_m))))

    peaks.sort(key=lambda item: item[0])
    return peaks


def _raster_mask_to_polygon(mask: np.ndarray, u0: float, v0: float, cell: float) -> Polygon:
    """Convert a 2D boolean raster mask (no interior holes, no thin noise whiskers) into a clean Shapely Polygon."""
    filled = ndimage.binary_fill_holes(mask)
    # Remove 1-2 cell thin noise whiskers/spikes poking outside the room while preserving real alcoves (>=0.3m wide)
    opened = ndimage.binary_opening(filled, structure=np.ones((3, 3), dtype=bool), iterations=1)
    if np.any(opened):
        filled = ndimage.binary_fill_holes(opened)
    labeled, ncomp = ndimage.label(filled)
    if ncomp == 0:
        raise ValueError("Empty floor raster mask")
    sizes = ndimage.sum(filled, labeled, range(1, ncomp + 1))
    max_size = float(np.max(sizes))
    keep = np.zeros_like(filled, dtype=bool)
    for cid, sz in enumerate(sizes, start=1):
        if sz >= max(max_size * 0.05, 16):
            keep |= labeled == cid

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
            u_prev = e_prev / l_prev
            u_next = e_next / l_next
            if float(np.dot(u_prev, u_next)) > 0.96 and max(l_prev, l_next) >= min_neighbor_m:
                if l_prev >= l_next:
                    n_perp = np.array([-u_prev[1], u_prev[0]])
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


def _extract_level_polygon_uv(band_uv: np.ndarray, *, cell: float = 0.12) -> Polygon | None:
    """Extract a clean hole-filled orthogonalized 2D polygon for an upper floor or secondary ceiling."""
    if len(band_uv) < 50:
        return None
    pad = 0.70
    u0 = float(band_uv[:, 0].min()) - pad
    v0 = float(band_uv[:, 1].min()) - pad
    u1 = float(band_uv[:, 0].max()) + pad
    v1 = float(band_uv[:, 1].max()) + pad
    nu = int(np.ceil((u1 - u0) / cell)) + 2
    nv = int(np.ceil((v1 - v0) / cell)) + 2
    grid = np.zeros((nu, nv), dtype=bool)
    iu = np.clip(np.floor((band_uv[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
    iv = np.clip(np.floor((band_uv[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
    grid[iu, iv] = True
    closed = ndimage.binary_closing(grid, structure=np.ones((5, 5), dtype=bool), iterations=2)
    try:
        raw = _raster_mask_to_polygon(closed, u0, v0, cell)
    except ValueError:
        return None
    if raw.is_empty or raw.area < 1.0:
        return None
    return _snap_orthogonal_polygon(raw, simplify_tol=0.14)


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
        out_n = np.array([t[1], -t[0]], dtype=np.float64)
        c0 = float(np.dot(p0, out_n))

        rel = wall_uv - p0[None, :]
        s_proj = rel @ t
        d_perp = rel @ out_n
        mask = (s_proj >= -0.10) & (s_proj <= length + 0.10) & (np.abs(d_perp) <= search_band_m)
        cnt = int(np.sum(mask))
        snapped = False
        c_new = c0
        if cnt >= min_support_pts:
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


def _square_spurious_corner_bevels(
    poly_uv: Polygon,
    wall_uvz: np.ndarray,
    *,
    max_bevel_len_m: float = 1.40,
    min_flank_len_m: float = 0.90,
) -> tuple[Polygon, int]:
    """Replace spurious diagonal corner chamfers (caused by corner clutter/scan shadows) with exact 90-deg corners.

    A genuine 45-degree architectural wall has substantial wall points strictly along the diagonal
    face away from the extrapolated orthogonal walls and is preserved untouched.
    """
    coords = np.asarray(list(orient(poly_uv, sign=1.0).exterior.coords)[:-1], dtype=np.float64)
    if len(coords) < 5 or len(wall_uvz) < 200:
        return poly_uv, 0

    wall_uv = wall_uvz[:, :2]
    squared_count = 0

    for _ in range(6):
        n = len(coords)
        if n < 5:
            break
        changed = False
        for i in range(n):
            p_prev = coords[(i - 1) % n]
            p0 = coords[i]
            p1 = coords[(i + 1) % n]
            p_next = coords[(i + 2) % n]

            e_prev = p0 - p_prev
            e_cur = p1 - p0
            e_next = p_next - p1
            l_prev = float(np.linalg.norm(e_prev))
            l_cur = float(np.linalg.norm(e_cur))
            l_next = float(np.linalg.norm(e_next))

            if l_cur > max_bevel_len_m or l_prev < min_flank_len_m or l_next < min_flank_len_m:
                continue
            t_prev = e_prev / l_prev
            t_cur = e_cur / max(l_cur, 1e-9)
            t_next = e_next / l_next

            # Check if flanking edges are nearly perpendicular (forming a real 90-deg corner)
            if abs(float(np.dot(t_prev, t_next))) > 0.22:
                continue
            # Check if current short edge is diagonal relative to its flanking walls
            ang_prev = abs(float(np.dot(t_cur, t_prev)))
            ang_next = abs(float(np.dot(t_cur, t_next)))
            if max(ang_prev, ang_next) > 0.95:
                continue

            # Solve exact intersection of extended flanking wall lines
            n_prev = np.array([t_prev[1], -t_prev[0]], dtype=np.float64)
            n_next = np.array([t_next[1], -t_next[0]], dtype=np.float64)
            a_mat = np.vstack([n_prev, n_next])
            det = float(np.linalg.det(a_mat))
            if abs(det) < 0.25:
                continue
            c_prev = float(np.dot(p0, n_prev))
            c_next = float(np.dot(p1, n_next))
            p_corner = np.linalg.solve(a_mat, np.array([c_prev, c_next], dtype=np.float64))
            if float(np.linalg.norm(p_corner - p0)) > max_bevel_len_m * 1.3 or float(np.linalg.norm(p_corner - p1)) > max_bevel_len_m * 1.3:
                continue

            # Compare wall evidence on the diagonal face vs wall evidence continuing along the
            # two extrapolated 90-degree corner legs [p0 -> p_corner] and [p_corner -> p1]!
            # Scale-invariant: handles both sparse synthetic clouds and 26M-pt LiDAR scans cleanly.
            n_cur = np.array([t_cur[1], -t_cur[0]], dtype=np.float64)
            rel_cur = wall_uv - p0[None, :]
            s_cur = rel_cur @ t_cur
            d_cur = np.abs(rel_cur @ n_cur)
            on_diag = (s_cur >= 0.08) & (s_cur <= l_cur - 0.08) & (d_cur <= 0.10)
            if int(np.sum(on_diag)) > 0:
                sub_uv = wall_uv[on_diag]
                d_to_prev = np.abs(sub_uv @ n_prev - c_prev)
                d_to_next = np.abs(sub_uv @ n_next - c_next)
                strictly_diag_pts = int(np.sum((d_to_prev > 0.14) & (d_to_next > 0.14)))
            else:
                strictly_diag_pts = 0

            leg1_len = float(np.linalg.norm(p_corner - p0))
            leg2_len = float(np.linalg.norm(p1 - p_corner))
            rel1 = wall_uv - p0[None, :]
            s1 = rel1 @ t_prev
            d1 = np.abs(rel1 @ n_prev)
            on_leg1 = int(np.sum((s1 >= 0.05) & (s1 <= leg1_len + 0.08) & (d1 <= 0.12)))

            rel2 = wall_uv - p_corner[None, :]
            s2 = rel2 @ t_next
            d2 = np.abs(rel2 @ n_next)
            on_leg2 = int(np.sum((s2 >= -0.08) & (s2 <= leg2_len - 0.05) & (d2 <= 0.12)))
            orth_corner_pts = on_leg1 + on_leg2

            # Preserve diagonal ONLY if it has genuine standalone diagonal wall support and NOT dominated by 90-deg corner walls
            if strictly_diag_pts >= 400 and orth_corner_pts < strictly_diag_pts * 1.5:
                continue

            # Replace p0 and p1 with the single crisp 90-degree corner p_corner
            j = (i + 1) % n
            if j > i:
                new_list = [coords[k] for k in range(n) if k not in (i, j)]
                new_list.insert(i, p_corner)
            else:
                # Wrap-around edge (i == n-1, j == 0)
                new_list = [p_corner] + [coords[k] for k in range(1, n - 1)]
            cand = Polygon(new_list).buffer(0)
            if isinstance(cand, MultiPolygon):
                cand = max(cand.geoms, key=lambda g: g.area)
            if not cand.is_empty and cand.area >= poly_uv.area * 0.90:
                coords = np.asarray(list(orient(cand, sign=1.0).exterior.coords)[:-1], dtype=np.float64)
                squared_count += 1
                changed = True
                break
        if not changed:
            break

    # Also collapse tiny collinear/micro-step vertices (< 0.09 m) produced during snapping
    cleaned = _remove_small_raster_jogs(coords, max_jog_m=0.12, min_neighbor_m=0.80)
    final_poly = Polygon(cleaned).buffer(0)
    if isinstance(final_poly, MultiPolygon):
        final_poly = max(final_poly.geoms, key=lambda g: g.area)
    if final_poly.is_empty or final_poly.area < poly_uv.area * 0.90:
        return orient(poly_uv, sign=1.0), squared_count
    return orient(Polygon(final_poly.exterior.coords), sign=1.0), squared_count


def analyze_floor_and_ceiling(
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
) -> FloorCeilingResult:
    """Estimate principal frame, multi-storey floor/ceiling elevations, and hole-free snapped room footprint."""
    frame = estimate_room_frame(points_local, work_labels)
    uvz = frame.pts_to_uvz(points_local)

    floor_pts = uvz[work_labels == 1]
    if len(floor_pts) == 0:
        raise ValueError("Cannot build room domain: work_labels has 0 floor (class 1) points")

    # Detect all distinct storey floor elevations (separated by >= 1.5m)
    floor_peaks = _detect_z_peaks(
        floor_pts,
        bin_size=0.04,
        min_sep_z=1.5,
        min_pts=max(1000, min(5000, int(len(floor_pts) * 0.02))),
        min_area_m2=3.0,
        band_half_m=0.16,
    )
    # Primary floor is the lowest elevation slab (1F)
    floor_z = float(floor_peaks[0][0])

    # Detect all distinct ceiling elevations (main roof + true lower/higher side-wing ceilings >= 0.85m apart)
    ceil_pts = uvz[work_labels == 11]
    ceiling_levels: list[CeilingLevel] = []
    ceiling_z: float | None = None
    if len(ceil_pts) >= 200:
        ceil_peaks = _detect_z_peaks(
            ceil_pts,
            bin_size=0.04,
            min_sep_z=0.85,
            min_pts=max(800, min(2000, int(len(ceil_pts) * 0.02))),
            min_area_m2=2.0,
            band_half_m=0.14,
        )
        # Primary ceiling is the dominant ceiling peak (largest point count)
        dom_ceil_idx = int(np.argmax([cnt for _zc, cnt in ceil_peaks]))
        ceiling_z = float(ceil_peaks[dom_ceil_idx][0])
        raw_clvs: list[CeilingLevel] = []
        primary_cpoly: Polygon | None = None
        for cid, (zc, cnt) in enumerate(ceil_peaks):
            band = ceil_pts[np.abs(ceil_pts[:, 2] - zc) <= 0.16, :2]
            cpoly = _extract_level_polygon_uv(band, cell=0.12)
            if cpoly is not None:
                is_prim = cid == dom_ceil_idx
                if is_prim:
                    primary_cpoly = cpoly
                raw_clvs.append(
                    CeilingLevel(
                        level_id=cid,
                        z_bottom=float(zc),
                        polygon_uv=cpoly,
                        point_count=int(cnt),
                        is_primary=is_prim,
                    )
                )
        # Keep primary ceiling + genuine lower side-wing ceilings (below primary roof and >=50% outside primary XY)
        for clv in raw_clvs:
            if clv.is_primary or primary_cpoly is None:
                ceiling_levels.append(clv)
            else:
                if clv.z_bottom > ceiling_z - 0.80:
                    continue
                outside_area = float(clv.polygon_uv.difference(primary_cpoly).area)
                if outside_area >= max(1.5, float(clv.polygon_uv.area) * 0.50):
                    ceiling_levels.append(clv)

    # Build Level-0 2D occupancy grid combining 1F floor + attached columns + stairs shadow footprint
    struct_mask = (
        (work_labels == 1)
        | ((work_labels == 2) & (uvz[:, 2] <= floor_z + 2.5))
        | (work_labels == 6)
        | (work_labels == 13)
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

    # Capture full 1F floor extent (allowing up to +-0.35m for slight SLAM tilt or 1-2 step raised vestibules,
    # while rejecting >=0.40m windowsill ledges and staying safely below any 2F floor level!)
    floor_upper_band = floor_z + 0.35
    if len(floor_peaks) >= 2:
        floor_upper_band = min(floor_upper_band, 0.5 * (floor_z + float(floor_peaks[1][0])))
    floor_band = floor_pts[(floor_pts[:, 2] >= floor_z - 0.35) & (floor_pts[:, 2] <= floor_upper_band)]
    col_pts = uvz[(work_labels == 4) & (uvz[:, 2] >= floor_z - 0.15) & (uvz[:, 2] <= floor_z + 2.5)]
    stair_pts = uvz[work_labels == 13]

    grid = np.zeros((nu, nv), dtype=bool)
    for pts_add in (floor_band, col_pts, stair_pts):
        if len(pts_add) > 0:
            iu = np.clip(np.floor((pts_add[:, 0] - u0) / cell).astype(np.int64), 0, nu - 1)
            iv = np.clip(np.floor((pts_add[:, 1] - v0) / cell).astype(np.int64), 0, nv - 1)
            grid[iu, iv] = True

    struct = np.ones((5, 5), dtype=bool)
    closed = ndimage.binary_closing(grid, structure=struct, iterations=2)
    raw_poly = _raster_mask_to_polygon(closed, u0, v0, cell)
    ortho_poly = _snap_orthogonal_polygon(raw_poly)

    wall_uvz = uvz[work_labels == 2]
    snapped_poly, edge_logs = _refine_footprint_edges_to_walls(ortho_poly, wall_uvz)
    snapped_poly, squared_bevels = _square_spurious_corner_bevels(snapped_poly, wall_uvz)
    if squared_bevels > 0:
        snapped_poly, edge_logs = _refine_footprint_edges_to_walls(snapped_poly, wall_uvz)

    # Assemble structured floor_levels (Level 0 = snapped primary footprint, Level k>=1 = upper mezzanines)
    floor_levels: list[FloorLevel] = [
        FloorLevel(
            level_id=0,
            z_top=floor_z,
            polygon_uv=snapped_poly,
            point_count=int(floor_peaks[0][1]),
            is_primary=True,
        )
    ]
    for fid, (zf, cnt) in enumerate(floor_peaks[1:], start=1):
        band = floor_pts[np.abs(floor_pts[:, 2] - zf) <= 0.18, :2]
        upoly = _extract_level_polygon_uv(band, cell=0.12)
        if upoly is not None:
            clipped = upoly.intersection(snapped_poly.buffer(0.05)).buffer(0)
            if isinstance(clipped, MultiPolygon):
                clipped = max(clipped.geoms, key=lambda g: g.area)
            if isinstance(clipped, Polygon) and not clipped.is_empty and clipped.area >= 2.0:
                floor_levels.append(
                    FloorLevel(
                        level_id=fid,
                        z_top=float(zf),
                        polygon_uv=orient(Polygon(clipped.exterior.coords).buffer(0), sign=1.0),
                        point_count=int(cnt),
                        is_primary=False,
                    )
                )

    return FloorCeilingResult(
        frame=frame,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        inner_footprint_uv=snapped_poly,
        floor_levels=floor_levels,
        ceiling_levels=ceiling_levels,
        audit={
            "principal_angle_deg": round(float(np.degrees(frame.angle_rad)), 3),
            "floor_z_local": round(floor_z, 4),
            "ceiling_z_local": round(ceiling_z, 4) if ceiling_z is not None else None,
            "clear_height_m": round(ceiling_z - floor_z, 4) if ceiling_z is not None else None,
            "floor_levels": [
                {
                    "level_id": lv.level_id,
                    "z_top_local": round(lv.z_top, 4),
                    "area_m2": round(float(lv.polygon_uv.area), 3),
                    "point_count": lv.point_count,
                    "is_primary": lv.is_primary,
                }
                for lv in floor_levels
            ],
            "ceiling_levels": [
                {
                    "level_id": lv.level_id,
                    "z_bottom_local": round(lv.z_bottom, 4),
                    "area_m2": round(float(lv.polygon_uv.area), 3),
                    "point_count": lv.point_count,
                    "is_primary": lv.is_primary,
                }
                for lv in ceiling_levels
            ],
            "raw_floor_area_m2": round(float(raw_poly.area), 3),
            "snapped_inner_area_m2": round(float(snapped_poly.area), 3),
            "footprint_vertex_count": len(list(snapped_poly.exterior.coords)) - 1,
            "squared_corner_bevels": squared_bevels,
            "edge_snapping": edge_logs,
        },
    )


def build_floor_and_ceiling_parts(
    fc: FloorCeilingResult,
    outer_shell_uv: Polygon,
    cfg: ArchitectureConfig,
    *,
    inner_uv: Polygon | None = None,
    upper_floor_cutouts: list[tuple[Polygon, float, float]] | None = None,
) -> list[MeshPart]:
    """Extrude watertight floor and ceiling slabs (1F primary + upper mezzanines + secondary ceilings)."""
    parts: list[MeshPart] = []
    inner_dom = (inner_uv if inner_uv is not None else fc.inner_footprint_uv).buffer(-0.004)
    cutouts = upper_floor_cutouts or []

    # 1. Primary 1F floor slab spanning outer_shell_uv, flush beneath exterior walls
    fv_uv, ft = extrude_polygon_solid(outer_shell_uv, fc.floor_z - cfg.floor_thickness_m, fc.floor_z)
    parts.append(
        MeshPart(
            part_id="floor_primary",
            class_id=1,
            role="floor",
            vertices=fc.frame.uvz_to_xyz(fv_uv),
            triangles=ft,
            evidence_kind="measured",
            metadata={
                "level_id": 0,
                "z_top_local": round(fc.floor_z, 4),
                "thickness_m": cfg.floor_thickness_m,
                "outer_area_m2": round(float(outer_shell_uv.area), 3),
            },
        )
    )

    # 2. Upper mezzanine floor slabs (clipped strictly inside inner_uv and minus stairwells / tall blockers)
    upper_slab_bands: list[tuple[Polygon, float, float]] = []
    for lv in fc.floor_levels:
        if lv.is_primary:
            continue
        z_bot = lv.z_top - cfg.floor_thickness_m
        z_top = lv.z_top
        slab_poly = lv.polygon_uv.intersection(inner_dom).buffer(0)
        rel_cuts = [
            cpoly.buffer(0.004)
            for cpoly, cz0, cz1 in cutouts
            if cz0 <= z_top + 0.05 and cz1 >= z_bot - 0.05
        ]
        if rel_cuts:
            slab_poly = slab_poly.difference(unary_union(rel_cuts)).buffer(0)
        geoms = (
            list(slab_poly.geoms)
            if isinstance(slab_poly, MultiPolygon)
            else ([slab_poly] if isinstance(slab_poly, Polygon) else [])
        )
        sub_idx = 0
        for g in geoms:
            if g.is_empty or g.area < 0.50:
                continue
            g_clean = orient(g.buffer(0), sign=1.0)
            if isinstance(g_clean, MultiPolygon):
                g_clean = max(g_clean.geoms, key=lambda x: x.area)
            try:
                uv_v, uv_t = extrude_polygon_solid(g_clean, z_bot, z_top)
            except ValueError:
                continue
            pid = f"floor_level_{lv.level_id}" if sub_idx == 0 else f"floor_level_{lv.level_id}_{sub_idx}"
            parts.append(
                MeshPart(
                    part_id=pid,
                    class_id=1,
                    role="floor_upper",
                    vertices=fc.frame.uvz_to_xyz(uv_v),
                    triangles=uv_t,
                    evidence_kind="measured",
                    metadata={
                        "level_id": lv.level_id,
                        "z_top_local": round(z_top, 4),
                        "z_bottom_local": round(z_bot, 4),
                        "thickness_m": cfg.floor_thickness_m,
                        "area_m2": round(float(g_clean.area), 3),
                    },
                )
            )
            upper_slab_bands.append((g_clean, z_bot, z_top))
            sub_idx += 1

    # 3. Primary roof ceiling slab spanning outer_shell_uv, flush atop exterior walls
    if fc.ceiling_z is not None:
        cv_uv, ct = extrude_polygon_solid(outer_shell_uv, fc.ceiling_z, fc.ceiling_z + cfg.ceiling_thickness_m)
        parts.append(
            MeshPart(
                part_id="ceiling_primary",
                class_id=11,
                role="ceiling",
                vertices=fc.frame.uvz_to_xyz(cv_uv),
                triangles=ct,
                evidence_kind="measured",
                metadata={
                    "z_bottom_local": round(fc.ceiling_z, 4),
                    "thickness_m": cfg.ceiling_thickness_m,
                    "outer_area_m2": round(float(outer_shell_uv.area), 3),
                },
            )
        )

    # 4. Secondary lower-wing ceiling slabs (strictly inside inner_uv, disjoint from primary ceiling & upper floors)
    for clv in fc.ceiling_levels:
        if clv.is_primary or fc.ceiling_z is None:
            continue
        zc_bot = clv.z_bottom
        zc_top = clv.z_bottom + cfg.ceiling_thickness_m
        # Skip if overlapping primary ceiling vertically
        if zc_top >= fc.ceiling_z - 0.02 and zc_bot <= fc.ceiling_z + cfg.ceiling_thickness_m + 0.02:
            continue
        cpoly = clv.polygon_uv.intersection(inner_dom).buffer(0)
        # Subtract any upper floor slab or cutout overlapping vertically
        block_z = [
            p.buffer(0.006)
            for p, zb, zt in (*upper_slab_bands, *cutouts)
            if zb <= zc_top + 0.02 and zt >= zc_bot - 0.02
        ]
        if block_z:
            cpoly = cpoly.difference(unary_union(block_z)).buffer(0)
        geoms = (
            list(cpoly.geoms)
            if isinstance(cpoly, MultiPolygon)
            else ([cpoly] if isinstance(cpoly, Polygon) else [])
        )
        sub_idx = 0
        for g in geoms:
            if g.is_empty or g.area < 0.80:
                continue
            g_clean = orient(g.buffer(0), sign=1.0)
            if isinstance(g_clean, MultiPolygon):
                g_clean = max(g_clean.geoms, key=lambda x: x.area)
            try:
                cv_uv, ct = extrude_polygon_solid(g_clean, zc_bot, zc_top)
            except ValueError:
                continue
            pid = f"ceiling_level_{clv.level_id}" if sub_idx == 0 else f"ceiling_level_{clv.level_id}_{sub_idx}"
            parts.append(
                MeshPart(
                    part_id=pid,
                    class_id=11,
                    role="ceiling_secondary",
                    vertices=fc.frame.uvz_to_xyz(cv_uv),
                    triangles=ct,
                    evidence_kind="measured",
                    metadata={
                        "level_id": clv.level_id,
                        "z_bottom_local": round(zc_bot, 4),
                        "z_top_local": round(zc_top, 4),
                        "thickness_m": cfg.ceiling_thickness_m,
                        "area_m2": round(float(g_clean.area), 3),
                    },
                )
            )
            sub_idx += 1

    return parts
