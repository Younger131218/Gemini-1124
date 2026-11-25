"""Closed manifold 3D solid builders (boxes and 2D-to-3D extrusions with holes)."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import orient

try:
    from shapely import constrained_delaunay_triangles as _cdt
except ImportError:  # pragma: no cover
    _cdt = None


def signed_mesh_volume(vertices: np.ndarray, triangles: np.ndarray) -> float:
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    if len(t) == 0 or len(v) == 0:
        return 0.0
    # Shift by v[0] (exact translation invariance!) to eliminate catastrophic cancellation at large UTM world coords!
    v_rel = v - v[0]
    v0 = v_rel[t[:, 0]]
    v1 = v_rel[t[:, 1]]
    v2 = v_rel[t[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", v0, np.cross(v1, v2))) / 6.0)


def make_oriented_box(
    center_xyz: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    axis_w: np.ndarray,
    size_uvw: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed manifold oriented box with guaranteed outward CCW normals."""
    c = np.asarray(center_xyz, dtype=np.float64).reshape(3)
    u = np.asarray(axis_u, dtype=np.float64).reshape(3)
    v = np.asarray(axis_v, dtype=np.float64).reshape(3)
    w = np.asarray(axis_w, dtype=np.float64).reshape(3)
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    w = w / max(float(np.linalg.norm(w)), 1e-12)
    if float(np.dot(np.cross(u, v), w)) < 0:
        w = -w

    hu, hv, hw = 0.5 * float(size_uvw[0]), 0.5 * float(size_uvw[1]), 0.5 * float(size_uvw[2])
    if hu <= 0 or hv <= 0 or hw <= 0:
        raise ValueError(f"Box extents must be positive, got {size_uvw}")

    signs = np.array(
        [
            [-1, -1, -1],
            [+1, -1, -1],
            [+1, +1, -1],
            [-1, +1, -1],
            [-1, -1, +1],
            [+1, -1, +1],
            [+1, +1, +1],
            [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    basis = np.vstack([u * hu, v * hv, w * hw])  # (3, 3)
    verts = c[None, :] + signs @ basis
    tris = np.array(
        [
            [0, 2, 1], [0, 3, 2],  # bottom (-w)
            [4, 5, 6], [4, 6, 7],  # top (+w)
            [0, 1, 5], [0, 5, 4],  # -v
            [1, 2, 6], [1, 6, 5],  # +u
            [2, 3, 7], [2, 7, 6],  # +v
            [3, 0, 4], [3, 4, 7],  # -u
        ],
        dtype=np.uint32,
    )
    if signed_mesh_volume(verts, tris) < 0:
        tris = tris[:, [0, 2, 1]]
    return verts, tris


def make_axis_aligned_box(
    xyz_min: tuple[float, float, float] | np.ndarray,
    xyz_max: tuple[float, float, float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mn = np.asarray(xyz_min, dtype=np.float64).reshape(3)
    mx = np.asarray(xyz_max, dtype=np.float64).reshape(3)
    center = 0.5 * (mn + mx)
    size = mx - mn
    return make_oriented_box(
        center,
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        (float(size[0]), float(size[1]), float(size[2])),
    )


def _orthonormal_basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.asarray(axis, dtype=np.float64).reshape(3)
    wn = float(np.linalg.norm(w))
    if wn < 1e-12:
        raise ValueError("Cylinder/cone axis cannot be zero")
    w = w / wn
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64) if abs(w[0]) < 0.85 else np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(w, ref)
    u /= max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(w, u)
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return u, v, w


def make_oriented_cylinder(
    p0_xyz: np.ndarray,
    p1_xyz: np.ndarray,
    radius: float,
    *,
    segments: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed manifold cylinder along ANY 3D axis (vertical, horizontal, or oblique; B18)."""
    p0 = np.asarray(p0_xyz, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1_xyz, dtype=np.float64).reshape(3)
    r = float(radius)
    seg = max(6, int(segments))
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 1e-6 or r <= 1e-6:
        raise ValueError(f"Invalid cylinder length={length} radius={r}")
    u, v, _w = _orthonormal_basis_from_axis(axis)

    angles = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    ring_offsets = (cos_a[:, None] * u[None, :] + sin_a[:, None] * v[None, :]) * r

    # Layout: [0..seg-1] bottom ring, [seg..2*seg-1] top ring, [2*seg] bottom center, [2*seg+1] top center
    verts = np.vstack([p0[None, :] + ring_offsets, p1[None, :] + ring_offsets, p0[None, :], p1[None, :]])
    c_bot = 2 * seg
    c_top = 2 * seg + 1
    tris: list[tuple[int, int, int]] = []
    for i in range(seg):
        j = (i + 1) % seg
        # Bottom cap (outward -w)
        tris.append((c_bot, j, i))
        # Top cap (outward +w)
        tris.append((c_top, seg + i, seg + j))
        # Side quad (2 tris, outward radial)
        tris.append((i, j, seg + j))
        tris.append((i, seg + j, seg + i))
    tri_arr = np.asarray(tris, dtype=np.uint32)
    if signed_mesh_volume(verts, tri_arr) < 0:
        tri_arr = tri_arr[:, [0, 2, 1]]
    return verts, tri_arr


def make_oriented_cone_or_pyramid(
    base_center_xyz: np.ndarray,
    apex_xyz: np.ndarray,
    base_radius: float,
    *,
    sides: int = 3,
    phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed manifold triangular pyramid (sides=3), square pyramid (sides=4), or cone (sides>=12) (B18)."""
    bc = np.asarray(base_center_xyz, dtype=np.float64).reshape(3)
    ap = np.asarray(apex_xyz, dtype=np.float64).reshape(3)
    r = float(base_radius)
    n = max(3, int(sides))
    axis = ap - bc
    height = float(np.linalg.norm(axis))
    if height <= 1e-6 or r <= 1e-6:
        raise ValueError(f"Invalid pyramid/cone height={height} radius={r}")
    u, v, _w = _orthonormal_basis_from_axis(axis)

    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + float(phase_rad)
    ring_offsets = (np.cos(angles)[:, None] * u[None, :] + np.sin(angles)[:, None] * v[None, :]) * r

    # Layout: [0..n-1] base ring, [n] base center, [n+1] apex
    verts = np.vstack([bc[None, :] + ring_offsets, bc[None, :], ap[None, :]])
    c_bot = n
    apex_i = n + 1
    tris: list[tuple[int, int, int]] = []
    for i in range(n):
        j = (i + 1) % n
        tris.append((c_bot, j, i))
        tris.append((i, j, apex_i))
    tri_arr = np.asarray(tris, dtype=np.uint32)
    if signed_mesh_volume(verts, tri_arr) < 0:
        tri_arr = tri_arr[:, [0, 2, 1]]
    return verts, tri_arr


def _clean_ring(coords: list[tuple[float, float]], eps: float = 1e-6) -> list[tuple[float, float]]:
    pts = [(float(x), float(y)) for x, y in coords[:-1]]  # drop closing duplicate
    out: list[tuple[float, float]] = []
    for p in pts:
        if not out or (abs(p[0] - out[-1][0]) > eps or abs(p[1] - out[-1][1]) > eps):
            out.append(p)
    if len(out) >= 2 and abs(out[0][0] - out[-1][0]) <= eps and abs(out[0][1] - out[-1][1]) <= eps:
        out.pop()
    return out


def extrude_polygon_solid(
    polygon: Polygon,
    z_min: float,
    z_max: float,
    *,
    snap_eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a 2D Shapely Polygon (convex, concave, or with holes) into a watertight 3D solid."""
    if z_max <= z_min:
        raise ValueError(f"z_max ({z_max}) must exceed z_min ({z_min})")
    poly = polygon.buffer(0) if not polygon.is_valid else polygon
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or poly.area < 1e-8:
        raise ValueError("Cannot extrude empty or near-zero area polygon")

    # Ensure exterior is CCW and interior holes are CW
    poly = orient(poly, sign=1.0)

    ext_ring = _clean_ring(list(poly.exterior.coords), eps=snap_eps)
    int_rings = [_clean_ring(list(r.coords), eps=snap_eps) for r in poly.interiors]
    int_rings = [r for r in int_rings if len(r) >= 3]
    if len(ext_ring) < 3:
        raise ValueError("Exterior ring has fewer than 3 vertices after cleaning")

    clean_poly = orient(Polygon(ext_ring, int_rings).buffer(0), sign=1.0)
    if isinstance(clean_poly, MultiPolygon):
        clean_poly = max(clean_poly.geoms, key=lambda g: g.area)
    clean_poly = orient(clean_poly.simplify(1e-4, preserve_topology=True).buffer(0), sign=1.0)
    if isinstance(clean_poly, MultiPolygon):
        clean_poly = max(clean_poly.geoms, key=lambda g: g.area)

    # Map every unique 2D vertex (quantized) to a bottom index i and top index i + N
    vert_map: dict[tuple[int, int], int] = {}
    xy_list: list[tuple[float, float]] = []

    def get_vid(x: float, y: float) -> int:
        key = (int(round(x / snap_eps)), int(round(y / snap_eps)))
        vid = vert_map.get(key)
        if vid is None:
            vid = len(xy_list)
            vert_map[key] = vid
            xy_list.append((float(x), float(y)))
        return vid

    # Triangulate the 2D face using GEOS constrained Delaunay
    if _cdt is None:
        raise RuntimeError("shapely.constrained_delaunay_triangles is required (shapely>=2.1)")
    tri_collection = _cdt(clean_poly)

    cap_tris_2d: list[tuple[int, int, int]] = []
    for geom in getattr(tri_collection, "geoms", [tri_collection]):
        if not isinstance(geom, Polygon) or geom.area < 1e-10:
            continue
        rcoords = list(geom.exterior.coords)[:3]
        if len(rcoords) < 3:
            continue
        i0 = get_vid(rcoords[0][0], rcoords[0][1])
        i1 = get_vid(rcoords[1][0], rcoords[1][1])
        i2 = get_vid(rcoords[2][0], rcoords[2][1])
        if len({i0, i1, i2}) < 3:
            continue
        p0, p1, p2 = xy_list[i0], xy_list[i1], xy_list[i2]
        cross2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
        if abs(cross2) < 1e-11:
            continue
        if cross2 < 0:
            i1, i2 = i2, i1
        cap_tris_2d.append((i0, i1, i2))

    if not cap_tris_2d:
        raise ValueError("2D triangulation produced zero triangles")

    # Derive side-wall edges DIRECTLY from the unmatched boundary half-edges of cap_tris_2d!
    # By graph duality, this mathematically guarantees 0 T-junctions and 0 non-manifold edges.
    from collections import Counter

    half_edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in cap_tris_2d:
        half_edges[(a, b)] += 1
        half_edges[(b, c)] += 1
        half_edges[(c, a)] += 1

    boundary_edges: list[tuple[int, int]] = []
    for (a, b), cnt in half_edges.items():
        net = cnt - half_edges.get((b, a), 0)
        for _ in range(max(0, net)):
            boundary_edges.append((a, b))

    n = len(xy_list)
    xy = np.asarray(xy_list, dtype=np.float64)
    verts = np.empty((2 * n, 3), dtype=np.float64)
    verts[:n, :2] = xy
    verts[:n, 2] = float(z_min)
    verts[n:, :2] = xy
    verts[n:, 2] = float(z_max)

    tris: list[tuple[int, int, int]] = []
    # Top cap (+Z normal -> CCW)
    for a, b, c in cap_tris_2d:
        tris.append((a + n, b + n, c + n))
    # Bottom cap (-Z normal -> CW)
    for a, b, c in cap_tris_2d:
        tris.append((a, c, b))
    # Side walls along every boundary half-edge (a -> b) of the CCW cap
    for a, b in boundary_edges:
        tris.append((a, b, b + n))
        tris.append((a, b + n, a + n))

    tri_arr = np.asarray(tris, dtype=np.uint32)
    if signed_mesh_volume(verts, tri_arr) < 0:
        tri_arr = tri_arr[:, [0, 2, 1]]
    return verts, tri_arr
