"""Stage 3 B18 regular clutter synthetic unit tests (boxes, vertical/oblique cylinders, triangular pyramid + semantic exclusions)."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from ipcr.geometry_primitives import make_oriented_cone_or_pyramid, make_oriented_cylinder, signed_mesh_volume
from ipcr.pipeline import run_pipeline
from ipcr.validator.glb_contract import _check_part_geometry
from ipcr.config import ValidationConfig


def _write_las_rgb(path: Path, pts: np.ndarray, cls: np.ndarray, rgb: np.ndarray | None = None) -> None:
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.offsets = np.array([0.0, 0.0, 0.0])
    hdr.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(hdr)
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    las.classification = cls.astype(np.uint8)
    if rgb is not None:
        las.red = (rgb[:, 0].astype(np.uint16) << 8)
        las.green = (rgb[:, 1].astype(np.uint16) << 8)
        las.blue = (rgb[:, 2].astype(np.uint16) << 8)
    las.write(str(path))


def _grid_xy(x0: float, x1: float, y0: float, y1: float, z: float, step: float = 0.08) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-6, step)
    ys = np.arange(y0, y1 + 1e-6, step)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, z)])


def _grid_wall_x(x0: float, x1: float, y: float, z0: float, z1: float, step: float = 0.08) -> np.ndarray:
    xs = np.arange(x0, x1 + 1e-6, step)
    zs = np.arange(z0, z1 + 1e-6, step)
    xx, zz = np.meshgrid(xs, zs)
    return np.column_stack([xx.ravel(), np.full(xx.size, y), zz.ravel()])


def _grid_wall_y(x: float, y0: float, y1: float, z0: float, z1: float, step: float = 0.08) -> np.ndarray:
    ys = np.arange(y0, y1 + 1e-6, step)
    zs = np.arange(z0, z1 + 1e-6, step)
    yy, zz = np.meshgrid(ys, zs)
    return np.column_stack([np.full(yy.size, x), yy.ravel(), zz.ravel()])


def _sample_box_surface(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float, step: float = 0.03) -> np.ndarray:
    return np.vstack([
        _grid_xy(x0, x1, y0, y1, z1, step=step),
        _grid_wall_x(x0, x1, y0, z0, z1, step=step),
        _grid_wall_x(x0, x1, y1, z0, z1, step=step),
        _grid_wall_y(x0, y0, y1, z0, z1, step=step),
        _grid_wall_y(x1, y0, y1, z0, z1, step=step),
    ])


def _sample_cylinder_surface(p0: np.ndarray, p1: np.ndarray, r: float, n_ang: int = 36, n_ax: int = 24) -> np.ndarray:
    axis = p1 - p0
    L = float(np.linalg.norm(axis))
    w = axis / L
    ref = np.array([1.0, 0.0, 0.0]) if abs(w[0]) < 0.85 else np.array([0.0, 1.0, 0.0])
    u = np.cross(w, ref)
    u /= np.linalg.norm(u)
    v = np.cross(w, u)
    angs = np.linspace(0.0, 2.0 * np.pi, n_ang, endpoint=False)
    ts = np.linspace(0.0, L, n_ax)
    pts = []
    for t in ts:
        c = p0 + w * t
        ring = c[None, :] + (np.cos(angs)[:, None] * u[None, :] + np.sin(angs)[:, None] * v[None, :]) * r
        pts.append(ring)
    return np.vstack(pts)


def _sample_tri_pyramid(bc: np.ndarray, ap: np.ndarray, r_base: float, n_z: int = 20, n_edge: int = 12) -> np.ndarray:
    pts = []
    h_vec = ap - bc
    angs = np.linspace(0.0, 2.0 * np.pi, 3, endpoint=False)
    base_v = np.column_stack([np.cos(angs) * r_base, np.sin(angs) * r_base, np.zeros(3)])
    for frac in np.linspace(0.0, 0.96, n_z):
        scale = 1.0 - frac
        z_off = h_vec * frac
        ring = bc[None, :] + z_off[None, :] + base_v * scale
        for i in range(3):
            a = ring[i]
            b = ring[(i + 1) % 3]
            for s in np.linspace(0.0, 1.0, n_edge, endpoint=False):
                pts.append(a * (1.0 - s) + b * s)
    return np.asarray(pts, dtype=np.float64)


def test_cylinder_and_pyramid_primitives_are_closed_positive_manifolds() -> None:
    vcfg = ValidationConfig()
    vc, tc = make_oriented_cylinder(np.array([1.0, 2.0, 0.0]), np.array([1.5, 2.3, 1.2]), 0.18, segments=20)
    chk_c = _check_part_geometry(vc, tc, vcfg)
    assert chk_c["passed"], chk_c
    assert signed_mesh_volume(vc, tc) > 0.1

    vp, tp = make_oriented_cone_or_pyramid(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.9]), 0.35, sides=3)
    chk_p = _check_part_geometry(vp, tp, vcfg)
    assert chk_p["passed"], chk_p
    assert signed_mesh_volume(vp, tp) > 0.04


def test_stage3_regular_clutter_primitives_and_semantic_exclusions(tmp_path: Path) -> None:
    # 8m x 7m x 3.2m room containing:
    # POSITIVE B18 targets (must ALL be built, 0 penetration):
    #   1. Floor box (0.50 x 0.40 x 0.60m)
    #   2. Tabletop monitor resting on a table (0.55 x 0.08 x 0.35m)
    #   3. Vertical cylinder (r=0.20m, h=0.90m)
    #   4. Oblique 35-deg slanted cylinder (r=0.12m, L=1.10m)
    #   5. Triangular pyramid (r_base=0.32m, h=0.75m)
    # NEGATIVE B18 targets (must ALL be rejected!):
    #   6. Green RGB potted plant (excluded_semantic_plant_rgb)
    #   7. Irregular porous pedestrian silhouette (excluded_semantic_pedestrian)
    #   8. Sparse 60-deg arc / single thin wall fragment (rejected)
    rng = np.random.default_rng(42)
    floor = _grid_xy(0.0, 8.0, 0.0, 7.0, z=0.0, step=0.08)
    ceil = _grid_xy(0.0, 8.0, 0.0, 7.0, z=3.2, step=0.08)
    walls = np.vstack([
        _grid_wall_x(0.0, 8.0, 0.0, 0.0, 3.2),
        _grid_wall_x(0.0, 8.0, 7.0, 0.0, 3.2),
        _grid_wall_y(0.0, 0.0, 7.0, 0.0, 3.2),
        _grid_wall_y(8.0, 0.0, 7.0, 0.0, 3.2),
    ])
    table = _grid_xy(1.0, 2.8, 1.0, 1.8, z=0.75, step=0.04)

    c_box = _sample_box_surface(1.0, 1.5, 4.5, 4.9, 0.0, 0.60, step=0.03)
    c_mon = _sample_box_surface(1.5, 2.05, 1.32, 1.40, 0.76, 1.12, step=0.025)
    c_vcyl = _sample_cylinder_surface(np.array([3.5, 5.2, 0.0]), np.array([3.5, 5.2, 0.90]), 0.20)
    c_ocyl = _sample_cylinder_surface(np.array([5.5, 1.5, 0.25]), np.array([6.2, 1.5, 1.15]), 0.12)
    c_pyr = _sample_tri_pyramid(np.array([6.2, 5.2, 0.0]), np.array([6.2, 5.2, 0.75]), 0.32)

    # Negative 1: Green potted plant (irregular canopy + green RGB)
    plant_pot = _sample_cylinder_surface(np.array([4.0, 2.5, 0.0]), np.array([4.0, 2.5, 0.35]), 0.14, n_ang=16, n_ax=8)
    plant_leaves = np.column_stack([
        rng.normal(4.0, 0.22, 600),
        rng.normal(2.5, 0.22, 600),
        rng.uniform(0.35, 1.45, 600),
    ])
    c_plant = np.vstack([plant_pot, plant_leaves])
    rgb_plant = np.vstack([
        np.tile(np.array([140, 120, 100], dtype=np.uint8), (len(plant_pot), 1)),
        np.tile(np.array([45, 155, 55], dtype=np.uint8), (len(plant_leaves), 1)),
    ])

    # Negative 2: Irregular floor-standing pedestrian blob (1.72m tall, non-planar torso/limbs)
    c_ped = np.column_stack([
        rng.normal(2.2, 0.15, 700),
        rng.normal(3.2, 0.10, 700),
        rng.uniform(0.0, 1.72, 700),
    ])

    # Assemble cloud + RGB
    neutral_blocks = [
        (floor, 1), (ceil, 0), (walls, 2), (table, 8),
        (c_box, 12), (c_mon, 12), (c_vcyl, 12), (c_ocyl, 12), (c_pyr, 12), (c_ped, 12),
    ]
    pts_list = [b[0] for b in neutral_blocks] + [c_plant]
    cls_list = [np.full(len(b[0]), b[1], dtype=np.uint8) for b in neutral_blocks] + [np.full(len(c_plant), 12, dtype=np.uint8)]
    rgb_list = [np.full((len(b[0]), 3), 130, dtype=np.uint8) for b in neutral_blocks] + [rgb_plant]

    las_file = tmp_path / "synth_stage3.las"
    _write_las_rgb(las_file, np.vstack(pts_list), np.concatenate(cls_list), np.vstack(rgb_list))
    out_dir = tmp_path / "out_s3"
    report = run_pipeline(las_file, out_dir, stage="full", library_root=None, render_preview=False)

    assert report["status"] == "PASS", report["relation_audit"]
    assert report["glb_validation"]["status"] == "PASS"
    assert report["relation_audit"]["unexpected_penetration_count"] == 0

    cl = report["clutter"]
    emitted_types = cl["by_primitive_type"]
    assert cl["emitted_clutter"] == 5, cl
    assert emitted_types.get("box", 0) == 2, emitted_types
    assert emitted_types.get("cylinder_vertical", 0) == 1, emitted_types
    assert emitted_types.get("cylinder_oblique", 0) == 1, emitted_types
    assert emitted_types.get("triangular_pyramid", 0) == 1, emitted_types

    rej_reasons = {r["reason"] for r in cl["rejected"]}
    assert "excluded_semantic_plant_rgb" in rej_reasons, cl["rejected"]
    assert any("pedestrian" in r or "irregular" in r for r in rej_reasons), cl["rejected"]
