"""Stage 2 synthetic tests: L-table/rect-table, under-table & wall cabinets, sofa vs false strip, multi-chair row & 0 penetrations."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from ipcr.pipeline import run_pipeline


def _write_las(path: Path, pts: np.ndarray, cls: np.ndarray) -> None:
    hdr = laspy.LasHeader(point_format=3, version="1.2")
    hdr.scales = np.array([0.001, 0.001, 0.001])
    hdr.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(hdr)
    las.x = pts[:, 0]
    las.y = pts[:, 1]
    las.z = pts[:, 2]
    las.classification = cls.astype(np.uint8)
    las.write(str(path))


def _grid_xy(x0: float, x1: float, y0: float, y1: float, z: float, step: float = 0.07) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def _grid_wall_x(x0: float, x1: float, y: float, z0: float, z1: float, step: float = 0.07) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    zs = np.arange(z0, z1, step)
    gx, gz = np.meshgrid(xs, zs)
    return np.column_stack([gx.ravel(), np.full(gx.size, y), gz.ravel()])


def _grid_wall_y(x: float, y0: float, y1: float, z0: float, z1: float, step: float = 0.07) -> np.ndarray:
    ys = np.arange(y0, y1, step)
    zs = np.arange(z0, z1, step)
    gy, gz = np.meshgrid(ys, zs)
    return np.column_stack([np.full(gy.size, x), gy.ravel(), gz.ravel()])


def _make_chair_cloud(cx: float, cy: float, back_side: str = "-x") -> np.ndarray:
    seat = _grid_xy(cx - 0.22, cx + 0.22, cy - 0.22, cy + 0.22, z=0.45, step=0.04)
    if back_side == "-x":
        back = _grid_wall_y(cx - 0.24, cy - 0.22, cy + 0.22, 0.45, 0.96, step=0.04)
    else:
        back = _grid_wall_y(cx + 0.24, cy - 0.22, cy + 0.22, 0.45, 0.96, step=0.04)
    return np.vstack([seat, back])


def test_stage2_furniture_zero_penetration_and_multi_chair_splitting(tmp_path: Path) -> None:
    # 8m x 7m x 3.2m conference room with:
    # - 1 rectangular conference table (x in [2.8, 4.4], y in [1.5, 4.5], z_top=0.76)
    # - 1 under-table pedestal cabinet underneath it (x in [3.2, 4.0], y in [2.7, 3.1], z in [0.0, 0.60])
    # - 1 freestanding tall cabinet near north wall (x in [0.4, 2.0], y in [6.2, 6.8], z in [0.0, 1.9])
    # - 3 touching chairs in a row along the west side of the conference table (must split into >=2-3 chairs!)
    # - 1 real sofa along east wall (x in [6.8, 7.8], y in [2.0, 4.2], z in [0.0, 0.85])
    # - 1 paper-thin false sofa strip along south wall (must be rejected!)
    floor = _grid_xy(0.0, 8.0, 0.0, 7.0, z=0.0, step=0.08)
    ceil = _grid_xy(0.0, 8.0, 0.0, 7.0, z=3.2, step=0.08)
    walls = np.vstack([
        _grid_wall_x(0.0, 8.0, 0.0, 0.0, 3.2),
        _grid_wall_x(0.0, 8.0, 7.0, 0.0, 3.2),
        _grid_wall_y(0.0, 0.0, 7.0, 0.0, 3.2),
        _grid_wall_y(8.0, 0.0, 7.0, 0.0, 3.2),
    ])

    table = np.vstack([
        _grid_xy(2.8, 4.4, 1.5, 4.5, z=0.76, step=0.04),
        _grid_wall_y(2.9, 1.6, 1.8, 0.0, 0.72, step=0.05),
    ])
    cab_under = np.vstack([
        _grid_xy(3.2, 4.0, 2.7, 3.1, z=0.58, step=0.03),
        _grid_wall_x(3.2, 4.0, 2.7, 0.02, 0.58, step=0.03),
        _grid_wall_x(3.2, 4.0, 3.1, 0.02, 0.58, step=0.03),
    ])
    cab_tall = np.vstack([
        _grid_xy(0.4, 2.0, 6.2, 6.8, z=1.90, step=0.05),
        _grid_wall_x(0.4, 2.0, 6.2, 0.02, 1.90, step=0.05),
        _grid_wall_y(0.4, 6.2, 6.8, 0.02, 1.90, step=0.05),
        _grid_wall_y(2.0, 6.2, 6.8, 0.02, 1.90, step=0.05),
    ])
    # 3 chairs along west side of table (x=2.45, y=1.9, 2.65, 3.40 — slightly touching at seat level!)
    chairs = np.vstack([
        _make_chair_cloud(2.45, 1.90, "-x"),
        _make_chair_cloud(2.45, 2.65, "-x"),
        _make_chair_cloud(2.45, 3.40, "-x"),
    ])
    # Real sofa along east wall + paper-thin false sofa strip along south wall
    sofa_real = np.vstack([
        _grid_xy(6.8, 7.8, 2.0, 4.2, z=0.42, step=0.05),
        _grid_wall_y(7.75, 2.0, 4.2, 0.02, 0.88, step=0.05),
        _grid_wall_y(6.82, 2.0, 4.2, 0.02, 0.42, step=0.05),
    ])
    sofa_fake = _grid_wall_x(1.0, 2.2, 0.04, 0.02, 0.40, step=0.03)  # 0m depth paper-thin strip!

    pts = np.vstack([floor, ceil, walls, table, cab_under, cab_tall, chairs, sofa_real, sofa_fake])
    cls = np.concatenate([
        np.full(len(floor), 1),
        np.full(len(ceil), 0),
        np.full(len(walls), 2),
        np.full(len(table), 8),
        np.full(len(cab_under), 9),
        np.full(len(cab_tall), 9),
        np.full(len(chairs), 7),
        np.full(len(sofa_real), 10),
        np.full(len(sofa_fake), 10),
    ])

    las_file = tmp_path / "synth_stage2.las"
    _write_las(las_file, pts, cls)
    out_dir = tmp_path / "out_s2"
    report = run_pipeline(las_file, out_dir, stage="furniture", library_root=None, render_preview=False)

    assert report["status"] == "PASS", report["relation_audit"]
    assert report["glb_validation"]["status"] == "PASS"
    assert report["relation_audit"]["unexpected_penetration_count"] == 0

    furn = report["furniture"]
    assert furn["tables"]["emitted_tables"] == 1
    assert furn["cabinets"]["emitted_cabinets"] == 2  # tall cabinet + under-table pedestal, 0 penetration!
    assert furn["sofas"]["emitted_sofas"] == 1  # real sofa emitted, thin false strip rejected!
    assert furn["chairs"]["emitted_chairs"] == 3, furn["chairs"]
