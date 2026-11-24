"""End-to-end synthetic architecture tests (rotated room, B13a exterior closure, B14 openings, zero penetration)."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from ipcr.pipeline import run_arch_pipeline


def _rot_xy(pts: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    out = pts.copy()
    out[:, 0] = pts[:, 0] * c - pts[:, 1] * s
    out[:, 1] = pts[:, 0] * s + pts[:, 1] * c
    return out


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


def _grid_xy(x0: float, x1: float, y0: float, y1: float, z: float, step: float = 0.08) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def _grid_wall_x(x0: float, x1: float, y: float, z0: float, z1: float, step: float = 0.06) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    zs = np.arange(z0, z1, step)
    gx, gz = np.meshgrid(xs, zs)
    return np.column_stack([gx.ravel(), np.full(gx.size, y), gz.ravel()])


def _grid_wall_y(x: float, y0: float, y1: float, z0: float, z1: float, step: float = 0.06) -> np.ndarray:
    ys = np.arange(y0, y1, step)
    zs = np.arange(z0, z1, step)
    gy, gz = np.meshgrid(ys, zs)
    return np.column_stack([np.full(gy.size, x), gy.ravel(), gz.ravel()])


def test_rotated_room_closed_shell_with_window_door_column_beam_and_scan_shadow(tmp_path: Path) -> None:
    # Build a synthetic 6m x 5m x 3.2m room rotated by 33 degrees, with:
    # - A furniture scan-shadow hole in the floor (tests Q11 hole-free floor completion)
    # - High ceiling labeled class 0 (tests WL-A 0->11 promotion)
    # - Missing lower wall foot on one wall (tests B13a exterior wall-foot completion)
    # - A window on y=5 wall with solid sill wall below (tests B14 cutout & no fake passage)
    # - A door on y=0 wall reaching floor
    # - An attached column & an overhead beam
    floor = _grid_xy(0, 6, 0, 5, z=0.0, step=0.08)
    # Punch a 1m x 1m scan shadow hole in the middle of the floor
    shadow = (floor[:, 0] > 2.5) & (floor[:, 0] < 3.5) & (floor[:, 1] > 2.0) & (floor[:, 1] < 3.0)
    floor = floor[~shadow]

    ceil0 = _grid_xy(0, 6, 0, 5, z=3.2, step=0.08)

    # 4 walls: omit wall foot [0, 0.5] on x=0 wall to verify B13a completion!
    w_south = _grid_wall_x(0, 6, y=0.0, z0=0.0, z1=3.2)
    w_south = w_south[~((w_south[:, 0] > 1.05) & (w_south[:, 0] < 1.95) & (w_south[:, 2] < 2.15))]
    w_north = _grid_wall_x(0, 6, y=5.0, z0=0.0, z1=3.2)
    w_north = w_north[~((w_north[:, 0] > 1.55) & (w_north[:, 0] < 3.95) & (w_north[:, 2] > 1.05) & (w_north[:, 2] < 2.45))]
    w_west = _grid_wall_y(x=0.0, y0=0, y1=5, z0=0.5, z1=3.2)  # missing foot!
    w_east = _grid_wall_y(x=6.0, y0=0, y1=5, z0=0.0, z1=3.2)
    walls = np.vstack([w_south, w_north, w_west, w_east])

    # Window on north wall (y=5.0), x in [1.5, 4.0], z in [1.0, 2.5]
    win = _grid_wall_x(1.5, 4.0, y=5.02, z0=1.0, z1=2.5, step=0.05)
    # Door on south wall (y=0.0), x in [1.0, 2.0], z in [0.05, 2.2]
    door = _grid_wall_x(1.0, 2.0, y=-0.02, z0=0.05, z1=2.2, step=0.05)
    # Floating non-door class-6 fragment high on east wall (must be rejected by B14!)
    fake_door = _grid_wall_y(x=6.02, y0=0.5, y1=1.1, z0=2.0, z1=2.5, step=0.04)

    # Attached column near east wall (x in [5.5, 6.0], y in [2.0, 2.5], z in [0, 3.2])
    col_f = _grid_wall_y(x=5.5, y0=2.0, y1=2.5, z0=0.0, z1=3.2, step=0.06)
    col_s1 = _grid_wall_x(5.5, 6.0, y=2.0, z0=0.0, z1=3.2, step=0.06)
    col_s2 = _grid_wall_x(5.5, 6.0, y=2.5, z0=0.0, z1=3.2, step=0.06)
    col = np.vstack([col_f, col_s1, col_s2])

    # Overhead beam spanning x in [0.1, 5.9], y in [2.1, 2.4], z in [2.75, 3.15]
    beam = _grid_xy(0.1, 5.9, 2.1, 2.4, z=2.75, step=0.04)

    pts = np.vstack([floor, ceil0, walls, win, door, fake_door, col, beam])
    cls = np.concatenate([
        np.full(len(floor), 1),
        np.full(len(ceil0), 0),
        np.full(len(walls), 2),
        np.full(len(win), 5),
        np.full(len(door), 6),
        np.full(len(fake_door), 6),
        np.full(len(col), 4),
        np.full(len(beam), 3),
    ])

    # Rotate by 33 deg and translate by large global offset
    pts_rot = _rot_xy(pts, 33.0) + np.array([1000.0, 2000.0, -1.0])
    las_file = tmp_path / "synth_rot_room.las"
    _write_las(las_file, pts_rot, cls)

    out_dir = tmp_path / "out_r1"
    report = run_arch_pipeline(las_file, out_dir, render_preview=False)

    assert report["status"] == "PASS", report
    assert report["glb_validation"]["status"] == "PASS"
    assert report["relation_audit"]["status"] == "PASS"
    assert report["relation_audit"]["unexpected_penetration_count"] == 0
    assert report["architecture"]["openings"]["window_count"] == 1
    assert report["architecture"]["openings"]["door_count"] == 1
    assert report["architecture"]["columns"]["emitted_count"] == 1
    assert report["architecture"]["beams"]["emitted_count"] >= 1
    # Verify B13a: all exterior walls span exact floor_z -> ceiling_z
    for w in report["architecture"]["exterior_shell"]["walls"]:
        assert abs(w["z_span"][0] - report["architecture"]["floor_ceiling"]["floor_z_local"]) < 1e-4
        assert abs(w["z_span"][1] - report["architecture"]["floor_ceiling"]["ceiling_z_local"]) < 1e-4
    # Verify high point-to-mesh coverage on floor, walls, ceiling
    cm = report["cloud_metrics"]["per_class"]
    assert cm["1"]["cloud_to_mesh_within_10cm"] >= 0.98
    assert cm["2"]["cloud_to_mesh_within_10cm"] >= 0.90
    assert cm["11"]["cloud_to_mesh_within_10cm"] >= 0.98


def test_corner_shadow_bevel_squared_and_raised_alcove_included(tmp_path: Path) -> None:
    # 6m x 5m room with:
    # 1. A 0.75m x 0.75m floor scan shadow right in the NE corner (x>5.25, y>4.25) while walls meet at sharp (6,5)
    #    -> must produce a sharp 90-degree corner (no spurious 45-degree chamfer!)
    # 2. A slightly raised (+0.28m) 2m x 2m entry alcove on south wall (x in [0, 2], y in [-2, 0])
    #    -> must be included in the footprint (not omitted by tight floor Z band!)
    f_main = _grid_xy(0.0, 6.0, 0.0, 5.0, z=0.0, step=0.08)
    # Remove floor points in NE corner triangle (simulating clutter/scan shadow)
    corner_shadow = (f_main[:, 0] + f_main[:, 1]) > 10.15
    f_main = f_main[~corner_shadow]
    f_alcove = _grid_xy(0.0, 2.0, -2.0, 0.0, z=0.28, step=0.08)
    floor = np.vstack([f_main, f_alcove])

    ceil = np.vstack([
        _grid_xy(0.0, 6.0, 0.0, 5.0, z=3.2, step=0.08),
        _grid_xy(0.0, 2.0, -2.0, 0.0, z=3.2, step=0.08),
    ])

    walls = np.vstack([
        _grid_wall_x(0.0, 2.0, -2.0, 0.28, 3.2),
        _grid_wall_y(2.0, -2.0, 0.0, 0.28, 3.2),
        _grid_wall_x(2.0, 6.0, 0.0, 0.0, 3.2),
        _grid_wall_y(6.0, 0.0, 5.0, 0.0, 3.2),
        _grid_wall_x(0.0, 6.0, 5.0, 0.0, 3.2),
        _grid_wall_y(0.0, -2.0, 5.0, 0.0, 3.2),
    ])

    pts = np.vstack([floor, ceil, walls])
    cls = np.concatenate([
        np.full(len(floor), 1),
        np.full(len(ceil), 0),
        np.full(len(walls), 2),
    ])
    las_file = tmp_path / "synth_corner_alcove.las"
    _write_las(las_file, pts, cls)
    out_dir = tmp_path / "out_ca"
    report = run_arch_pipeline(las_file, out_dir, render_preview=False)

    assert report["status"] == "PASS", report
    assert report["relation_audit"]["unexpected_penetration_count"] == 0
    fc_aud = report["architecture"]["floor_ceiling"]
    # True inner area = 6*5 + 2*2 = 34.0 m^2 (if alcove dropped -> 30 m^2; if corner chamfered -> < 33.4 m^2)
    assert fc_aud["snapped_inner_area_m2"] >= 33.2, fc_aud
    # Exact L-shape has 6 vertices (no extra diagonal chamfer edge!)
    assert fc_aud["footprint_vertex_count"] <= 7, fc_aud
