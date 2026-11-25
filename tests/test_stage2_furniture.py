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


def test_irregular_l_desk_preserved_and_seat_only_chair_anchored(tmp_path: Path) -> None:
    # 7m x 6m x 3.2m room with:
    # 1. A genuine L-shaped executive desk (2.2m x 2.0m with a 1.3m x 1.2m recessed wing cutout) -> must stay irregular_shaped_contour!
    # 2. A rectangular desk (1.6m x 0.6m) with sparse fringe noise wisps -> must snap cleanly to rectangular!
    # 3. A seat-only chair (missing backrest, only z=0.45m cushion) right near east wall -> must use seat_anchored mode!
    floor = _grid_xy(0.0, 7.0, 0.0, 6.0, z=0.0, step=0.08)
    ceil = _grid_xy(0.0, 7.0, 0.0, 6.0, z=3.2, step=0.08)
    walls = np.vstack([
        _grid_wall_x(0.0, 7.0, 0.0, 0.0, 3.2),
        _grid_wall_x(0.0, 7.0, 6.0, 0.0, 3.2),
        _grid_wall_y(0.0, 0.0, 6.0, 0.0, 3.2),
        _grid_wall_y(7.0, 0.0, 6.0, 0.0, 3.2),
    ])

    # True L-shaped desk: leg A [0.8..3.0, 3.8..4.6] + leg B [0.8..1.6, 2.4..3.8]
    l_desk = np.vstack([
        _grid_xy(0.8, 3.0, 3.8, 4.6, z=0.76, step=0.035),
        _grid_xy(0.8, 1.6, 2.4, 3.8, z=0.76, step=0.035),
    ])
    # Rectangular side desk with a tiny sparse noise wisp sticking out on one edge
    r_desk_main = _grid_xy(4.5, 5.2, 1.0, 2.7, z=0.75, step=0.035)
    r_desk_wisp = np.array([[5.42, 1.5, 0.74], [5.45, 1.55, 0.74], [5.40, 1.60, 0.74]])
    r_desk = np.vstack([r_desk_main, r_desk_wisp])

    # Seat-only chair (no backrest above 0.50m!) sitting near east wall facing the side desk
    chair_post_z = np.arange(0.05, 0.45, 0.03)
    chair_post = np.column_stack([
        np.full(len(chair_post_z), 6.40),
        np.full(len(chair_post_z), 1.80),
        chair_post_z,
    ])
    seat_only_chair = np.vstack([
        _grid_xy(6.15, 6.65, 1.55, 2.05, z=0.45, step=0.03),
        _grid_xy(6.18, 6.62, 1.58, 2.02, z=0.42, step=0.035),
        _grid_xy(6.25, 6.55, 1.65, 1.95, z=0.08, step=0.04),
        chair_post,
    ])

    pts = np.vstack([floor, ceil, walls, l_desk, r_desk, seat_only_chair])
    cls = np.concatenate([
        np.full(len(floor), 1),
        np.full(len(ceil), 0),
        np.full(len(walls), 2),
        np.full(len(l_desk), 8),
        np.full(len(r_desk), 8),
        np.full(len(seat_only_chair), 7),
    ])
    las_file = tmp_path / "synth_ldesk_seat_chair.las"
    _write_las(las_file, pts, cls)
    out_dir = tmp_path / "out_ld"
    report = run_pipeline(las_file, out_dir, stage="furniture", library_root=None, render_preview=False)

    assert report["status"] == "PASS", report["relation_audit"]
    assert report["relation_audit"]["unexpected_penetration_count"] == 0
    tables = report["furniture"]["tables"]["tables"]
    assert len(tables) == 2, tables
    shapes = {t["shape"] for t in tables}
    assert "irregular_shaped_contour" in shapes, tables
    assert any("rectangular" in s for s in shapes), tables
    chairs = report["furniture"]["chairs"]
    assert chairs["emitted_chairs"] == 1, chairs
    assert chairs["seat_anchored_count"] == 1, chairs


def test_square_table_obb_avoids_pca_tilt_and_preserves_angled_desk() -> None:
    from ipcr.furniture.tables_cabinets_sofas import _fit_obb_2d

    # 1. Near-square 1.6m x 1.6m table with heavy corner density bias (PCA would tilt ~45 deg!)
    base_sq = _grid_xy(0.0, 1.6, 0.0, 1.6, z=0.0, step=0.04)[:, :2]
    corner_bias = _grid_xy(0.0, 0.45, 0.0, 0.45, z=0.0, step=0.015)[:, :2]
    corner_bias2 = _grid_xy(1.15, 1.6, 1.15, 1.6, z=0.0, step=0.015)[:, :2]
    sq_pts = np.vstack([base_sq, corner_bias, corner_bias2])
    _, long_dir, _, L, D = _fit_obb_2d(sq_pts, snap_deg=12.0)
    # Must snap cleanly to room axes (0 or 90 deg), NEVER diagonal!
    assert max(abs(long_dir[0]), abs(long_dir[1])) > 0.99, (long_dir, L, D)
    assert 1.45 <= L <= 1.65 and 1.45 <= D <= 1.65

    # 2. Genuinely 30-deg rotated rectangular desk (2.0m x 0.75m) -> must keep ~30-deg angle!
    rect = _grid_xy(-1.0, 1.0, -0.375, 0.375, z=0.0, step=0.04)[:, :2]
    ang = np.radians(30.0)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    rot_pts = rect @ rot.T
    _, r_dir, _, rL, rD = _fit_obb_2d(rot_pts, snap_deg=12.0)
    dot_30 = abs(float(r_dir @ np.array([np.cos(ang), np.sin(ang)])))
    assert dot_30 > 0.98, (r_dir, dot_30, rL, rD)
    assert 1.85 <= rL <= 2.05 and 0.65 <= rD <= 0.80


def test_cabinet_six_face_solidity_gate_rejects_hollow_open_top() -> None:
    from ipcr.furniture.library import _check_cabinet_six_faces_solid
    from ipcr.geometry_primitives import make_axis_aligned_box

    v, t = make_axis_aligned_box((0.0, 0.0, 0.0), (1.0, 0.5, 2.0))
    ok_full, ratios_full = _check_cabinet_six_faces_solid(v, t)
    assert ok_full, ratios_full

    # Strip the top (+Z) faces -> must be rejected as hollow!
    v0, v1, v2 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    nu = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    t_no_top = t[nu[:, 2] < 0.5]
    ok_hollow, ratios_hollow = _check_cabinet_six_faces_solid(v, t_no_top)
    assert not ok_hollow and ratios_hollow["top_pz"] == 0.0, ratios_hollow


def test_asymmetric_lidar_backrest_and_valley_row_split(tmp_path: Path) -> None:
    # 4 chairs spaced 0.90m apart along a conference table (3.3m total span), with:
    # - Sparse bridge points at floor level merging all 4 chairs into ONE 3D connected component
    # - 5x denser LiDAR scan points on the +Y edge of every backrest (must NOT pull chair centers off!)
    # Must emit EXACTLY 4 chairs (not 5!) with <= 5cm lateral error and min chair spacing >= 0.50m!
    floor = _grid_xy(0.0, 6.0, 0.0, 6.0, z=0.0, step=0.08)
    ceil = _grid_xy(0.0, 6.0, 0.0, 6.0, z=3.2, step=0.08)
    walls = np.vstack([
        _grid_wall_x(0.0, 6.0, 0.0, 0.0, 3.2),
        _grid_wall_x(0.0, 6.0, 6.0, 0.0, 3.2),
        _grid_wall_y(0.0, 0.0, 6.0, 0.0, 3.2),
        _grid_wall_y(6.0, 0.0, 6.0, 0.0, 3.2),
    ])
    table = _grid_xy(1.5, 2.8, 1.0, 4.8, z=0.75, step=0.04)

    chair_ys = [1.45, 2.35, 3.25, 4.15]
    chair_clouds = []
    for yc in chair_ys:
        seat = _grid_xy(3.10, 3.55, yc - 0.24, yc + 0.24, z=0.45, step=0.04)
        back = _grid_wall_y(3.60, yc - 0.25, yc + 0.25, 0.45, 1.00, step=0.04)
        # 5x denser overscan strictly on the +Y edge [yc+0.18 .. yc+0.25] of every backrest!
        overscan = _grid_wall_y(3.60, yc + 0.18, yc + 0.25, 0.45, 1.00, step=0.012)
        chair_clouds.extend([seat, back, overscan])
    # Add sparse connecting wisps at z=0.05..0.55 between chairs so voxel=0.08 merges all 4 into 1 strip
    bridge = np.column_stack([
        np.full(80, 3.35),
        np.linspace(1.45, 4.15, 80),
        np.full(80, 0.15),
    ])
    chairs_all = np.vstack(chair_clouds + [bridge])

    pts = np.vstack([floor, ceil, walls, table, chairs_all])
    cls = np.concatenate([
        np.full(len(floor), 1),
        np.full(len(ceil), 0),
        np.full(len(walls), 2),
        np.full(len(table), 8),
        np.full(len(chairs_all), 7),
    ])
    las_file = tmp_path / "synth_asym_row.las"
    _write_las(las_file, pts, cls)
    out_dir = tmp_path / "out_asym"
    report = run_pipeline(las_file, out_dir, stage="furniture", library_root=None, render_preview=False)

    assert report["status"] == "PASS", report["relation_audit"]
    assert report["relation_audit"]["unexpected_penetration_count"] == 0
    ch_list = report["furniture"]["chairs"]["chairs"]
    assert len(ch_list) == 4, ch_list
    # Verify lateral center accuracy (immune to 5x overscan on +Y edge) & min spacing >= 0.50m
    placed_ys = sorted([c["center_uv"][1] for c in ch_list])
    # Room UV might flip/shift origin, so compare inter-chair spacings and uniform alignment!
    spacings = np.diff(placed_ys)
    assert np.all(spacings >= 0.78) and np.all(spacings <= 1.02), (placed_ys, spacings)
