"""Synthetic unit tests for multi-storey slabs, local-storey beams/columns, B16 stairs/railings, and open doors."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np

from ipcr.pipeline import run_arch_pipeline


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


def test_multistorey_mezzanine_stairs_railing_and_local_beam(tmp_path: Path) -> None:
    # 10m x 8m x 6.0m two-storey hall:
    # - 1F floor at z=0.0, 2F mezzanine floor at z=3.2 over y in [0, 3.5]
    # - Main ceiling at z=6.0 (labeled class 0 -> promoted to 11 by WL-A)
    # - 1F beam under mezzanine at z in [2.65, 3.05] (must cap below mezzanine 3.08, NOT stretch to 6.0!)
    # - Supporting column under mezzanine at x in [4.5, 5.0], y in [3.0, 3.5], z in [0, 3.05]
    # - Straight stair flight rising from x=6.0 -> x=1.5 along y in [0.2, 1.5], z=0.0 -> 3.2
    # - Sloped stair railing along y=1.55 from x=6.0 -> x=1.5
    f1 = _grid_xy(0.0, 10.0, 0.0, 8.0, z=0.0, step=0.08)
    f2 = _grid_xy(0.0, 10.0, 0.0, 3.5, z=3.2, step=0.08)
    ceil = _grid_xy(0.0, 10.0, 0.0, 8.0, z=6.0, step=0.10)

    w_s = _grid_wall_x(0.0, 10.0, 0.0, 0.0, 6.0)
    w_n = _grid_wall_x(0.0, 10.0, 8.0, 0.0, 6.0)
    w_w = _grid_wall_y(0.0, 0.0, 8.0, 0.0, 6.0)
    w_e = _grid_wall_y(10.0, 0.0, 8.0, 0.0, 6.0)
    walls = np.vstack([w_s, w_n, w_w, w_e])

    # Supporting 1F column under mezzanine edge
    col = np.vstack([
        _grid_wall_x(4.5, 5.0, 3.5, 0.0, 3.05, step=0.06),
        _grid_wall_x(4.5, 5.0, 3.0, 0.0, 3.05, step=0.06),
        _grid_wall_y(4.5, 3.0, 3.5, 0.0, 3.05, step=0.06),
        _grid_wall_y(5.0, 3.0, 3.5, 0.0, 3.05, step=0.06),
    ])

    # 1F beam under mezzanine edge (y in [2.9, 3.4], x in [0.2, 9.8], z=2.65..3.05)
    beam = np.vstack([
        _grid_xy(0.2, 9.8, 2.9, 3.4, z=2.65, step=0.05),
        _grid_xy(0.2, 9.8, 2.9, 3.4, z=3.02, step=0.05),
    ])

    # Straight stair flight rising in -X direction from x=6.0 (z=0.05) to x=1.5 (z=3.18), y in [0.2, 1.4]
    stair_list = []
    for k in range(18):
        xk = 6.0 - k * (4.5 / 18.0)
        zk = (k + 1) * (3.18 / 18.0)
        stair_list.append(_grid_xy(xk - 0.25, xk, 0.2, 1.4, z=zk, step=0.05))
    stairs = np.vstack(stair_list)

    # Sloped stair railing along y=1.55 from x=6.0 -> x=1.5 (top z = stair_z + 0.95)
    rail_list = []
    for s_val in np.linspace(1.6, 5.9, 60):
        frac = (6.0 - s_val) / 4.5
        z_base = frac * 3.18
        for zv in np.linspace(z_base + 0.05, z_base + 0.95, 12):
            rail_list.append([s_val, 1.55, zv])
            rail_list.append([s_val, 1.57, zv])
    rail = np.asarray(rail_list, dtype=np.float64)

    pts = np.vstack([f1, f2, ceil, walls, col, beam, stairs, rail])
    cls = np.concatenate([
        np.full(len(f1), 1),
        np.full(len(f2), 1),
        np.full(len(ceil), 0),
        np.full(len(walls), 2),
        np.full(len(col), 4),
        np.full(len(beam), 3),
        np.full(len(stairs), 13),
        np.full(len(rail), 14),
    ])

    las_file = tmp_path / "synth_multistorey.las"
    _write_las(las_file, pts, cls)
    out_dir = tmp_path / "out_ms"
    report = run_arch_pipeline(las_file, out_dir, render_preview=False)

    assert report["status"] == "PASS", report["relation_audit"]
    assert report["glb_validation"]["status"] == "PASS"
    assert report["relation_audit"]["unexpected_penetration_count"] == 0

    # Verify multi-storey floor detection (1F + 2F mezzanine)
    flvs = report["architecture"]["floor_ceiling"]["floor_levels"]
    assert len(flvs) == 2, flvs
    assert abs((flvs[1]["z_top_local"] - flvs[0]["z_top_local"]) - 3.2) < 0.08

    # Verify 1F beams cap below mezzanine (~flvs[1].z_top - 0.12), NOT stretched to 6.0m roof!
    beams = report["architecture"]["beams"]["beams"]
    assert len(beams) >= 1
    for b in beams:
        assert b["z_span"][1] < flvs[1]["z_top_local"] - 0.05, f"Beam stretched above mezzanine: {b}"

    # Verify B16 stair flight & open handrail/baluster railing built!
    sr = report["architecture"]["stairs_railings"]
    assert sr["status"] == "PASS", sr
    assert sr["emitted_stair_flights"] >= 1
    assert sr["emitted_railing_groups"] >= 1
    assert sr["emitted_railing_parts"] >= 3  # 1 handrail + >=2 balusters


def test_open_swing_door_and_wall_void_door_rules(tmp_path: Path) -> None:
    # 6m x 5m x 3.0m room with:
    # 1. Open-swing door on east wall (x=6.0): header on wall y in [1.0, 2.0], z in [1.8, 2.3]
    #    + 90-degree inward-swung door leaf at y=2.0, x in [5.05, 5.98], z in [0.02, 2.1] (Option A closed leaf!)
    # 2. Occluded upper-only door on east wall y in [3.0, 4.0], z in [1.7, 2.3] with REAL wall void below
    # 3. Floating fake door speck on west wall (x=0.0) y in [2.0, 2.8], z in [1.8, 2.3] with SOLID wall below (must be rejected!)
    floor = _grid_xy(0.0, 6.0, 0.0, 5.0, z=0.0, step=0.08)
    ceil = _grid_xy(0.0, 6.0, 0.0, 5.0, z=3.0, step=0.08)

    w_s = _grid_wall_x(0.0, 6.0, 0.0, 0.0, 3.0)
    w_n = _grid_wall_x(0.0, 6.0, 5.0, 0.0, 3.0)
    w_w = _grid_wall_y(0.0, 0.0, 5.0, 0.0, 3.0)  # solid west wall (no void under fake door!)
    w_e = _grid_wall_y(6.0, 0.0, 5.0, 0.0, 3.0)
    # Cut real doorway voids in east wall for BOTH real doors!
    w_e = w_e[~((w_e[:, 1] > 1.05) & (w_e[:, 1] < 1.95) & (w_e[:, 2] < 2.25))]
    w_e = w_e[~((w_e[:, 1] > 3.05) & (w_e[:, 1] < 3.95) & (w_e[:, 2] < 2.25))]
    walls = np.vstack([w_s, w_n, w_w, w_e])

    # Door 1: header on east wall + separate 90-deg open leaf swinging inward at hinge y=2.0 (exercises Rule D1!)
    d1_hdr = _grid_wall_y(6.02, 1.0, 2.0, 1.8, 2.35, step=0.03)
    d1_leaf = _grid_wall_x(5.05, 5.82, 1.98, 0.02, 1.65, step=0.04)
    # Door 2: upper-only transom/header on east wall with wall void below (exercises Rule D2 accept!)
    d2_hdr = _grid_wall_y(6.02, 3.0, 4.0, 1.7, 2.35, step=0.03)
    # Fake Door 3: upper fragment on solid west wall (exercises Rule D2 reject!)
    d3_fake = _grid_wall_y(-0.02, 2.0, 2.8, 1.8, 2.35, step=0.03)

    pts = np.vstack([floor, ceil, walls, d1_hdr, d1_leaf, d2_hdr, d3_fake])
    cls = np.concatenate([
        np.full(len(floor), 1),
        np.full(len(ceil), 0),
        np.full(len(walls), 2),
        np.full(len(d1_hdr), 6),
        np.full(len(d1_leaf), 6),
        np.full(len(d2_hdr), 6),
        np.full(len(d3_fake), 6),
    ])

    las_file = tmp_path / "synth_open_doors.las"
    _write_las(las_file, pts, cls)
    out_dir = tmp_path / "out_doors"
    report = run_arch_pipeline(las_file, out_dir, render_preview=False)

    assert report["status"] == "PASS", report
    assert report["relation_audit"]["unexpected_penetration_count"] == 0
    assert report["architecture"]["openings"]["door_count"] == 2, report["architecture"]["openings"]
    reasons = [r["reason"] for r in report["architecture"]["openings"]["rejected"]]
    assert "failed_b14_door_solid_wall_below" in reasons, report["architecture"]["openings"]


def test_dual_angle_trapezoid_higher_stairwell_roof_and_u_turn_stairs(tmp_path: Path) -> None:
    """Regression gate for G8 (mode-voting room frame + slanted wall preservation),
    G9 (higher stairwell ceiling + raised stairwell walls + WL-A2), and G10 (U-turn multi-flight stairs).
    """
    # Build a 10m x 8m hall rotated by 30 deg with:
    # - Main 1F ceiling at z=3.2 over y in [0, 5.0] (with central patch mislabeled as class 1!)
    # - Higher 2F stairwell ceiling at z=5.2 over y in [5.0, 8.0], x in [4.0, 10.0]
    # - Slanted west wall from (0.0, 0.0) to (1.8, 5.0) (~20 deg off-axis — must stay straight, NOT snap to orthogonal!)
    # - 2-flight U-turn staircase inside the stairwell wing x in [4.5, 9.5], y in [4.8, 7.6]
    #   Flight 1: x in [4.7, 6.2], y climbs 4.8 -> 6.8, z climbs 0.0 -> 1.8
    #   Mid-landing: x in [4.7, 9.2], y in [6.8, 7.6], z = 1.8
    #   Flight 2: x in [7.7, 9.2], y climbs 6.8 -> 4.8, z climbs 1.8 -> 3.6
    def _in_trap(xy: np.ndarray) -> np.ndarray:
        x, y = xy[:, 0], xy[:, 1]
        # Slanted west edge: x >= 1.8 * (y / 5.0) for y <= 5, and x >= 4.0 for y > 5
        west_x = np.where(y <= 5.0, 1.8 * (y / 5.0), 4.0)
        return (x >= west_x) & (x <= 10.0) & (y >= 0.0) & (y <= 8.0)

    raw_f = _grid_xy(0.0, 10.0, 0.0, 8.0, z=0.0, step=0.10)
    floor = raw_f[_in_trap(raw_f)]

    raw_c1 = _grid_xy(0.0, 10.0, 0.0, 5.0, z=3.2, step=0.10)
    c1_all = raw_c1[_in_trap(raw_c1)]
    c1_mislab_mask = (c1_all[:, 0] >= 4.0) & (c1_all[:, 0] <= 7.0) & (c1_all[:, 1] >= 1.2) & (c1_all[:, 1] <= 3.5)
    c1_cls0 = c1_all[~c1_mislab_mask]
    c1_cls1 = c1_all[c1_mislab_mask]  # AI mislabeled ceiling as floor!

    c2_hi = _grid_xy(4.0, 10.0, 5.1, 8.0, z=5.2, step=0.10)

    # Perimeter walls (1F height 3.2m on lobby, 5.2m around stairwell!)
    w_south = _grid_wall_x(0.0, 10.0, 0.0, 0.0, 3.2)
    w_east_1f = _grid_wall_y(10.0, 0.0, 5.0, 0.0, 3.2)
    w_east_2f = _grid_wall_y(10.0, 5.0, 8.0, 0.0, 5.2)
    w_north_2f = _grid_wall_x(4.0, 10.0, 8.0, 0.0, 5.2)
    w_west_2f = _grid_wall_y(4.0, 5.0, 8.0, 0.0, 5.2)
    w_step_n = _grid_wall_x(1.8, 4.0, 5.0, 0.0, 3.2)
    # Slanted west lobby wall: (0, 0) -> (1.8, 5.0)
    ts_w = np.linspace(0.0, 1.0, 75)
    zs_w = np.arange(0.0, 3.2, 0.08)
    gt, gz = np.meshgrid(ts_w, zs_w)
    w_slant = np.column_stack([1.8 * gt.ravel(), 5.0 * gt.ravel(), gz.ravel()])
    walls = np.vstack([w_south, w_east_1f, w_east_2f, w_north_2f, w_west_2f, w_step_n, w_slant])

    # U-turn 2-flight staircase + landing
    st_parts = []
    for k in range(10):
        yk = 4.6 + k * 0.22
        zk = (k + 1) * 0.18
        st_parts.append(_grid_xy(4.7, 6.2, yk, yk + 0.22, z=zk, step=0.06))
    st_parts.append(_grid_xy(4.7, 9.2, 6.8, 7.6, z=1.80, step=0.06))
    for k in range(10):
        yk = 6.8 - (k + 1) * 0.22
        zk = 1.80 + (k + 1) * 0.18
        st_parts.append(_grid_xy(7.7, 9.2, yk, yk + 0.22, z=zk, step=0.06))
    stairs = np.vstack(st_parts)

    pts_local = np.vstack([floor, c1_cls0, c1_cls1, c2_hi, walls, stairs])
    # Rotate entire scene by +30 deg to verify mode-voting principal frame lock!
    ang = np.radians(30.0)
    rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts_rot = pts_local.copy()
    pts_rot[:, :2] = pts_local[:, :2] @ rot.T

    cls = np.concatenate([
        np.full(len(floor), 1),
        np.full(len(c1_cls0), 0),
        np.full(len(c1_cls1), 1),
        np.full(len(c2_hi), 0),
        np.full(len(walls), 2),
        np.full(len(stairs), 13),
    ])

    las_file = tmp_path / "synth_trapezoid_uturn.las"
    _write_las(las_file, pts_rot, cls)
    out_dir = tmp_path / "out_trap"
    report = run_arch_pipeline(las_file, out_dir, render_preview=False)

    assert report["status"] == "PASS", report["relation_audit"]
    assert report["glb_validation"]["status"] == "PASS"
    assert report["relation_audit"]["unexpected_penetration_count"] == 0

    fc_rep = report["architecture"]["floor_ceiling"]
    # Principal angle must lock onto the dominant 30-deg wall family (mod 90 within 3 deg), NOT average with slanted wall!
    ang_mod = fc_rep["principal_angle_deg"] % 90.0
    ang_err = min(abs(ang_mod - 30.0), 90.0 - abs(ang_mod - 30.0))
    assert ang_err < 3.5, f"Mode-voting frame drifted: {fc_rep['principal_angle_deg']} deg"
    # WL-A2 must keep floor_levels == 1 (no phantom 2F floor from mislabeled 1F ceiling!)
    assert len(fc_rep["floor_levels"]) == 1, fc_rep["floor_levels"]
    # Higher stairwell ceiling (+5.2m) must be detected alongside primary (+3.2m)
    assert len(fc_rep["ceiling_levels"]) == 2, fc_rep["ceiling_levels"]
    # U-turn staircase must decompose into >= 2 flights + landing
    sr = report["architecture"]["stairs_railings"]
    assert sr["emitted_stair_flights"] >= 2, sr
    assert sr["emitted_stair_landings"] >= 1, sr
