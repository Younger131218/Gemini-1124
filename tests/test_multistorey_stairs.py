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
