"""Unit tests (positive & negative cases) for the optimized Working Label Layer."""

from __future__ import annotations

import numpy as np

from ipcr.config import WorkingLabelConfig
from ipcr.types import SceneCloud
from ipcr.working_labels import (
    RULE_WL_A_CEILING_0_TO_11,
    RULE_WL_B_ISOLATED_NOISE,
    RULE_WL_C1_FLUSH_COLUMN_TO_WALL,
    build_working_labels,
    resolve_mixed_component_dominance,
)


def _make_cloud(pts: np.ndarray, labels: np.ndarray) -> SceneCloud:
    pts64 = np.asarray(pts, dtype=np.float64)
    lbl = np.asarray(labels, dtype=np.uint8)
    return SceneCloud(
        source_path="synthetic.las",
        sha256="0" * 64,
        las_version="1.2",
        point_format_id=3,
        origin_xyz=np.zeros(3),
        points_local=pts64,
        raw_labels=lbl,
        global_min=pts64.min(axis=0),
        global_max=pts64.max(axis=0),
    )


def _grid_plane_xy(x0: float, x1: float, y0: float, y1: float, z: float, step: float = 0.1) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    ys = np.arange(y0, y1, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])


def _grid_wall_xz(x0: float, x1: float, y: float, z0: float, z1: float, step: float = 0.08) -> np.ndarray:
    xs = np.arange(x0, x1, step)
    zs = np.arange(z0, z1, step)
    gx, gz = np.meshgrid(xs, zs)
    return np.column_stack([gx.ravel(), np.full(gx.size, y), gz.ravel()])


def test_wl_a_promotes_high_ceiling_slab_but_keeps_low_class0_untouched() -> None:
    floor = _grid_plane_xy(0, 6, 0, 6, z=0.0, step=0.1)
    ceil0 = _grid_plane_xy(0, 6, 0, 6, z=3.2, step=0.1)
    low0 = _grid_plane_xy(1, 3, 1, 3, z=0.8, step=0.1)  # low class 0 must stay 0!
    pts = np.vstack([floor, ceil0, low0])
    raw = np.concatenate([
        np.full(len(floor), 1, dtype=np.uint8),
        np.full(len(ceil0), 0, dtype=np.uint8),
        np.full(len(low0), 0, dtype=np.uint8),
    ])
    cloud = _make_cloud(pts, raw)
    raw_before = cloud.raw_labels.copy()

    state = build_working_labels(cloud, WorkingLabelConfig())
    # Raw labels MUST be untouched
    assert np.array_equal(cloud.raw_labels, raw_before)
    # High ceiling promoted to 11
    n_f, n_c = len(floor), len(ceil0)
    assert np.all(state.work_labels[n_f : n_f + n_c] == 11)
    assert np.all(state.rule_ids[n_f : n_f + n_c] == RULE_WL_A_CEILING_0_TO_11)
    # Low class 0 preserved as 0
    assert np.all(state.work_labels[n_f + n_c :] == 0)


def test_wl_b_quarantines_floating_micro_stairs_but_keeps_real_staircase() -> None:
    floor = _grid_plane_xy(0, 6, 0, 6, z=0.0, step=0.1)
    fake_stairs = np.column_stack([
        np.linspace(1.0, 1.4, 90),
        np.linspace(1.0, 1.3, 90),
        np.full(90, 2.5),
    ])
    pts = np.vstack([floor, fake_stairs])
    raw = np.concatenate([np.full(len(floor), 1, dtype=np.uint8), np.full(len(fake_stairs), 13, dtype=np.uint8)])
    state = build_working_labels(_make_cloud(pts, raw))
    assert np.all(state.work_labels[len(floor) :] == 0)
    assert np.all(state.rule_ids[len(floor) :] == RULE_WL_B_ISOLATED_NOISE)

    # Negative case: real staircase with >500 points and 2.5m rise from floor MUST stay 13
    steps = [
        _grid_plane_xy(1.0 + i * 0.25, 1.25 + i * 0.25, 0.0, 1.2, z=i * 0.2, step=0.05)
        for i in range(12)
    ]
    real_stairs = np.vstack(steps)
    pts2 = np.vstack([floor, real_stairs])
    raw2 = np.concatenate([np.full(len(floor), 1, dtype=np.uint8), np.full(len(real_stairs), 13, dtype=np.uint8)])
    state2 = build_working_labels(_make_cloud(pts2, raw2))
    assert np.all(state2.work_labels[len(floor) :] == 13)


def test_wl_c1_fixes_flush_wall_column_bleed_but_keeps_protruding_true_column() -> None:
    floor = _grid_plane_xy(0, 8, 0, 6, z=0.0, step=0.15)
    wall = _grid_wall_xz(0, 8, y=0.0, z0=0.0, z1=3.0, step=0.08)
    # Flush mislabeled column patch lying inside wall plane (y = 0.01)
    fake_col = _grid_wall_xz(2.0, 2.6, y=0.01, z0=0.2, z1=2.6, step=0.06)
    # Real attached column protruding 0.35m into room (y from 0.0 to 0.35)
    real_col_front = _grid_wall_xz(5.5, 6.0, y=0.35, z0=0.0, z1=3.0, step=0.06)
    real_col_side = np.column_stack([
        np.full(150, 5.5),
        np.tile(np.linspace(0.0, 0.35, 10), 15),
        np.repeat(np.linspace(0.0, 3.0, 15), 10),
    ])
    real_col = np.vstack([real_col_front, real_col_side])

    pts = np.vstack([floor, wall, fake_col, real_col])
    raw = np.concatenate([
        np.full(len(floor), 1, dtype=np.uint8),
        np.full(len(wall), 2, dtype=np.uint8),
        np.full(len(fake_col), 4, dtype=np.uint8),
        np.full(len(real_col), 4, dtype=np.uint8),
    ])
    state = build_working_labels(_make_cloud(pts, raw))
    i_fake = len(floor) + len(wall)
    i_real = i_fake + len(fake_col)

    assert np.all(state.work_labels[i_fake:i_real] == 2)
    assert np.all(state.rule_ids[i_fake:i_real] == RULE_WL_C1_FLUSH_COLUMN_TO_WALL)
    assert np.all(state.work_labels[i_real:] == 4)


def test_wl_c2_dominance_resolves_q04_bleed_and_preserves_ambiguous_mix() -> None:
    # Q04 style: 44081 clutter + 1020 cabinet -> relabel minor bleed to clutter
    q04 = np.concatenate([np.full(44081, 12, dtype=np.uint8), np.full(1020, 9, dtype=np.uint8)])
    r_q04 = resolve_mixed_component_dominance(q04, dominance_ratio=0.90)
    assert r_q04["decision"] == "relabel_minor_bleed"
    assert r_q04["dominant_class"] == 12

    # Ambiguous 60/40 mix -> must NOT force relabel
    amb = np.concatenate([np.full(600, 8, dtype=np.uint8), np.full(400, 9, dtype=np.uint8)])
    r_amb = resolve_mixed_component_dominance(amb, dominance_ratio=0.90)
    assert r_amb["decision"] == "ambiguous_keep"
