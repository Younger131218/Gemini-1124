"""Tests for closed solid extrusion, R23 Z-up/Y-up export, and independent validation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from ipcr.export_glb import export_glb_pair
from ipcr.geometry_primitives import extrude_polygon_solid, make_axis_aligned_box
from ipcr.types import MeshPart, SceneBuildResult
from ipcr.validator.glb_contract import validate_glb_pair


def test_r23_pair_export_and_validation_with_large_origin_and_holed_floor(tmp_path: Path) -> None:
    # Concave L-shaped floor WITH an interior stair-well hole
    outer = [(0, 0), (8, 0), (8, 4), (4, 4), (4, 7), (0, 7)]
    hole = [(1.5, 1.5), (3.0, 1.5), (3.0, 3.0), (1.5, 3.0)]
    floor_poly = Polygon(outer, [hole])
    fv, ft = extrude_polygon_solid(floor_poly, -0.12, 0.0)
    cv, ct = extrude_polygon_solid(Polygon(outer), 3.0, 3.12)
    wv, wt = make_axis_aligned_box((0.0, -0.12, -0.12), (8.0, 0.0, 3.12))

    origin = np.array([500123.0, 3400456.0, 25.0], dtype=np.float64)
    scene = SceneBuildResult(
        origin_xyz=origin,
        parts=[
            MeshPart("floor_0", 1, "floor", fv, ft, "measured"),
            MeshPart("wall_0", 2, "exterior_wall", wv, wt, "measured"),
            MeshPart("ceiling_0", 11, "ceiling", cv, ct, "measured"),
        ],
    )

    exp = export_glb_pair(scene, tmp_path / "run_01", "synth_room")
    rep = validate_glb_pair(exp["zup_path"], exp["yup_path"], require_ceiling=True)
    assert rep["status"] == "PASS", rep
    assert rep["part_count"] == 3


def test_export_refuses_overwrite(tmp_path: Path) -> None:
    v, t = make_axis_aligned_box((0, 0, 0), (1, 1, 1))
    scene = SceneBuildResult(
        origin_xyz=np.zeros(3),
        parts=[MeshPart("c", 11, "ceiling", v, t, "measured")],
    )
    export_glb_pair(scene, tmp_path / "out", "dup")
    with pytest.raises(FileExistsError):
        export_glb_pair(scene, tmp_path / "out", "dup")


def test_validator_rejects_open_non_closed_mesh_and_zero_area_triangle(tmp_path: Path) -> None:
    v, t = make_axis_aligned_box((0, 0, 0), (1, 1, 1))
    # Drop one triangle -> open boundary edges
    open_t = t[:-1]
    scene = SceneBuildResult(
        origin_xyz=np.zeros(3),
        parts=[MeshPart("broken_ceiling", 11, "ceiling", v, open_t, "measured")],
    )
    exp = export_glb_pair(scene, tmp_path / "bad_open", "bad")
    rep = validate_glb_pair(exp["zup_path"], exp["yup_path"], require_ceiling=True)
    assert rep["status"] == "FAIL"
    assert len(rep["failed_parts"]) == 1
    assert rep["failed_parts"][0]["closed_manifold"] is False

    # Degenerate zero-area triangle added
    deg_v = np.vstack([v, np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])])
    deg_t = np.vstack([t, np.array([[8, 9, 10]], dtype=np.uint32)])
    scene2 = SceneBuildResult(
        origin_xyz=np.zeros(3),
        parts=[MeshPart("deg_ceiling", 11, "ceiling", deg_v, deg_t, "measured")],
    )
    exp2 = export_glb_pair(scene2, tmp_path / "bad_deg", "bad2")
    rep2 = validate_glb_pair(exp2["zup_path"], exp2["yup_path"], require_ceiling=True)
    assert rep2["status"] == "FAIL"
    assert rep2["failed_parts"][0]["zero_area_triangles"] >= 1
