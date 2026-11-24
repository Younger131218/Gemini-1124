"""Tests for real triangle-mesh penetration and full volumetric containment detection."""

from __future__ import annotations

import numpy as np

from ipcr.geometry_primitives import make_axis_aligned_box
from ipcr.types import MeshPart
from ipcr.validator.mesh_relations import audit_part_pair_intersections, meshes_penetrate


def test_piercing_and_full_containment_detected_while_face_touch_allowed() -> None:
    va, ta = make_axis_aligned_box((0.0, 0.0, 0.0), (2.0, 2.0, 0.1))  # tabletop
    # 1. Piercing box (armrest piercing through tabletop)
    vb, tb = make_axis_aligned_box((0.5, 0.5, -0.2), (0.7, 0.7, 0.2))
    r1 = meshes_penetrate(va, ta, vb, tb)
    assert r1["penetrates"] is True
    assert r1["piercing_pairs"] >= 1

    # 2. Full containment (small solid completely inside tabletop, zero surface crossing!)
    vc, tc = make_axis_aligned_box((0.5, 0.5, 0.02), (0.7, 0.7, 0.08))
    r2 = meshes_penetrate(va, ta, vc, tc)
    assert r2["penetrates"] is True
    assert r2["contained_b_in_a"] is True

    # 3. Clean face-to-face touch (leg resting under tabletop at z=0.0, no overlap)
    vd, td = make_axis_aligned_box((0.1, 0.1, -0.7), (0.2, 0.2, 0.0))
    r3 = meshes_penetrate(va, ta, vd, td)
    assert r3["penetrates"] is False


def test_audit_part_pair_intersections_flags_unexpected_only() -> None:
    va, ta = make_axis_aligned_box((0, 0, 0), (2, 1, 1))
    vb, tb = make_axis_aligned_box((1.5, 0, 0), (3, 1, 1))
    parts = [
        MeshPart("w1", 2, "exterior_wall", va, ta, "measured"),
        MeshPart("c1", 7, "chair", vb, tb, "measured"),
    ]
    res = audit_part_pair_intersections(parts)
    assert res["status"] == "FAIL"
    assert res["unexpected_penetration_count"] == 1
