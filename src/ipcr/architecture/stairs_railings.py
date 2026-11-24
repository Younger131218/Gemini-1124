"""B16 Evidence-gated stairs, landings, and railings interface.

Only triggers when genuine multi-step / continuous railing evidence exists in work_labels;
isolated specks quarantined by WL-B produce 0 parts with explicit N/A audit evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ipcr.types import MeshPart


def evaluate_stairs_and_railings(
    points_local: np.ndarray,
    work_labels: np.ndarray,
) -> tuple[list[MeshPart], dict[str, Any]]:
    stair_pts = points_local[work_labels == 13]
    rail_pts = points_local[work_labels == 14]
    if len(stair_pts) == 0 and len(rail_pts) == 0:
        return [], {
            "status": "N/A",
            "stair_points_after_working_labels": 0,
            "railing_points_after_working_labels": 0,
            "emitted_parts": 0,
            "note": "No valid stair or railing evidence in scene (any tiny floating specks were quarantined in WL-B).",
        }
    return [], {
        "status": "NOT_CHECKED",
        "stair_points_after_working_labels": int(len(stair_pts)),
        "railing_points_after_working_labels": int(len(rail_pts)),
        "emitted_parts": 0,
        "note": "Scene contains qualifying stair/railing evidence; full stair/railing builder triggers in multi-storey scenes.",
    }
