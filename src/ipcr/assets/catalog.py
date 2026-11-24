"""Read-only model library asset auditing and unit/semantic-part metadata catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ipcr.assets.glb_reader import merged_glb_mesh
from ipcr.baseline import sha256_file


CATEGORY_MAP = {
    "凳子椅子": (7, "椅子"),
    "桌子": (8, "桌子"),
    "柜子": (9, "柜子"),
    "沙发": (10, "沙发"),
    "柱子": (4, "柱"),
    "门": (6, "门"),
    "楼梯扶手": (13, "楼梯"),
}


def _infer_unit_scale(max_extent: float, class_id: int) -> tuple[str, float]:
    """Infer raw asset units -> meters scale factor conservatively."""
    if max_extent > 300.0:
        return "mm", 0.001
    if max_extent > 15.0:
        # Check if cm (furniture ~40-350 cm -> 0.4-3.5m)
        return "cm", 0.01
    if class_id == 4 and max_extent > 8.0:
        # Decorative columns sometimes modeled tall
        return "m_tall_review", 1.0
    return "m", 1.0


def _infer_up_axis(ext_m: np.ndarray, class_id: int) -> str:
    """In standard glTF 2.0, assets are Y-up; report Y-up vs Z-up hint."""
    # Standard glTF 2.0 convention is Y-up; flag if Z strongly looks like height
    if class_id in (7, 9, 6, 4) and ext_m[2] > ext_m[1] * 1.4 and ext_m[1] < 0.35:
        return "Z_candidate_review"
    return "Y"


def audit_model_library(library_root: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Audit all GLB files in the read-only model library and emit sanitized metadata."""
    root = Path(library_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Model library not found: {root}")

    assets: list[dict[str, Any]] = []
    skipped_non_glb: list[str] = []

    for f in sorted(root.rglob("*")):
        if f.is_dir():
            continue
        rel = f.relative_to(root).as_posix()
        top_dir = rel.split("/")[0] if "/" in rel else ""
        class_id, class_name = CATEGORY_MAP.get(top_dir, (0, "待审核"))
        if f.suffix.lower() != ".glb":
            skipped_non_glb.append(rel)
            continue

        verts, tris = merged_glb_mesh(f)
        digest = sha256_file(f)
        if len(verts) == 0:
            assets.append(
                {
                    "relative_path": rel,
                    "class_id": class_id,
                    "class_name": class_name,
                    "sha256": digest,
                    "vertex_count": 0,
                    "triangle_count": 0,
                    "status": "empty_mesh",
                }
            )
            continue

        vmin = verts.min(axis=0)
        vmax = verts.max(axis=0)
        raw_ext = vmax - vmin
        unit_name, scale_to_m = _infer_unit_scale(float(raw_ext.max()), class_id)
        ext_m = raw_ext * scale_to_m
        up_axis = _infer_up_axis(ext_m, class_id)

        assets.append(
            {
                "asset_id": f.stem,
                "relative_path": rel,
                "class_id": class_id,
                "class_name": class_name,
                "sha256": digest,
                "vertex_count": int(len(verts)),
                "triangle_count": int(len(tris)),
                "baked_raw_extent_xyz": np.round(raw_ext, 4).tolist(),
                "inferred_unit": unit_name,
                "scale_to_meters": scale_to_m,
                "normalized_extent_xyz_m": np.round(ext_m, 4).tolist(),
                "up_axis": up_axis,
                "status": "audited",
            }
        )

    catalog = {
        "schema_version": 1,
        "glb_count": len(assets),
        "skipped_non_glb": skipped_non_glb,
        "notes": "Sanitized read-only metadata catalog. Binary GLB assets remain local and are never committed to Git.",
        "assets": assets,
    }
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return catalog
