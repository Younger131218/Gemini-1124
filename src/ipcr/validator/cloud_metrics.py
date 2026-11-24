"""Bidirectional point-cloud <-> mesh surface metrics (B11/B23)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from ipcr.config import ValidationConfig
from ipcr.types import MeshPart


def sample_mesh_surface(
    parts: list[MeshPart],
    sample_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Area-weighted uniform surface sampling across a list of MeshParts."""
    if not parts:
        return np.zeros((0, 3), dtype=np.float64)
    v0_all, v1_all, v2_all = [], [], []
    for p in parts:
        if len(p.triangles) == 0:
            continue
        t = p.triangles.astype(np.int64)
        v0_all.append(p.vertices[t[:, 0]])
        v1_all.append(p.vertices[t[:, 1]])
        v2_all.append(p.vertices[t[:, 2]])
    if not v0_all:
        return np.zeros((0, 3), dtype=np.float64)
    v0 = np.vstack(v0_all)
    v1 = np.vstack(v1_all)
    v2 = np.vstack(v2_all)
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    total_area = float(areas.sum())
    if total_area <= 1e-12:
        return np.zeros((0, 3), dtype=np.float64)
    probs = areas / total_area
    tri_idx = rng.choice(len(areas), size=sample_count, replace=True, p=probs)
    r1 = np.sqrt(rng.random(sample_count))[:, None]
    r2 = rng.random(sample_count)[:, None]
    pts = (1.0 - r1) * v0[tri_idx] + (r1 * (1.0 - r2)) * v1[tri_idx] + (r1 * r2) * v2[tri_idx]
    return pts


def compute_bidirectional_class_metrics(
    points_local: np.ndarray,
    labels: np.ndarray,
    parts: list[MeshPart],
    *,
    target_classes: list[int] | None = None,
    cfg: ValidationConfig | None = None,
) -> dict[str, Any]:
    """Compute per-class bidirectional point->mesh and mesh->point distances."""
    vcfg = cfg or ValidationConfig()
    rng = np.random.default_rng(vcfg.random_seed)
    parts_by_cls: dict[int, list[MeshPart]] = {}
    for p in parts:
        parts_by_cls.setdefault(int(p.class_id), []).append(p)

    classes = target_classes if target_classes is not None else sorted(set(parts_by_cls.keys()))
    per_class: dict[str, Any] = {}

    for cid in classes:
        cls_pts = points_local[labels == cid]
        cls_parts = parts_by_cls.get(cid, [])
        if len(cls_pts) == 0 and not cls_parts:
            continue
        if len(cls_pts) == 0:
            per_class[str(cid)] = {"source_points": 0, "emitted_parts": len(cls_parts), "note": "no_source_points"}
            continue
        if not cls_parts:
            per_class[str(cid)] = {
                "source_points": int(len(cls_pts)),
                "emitted_parts": 0,
                "cloud_to_mesh_within_10cm": 0.0,
                "note": "not_emitted",
            }
            continue

        # Subsample source points deterministically
        if len(cls_pts) > vcfg.cloud_sample_limit_per_class:
            idx = rng.choice(len(cls_pts), size=vcfg.cloud_sample_limit_per_class, replace=False)
            src_sample = cls_pts[idx]
        else:
            src_sample = cls_pts

        mesh_sample = sample_mesh_surface(cls_parts, vcfg.mesh_sample_limit_per_class, rng)
        if len(mesh_sample) == 0:
            per_class[str(cid)] = {"source_points": int(len(cls_pts)), "emitted_parts": len(cls_parts), "note": "zero_area_mesh"}
            continue

        # Dense mesh surface KD-tree for fast accurate point-to-surface approximation
        mesh_tree = cKDTree(mesh_sample)
        d_c2m, _ = mesh_tree.query(src_sample, workers=-1)

        src_tree = cKDTree(src_sample)
        d_m2c, _ = src_tree.query(mesh_sample, workers=-1)

        per_class[str(cid)] = {
            "source_points": int(len(cls_pts)),
            "sampled_source_points": int(len(src_sample)),
            "emitted_parts": len(cls_parts),
            "cloud_to_mesh_within_10cm": round(float(np.mean(d_c2m <= vcfg.coverage_threshold_m)), 4),
            "cloud_to_mesh_p50_m": round(float(np.percentile(d_c2m, 50)), 4),
            "cloud_to_mesh_p95_m": round(float(np.percentile(d_c2m, 95)), 4),
            "mesh_to_cloud_far_gt_20cm": round(float(np.mean(d_m2c > vcfg.far_surface_threshold_m)), 4),
            "mesh_to_cloud_p50_m": round(float(np.percentile(d_m2c, 50)), 4),
            "mesh_to_cloud_p95_m": round(float(np.percentile(d_m2c, 95)), 4),
            "note": "mesh_to_cloud includes hidden/thickness prior faces; interpret alongside visual review",
        }

    return {
        "coverage_threshold_m": vcfg.coverage_threshold_m,
        "far_surface_threshold_m": vcfg.far_surface_threshold_m,
        "per_class": per_class,
    }
