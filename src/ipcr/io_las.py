"""Read-only LAS 1.2-1.4 loader with SHA-256 fingerprint and float64 localization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import laspy
import numpy as np

from ipcr.baseline import sha256_file
from ipcr.config import load_class_ids
from ipcr.types import SceneCloud


def load_las(path: str | Path, *, origin_xyz: np.ndarray | None = None) -> SceneCloud:
    """Read an uncompressed classified LAS file without modifying it."""
    las_path = Path(path).resolve()
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_path}")
    if las_path.suffix.lower() != ".las":
        raise ValueError(f"Only uncompressed .las is supported, got: {las_path.suffix}")

    digest = sha256_file(las_path)
    las = laspy.read(str(las_path))
    hdr = las.header

    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    raw_labels = np.asarray(las.classification, dtype=np.uint8).copy()

    gmin = np.array([float(hdr.mins[0]), float(hdr.mins[1]), float(hdr.mins[2])], dtype=np.float64)
    gmax = np.array([float(hdr.maxs[0]), float(hdr.maxs[1]), float(hdr.maxs[2])], dtype=np.float64)

    if origin_xyz is None:
        # Round center to 1m grid for clean reproducible compensation translation
        center = 0.5 * (gmin + gmax)
        origin = np.round(center).astype(np.float64)
    else:
        origin = np.asarray(origin_xyz, dtype=np.float64).reshape(3)

    points_local = np.empty((len(x), 3), dtype=np.float64)
    points_local[:, 0] = x - origin[0]
    points_local[:, 1] = y - origin[1]
    points_local[:, 2] = z - origin[2]

    rgb_uint8: np.ndarray | None = None
    dims = set(las.point_format.dimension_names)
    if {"red", "green", "blue"}.issubset(dims):
        r = np.asarray(las.red)
        g = np.asarray(las.green)
        b = np.asarray(las.blue)
        if max(int(r.max(initial=0)), int(g.max(initial=0)), int(b.max(initial=0))) > 255:
            rgb_uint8 = np.column_stack([(r >> 8), (g >> 8), (b >> 8)]).astype(np.uint8)
        else:
            rgb_uint8 = np.column_stack([r, g, b]).astype(np.uint8)

    return SceneCloud(
        source_path=str(las_path),
        sha256=digest,
        las_version=f"{hdr.version.major}.{hdr.version.minor}",
        point_format_id=int(hdr.point_format.id),
        origin_xyz=origin,
        points_local=points_local,
        raw_labels=raw_labels,
        global_min=gmin,
        global_max=gmax,
        rgb_uint8=rgb_uint8,
    )


def preflight_summary(cloud: SceneCloud) -> dict[str, Any]:
    """Generate a deterministic, non-destructive preflight audit of a LAS input."""
    names = load_class_ids()
    counts = cloud.class_counts()
    z_world = cloud.points_local[:, 2] + cloud.origin_xyz[2]
    classes_detail: dict[str, Any] = {}
    for cid, cnt in sorted(counts.items()):
        mask = cloud.raw_labels == cid
        zk = z_world[mask]
        p05, p50, p95 = np.percentile(zk, [5, 50, 95])
        classes_detail[str(cid)] = {
            "name": names.get(cid, "未知"),
            "count": cnt,
            "ratio": round(cnt / max(cloud.point_count, 1), 6),
            "z_min": round(float(zk.min()), 4),
            "z_p05": round(float(p05), 4),
            "z_p50": round(float(p50), 4),
            "z_p95": round(float(p95), 4),
            "z_max": round(float(zk.max()), 4),
        }

    warnings: list[str] = []
    if len(counts) <= 1:
        warnings.append("SINGLE_CLASS_INPUT: input has <=1 distinct label; verify if AI classification was applied.")
    if counts.get(1, 0) == 0:
        warnings.append("NO_FLOOR_CLASS_1: class 1 (地面) has 0 points.")
    if counts.get(11, 0) == 0 and counts.get(0, 0) > 1000:
        warnings.append("CEILING_MAY_BE_IN_CLASS_0: class 11 is empty while class 0 has substantial points.")

    return {
        "sha256": cloud.sha256,
        "las_version": cloud.las_version,
        "point_format_id": cloud.point_format_id,
        "point_count": cloud.point_count,
        "origin_xyz": cloud.origin_xyz.tolist(),
        "global_min": np.round(cloud.global_min, 4).tolist(),
        "global_max": np.round(cloud.global_max, 4).tolist(),
        "span_xyz_m": np.round(cloud.global_max - cloud.global_min, 4).tolist(),
        "classes": classes_detail,
        "warnings": warnings,
    }
