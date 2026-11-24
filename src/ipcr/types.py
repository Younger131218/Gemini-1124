"""Core scene, geometry, working-label, and audit data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SceneCloud:
    """Read-only point cloud localized in float64 to avoid float32 precision loss."""

    source_path: str
    sha256: str
    las_version: str
    point_format_id: int
    origin_xyz: np.ndarray  # shape (3,), float64 global world offset
    points_local: np.ndarray  # shape (N, 3), float64 (global - origin_xyz)
    raw_labels: np.ndarray  # shape (N,), uint8 — NEVER mutated
    global_min: np.ndarray  # shape (3,), float64
    global_max: np.ndarray  # shape (3,), float64

    @property
    def point_count(self) -> int:
        return int(len(self.raw_labels))

    def class_counts(self, labels: np.ndarray | None = None) -> dict[int, int]:
        arr = self.raw_labels if labels is None else labels
        u, c = np.unique(arr, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}


@dataclass
class WorkingLabelAdjustment:
    rule_id: int
    rule_name: str
    raw_class: int
    work_class: int
    point_count: int
    bbox_min_local: list[float]
    bbox_max_local: list[float]
    evidence: dict[str, Any]


@dataclass
class WorkingLabelState:
    raw_labels: np.ndarray  # shape (N,), uint8 read-only reference
    work_labels: np.ndarray  # shape (N,), uint8
    rule_ids: np.ndarray  # shape (N,), uint8 (0 = untouched)
    adjustments: list[WorkingLabelAdjustment] = field(default_factory=list)
    ambiguous_clusters: list[dict[str, Any]] = field(default_factory=list)

    def to_audit_dict(self, class_names: dict[int, str]) -> dict[str, Any]:
        ru, rc = np.unique(self.raw_labels, return_counts=True)
        wu, wc = np.unique(self.work_labels, return_counts=True)
        changed = int(np.sum(self.raw_labels != self.work_labels))
        total = int(len(self.raw_labels))
        return {
            "total_points": total,
            "changed_points": changed,
            "changed_ratio": round(changed / max(total, 1), 6),
            "raw_counts": {str(int(k)): {"name": class_names.get(int(k), "?"), "count": int(v)} for k, v in zip(ru, rc)},
            "work_counts": {str(int(k)): {"name": class_names.get(int(k), "?"), "count": int(v)} for k, v in zip(wu, wc)},
            "adjustments": [
                {
                    "rule_id": a.rule_id,
                    "rule_name": a.rule_name,
                    "raw_class": a.raw_class,
                    "work_class": a.work_class,
                    "point_count": a.point_count,
                    "bbox_min_local": a.bbox_min_local,
                    "bbox_max_local": a.bbox_max_local,
                    "evidence": a.evidence,
                }
                for a in self.adjustments
            ],
            "ambiguous_clusters": self.ambiguous_clusters,
        }


@dataclass
class MeshPart:
    """Single closed solid semantic part in localized Z-up scene coordinates."""

    part_id: str
    class_id: int
    role: str
    vertices: np.ndarray  # (V, 3) float64 in scene local Z-up coordinates
    triangles: np.ndarray  # (T, 3) uint32 CCW outward normals
    evidence_kind: str  # "measured" | "inferred_completion" | "template_prior"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WallSegment:
    """Measured or completed finite wall segment in local Z-up coordinates."""

    wall_id: str
    p0_xy: np.ndarray  # (2,) float64 start centerline
    p1_xy: np.ndarray  # (2,) float64 end centerline
    normal_xy: np.ndarray  # (2,) float64 unit normal (pointing outward if exterior)
    tangent_xy: np.ndarray  # (2,) float64 unit tangent p0 -> p1
    thickness_m: float
    z_min: float
    z_max: float
    is_exterior: bool
    point_count: int
    measured_z_min: float
    measured_z_max: float
    anchor_mode: str  # "dual_face" | "single_face_outward" | "inferred_closure"
    openings: list["OpeningSpec"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.p1_xy - self.p0_xy))


@dataclass
class OpeningSpec:
    """Shared door/window/passage opening bound to its physical host wall."""

    opening_id: str
    kind: str  # "window" | "door" | "passage"
    class_id: int  # 5=window, 6=door, 0=passage
    host_wall_id: str
    s_min: float  # along host tangent from p0_xy
    s_max: float
    z_min: float
    z_max: float
    point_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneBuildResult:
    """Complete scene parts and structured provenance before GLB export."""

    origin_xyz: np.ndarray  # (3,) float64
    parts: list[MeshPart]
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[OpeningSpec] = field(default_factory=list)
    floor_z: float = 0.0
    ceiling_z: float | None = None
    main_angle_rad: float = 0.0
    audit: dict[str, Any] = field(default_factory=dict)
