"""Explicit strongly-typed pipeline configuration (no hidden env-var caches)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ipcr.baseline import project_root


@dataclass
class WorkingLabelConfig:
    ceiling_min_height_above_floor_m: float = 2.0
    ceiling_plane_distance_m: float = 0.06
    ceiling_normal_min_abs_z: float = 0.96
    ceiling_min_area_m2: float = 2.0
    ceiling_min_floor_area_ratio: float = 0.15
    ceiling_voxel_size_m: float = 0.08
    stair_railing_min_points: int = 500
    stair_railing_min_vertical_span_m: float = 0.45
    floor_outlier_below_m: float = 0.25
    beam_outlier_below_floor_clearance_m: float = 1.2
    column_wall_flush_threshold_m: float = 0.06
    column_wall_flush_ratio: float = 0.92
    column_min_protrusion_m: float = 0.08
    column_min_points: int = 200
    mixed_dominance_ratio: float = 0.90


@dataclass
class ArchitectureConfig:
    floor_thickness_m: float = 0.12
    ceiling_thickness_m: float = 0.12
    default_wall_thickness_m: float = 0.12
    min_wall_thickness_m: float = 0.06
    max_wall_thickness_m: float = 0.35
    wall_voxel_size_m: float = 0.04
    wall_plane_distance_m: float = 0.045
    wall_normal_max_abs_z: float = 0.20
    wall_min_length_m: float = 0.45
    wall_min_height_m: float = 0.60
    wall_min_points: int = 400
    wall_corner_snap_m: float = 0.35
    wall_slab_overlap_m: float = 0.01
    exterior_wall_foot_gap_snap_m: float = 1.20
    opening_host_max_dist_m: float = 0.30
    opening_min_width_m: float = 0.35
    opening_min_height_m: float = 0.40
    opening_window_min_points: int = 500
    opening_door_min_points: int = 400
    window_frame_width_m: float = 0.05
    window_frame_depth_m: float = 0.08
    window_glass_thickness_m: float = 0.02
    door_frame_width_m: float = 0.06
    door_frame_depth_m: float = 0.10
    door_leaf_thickness_m: float = 0.045
    beam_voxel_size_m: float = 0.06
    beam_min_points: int = 500
    beam_min_width_m: float = 0.10
    beam_min_height_m: float = 0.08
    column_voxel_size_m: float = 0.04
    column_min_points: int = 400
    column_min_section_m: float = 0.14
    column_min_height_m: float = 1.0


@dataclass
class FurnitureConfig:
    enabled: bool = True
    model_library_path: str | None = None
    table_min_points: int = 800
    table_min_area_m2: float = 0.20
    table_min_height_m: float = 0.42
    table_max_height_m: float = 1.35
    table_top_thickness_m: float = 0.048
    table_leg_size_m: float = 0.055
    cabinet_min_points: int = 800
    cabinet_min_depth_m: float = 0.16
    cabinet_min_width_m: float = 0.28
    cabinet_min_height_m: float = 0.35
    sofa_min_points: int = 1000
    sofa_min_depth_m: float = 0.48
    sofa_min_width_m: float = 0.70
    sofa_min_height_m: float = 0.50
    chair_min_points: int = 250
    chair_min_height_m: float = 0.38
    chair_max_height_m: float = 1.35
    chair_row_slot_m: float = 0.68
    clearance_m: float = 0.010


@dataclass
class ClutterConfig:
    enabled: bool = True
    min_points: int = 250
    voxel_cluster_size_m: float = 0.055
    min_extent_m: float = 0.035
    min_major_span_m: float = 0.16
    min_volume_m3: float = 0.002
    plant_green_ratio_threshold: float = 0.16
    regularity_max_surface_p90_m: float = 0.065
    regularity_max_surface_p50_m: float = 0.036
    min_cylinder_arc_deg: float = 120.0
    cylinder_segments: int = 20
    clearance_m: float = 0.008


@dataclass
class ValidationConfig:
    min_triangle_area_m2: float = 1e-10
    min_solid_volume_m3: float = 1e-7
    cloud_sample_limit_per_class: int = 15000
    mesh_sample_limit_per_class: int = 15000
    coverage_threshold_m: float = 0.10
    far_surface_threshold_m: float = 0.20
    random_seed: int = 20261124


@dataclass
class PipelineConfig:
    schema_version: int = 1
    working_labels: WorkingLabelConfig = field(default_factory=WorkingLabelConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    furniture: FurnitureConfig = field(default_factory=FurnitureConfig)
    clutter: ClutterConfig = field(default_factory=ClutterConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            working_labels=WorkingLabelConfig(**data.get("working_labels", {})),
            architecture=ArchitectureConfig(**data.get("architecture", {})),
            furniture=FurnitureConfig(**data.get("furniture", {})),
            clutter=ClutterConfig(**data.get("clutter", {})),
            validation=ValidationConfig(**data.get("validation", {})),
        )

    @classmethod
    def load(cls, path: Path | None = None) -> "PipelineConfig":
        cfg_path = path if path is not None else project_root() / "configs" / "default_pipeline.json"
        return cls.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))


def load_class_ids(root: Path | None = None) -> dict[int, str]:
    base = root if root is not None else project_root()
    raw = json.loads((base / "configs" / "class_ids.json").read_text(encoding="utf-8"))
    return {int(k): str(v) for k, v in raw.items()}


def load_r23_palette(root: Path | None = None) -> dict[str, Any]:
    base = root if root is not None else project_root()
    return json.loads((base / "configs" / "r23_palette.json").read_text(encoding="utf-8"))
