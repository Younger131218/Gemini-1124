"""End-to-end Stage 1 (architecture) and Stage 2 (architecture + furniture) reconstruction pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ipcr import __version__
from ipcr.architecture.beams_columns import build_beam_parts, build_column_parts
from ipcr.architecture.frame_floors import analyze_floor_and_ceiling, build_floor_and_ceiling_parts
from ipcr.architecture.openings import build_opening_parts, detect_and_bind_openings
from ipcr.architecture.stairs_railings import build_stairs_and_railings
from ipcr.architecture.walls_shell import (
    build_exterior_wall_parts,
    build_interior_wall_parts,
    plan_exterior_wall_shell,
)
from ipcr.baseline import verify_frozen_baseline
from ipcr.config import PipelineConfig, load_class_ids
from ipcr.export_glb import export_glb_pair
from ipcr.furniture import (
    FurnitureLibrary,
    build_cabinet_parts,
    build_chair_parts,
    build_sofa_parts,
    build_table_parts,
)
from ipcr.io_las import load_las, preflight_summary
from ipcr.types import MeshPart, SceneBuildResult
from ipcr.validator.cloud_metrics import compute_bidirectional_class_metrics
from ipcr.validator.glb_contract import validate_glb_pair
from ipcr.validator.mesh_relations import audit_part_pair_intersections
from ipcr.visualize import render_arch_review_png
from ipcr.working_labels import build_working_labels


def run_pipeline(
    las_path: str | Path,
    output_dir: str | Path,
    *,
    stage: str = "furniture",
    cfg: PipelineConfig | None = None,
    library_root: str | Path | None = None,
    basename: str | None = None,
    render_preview: bool = True,
) -> dict[str, Any]:
    """Run Stage 1 ('arch') or Stage 2 ('furniture') reconstruction with independent post-export validation."""
    if stage not in ("arch", "furniture"):
        raise ValueError(f"Unsupported stage: {stage!r} (expected 'arch' or 'furniture')")

    t0 = time.perf_counter()
    baseline_res = verify_frozen_baseline()
    pcfg = cfg or PipelineConfig.load()

    out_dir = Path(output_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read-only LAS load & preflight
    cloud = load_las(las_path)
    preflight = preflight_summary(cloud)
    bname = basename or Path(las_path).stem

    # 2. Working Label Layer (non-destructive, auditable)
    wl_state = build_working_labels(cloud, pcfg.working_labels)
    wl_audit = wl_state.to_audit_dict(load_class_ids())
    (out_dir / "working_labels_audit.json").write_text(
        json.dumps(wl_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 3. Stage 1 Architecture: frame, floor/ceiling domain, closed B13a exterior shell
    fc = analyze_floor_and_ceiling(cloud.points_local, wl_state.work_labels, pcfg.architecture)
    shell = plan_exterior_wall_shell(fc, cloud.points_local, wl_state.work_labels, pcfg.architecture)

    # 4. B14 Door/Window identity & unified host opening binding
    openings, openings_audit = detect_and_bind_openings(
        fc, shell, cloud.points_local, wl_state.work_labels, pcfg.architecture
    )

    # 5. Extrude watertight architectural solids in strict collision-free DAG order:
    #    interior_walls -> columns -> stairs/railings -> beams -> floor/ceiling slabs (with stairwell/column cutouts)
    int_wall_parts, int_wall_polys_uv, int_wall_cutouts, int_wall_audit = build_interior_wall_parts(
        fc, shell, cloud.points_local, wl_state.work_labels, pcfg.architecture
    )
    col_parts, col_polys_uv, col_cutouts, col_audit = build_column_parts(
        fc, shell, int_wall_polys_uv, cloud.points_local, wl_state.work_labels, pcfg.architecture
    )
    stair_parts, stair_well_cutouts, stair_beam_blockers, stair_audit = build_stairs_and_railings(
        fc, shell, int_wall_polys_uv, col_polys_uv, cloud.points_local, wl_state.work_labels, pcfg.architecture
    )
    beam_parts, beam_audit = build_beam_parts(
        fc,
        shell,
        col_polys_uv,
        int_wall_polys_uv,
        cloud.points_local,
        wl_state.work_labels,
        pcfg.architecture,
        extra_blocker_bands=stair_beam_blockers,
    )

    upper_floor_cutouts = [*stair_well_cutouts, *col_cutouts, *int_wall_cutouts]
    parts: list[MeshPart] = []
    parts.extend(
        build_floor_and_ceiling_parts(
            fc,
            shell.outer_uv,
            pcfg.architecture,
            inner_uv=shell.inner_uv,
            upper_floor_cutouts=upper_floor_cutouts,
        )
    )
    parts.extend(build_exterior_wall_parts(fc, shell))
    parts.extend(build_opening_parts(fc, shell, openings, pcfg.architecture))
    parts.extend(int_wall_parts)
    parts.extend(col_parts)
    parts.extend(beam_parts)
    parts.extend(stair_parts)

    # 6. Stage 2 Furniture: Tables (8) -> Cabinets (9) -> Sofas (10) -> Library-Matched Chairs (7)
    furniture_audit: dict[str, Any] | None = None
    if stage == "furniture" and pcfg.furniture.enabled:
        lib_path = library_root if library_root is not None else pcfg.furniture.model_library_path
        library = FurnitureLibrary(library_root=lib_path)

        static_blockers = [*int_wall_polys_uv, *col_polys_uv]
        z_blockers = [*stair_beam_blockers]

        table_parts, table_specs, table_audit = build_table_parts(
            fc, shell, static_blockers, z_blockers, cloud.points_local, wl_state.work_labels, pcfg.furniture,
            library=library,
        )
        parts.extend(table_parts)

        cab_parts, cab_bands, cab_audit = build_cabinet_parts(
            fc, shell, static_blockers, z_blockers, table_specs, cloud.points_local, wl_state.work_labels, pcfg.furniture,
            library=library,
        )
        parts.extend(cab_parts)

        sofa_blockers = [*z_blockers, *cab_bands, *[(ts.poly_uv, ts.z_floor, ts.z_top) for ts in table_specs]]
        sofa_parts, sofa_bands, sofa_audit = build_sofa_parts(
            fc, shell, static_blockers, sofa_blockers, cloud.points_local, wl_state.work_labels, pcfg.furniture,
            library=library,
        )
        parts.extend(sofa_parts)

        chair_parts, chair_audit = build_chair_parts(
            fc, shell, parts, table_specs, library, cloud.points_local, wl_state.work_labels, pcfg.furniture
        )
        parts.extend(chair_parts)

        furniture_audit = {
            "library": library.to_audit_dict(),
            "tables": table_audit,
            "cabinets": cab_audit,
            "sofas": sofa_audit,
            "chairs": chair_audit,
        }

    scene = SceneBuildResult(
        origin_xyz=cloud.origin_xyz,
        parts=parts,
        walls=shell.walls,
        openings=openings,
        floor_z=fc.floor_z,
        ceiling_z=fc.ceiling_z,
        main_angle_rad=fc.frame.angle_rad,
    )

    stage_label = "stage2_arch_and_furniture" if stage == "furniture" else "stage1_architecture"

    # 7. Export R23 paired Z-up & Y-up GLBs
    export_info = export_glb_pair(
        scene,
        out_dir,
        bname,
        scene_extras={
            "ipcr_version": __version__,
            "stage": stage_label,
            "baseline_version": baseline_res.version,
            "baseline_sha256": baseline_res.actual_sha256,
            "input_las_sha256": cloud.sha256,
        },
    )

    # 8. Independent post-export validation (B22-B26)
    glb_val = validate_glb_pair(
        export_info["zup_path"],
        export_info["yup_path"],
        cfg=pcfg.validation,
        require_ceiling=(fc.ceiling_z is not None),
    )
    rel_audit = audit_part_pair_intersections(parts, eps=1e-3)

    eval_classes = [1, 2, 3, 4, 5, 6, 11]
    if any(p.class_id == 13 for p in parts) or stair_audit.get("stair_points_after_working_labels", 0) > 0:
        eval_classes.append(13)
    if any(p.class_id == 14 for p in parts) or stair_audit.get("railing_points_after_working_labels", 0) > 0:
        eval_classes.append(14)
    if stage == "furniture":
        for fcid in (7, 8, 9, 10):
            if any(p.class_id == fcid for p in parts) or int((wl_state.work_labels == fcid).sum()) > 0:
                eval_classes.append(fcid)
    eval_classes = sorted(set(eval_classes))

    cloud_metrics = compute_bidirectional_class_metrics(
        cloud.points_local, wl_state.work_labels, parts, target_classes=eval_classes, cfg=pcfg.validation
    )

    preview_path = None
    if render_preview:
        preview_path = render_arch_review_png(cloud, wl_state, fc, shell, parts, out_dir / "review_overlay.png")

    elapsed = round(time.perf_counter() - t0, 3)
    overall_pass = glb_val["status"] == "PASS" and rel_audit["status"] == "PASS"

    not_checked = [
        "Stage 3 regular clutter (class 12) not yet built",
        "User visual overlay acceptance in host software pending",
    ]
    if stage == "arch":
        not_checked.insert(0, "Stage 2 furniture (classes 7, 8, 9, 10) not built in stage=arch")

    report: dict[str, Any] = {
        "status": "PASS" if overall_pass else "FAIL",
        "stage": stage_label,
        "user_visual_acceptance": "pending_review",
        "formal_release_approved": False,
        "elapsed_seconds": elapsed,
        "ipcr_version": __version__,
        "baseline": {
            "version": baseline_res.version,
            "sha256": baseline_res.actual_sha256,
            "verified": baseline_res.passed,
        },
        "config": pcfg.to_dict(),
        "preflight": preflight,
        "working_labels_summary": {
            "changed_points": wl_audit["changed_points"],
            "changed_ratio": wl_audit["changed_ratio"],
            "adjustment_count": len(wl_audit["adjustments"]),
            "ambiguous_cluster_count": len(wl_audit["ambiguous_clusters"]),
        },
        "architecture": {
            "floor_ceiling": fc.audit,
            "exterior_shell": shell.audit,
            "interior_walls": int_wall_audit,
            "openings": openings_audit,
            "columns": col_audit,
            "beams": beam_audit,
            "stairs_railings": stair_audit,
        },
        "furniture": furniture_audit,
        "export": export_info,
        "glb_validation": glb_val,
        "relation_audit": rel_audit,
        "cloud_metrics": cloud_metrics,
        "review_overlay_png": preview_path,
        "not_checked_scope": not_checked,
    }

    (out_dir / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def run_arch_pipeline(
    las_path: str | Path,
    output_dir: str | Path,
    *,
    cfg: PipelineConfig | None = None,
    basename: str | None = None,
    render_preview: bool = True,
) -> dict[str, Any]:
    """Backward-compatible Stage 1 architecture-only runner."""
    return run_pipeline(
        las_path,
        output_dir,
        stage="arch",
        cfg=cfg,
        basename=basename,
        render_preview=render_preview,
    )
