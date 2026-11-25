"""IPCR v2 Command-Line Interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ipcr.assets.catalog import audit_model_library
from ipcr.baseline import project_root, verify_frozen_baseline
from ipcr.config import PipelineConfig
from ipcr.io_las import load_las, preflight_summary
from ipcr.validator.glb_contract import validate_glb_pair


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ipcr", description="IPCR v2 indoor point-cloud reconstruction CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify-baseline", help="Verify frozen baseline v1.1 SHA-256")

    p_audit = sub.add_parser("audit-assets", help="Audit read-only GLB model library into sanitized metadata catalog")
    p_audit.add_argument("--library", required=True, help="Path to read-only model library directory")
    p_audit.add_argument(
        "--out",
        default=str(project_root() / "assets_meta" / "library_catalog.json"),
        help="Output JSON metadata catalog path",
    )

    p_pre = sub.add_parser("preflight", help="Inspect classified LAS file without modifying it")
    p_pre.add_argument("--las", required=True, help="Path to uncompressed .las file")

    p_val = sub.add_parser("validate", help="Independently validate paired Z-up / Y-up GLB files")
    p_val.add_argument("--zup", required=True, help="Path to global_zup.glb")
    p_val.add_argument("--yup", required=True, help="Path to global_yup.glb")
    p_val.add_argument("--no-require-ceiling", action="store_true")

    p_run = sub.add_parser("run", help="Run end-to-end reconstruction pipeline from raw LAS")
    p_run.add_argument("--las", required=True, help="Path to classified uncompressed .las")
    p_run.add_argument("--out", required=True, help="New output directory (never overwrites existing)")
    p_run.add_argument("--stage", choices=["arch", "furniture", "full"], default="full", help="Pipeline stage target")
    p_run.add_argument("--library", default=None, help="Path to read-only GLB model library directory")
    p_run.add_argument("--config", default=None, help="Optional pipeline config JSON path")
    p_run.add_argument("--basename", default=None, help="Optional output basename (defaults to LAS stem)")

    args = parser.parse_args(argv)

    if args.cmd == "verify-baseline":
        res = verify_frozen_baseline()
        print(json.dumps({
            "version": res.version,
            "path": str(res.path),
            "sha256": res.actual_sha256,
            "passed": res.passed,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "audit-assets":
        verify_frozen_baseline()
        cat = audit_model_library(args.library, args.out)
        print(json.dumps({
            "glb_count": cat["glb_count"],
            "skipped_non_glb": cat["skipped_non_glb"],
            "written_to": args.out,
        }, ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "preflight":
        verify_frozen_baseline()
        cloud = load_las(args.las)
        print(json.dumps(preflight_summary(cloud), ensure_ascii=False, indent=2))
        return 0

    if args.cmd == "validate":
        verify_frozen_baseline()
        rep = validate_glb_pair(args.zup, args.yup, require_ceiling=not args.no_require_ceiling)
        print(json.dumps(rep, ensure_ascii=False, indent=2))
        return 0 if rep["status"] == "PASS" else 2

    if args.cmd == "run":
        verify_frozen_baseline()
        from ipcr.pipeline import run_pipeline

        cfg = PipelineConfig.load(Path(args.config) if args.config else None)
        report = run_pipeline(
            args.las,
            args.out,
            stage=args.stage,
            cfg=cfg,
            library_root=args.library,
            basename=args.basename,
        )
        print(json.dumps({
            "status": report["status"],
            "out_dir": str(Path(args.out).resolve()),
            "zup_glb": report["export"]["zup_path"],
            "yup_glb": report["export"]["yup_path"],
            "glb_validation": report["glb_validation"]["status"],
            "relation_audit": report["relation_audit"]["status"],
        }, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 2

    return 1


if __name__ == "__main__":
    sys.exit(main())
