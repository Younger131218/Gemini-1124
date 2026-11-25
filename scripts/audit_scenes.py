"""Automated multi-scene regression invariant gate checker (G1-G12).

Usage:
    python -X utf8 scripts/audit_scenes.py runs/tjherlou_... runs/office_... runs/dcjs_... runs/hall_...

Exits with code 0 ONLY if every scene passes ALL hard geometric & anti-regression invariants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def audit_run(run_dir: Path) -> list[str]:
    report_path = run_dir / "run_report.json"
    if not report_path.exists():
        return [f"Missing run_report.json in {run_dir}"]
    r = json.loads(report_path.read_text(encoding="utf-8"))
    errs: list[str] = []

    name = run_dir.name.lower()
    if r.get("status") != "PASS":
        errs.append(f"overall status={r.get('status')}")
    if r.get("glb_validation", {}).get("status") != "PASS":
        errs.append(f"glb_validation={r.get('glb_validation', {}).get('status')}")
    pen = r.get("relation_audit", {}).get("unexpected_penetration_count", -1)
    if pen != 0:
        errs.append(f"unexpected_penetration_count={pen} (must be 0)")

    inv = r.get("invariants_audit") or {}
    if inv.get("status") != "PASS":
        errs.append(f"invariants_audit={inv}")
    if not inv.get("single_storey_ceiling_sealed_ok", False):
        errs.append(f"single-storey ceiling has holes: {inv.get('ceiling_primary_hole_count')}")

    fc = r["architecture"]["floor_ceiling"]
    sr = r["architecture"]["stairs_railings"]
    n_fl = len(fc["floor_levels"])
    n_cl = len(fc["ceiling_levels"])
    n_walls = int(r["architecture"]["exterior_shell"]["exterior_wall_count"])

    max_vdist = float(inv.get("max_footprint_vertex_cloud_dist_m", 0.0))
    # Rectangular rooms (<=6 walls) legitimately square 90-deg corners over 30-45cm corner column/plant shadows;
    # complex multi-wall rooms (>6 walls) must stay within <=0.25m of real cloud (catches any outward spike!).
    vdist_limit = 0.50 if n_walls <= 6 else 0.25
    if max_vdist > vdist_limit:
        errs.append(f"footprint vertex spike detected: max_dist={max_vdist:.3f}m (>{vdist_limit:.2f}m)")

    # Scene-family sanity checks (protecting against historical regressions across all 4 benchmarks)
    if "tjherlou" in name or "office" in name:
        if int(inv.get("ceiling_primary_hole_count", -1)) != 0:
            errs.append(f"single-storey ceiling_primary_hole_count={inv.get('ceiling_primary_hole_count')} (must be 0!)")
        if n_fl != 1 or n_cl != 1:
            errs.append(f"expected 1 floor + 1 ceiling, got fl={n_fl}, cl={n_cl}")
    elif "dcjs" in name:
        if n_fl != 2 or n_cl != 2:
            errs.append(f"DCJS expected 2 floors + 2 ceilings (no +4.38m outdoor roof!), got fl={n_fl}, cl={n_cl}")
        if sr.get("emitted_stair_flights", 0) != 1:
            errs.append(f"DCJS expected 1 straight stair flight, got {sr.get('emitted_stair_flights')}")
    elif "hall" in name:
        if n_fl != 1:
            errs.append(f"Hall expected 1 floor (mid-landings merged via WL-D into class 13!), got fl={n_fl}")
        if n_cl != 2:
            errs.append(f"Hall expected 2 ceilings (+1.05m 1F + +2.08m 2F stairwell), got cl={n_cl}")
        if sr.get("emitted_stair_flights", 0) < 3 or sr.get("emitted_stair_landings", 0) < 2:
            errs.append(f"Hall expected 3 flights + 2 landings, got {sr.get('emitted_stair_flights')}F+{sr.get('emitted_stair_landings')}L")
        for g in inv.get("stair_consecutive_z_gaps_m") or []:
            if g > 0.05:
                errs.append(f"Hall stair consecutive Z gap too large: {g}m")

    return errs


def main(argv: list[str]) -> int:
    if not argv:
        print("Usage: python scripts/audit_scenes.py <run_dir_1> [run_dir_2 ...]")
        return 2
    any_fail = False
    for arg in argv:
        p = Path(arg)
        errs = audit_run(p)
        r = json.loads((p / "run_report.json").read_text(encoding="utf-8")) if (p / "run_report.json").exists() else {}
        inv = r.get("invariants_audit") or {}
        fc = (r.get("architecture") or {}).get("floor_ceiling") or {}
        sr = (r.get("architecture") or {}).get("stairs_railings") or {}
        status = "PASS" if not errs else "FAIL"
        print(
            f"[{status}] {p.name:28s} | pen={inv.get('unexpected_penetration_count')} "
            f"| ceil_holes={inv.get('ceiling_primary_hole_count')} "
            f"| max_fp_vdist={inv.get('max_footprint_vertex_cloud_dist_m', 0)*100:4.1f}cm "
            f"| fl={len(fc.get('floor_levels', []))} cl={len(fc.get('ceiling_levels', []))} "
            f"| st={sr.get('emitted_stair_flights', 0)}F+{sr.get('emitted_stair_landings', 0)}L rl={sr.get('emitted_railing_groups', 0)} "
            f"| st_zgaps={inv.get('stair_consecutive_z_gaps_m', [])}"
        )
        for e in errs:
            print(f"    -> FAIL: {e}")
            any_fail = True
    if any_fail:
        print("\nAUDIT FAILED: One or more G1-G12 anti-regression gates failed!")
        return 1
    print(f"\nALL {len(argv)} SCENES INVARIANTS PASS (G1-G12 verified)!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
