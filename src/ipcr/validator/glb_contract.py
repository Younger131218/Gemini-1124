"""Independent GLB read-back contract & geometric validity checker (B19-B22)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ipcr.assets.glb_reader import load_glb
from ipcr.baseline import sha256_file
from ipcr.config import ValidationConfig
from ipcr.export_glb import MATRIX_YUP_ROOT, MATRIX_ZUP_ROOT
from ipcr.geometry_primitives import signed_mesh_volume


def _check_part_geometry(
    verts: np.ndarray,
    tris: np.ndarray,
    cfg: ValidationConfig,
    weld_tol: float = 1e-5,
) -> dict[str, Any]:
    if len(tris) == 0 or len(verts) == 0:
        return {
            "passed": False,
            "reason": "empty_geometry",
            "triangle_count": int(len(tris)),
        }

    # Check index bounds
    if int(tris.max()) >= len(verts):
        return {"passed": False, "reason": "index_out_of_bounds"}

    # Check degenerate / zero-area triangles
    v0 = verts[tris[:, 0]]
    v1 = verts[tris[:, 1]]
    v2 = verts[tris[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    zero_area_count = int(np.sum(areas < cfg.min_triangle_area_m2))

    # Quantize vertices for topological edge & duplicate-face audit
    q = np.rint(verts / weld_tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    wt = inv[tris]

    # Duplicate faces (unordered vertex triple)
    face_keys = [tuple(sorted((int(a), int(b), int(c)))) for a, b, c in wt]
    dup_faces = sum(cnt - 1 for cnt in Counter(face_keys).values() if cnt > 1)

    # Directed half-edge manifold & watertightness check
    directed_edges: Counter[tuple[int, int]] = Counter()
    undirected_edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in wt:
        ia, ib, ic = int(a), int(b), int(c)
        if ia == ib or ib == ic or ic == ia:
            continue
        for u, w in ((ia, ib), (ib, ic), (ic, ia)):
            directed_edges[(u, w)] += 1
            undirected_edges[(min(u, w), max(u, w))] += 1

    boundary_edges = sum(1 for (u, w), cnt in directed_edges.items() if directed_edges.get((w, u), 0) != cnt)
    non_manifold_edges = sum(1 for cnt in undirected_edges.values() if cnt != 2)

    vol = signed_mesh_volume(verts, tris)
    closed = boundary_edges == 0 and non_manifold_edges == 0
    positive_vol = vol >= cfg.min_solid_volume_m3
    passed = closed and positive_vol and zero_area_count == 0 and dup_faces == 0

    return {
        "passed": passed,
        "triangle_count": int(len(tris)),
        "signed_volume_m3": round(float(vol), 6),
        "positive_volume": positive_vol,
        "closed_manifold": closed,
        "boundary_half_edges": int(boundary_edges),
        "non_manifold_edges": int(non_manifold_edges),
        "zero_area_triangles": zero_area_count,
        "duplicate_faces": int(dup_faces),
    }


def validate_glb_pair(
    zup_path: str | Path,
    yup_path: str | Path,
    *,
    cfg: ValidationConfig | None = None,
    require_ceiling: bool = True,
) -> dict[str, Any]:
    """Independently read Z-up and Y-up GLBs from disk and verify R23 contract + geometry."""
    vcfg = cfg or ValidationConfig()
    zp = Path(zup_path)
    yp = Path(yup_path)

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    if not zp.exists() or not yp.exists():
        add_check("files_exist", False, {"zup_exists": zp.exists(), "yup_exists": yp.exists()})
        return {"status": "FAIL", "checks": checks}

    add_check("files_exist", True, {"zup_sha256": sha256_file(zp), "yup_sha256": sha256_file(yp)})

    z_glb = load_glb(zp)
    y_glb = load_glb(yp)

    # 1. R23 structural hierarchy: 1 Scene, 2 Nodes, 1 Mesh
    for label, g, expected_mat in (("zup", z_glb.gltf, MATRIX_ZUP_ROOT), ("yup", y_glb.gltf, MATRIX_YUP_ROOT)):
        ok_struct = (
            len(g.get("scenes", [])) == 1
            and len(g.get("nodes", [])) == 2
            and len(g.get("meshes", [])) == 1
        )
        add_check(f"{label}_r23_hierarchy", ok_struct, {
            "scenes": len(g.get("scenes", [])),
            "nodes": len(g.get("nodes", [])),
            "meshes": len(g.get("meshes", [])),
        })
        root_node = g["nodes"][1]
        mat = np.asarray(root_node.get("matrix", []), dtype=np.float64)
        ok_mat = len(mat) == 16 and bool(np.allclose(mat, expected_mat, atol=1e-6))
        det = float(np.linalg.det(mat.reshape((4, 4), order="F")[:3, :3])) if len(mat) == 16 else 0.0
        add_check(f"{label}_root_matrix_right_handed", ok_mat and abs(det - 1.0) < 1e-6, {"det": round(det, 6)})

        # Check unique class per primitive and roughness=0.45
        prims = g["meshes"][0].get("primitives", [])
        cids = [p.get("extras", {}).get("class_id") for p in prims]
        ok_unique = all(c is not None for c in cids) and len(cids) == len(set(cids))
        add_check(f"{label}_unique_class_primitives", ok_unique, {"classes": cids})

        mats = g.get("materials", [])
        ok_rough = all(
            abs(float(m.get("pbrMetallicRoughness", {}).get("roughnessFactor", -1.0)) - 0.45) < 1e-5
            for m in mats
        )
        add_check(f"{label}_material_roughness_045", ok_rough, {"material_count": len(mats)})

    # 2. Ceiling presence check
    z_classes = [p.extras.get("class_id") for p in z_glb.primitives]
    if require_ceiling:
        add_check("with_ceiling_present", 11 in z_classes, {"classes": z_classes})

    # 3. Exact Z-up <-> Y-up geometric equivalence under Rx(-90 deg)
    # Rx(-90 deg) maps (x, y, z) -> (x, z, -y), so inverse maps (X, Y, Z) -> (X, -Z, Y)
    pair_ok = len(z_glb.primitives) == len(y_glb.primitives)
    max_world_diff = 0.0
    if pair_ok:
        for zp_prim, yp_prim in zip(z_glb.primitives, y_glb.primitives):
            if zp_prim.positions.shape != yp_prim.positions.shape or zp_prim.indices.shape != yp_prim.indices.shape:
                pair_ok = False
                break
            if not np.array_equal(zp_prim.indices, yp_prim.indices):
                pair_ok = False
                break
            y_back = np.empty_like(yp_prim.positions)
            y_back[:, 0] = yp_prim.positions[:, 0]
            y_back[:, 1] = -yp_prim.positions[:, 2]
            y_back[:, 2] = yp_prim.positions[:, 1]
            diff = float(np.max(np.abs(zp_prim.positions - y_back))) if len(y_back) else 0.0
            max_world_diff = max(max_world_diff, diff)
            if diff > 1e-4:
                pair_ok = False
    add_check("zup_yup_exact_equivalence", pair_ok, {"max_world_diff_m": round(max_world_diff, 8)})

    # 4. Per-part solid manifold / positive volume / zero-area / duplicate-face audit (on Z-up world coords)
    part_reports: list[dict[str, Any]] = []
    failed_parts: list[dict[str, Any]] = []
    for prim in z_glb.primitives:
        cid = prim.extras.get("class_id")
        parts_meta = prim.extras.get("parts", [])
        if not parts_meta:
            rep = _check_part_geometry(prim.positions, prim.indices, vcfg)
            rep["part_id"] = f"class_{cid}_merged"
            rep["class_id"] = cid
            part_reports.append(rep)
            if not rep["passed"]:
                failed_parts.append(rep)
            continue
        for pm in parts_meta:
            t0 = int(pm["triangle_start"])
            tc = int(pm["triangle_count"])
            sub_tris = prim.indices[t0 : t0 + tc]
            # Re-index local slice
            uverts, inv = np.unique(sub_tris.reshape(-1), return_inverse=True)
            sub_v = prim.positions[uverts]
            sub_t = inv.reshape(-1, 3).astype(np.uint32)
            rep = _check_part_geometry(sub_v, sub_t, vcfg)
            rep["part_id"] = pm.get("part_id", "unknown")
            rep["class_id"] = cid
            rep["role"] = pm.get("role", "")
            part_reports.append(rep)
            if not rep["passed"]:
                failed_parts.append(rep)

    add_check(
        "all_parts_closed_positive_valid",
        len(failed_parts) == 0 and len(part_reports) > 0,
        {
            "total_parts": len(part_reports),
            "failed_count": len(failed_parts),
            "failed_samples": failed_parts[:10],
        },
    )

    overall_pass = all(c["status"] == "PASS" for c in checks)
    return {
        "status": "PASS" if overall_pass else "FAIL",
        "checks": checks,
        "part_count": len(part_reports),
        "failed_parts": failed_parts,
    }
