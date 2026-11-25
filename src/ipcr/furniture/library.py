"""Read-only GLB model-library loader, SHA-256 verifier, in-memory B21 watertight repair & multi-asset selector (B10/B21)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ipcr.assets.glb_reader import load_glb, merged_glb_mesh
from ipcr.baseline import project_root, sha256_file
from ipcr.config import ValidationConfig
from ipcr.geometry_primitives import make_axis_aligned_box, signed_mesh_volume
from ipcr.validator.glb_contract import _check_part_geometry


@dataclass
class CanonicalLibraryAsset:
    """Verified watertight furniture asset in canonical Z-up frame (front=+Y, back=-Y, width=X, depth=Y, z_min=0)."""

    asset_id: str
    class_id: int
    relative_path: str
    sha256: str
    vertices: np.ndarray  # (V, 3) float64
    triangles: np.ndarray  # (T, 3) uint32
    extents_xyz: np.ndarray  # (3,) float64 [width_X, depth_Y, height_Z]
    is_library_mesh: bool = True
    repaired_in_memory: bool = False
    # Part-level canonical functional references (especially for chairs)
    backrest_width_m: float = 0.50
    backrest_y_center: float = -0.34
    seat_width_m: float = 0.56
    seat_depth_m: float = 0.56
    seat_y_center: float = 0.0
    seat_z_center: float = 0.45
    base_width_m: float = 0.70

    def transform_to_uvz(
        self,
        center_uv: np.ndarray,
        z_floor: float,
        front_dir_uv: np.ndarray,
        scale_x: float,
        scale_y: float,
        scale_z: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place this canonical asset at (center_uv, z_floor) facing front_dir_uv with bounded (scale_x, scale_y, scale_z)."""
        fy = np.asarray(front_dir_uv, dtype=np.float64)
        fy = fy / max(float(np.linalg.norm(fy)), 1e-9)
        rx = np.array([fy[1], -fy[0]], dtype=np.float64)

        v = self.vertices.copy()
        v[:, 0] *= float(scale_x)
        v[:, 1] *= float(scale_y)
        v[:, 2] *= float(scale_z)

        out = np.empty_like(v)
        out[:, 0] = center_uv[0] + v[:, 0] * rx[0] + v[:, 1] * fy[0]
        out[:, 1] = center_uv[1] + v[:, 0] * rx[1] + v[:, 1] * fy[1]
        out[:, 2] = z_floor + v[:, 2]
        return out, self.triangles.copy()


def repair_mesh_watertight_in_memory(
    verts: np.ndarray,
    tris: np.ndarray,
    *,
    weld_tol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """In-memory B21 watertight manifold repair (weld coincident verts, drop dup/zero faces, cap simple boundary loops)."""
    q = np.rint(verts / weld_tol).astype(np.int64)
    _, uniq_idx, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    n_u = len(uniq_idx)
    new_v = np.zeros((n_u, 3), dtype=np.float64)
    counts = np.bincount(inv)
    for ax in range(3):
        np.add.at(new_v[:, ax], inv, verts[:, ax])
    new_v /= counts[:, None]

    wt = inv[tris]
    valid = (wt[:, 0] != wt[:, 1]) & (wt[:, 1] != wt[:, 2]) & (wt[:, 2] != wt[:, 0])
    wt = wt[valid]
    if len(wt) == 0:
        return new_v, np.zeros((0, 3), dtype=np.uint32)
    v0, v1, v2 = new_v[wt[:, 0]], new_v[wt[:, 1]], new_v[wt[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    wt = wt[areas >= 1e-10]

    seen: set[tuple[int, int, int]] = set()
    keep: list[tuple[int, int, int]] = []
    for a, b, c in wt:
        ia, ib, ic = int(a), int(b), int(c)
        key = tuple(sorted((ia, ib, ic)))
        if key not in seen:
            seen.add(key)
            keep.append((ia, ib, ic))
    wt = np.asarray(keep, dtype=np.uint32)

    he: Counter[tuple[int, int]] = Counter()
    for a, b, c in wt:
        ia, ib, ic = int(a), int(b), int(c)
        he[(ia, ib)] += 1
        he[(ib, ic)] += 1
        he[(ic, ia)] += 1

    next_map: dict[int, list[int]] = defaultdict(list)
    for (u, w), cnt in he.items():
        diff = cnt - he.get((w, u), 0)
        if diff > 0:
            for _ in range(diff):
                next_map[w].append(u)

    cap_tris: list[tuple[int, int, int]] = []
    verts_list = [new_v[i] for i in range(len(new_v))]
    while True:
        start = next((k for k, vs in next_map.items() if vs), None)
        if start is None:
            break
        loop = [start]
        cur = next_map[start].pop()
        guard = 0
        while cur != start and guard < 2000:
            loop.append(cur)
            if not next_map[cur]:
                break
            cur = next_map[cur].pop()
            guard += 1
        if cur == start and len(loop) >= 3:
            c_idx = len(verts_list)
            c_pos = np.mean([verts_list[i] for i in loop], axis=0)
            verts_list.append(c_pos)
            for i in range(len(loop)):
                a = loop[i]
                b = loop[(i + 1) % len(loop)]
                cap_tris.append((a, b, c_idx))

    out_v = np.asarray(verts_list, dtype=np.float64)
    out_t = np.vstack([wt, np.asarray(cap_tris, dtype=np.uint32)]) if cap_tris else wt
    if signed_mesh_volume(out_v, out_t) < 0:
        out_t = out_t[:, [0, 2, 1]]
    return out_v, out_t


def _detect_backrest_on_positive_y(verts_zup: np.ndarray) -> bool:
    z_min, z_max = float(verts_zup[:, 2].min()), float(verts_zup[:, 2].max())
    h = z_max - z_min
    if h < 0.20:
        return False
    hi = verts_zup[verts_zup[:, 2] >= z_min + 0.70 * h]
    lo = verts_zup[(verts_zup[:, 2] >= z_min + 0.25 * h) & (verts_zup[:, 2] <= z_min + 0.55 * h)]
    if len(hi) == 0 or len(lo) == 0:
        return False
    return float(hi[:, 1].mean()) > float(lo[:, 1].mean())


def make_fallback_parametric_chair() -> tuple[list[tuple[np.ndarray, np.ndarray, str]], np.ndarray]:
    """Closed manifold multi-box structural chair used ONLY when local model library is unavailable."""
    parts: list[tuple[np.ndarray, np.ndarray, str]] = []
    parts.append((*make_axis_aligned_box((-0.26, -0.24, 0.42), (0.26, 0.25, 0.49)), "chair_seat"))
    parts.append((*make_axis_aligned_box((-0.25, -0.29, 0.494), (0.25, -0.22, 0.95)), "chair_back"))
    for sx in (-0.23, 0.19):
        for sy in (-0.23, 0.19):
            parts.append((*make_axis_aligned_box((sx, sy, 0.0), (sx + 0.04, sy + 0.04, 0.418)), "chair_leg"))
    return parts, np.array([0.52, 0.54, 0.95], dtype=np.float64)


class FurnitureLibrary:
    """Verified read-only furniture asset registry with in-memory B21 watertight repair & multi-asset matching."""

    def __init__(self, library_root: str | Path | None = None, catalog_path: str | Path | None = None) -> None:
        self.library_root = Path(library_root).resolve() if library_root is not None else None
        self.catalog_path = (
            Path(catalog_path).resolve()
            if catalog_path is not None
            else project_root() / "assets_meta" / "library_catalog.json"
        )
        self.verified_assets: dict[int, list[CanonicalLibraryAsset]] = {7: [], 8: [], 9: [], 10: []}
        self.audit_entries: list[dict[str, Any]] = []
        self._load()

    def _canonicalize_and_register(
        self,
        asset_id: str,
        cid: int,
        rel: str,
        sha: str,
        vz: np.ndarray,
        tris: np.ndarray,
        repaired: bool,
        handle_side_xy: np.ndarray | None,
        vcfg: ValidationConfig,
    ) -> dict[str, Any]:
        vmin = vz.min(axis=0)
        vmax = vz.max(axis=0)
        cx, cy = 0.5 * (vmin[0] + vmax[0]), 0.5 * (vmin[1] + vmax[1])
        vz = vz.copy()
        vz[:, 0] -= cx
        vz[:, 1] -= cy
        vz[:, 2] -= vmin[2]
        hxy = handle_side_xy - np.array([cx, cy], dtype=np.float64) if handle_side_xy is not None else None

        if cid in (7, 10):
            if _detect_backrest_on_positive_y(vz):
                vz[:, 0] = -vz[:, 0]
                vz[:, 1] = -vz[:, 1]
            # Re-center chair/sofa X onto upper backrest+seat symmetry axis (removes off-center caster star bias!)
            h_cur = float(vz[:, 2].max())
            body = vz[vz[:, 2] >= 0.35 * h_cur]
            if len(body) >= 20:
                x_sym = 0.5 * (float(body[:, 0].min()) + float(body[:, 0].max()))
                vz[:, 0] -= x_sym
        elif cid == 9:
            ext = vz.max(axis=0) - vz.min(axis=0)
            if ext[1] > ext[0] * 1.15:
                if hxy is not None and hxy[0] > 0:
                    vz = np.column_stack([-vz[:, 1], vz[:, 0], vz[:, 2]])
                else:
                    vz = np.column_stack([vz[:, 1], -vz[:, 0], vz[:, 2]])
            elif hxy is not None and hxy[1] < 0:
                vz[:, 0] = -vz[:, 0]
                vz[:, 1] = -vz[:, 1]
        elif cid == 8:
            # Ensure table long axis is along X (width >= depth)
            ext = vz.max(axis=0) - vz.min(axis=0)
            if ext[1] > ext[0] * 1.05:
                vz = np.column_stack([-vz[:, 1], vz[:, 0], vz[:, 2]])

        if signed_mesh_volume(vz, tris) < 0:
            tris = tris[:, [0, 2, 1]]

        chk = _check_part_geometry(vz, tris, vcfg)
        ext = vz.max(axis=0) - vz.min(axis=0)
        h = float(ext[2])

        # Compute exact part-level functional metrics (base / seat / backrest)
        base_sl = vz[vz[:, 2] <= 0.30 * h]
        seat_sl = vz[(vz[:, 2] >= 0.35 * h) & (vz[:, 2] <= 0.55 * h)]
        back_sl = vz[vz[:, 2] >= 0.68 * h]

        base_w = float(base_sl[:, 0].max() - base_sl[:, 0].min()) if len(base_sl) else float(ext[0])
        seat_w = float(seat_sl[:, 0].max() - seat_sl[:, 0].min()) if len(seat_sl) else float(ext[0])
        seat_d = float(seat_sl[:, 1].max() - seat_sl[:, 1].min()) if len(seat_sl) else float(ext[1])
        seat_yc = float(seat_sl[:, 1].mean()) if len(seat_sl) else 0.0
        seat_zc = float(seat_sl[:, 2].mean()) if len(seat_sl) else 0.45 * h
        back_w = float(back_sl[:, 0].max() - back_sl[:, 0].min()) if len(back_sl) else float(ext[0])
        back_yc = float(back_sl[:, 1].mean()) if len(back_sl) else -0.34 * float(ext[1])

        rec = {
            "asset_id": asset_id,
            "class_id": cid,
            "sha256": sha,
            "triangle_count": int(len(tris)),
            "canonical_extents_xyz_m": np.round(ext, 4).tolist(),
            "part_metrics_m": {
                "base_w": round(base_w, 3),
                "seat_w": round(seat_w, 3),
                "seat_d": round(seat_d, 3),
                "seat_z": round(seat_zc, 3),
                "back_w": round(back_w, 3),
                "back_y": round(back_yc, 3),
            },
            "repaired_in_memory": repaired,
            "b21_watertight_passed": bool(chk["passed"]),
            "boundary_half_edges": int(chk.get("boundary_half_edges", 0)),
            "non_manifold_edges": int(chk.get("non_manifold_edges", 0)),
            "status": (
                "verified_ready_repaired"
                if (chk["passed"] and repaired)
                else ("verified_ready_raw" if chk["passed"] else "rejected_non_watertight_b21")
            ),
        }
        if chk["passed"]:
            self.verified_assets[cid].append(
                CanonicalLibraryAsset(
                    asset_id=asset_id,
                    class_id=cid,
                    relative_path=rel,
                    sha256=sha,
                    vertices=vz,
                    triangles=tris.astype(np.uint32),
                    extents_xyz=ext,
                    is_library_mesh=True,
                    repaired_in_memory=repaired,
                    backrest_width_m=back_w,
                    backrest_y_center=back_yc,
                    seat_width_m=seat_w,
                    seat_depth_m=seat_d,
                    seat_y_center=seat_yc,
                    seat_z_center=seat_zc,
                    base_width_m=base_w,
                )
            )
        return rec

    def _load(self) -> None:
        if not self.catalog_path.exists():
            self.audit_entries.append({"status": "catalog_missing", "path": str(self.catalog_path)})
            return
        cat = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        if self.library_root is None or not self.library_root.exists():
            self.audit_entries.append({
                "status": "library_root_unavailable",
                "note": "Model library path not supplied or missing; parametric rule fallbacks will be used per B10.",
            })
            return

        vcfg = ValidationConfig()
        for entry in cat.get("assets", []):
            cid = int(entry.get("class_id", 0))
            if cid not in (7, 8, 9, 10):
                continue
            ext_meta = entry.get("normalized_extent_xyz_m", [0, 0, 0])
            if cid == 8 and float(ext_meta[1]) > 1.50:
                self.audit_entries.append({"asset_id": entry["asset_id"], "class_id": cid, "status": "skipped_contains_props_b10"})
                continue
            if cid == 10 and float(ext_meta[1]) > 1.30:
                self.audit_entries.append({"asset_id": entry["asset_id"], "class_id": cid, "status": "skipped_contains_props_b10"})
                continue

            rel = str(entry["relative_path"])
            fpath = self.library_root / rel
            if not fpath.exists():
                self.audit_entries.append({"asset_id": entry["asset_id"], "class_id": cid, "status": "file_missing"})
                continue

            actual_sha = sha256_file(fpath)
            expected_sha = str(entry.get("sha256", ""))
            if expected_sha and actual_sha != expected_sha:
                self.audit_entries.append({"asset_id": entry["asset_id"], "class_id": cid, "status": "sha256_mismatch_rejected"})
                continue

            scale_m = float(entry.get("scale_to_meters", 1.0))
            v_raw, t_raw = merged_glb_mesh(fpath)
            if len(v_raw) == 0 or len(t_raw) == 0:
                continue

            vz = np.column_stack([v_raw[:, 0], -v_raw[:, 2], v_raw[:, 1]]).astype(np.float64) * scale_m
            tris = t_raw.astype(np.uint32).copy()

            chk_raw = _check_part_geometry(vz, tris, vcfg)
            if chk_raw["passed"]:
                rec = self._canonicalize_and_register(
                    str(entry["asset_id"]), cid, rel, actual_sha, vz, tris, False, None, vcfg
                )
                self.audit_entries.append(rec)
                continue

            parsed = load_glb(fpath)
            best_prim = None
            best_vol_box = 0.0
            handle_xy = None
            for pr in parsed.primitives:
                pv = np.column_stack([pr.positions[:, 0], -pr.positions[:, 2], pr.positions[:, 1]]).astype(np.float64) * scale_m
                pext = pv.max(axis=0) - pv.min(axis=0)
                vbox = float(pext[0] * pext[1] * pext[2])
                if vbox > best_vol_box:
                    best_vol_box = vbox
                    best_prim = (pv, pr.indices.astype(np.uint32))
                elif cid == 9 and vbox < 0.01 and float(pext[2]) < 0.25:
                    handle_xy = pv[:, :2].mean(axis=0)

            if best_prim is not None and len(best_prim[1]) <= 15000:
                rv, rt = repair_mesh_watertight_in_memory(best_prim[0], best_prim[1])
                rec = self._canonicalize_and_register(
                    str(entry["asset_id"]), cid, rel, actual_sha, rv, rt, True, handle_xy, vcfg
                )
                self.audit_entries.append(rec)
            else:
                self.audit_entries.append({
                    "asset_id": entry["asset_id"],
                    "class_id": cid,
                    "sha256": actual_sha,
                    "status": "rejected_non_watertight_b21",
                })

    def get_assets(self, class_id: int) -> list[CanonicalLibraryAsset]:
        return list(self.verified_assets.get(class_id, []))

    def get_best_asset(self, class_id: int) -> CanonicalLibraryAsset | None:
        pool = self.verified_assets.get(class_id, [])
        return pool[0] if pool else None

    def rank_assets(
        self,
        class_id: int,
        target_wdh: tuple[float, float, float],
        *,
        back_width_m: float | None = None,
        seat_only: bool = False,
    ) -> list[CanonicalLibraryAsset]:
        """Rank all verified library assets for class_id by geometric similarity to measured dimensions (B10)."""
        pool = self.verified_assets.get(class_id, [])
        if len(pool) <= 1:
            return list(pool)

        tw, td, th = float(target_wdh[0]), float(target_wdh[1]), float(target_wdh[2])

        def _score(a: CanonicalLibraryAsset) -> float:
            aw, ad, ah = float(a.extents_xyz[0]), float(a.extents_xyz[1]), float(a.extents_xyz[2])
            if class_id == 7:
                # Match chair height + backrest-width-to-height proportion!
                h_err = abs(ah - th) / max(th, 0.5)
                if seat_only:
                    # Compact lower-profile modern chair is ideal when backrest is low/missing
                    return h_err + 0.35 * abs(a.seat_width_m - tw) / max(tw, 0.35) + (0.0 if ah <= 1.02 else 0.30)
                if back_width_m is not None and back_width_m >= 0.30:
                    target_ratio = back_width_m / max(th, 0.5)
                    asset_ratio = a.backrest_width_m / max(ah, 0.5)
                    # Penalize wide wheel base relative to narrow backrest (prevents skinny backrest + bloated wheels!)
                    base_bloat = max(0.0, (a.base_width_m / max(a.backrest_width_m, 0.30)) - 1.35) * 0.45
                    return 1.4 * abs(asset_ratio - target_ratio) + 0.8 * h_err + base_bloat
                return h_err + 0.5 * abs(a.seat_width_m - tw) / max(tw, 0.35)
            # Cabinets / tables / sofas: match aspect & height ratios
            d_err = abs(ad - td) / max(td, 0.25)
            h_err = abs(ah - th) / max(th, 0.40)
            w_err = abs(aw - tw) / max(tw, 0.35)
            return d_err + h_err + 0.4 * w_err

        return sorted(pool, key=_score)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "library_root_configured": self.library_root is not None,
            "verified_counts_by_class": {str(k): len(v) for k, v in self.verified_assets.items()},
            "assets": self.audit_entries,
        }
