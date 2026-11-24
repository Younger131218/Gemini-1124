"""Read-only GLB model-library loader, SHA-256 verifier, and canonical Z-up/front-+Y normalizer (B10/B21)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ipcr.assets.glb_reader import merged_glb_mesh
from ipcr.baseline import project_root, sha256_file
from ipcr.config import ValidationConfig
from ipcr.geometry_primitives import make_axis_aligned_box, signed_mesh_volume
from ipcr.validator.glb_contract import _check_part_geometry


@dataclass
class CanonicalLibraryAsset:
    """Verified watertight furniture asset in canonical Z-up frame (front=+Y, backrest=-Y, centered XY, z_min=0)."""

    asset_id: str
    class_id: int
    relative_path: str
    sha256: str
    vertices: np.ndarray  # (V, 3) float64
    triangles: np.ndarray  # (T, 3) uint32
    extents_xyz: np.ndarray  # (3,) float64 [width_X, depth_Y, height_Z]
    is_library_mesh: bool = True

    def transform_to_uvz(
        self,
        center_uv: np.ndarray,
        z_floor: float,
        front_dir_uv: np.ndarray,
        scale_xy: float,
        scale_z: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Place this canonical asset at (center_uv, z_floor) facing front_dir_uv with bounded (scale_xy, scale_z)."""
        fy = np.asarray(front_dir_uv, dtype=np.float64)
        fy = fy / max(float(np.linalg.norm(fy)), 1e-9)
        # Right-handed 2D basis: +X_local maps to right = (fy_y, -fy_x), +Y_local maps to front = fy
        rx = np.array([fy[1], -fy[0]], dtype=np.float64)

        v = self.vertices.copy()
        v[:, 0] *= float(scale_xy)
        v[:, 1] *= float(scale_xy)
        v[:, 2] *= float(scale_z)

        out = np.empty_like(v)
        out[:, 0] = center_uv[0] + v[:, 0] * rx[0] + v[:, 1] * fy[0]
        out[:, 1] = center_uv[1] + v[:, 0] * rx[1] + v[:, 1] * fy[1]
        out[:, 2] = z_floor + v[:, 2]
        return out, self.triangles.copy()


def _detect_backrest_on_positive_y(verts_zup: np.ndarray) -> bool:
    """Detect whether the upper backrest (top 30% Z) lies on +Y (needs 180-deg flip so front=+Y)."""
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
    """Closed manifold multi-box structural chair (canonical front=+Y, backrest=-Y, W=0.58, D=0.60, H=0.95)
    used ONLY when local model library is unavailable (e.g., offline CI tests).
    """
    parts: list[tuple[np.ndarray, np.ndarray, str]] = []
    # Seat cushion
    parts.append((*make_axis_aligned_box((-0.26, -0.24, 0.42), (0.26, 0.25, 0.49)), "chair_seat"))
    # Backrest (at -Y, cleanly 4mm above seat)
    parts.append((*make_axis_aligned_box((-0.25, -0.29, 0.494), (0.25, -0.22, 0.95)), "chair_back"))
    # 4 legs (cleanly 2mm below seat bottom)
    for sx in (-0.23, 0.19):
        for sy in (-0.23, 0.19):
            parts.append((*make_axis_aligned_box((sx, sy, 0.0), (sx + 0.04, sy + 0.04, 0.418)), "chair_leg"))
    return parts, np.array([0.52, 0.54, 0.95], dtype=np.float64)


class FurnitureLibrary:
    """Verified read-only furniture asset registry (B10/B21)."""

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

    def _load(self) -> None:
        if not self.catalog_path.exists():
            self.audit_entries.append({"status": "catalog_missing", "path": str(self.catalog_path)})
            return
        cat = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        if self.library_root is None or not self.library_root.exists():
            self.audit_entries.append({
                "status": "library_root_unavailable",
                "note": "Model library path not supplied or missing; parametric rule fallbacks will be used and logged per B10.",
            })
            return

        vcfg = ValidationConfig()
        for entry in cat.get("assets", []):
            cid = int(entry.get("class_id", 0))
            if cid not in (7, 8, 9, 10):
                continue
            rel = str(entry["relative_path"])
            fpath = self.library_root / rel
            if not fpath.exists():
                self.audit_entries.append({
                    "asset_id": entry["asset_id"],
                    "class_id": cid,
                    "status": "file_missing",
                })
                continue

            actual_sha = sha256_file(fpath)
            expected_sha = str(entry.get("sha256", ""))
            if expected_sha and actual_sha != expected_sha:
                self.audit_entries.append({
                    "asset_id": entry["asset_id"],
                    "class_id": cid,
                    "status": "sha256_mismatch_rejected",
                })
                continue

            v_raw, t_raw = merged_glb_mesh(fpath)
            if len(v_raw) == 0 or len(t_raw) == 0:
                continue

            scale_m = float(entry.get("scale_to_meters", 1.0))
            # Convert glTF Y-up -> Z-up: (x, y, z)_yup -> (x, -z, y)_zup
            vz = np.column_stack([v_raw[:, 0], -v_raw[:, 2], v_raw[:, 1]]).astype(np.float64) * scale_m
            tris = t_raw.astype(np.uint32).copy()

            # Center XY and ground z_min = 0
            vmin = vz.min(axis=0)
            vmax = vz.max(axis=0)
            vz[:, 0] -= 0.5 * (vmin[0] + vmax[0])
            vz[:, 1] -= 0.5 * (vmin[1] + vmax[1])
            vz[:, 2] -= vmin[2]

            # Canonicalize facing direction: front=+Y, backrest=-Y (for chairs/sofas)
            if cid in (7, 10) and _detect_backrest_on_positive_y(vz):
                vz[:, 0] = -vz[:, 0]
                vz[:, 1] = -vz[:, 1]

            if signed_mesh_volume(vz, tris) < 0:
                tris = tris[:, [0, 2, 1]]

            chk = _check_part_geometry(vz, tris, vcfg)
            ext = vz.max(axis=0) - vz.min(axis=0)
            rec = {
                "asset_id": entry["asset_id"],
                "class_id": cid,
                "sha256": actual_sha,
                "triangle_count": int(len(tris)),
                "canonical_extents_xyz_m": np.round(ext, 4).tolist(),
                "b21_watertight_passed": bool(chk["passed"]),
                "boundary_half_edges": int(chk.get("boundary_half_edges", 0)),
                "non_manifold_edges": int(chk.get("non_manifold_edges", 0)),
                "status": "verified_ready" if chk["passed"] else "rejected_non_watertight_b21",
            }
            self.audit_entries.append(rec)
            if chk["passed"]:
                self.verified_assets[cid].append(
                    CanonicalLibraryAsset(
                        asset_id=str(entry["asset_id"]),
                        class_id=cid,
                        relative_path=rel,
                        sha256=actual_sha,
                        vertices=vz,
                        triangles=tris,
                        extents_xyz=ext,
                        is_library_mesh=True,
                    )
                )

    def get_best_asset(self, class_id: int) -> CanonicalLibraryAsset | None:
        pool = self.verified_assets.get(class_id, [])
        return pool[0] if pool else None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "library_root_configured": self.library_root is not None,
            "verified_counts_by_class": {str(k): len(v) for k, v in self.verified_assets.items()},
            "assets": self.audit_entries,
        }
