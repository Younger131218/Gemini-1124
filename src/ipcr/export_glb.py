"""Strict R23 GLB 2.0 exporter: with-ceiling class-color Z-up and Y-up pair (B19-B21)."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np

from ipcr.baseline import sha256_file
from ipcr.config import load_class_ids, load_r23_palette
from ipcr.types import MeshPart, SceneBuildResult


# Column-major 4x4 matrices matching R23 template contract:
# Z-up root: Identity
MATRIX_ZUP_ROOT = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]

# Y-up root: Rx(-90 deg) in column-major order -> maps (x, y, z) to (x, z, -y), det = +1
# Column 0 = (1, 0, 0, 0)
# Column 1 = (0, 0, -1, 0)
# Column 2 = (0, 1, 0, 0)
# Column 3 = (0, 0, 0, 1)
MATRIX_YUP_ROOT = [
    1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, -1.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def _pad4(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
    rem = len(data) % 4
    return data if rem == 0 else data + pad_byte * (4 - rem)


def _encode_single_glb(
    parts_by_class: dict[int, list[MeshPart]],
    origin_xyz: np.ndarray,
    root_matrix: list[float],
    coord_convention: str,
    palette: dict[str, Any],
    class_names: dict[int, str],
    scene_extras: dict[str, Any],
) -> bytes:
    bin_buf = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    primitives: list[dict[str, Any]] = []

    roughness = float(palette.get("roughness_factor", 0.45))
    metallic = float(palette.get("metallic_factor", 0.0))
    class_palette = palette["classes"]

    def append_buffer(raw: bytes, target: int) -> int:
        # Align offset to 4 bytes
        rem = len(bin_buf) % 4
        if rem:
            bin_buf.extend(b"\x00" * (4 - rem))
        offset = len(bin_buf)
        bin_buf.extend(raw)
        bv_idx = len(buffer_views)
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": offset,
                "byteLength": len(raw),
                "target": target,
            }
        )
        return bv_idx

    for class_id in sorted(parts_by_class.keys()):
        parts = parts_by_class[class_id]
        if not parts:
            continue

        # Merge all parts of this class into one primitive, recording exact per-part index ranges
        verts_list: list[np.ndarray] = []
        tris_list: list[np.ndarray] = []
        part_ranges: list[dict[str, Any]] = []
        v_offset = 0
        t_offset = 0
        for p in parts:
            v32 = np.asarray(p.vertices, dtype=np.float32)
            t32 = np.asarray(p.triangles, dtype=np.uint32).reshape(-1, 3)
            if len(v32) == 0 or len(t32) == 0:
                continue
            verts_list.append(v32)
            tris_list.append(t32 + np.uint32(v_offset))
            part_ranges.append(
                {
                    "part_id": p.part_id,
                    "role": p.role,
                    "evidence": p.evidence_kind,
                    "vertex_start": v_offset,
                    "vertex_count": int(len(v32)),
                    "triangle_start": t_offset,
                    "triangle_count": int(len(t32)),
                    "metadata": p.metadata,
                }
            )
            v_offset += int(len(v32))
            t_offset += int(len(t32))

        if not verts_list:
            continue

        all_v = np.ascontiguousarray(np.vstack(verts_list), dtype="<f4")
        all_t = np.ascontiguousarray(np.vstack(tris_list).reshape(-1), dtype="<u4")

        v_bv = append_buffer(all_v.tobytes(), 34962)  # ARRAY_BUFFER
        t_bv = append_buffer(all_t.tobytes(), 34963)  # ELEMENT_ARRAY_BUFFER

        pos_acc_idx = len(accessors)
        accessors.append(
            {
                "bufferView": v_bv,
                "componentType": 5126,  # FLOAT
                "count": int(len(all_v)),
                "type": "VEC3",
                "min": [float(x) for x in all_v.min(axis=0)],
                "max": [float(x) for x in all_v.max(axis=0)],
            }
        )
        idx_acc_idx = len(accessors)
        accessors.append(
            {
                "bufferView": t_bv,
                "componentType": 5125,  # UNSIGNED_INT (uint32, prevents uint16 overflow)
                "count": int(len(all_t)),
                "type": "SCALAR",
                "min": [int(all_t.min())],
                "max": [int(all_t.max())],
            }
        )

        mat_idx = len(materials)
        cinfo = class_palette.get(
            str(class_id),
            {"name": class_names.get(class_id, f"class_{class_id}"), "rgba": [0.6, 0.6, 0.6, 1.0]},
        )
        materials.append(
            {
                "name": str(cinfo["name"]),
                "pbrMetallicRoughness": {
                    "baseColorFactor": [float(x) for x in cinfo["rgba"]],
                    "metallicFactor": metallic,
                    "roughnessFactor": roughness,
                },
                "doubleSided": False,
                "extras": {"class_id": int(class_id), "class_name": class_names.get(class_id, str(cinfo["name"]))},
            }
        )

        primitives.append(
            {
                "attributes": {"POSITION": pos_acc_idx},
                "indices": idx_acc_idx,
                "material": mat_idx,
                "mode": 4,  # TRIANGLES
                "extras": {
                    "class_id": int(class_id),
                    "class_name": class_names.get(class_id, str(cinfo["name"])),
                    "part_count": len(part_ranges),
                    "parts": part_ranges,
                },
            }
        )

    if not primitives:
        raise ValueError("Cannot export empty scene: zero valid primitives.")

    bin_bytes = _pad4(bytes(bin_buf), b"\x00")
    gltf: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "ipcr-v2",
            "extras": {
                "contract": "R23_v1.1",
                "coord_convention": coord_convention,
                "with_ceiling": 11 in parts_by_class and len(parts_by_class[11]) > 0,
                **scene_extras,
            },
        },
        "scene": 0,
        "scenes": [{"nodes": [1]}],
        "nodes": [
            {
                "name": "scene_mesh",
                "mesh": 0,
                "translation": [float(origin_xyz[0]), float(origin_xyz[1]), float(origin_xyz[2])],
            },
            {
                "name": f"root_{coord_convention}",
                "children": [0],
                "matrix": [float(x) for x in root_matrix],
            },
        ],
        "meshes": [{"name": "ipcr_scene", "primitives": primitives}],
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_bytes)}],
    }

    json_bytes = _pad4(json.dumps(gltf, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), b" ")
    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    header = struct.pack("<III", 0x46546C67, 2, total_len)
    chunk0 = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    chunk1 = struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes
    return header + chunk0 + chunk1


def export_glb_pair(
    scene: SceneBuildResult,
    output_dir: str | Path,
    basename: str,
    *,
    scene_extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export paired Z-up and Y-up with-ceiling class-color GLB files (B19-B21)."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zup_path = out_dir / f"{basename}_global_zup.glb"
    yup_path = out_dir / f"{basename}_global_yup.glb"
    if zup_path.exists() or yup_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing GLB output in {out_dir}")

    parts_by_class: dict[int, list[MeshPart]] = {}
    for p in scene.parts:
        parts_by_class.setdefault(int(p.class_id), []).append(p)

    palette = load_r23_palette()
    class_names = load_class_ids()
    extras = dict(scene_extras or {})

    zup_bytes = _encode_single_glb(
        parts_by_class, scene.origin_xyz, MATRIX_ZUP_ROOT, "global_zup", palette, class_names, extras
    )
    yup_bytes = _encode_single_glb(
        parts_by_class, scene.origin_xyz, MATRIX_YUP_ROOT, "global_yup", palette, class_names, extras
    )

    zup_path.write_bytes(zup_bytes)
    yup_path.write_bytes(yup_bytes)

    return {
        "zup_path": str(zup_path),
        "zup_sha256": sha256_file(zup_path),
        "zup_bytes": len(zup_bytes),
        "yup_path": str(yup_path),
        "yup_sha256": sha256_file(yup_path),
        "yup_bytes": len(yup_bytes),
        "classes_emitted": sorted(parts_by_class.keys()),
        "part_count": len(scene.parts),
    }
