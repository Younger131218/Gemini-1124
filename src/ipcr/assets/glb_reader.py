"""Pure-Python GLB 2.0 reader with full glTF node-hierarchy transform baking."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_COUNT = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}


@dataclass
class GlbPrimitiveData:
    mesh_index: int
    primitive_index: int
    material_index: int | None
    positions: np.ndarray  # (V, 3) float64 in glTF root world space
    indices: np.ndarray  # (T, 3) uint32
    extras: dict[str, Any]


@dataclass
class ParsedGlb:
    gltf: dict[str, Any]
    bin_chunk: bytes
    primitives: list[GlbPrimitiveData]


def _quat_to_mat4(q: list[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = np.sqrt(x * x + y * y + z * z + w * w)
    if n > 0:
        x, y, z, w = x / n, y / n, z / n, w / n
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = 1 - 2 * (y * y + z * z)
    m[0, 1] = 2 * (x * y - z * w)
    m[0, 2] = 2 * (x * z + y * w)
    m[1, 0] = 2 * (x * y + z * w)
    m[1, 1] = 1 - 2 * (x * x + z * z)
    m[1, 2] = 2 * (y * z - x * w)
    m[2, 0] = 2 * (x * z - y * w)
    m[2, 1] = 2 * (y * z + x * w)
    m[2, 2] = 1 - 2 * (x * x + y * y)
    return m


def _node_local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        # glTF stores 4x4 matrices in column-major order
        return np.asarray(node["matrix"], dtype=np.float64).reshape((4, 4), order="F")
    t = np.eye(4, dtype=np.float64)
    if "translation" in node:
        t[:3, 3] = np.asarray(node["translation"], dtype=np.float64)
    r = _quat_to_mat4(node["rotation"]) if "rotation" in node else np.eye(4, dtype=np.float64)
    s = np.eye(4, dtype=np.float64)
    if "scale" in node:
        sx, sy, sz = [float(v) for v in node["scale"]]
        s[0, 0], s[1, 1], s[2, 2] = sx, sy, sz
    return t @ r @ s


def read_accessor(gltf: dict[str, Any], bin_chunk: bytes, accessor_index: int) -> np.ndarray:
    acc = gltf["accessors"][accessor_index]
    bv = gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_COMPONENT_DTYPE[int(acc["componentType"])])
    ncomp = _TYPE_COUNT[str(acc["type"])]
    count = int(acc["count"])
    offset = int(bv.get("byteOffset", 0)) + int(acc.get("byteOffset", 0))
    stride = int(bv.get("byteStride", 0))
    item_size = dtype.itemsize * ncomp

    if stride == 0 or stride == item_size:
        arr = np.frombuffer(bin_chunk, dtype=dtype, count=count * ncomp, offset=offset)
        arr = arr.reshape((count, ncomp)) if ncomp > 1 else arr.reshape((count,))
        return arr.copy()

    out = np.empty((count, ncomp) if ncomp > 1 else (count,), dtype=dtype)
    for i in range(count):
        pos = offset + i * stride
        vals = np.frombuffer(bin_chunk, dtype=dtype, count=ncomp, offset=pos)
        out[i] = vals if ncomp > 1 else vals[0]
    return out


def load_glb(path: str | Path) -> ParsedGlb:
    """Parse GLB 2.0 and bake full node hierarchy transforms into world positions."""
    raw = Path(path).read_bytes()
    if len(raw) < 20:
        raise ValueError(f"File too small to be valid GLB: {path}")
    magic, version, total_len = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError(f"Invalid GLB 2.0 header in {path}: magic={hex(magic)}, version={version}")
    if total_len != len(raw):
        raise ValueError(f"GLB length header {total_len} != file size {len(raw)}")

    pos = 12
    json_chunk: bytes | None = None
    bin_chunk = b""
    while pos + 8 <= len(raw):
        clen, ctype = struct.unpack_from("<II", raw, pos)
        pos += 8
        cdata = raw[pos : pos + clen]
        pos += clen
        if ctype == 0x4E4F534A:
            json_chunk = cdata
        elif ctype == 0x004E4942:
            bin_chunk = cdata

    if json_chunk is None:
        raise ValueError(f"GLB missing JSON chunk: {path}")
    gltf = json.loads(json_chunk.decode("utf-8"))

    nodes = gltf.get("nodes", [])
    scenes = gltf.get("scenes", [])
    scene_idx = int(gltf.get("scene", 0))
    root_nodes = scenes[scene_idx].get("nodes", []) if scenes else list(range(len(nodes)))

    # Traverse scene graph to collect (mesh_idx, world_matrix)
    mesh_instances: list[tuple[int, np.ndarray]] = []

    def _visit(node_idx: int, parent_mat: np.ndarray) -> None:
        node = nodes[node_idx]
        world = parent_mat @ _node_local_matrix(node)
        if "mesh" in node:
            mesh_instances.append((int(node["mesh"]), world))
        for child_idx in node.get("children", []):
            _visit(int(child_idx), world)

    for rn in root_nodes:
        _visit(int(rn), np.eye(4, dtype=np.float64))

    primitives: list[GlbPrimitiveData] = []
    meshes = gltf.get("meshes", [])
    for mesh_idx, world in mesh_instances:
        det = float(np.linalg.det(world[:3, :3]))
        mesh = meshes[mesh_idx]
        for prim_idx, prim in enumerate(mesh.get("primitives", [])):
            mode = int(prim.get("mode", 4))
            if mode != 4:
                continue
            pos_acc = prim["attributes"]["POSITION"]
            v_local = read_accessor(gltf, bin_chunk, int(pos_acc)).astype(np.float64)
            v_world = (v_local @ world[:3, :3].T) + world[:3, 3]
            if "indices" in prim:
                idx = read_accessor(gltf, bin_chunk, int(prim["indices"])).astype(np.uint32)
            else:
                idx = np.arange(len(v_local), dtype=np.uint32)
            tris = idx.reshape(-1, 3)
            if det < 0:
                tris = tris[:, [0, 2, 1]]
            primitives.append(
                GlbPrimitiveData(
                    mesh_index=mesh_idx,
                    primitive_index=prim_idx,
                    material_index=prim.get("material"),
                    positions=v_world,
                    indices=tris,
                    extras=dict(prim.get("extras", {})),
                )
            )
    return ParsedGlb(gltf=gltf, bin_chunk=bin_chunk, primitives=primitives)


def merged_glb_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all TRIANGLES primitives from a GLB baked into a single (V, 3), (T, 3) mesh."""
    parsed = load_glb(path)
    if not parsed.primitives:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint32)
    vs = []
    ts = []
    offset = 0
    for p in parsed.primitives:
        vs.append(p.positions)
        ts.append(p.indices.astype(np.uint32) + np.uint32(offset))
        offset += len(p.positions)
    return np.vstack(vs), np.vstack(ts)
