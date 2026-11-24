"""B14 Door & window identity verification, vertical/horizontal fragment merge, and closed solids."""

from __future__ import annotations

from typing import Any

import numpy as np

from ipcr.architecture.frame_floors import FloorCeilingResult
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import ArchitectureConfig
from ipcr.geometry_primitives import make_oriented_box
from ipcr.types import MeshPart, OpeningSpec, WallSegment
from ipcr.working_labels import _voxel_connected_components_3d


def _match_cluster_to_host_wall(
    pts_uvz: np.ndarray,
    walls: list[WallSegment],
    max_dist_m: float,
) -> tuple[WallSegment | None, float, float, float]:
    """Find best host wall for a door/window cluster (requiring wall length >= opening width)."""
    best_wall: WallSegment | None = None
    best_score = -1.0
    best_bounds = (0.0, 0.0, 999.0)

    for w in walls:
        if w.length_m < 0.55:
            # Tiny corner-step edges are never standalone door/window hosts
            continue
        rel = pts_uvz[:, :2] - w.p0_xy[None, :]
        s = rel @ w.tangent_xy
        d = rel @ w.normal_xy
        in_d = (d >= -max_dist_m) & (d <= w.thickness_m + max_dist_m)
        in_s = (s >= -0.25) & (s <= w.length_m + 0.25)
        m = in_d & in_s
        inlier_ratio = float(np.mean(m))
        if inlier_ratio < 0.50:
            continue
        s_in = s[m]
        s_lo = float(np.clip(np.percentile(s_in, 2), 0.08, w.length_m - 0.08))
        s_hi = float(np.clip(np.percentile(s_in, 98), 0.08, w.length_m - 0.08))
        if s_hi - s_lo < 0.30:
            continue
        med_d = float(np.median(np.abs(d[m] - 0.5 * w.thickness_m)))
        # Prefer longer walls with closer perpendicular fit
        score = (inlier_ratio * min(w.length_m, 4.0)) / (0.05 + med_d)
        if score > best_score:
            best_score = score
            best_wall = w
            best_bounds = (s_lo, s_hi, med_d)

    return best_wall, best_bounds[0], best_bounds[1], best_bounds[2]


def _merge_wall_openings(openings: list[OpeningSpec], merge_s_gap_m: float = 0.22, merge_z_gap_m: float = 0.45) -> list[OpeningSpec]:
    """Merge horizontally or vertically adjacent same-class opening fragments (e.g. door leaf + transom)."""
    if len(openings) <= 1:
        return openings
    groups: dict[tuple[str, int], list[OpeningSpec]] = {}
    for o in openings:
        groups.setdefault((o.host_wall_id, o.class_id), []).append(o)

    merged_all: list[OpeningSpec] = []
    for (_wid, _cid), ops in groups.items():
        ops.sort(key=lambda x: x.s_min)
        changed = True
        while changed:
            changed = False
            out: list[OpeningSpec] = []
            used = [False] * len(ops)
            for i in range(len(ops)):
                if used[i]:
                    continue
                cur = ops[i]
                used[i] = True
                for j in range(i + 1, len(ops)):
                    if used[j]:
                        continue
                    nxt = ops[j]
                    s_gap = max(0.0, max(cur.s_min, nxt.s_min) - min(cur.s_max, nxt.s_max))
                    z_gap = max(0.0, max(cur.z_min, nxt.z_min) - min(cur.z_max, nxt.z_max))
                    s_overlap = min(cur.s_max, nxt.s_max) - max(cur.s_min, nxt.s_min)
                    z_overlap = min(cur.z_max, nxt.z_max) - max(cur.z_min, nxt.z_min)
                    # Horizontal neighbor (same band) OR vertical stack (e.g. upper transom + lower door)
                    if (s_gap <= merge_s_gap_m and z_overlap > 0.15) or (s_overlap > 0.20 and z_gap <= merge_z_gap_m):
                        cur = OpeningSpec(
                            opening_id=cur.opening_id,
                            kind=cur.kind,
                            class_id=cur.class_id,
                            host_wall_id=cur.host_wall_id,
                            s_min=min(cur.s_min, nxt.s_min),
                            s_max=max(cur.s_max, nxt.s_max),
                            z_min=min(cur.z_min, nxt.z_min),
                            z_max=max(cur.z_max, nxt.z_max),
                            point_count=cur.point_count + nxt.point_count,
                            metadata={**cur.metadata, "merged_count": cur.metadata.get("merged_count", 1) + 1},
                        )
                        used[j] = True
                        changed = True
                out.append(cur)
            ops = sorted(out, key=lambda x: x.s_min)
        merged_all.extend(ops)

    # Resolve any window-door overlap on the same wall (higher point count keeps shared strip)
    by_wall: dict[str, list[OpeningSpec]] = {}
    for o in merged_all:
        by_wall.setdefault(o.host_wall_id, []).append(o)

    final_ops: list[OpeningSpec] = []
    for _wid, ops in by_wall.items():
        ops.sort(key=lambda x: x.s_min)
        resolved: list[OpeningSpec] = []
        for o in ops:
            if resolved and o.s_min < resolved[-1].s_max + 0.04:
                prev = resolved[-1]
                if o.point_count <= prev.point_count:
                    o.s_min = prev.s_max + 0.05
                else:
                    prev.s_max = o.s_min - 0.05
            if o.s_max - o.s_min >= 0.35:
                resolved.append(o)
        final_ops.extend(resolved)
    return final_ops


def detect_and_bind_openings(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    points_local: np.ndarray,
    work_labels: np.ndarray,
    cfg: ArchitectureConfig,
) -> tuple[list[OpeningSpec], dict[str, Any]]:
    """Detect doors (6) and windows (5), verify B14 physical identity, and bind shared host openings."""
    uvz = fc.frame.pts_to_uvz(points_local)
    raw_openings: list[OpeningSpec] = []
    rejected: list[dict[str, Any]] = []

    for cid, kind, min_pts in (
        (5, "window", cfg.opening_window_min_points),
        (6, "door", cfg.opening_door_min_points),
    ):
        pts = uvz[work_labels == cid]
        if len(pts) < min_pts:
            continue
        comps = _voxel_connected_components_3d(pts, voxel_size=0.14)
        for idx_c, comp in enumerate(comps):
            if len(comp) < min_pts:
                continue
            sub = pts[comp]
            host, s_lo, s_hi, med_d = _match_cluster_to_host_wall(
                sub, shell.walls, max_dist_m=max(cfg.opening_host_max_dist_m, 0.45)
            )
            if host is None:
                rejected.append({
                    "kind": kind,
                    "class_id": cid,
                    "point_count": int(len(comp)),
                    "reason": "no_compatible_host_wall",
                })
                continue

            z_lo = float(np.percentile(sub[:, 2], 2))
            z_hi = float(np.percentile(sub[:, 2], 98))
            s_lo = max(s_lo, 0.14)  # Stay safely clear of corner miters
            s_hi = min(s_hi, host.length_m - 0.14)
            z_hi = min(z_hi, host.z_max - 0.05)

            if s_hi - s_lo < cfg.opening_min_width_m or z_hi - z_lo < 0.25:
                rejected.append({
                    "kind": kind,
                    "class_id": cid,
                    "host_wall_id": host.wall_id,
                    "point_count": int(len(comp)),
                    "reason": "below_min_fragment_size",
                })
                continue

            raw_openings.append(
                OpeningSpec(
                    opening_id=f"{kind}_raw_{len(raw_openings):02d}",
                    kind=kind,
                    class_id=cid,
                    host_wall_id=host.wall_id,
                    s_min=s_lo,
                    s_max=s_hi,
                    z_min=max(z_lo, fc.floor_z),
                    z_max=z_hi,
                    point_count=int(len(comp)),
                    metadata={"median_host_dist_m": round(med_d, 4), "raw_component_idx": idx_c},
                )
            )

    merged = _merge_wall_openings(raw_openings)

    # Apply B14 physical opening identity gate AFTER vertical/horizontal merge
    accepted: list[OpeningSpec] = []
    for o in merged:
        width = o.s_max - o.s_min
        if o.kind == "door":
            # A real door must reach near the floor (<= +0.65m above floor) and have usable height (>= 1.50m)
            if o.z_min <= fc.floor_z + 0.65:
                o.z_min = fc.floor_z
            height = o.z_max - o.z_min
            if o.z_min > fc.floor_z + 0.65 or height < 1.50 or width < 0.50:
                rejected.append({
                    "kind": "door",
                    "class_id": 6,
                    "host_wall_id": o.host_wall_id,
                    "point_count": o.point_count,
                    "width_m": round(width, 3),
                    "height_m": round(height, 3),
                    "z_min_above_floor_m": round(o.z_min - fc.floor_z, 3),
                    "reason": "failed_b14_door_physical_identity",
                })
                continue
        else:
            height = o.z_max - o.z_min
            if height < cfg.opening_min_height_m or width < cfg.opening_min_width_m:
                rejected.append({
                    "kind": "window",
                    "class_id": 5,
                    "host_wall_id": o.host_wall_id,
                    "point_count": o.point_count,
                    "width_m": round(width, 3),
                    "height_m": round(height, 3),
                    "reason": "below_min_window_size",
                })
                continue
        accepted.append(o)

    wall_by_id = {w.wall_id: w for w in shell.walls}
    for w in shell.walls:
        w.openings.clear()
    for i, op in enumerate(accepted):
        op.opening_id = f"{op.kind}_{i:02d}"
        wall_by_id[op.host_wall_id].openings.append(op)

    audit = {
        "accepted_count": len(accepted),
        "window_count": sum(1 for o in accepted if o.kind == "window"),
        "door_count": sum(1 for o in accepted if o.kind == "door"),
        "rejected_count": len(rejected),
        "openings": [
            {
                "opening_id": o.opening_id,
                "kind": o.kind,
                "host_wall_id": o.host_wall_id,
                "width_m": round(o.s_max - o.s_min, 3),
                "height_m": round(o.z_max - o.z_min, 3),
                "s_range": [round(o.s_min, 3), round(o.s_max, 3)],
                "z_range": [round(o.z_min, 3), round(o.z_max, 3)],
                "point_count": o.point_count,
            }
            for o in accepted
        ],
        "rejected": rejected,
    }
    return accepted, audit


def _box_in_wall_frame(
    fc: FloorCeilingResult,
    w: WallSegment,
    s0: float,
    s1: float,
    d0: float,
    d1: float,
    z0: float,
    z1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a closed oriented box parameterized in host wall (s, d_out, z) coordinates."""
    s_mid = 0.5 * (s0 + s1)
    d_mid = 0.5 * (d0 + d1)
    z_mid = 0.5 * (z0 + z1)
    c_uv = w.p0_xy + w.tangent_xy * s_mid + w.normal_xy * d_mid
    c_xyz = np.array([*fc.frame.uv_to_xy(c_uv), z_mid], dtype=np.float64)
    u_xyz = np.array([*fc.frame.uv_to_xy(w.tangent_xy), 0.0], dtype=np.float64)
    v_xyz = np.array([*fc.frame.uv_to_xy(w.normal_xy), 0.0], dtype=np.float64)
    w_xyz = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return make_oriented_box(c_xyz, u_xyz, v_xyz, w_xyz, (s1 - s0, d1 - d0, z1 - z0))


def build_opening_parts(
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    openings: list[OpeningSpec],
    cfg: ArchitectureConfig,
) -> list[MeshPart]:
    """Build closed non-overlapping frame & glass/leaf solids strictly inside each host opening."""
    wall_by_id = {w.wall_id: w for w in shell.walls}
    parts: list[MeshPart] = []

    for op in openings:
        w = wall_by_id[op.host_wall_id]
        gap = 0.002
        s0, s1 = op.s_min + gap, op.s_max - gap
        z0, z1 = op.z_min + gap, op.z_max - gap
        if s1 - s0 < 0.25 or z1 - z0 < 0.30:
            continue

        if op.kind == "window":
            fw = min(cfg.window_frame_width_m, 0.15 * (s1 - s0), 0.15 * (z1 - z0))
            fd = min(cfg.window_frame_depth_m, w.thickness_m - 0.004)
            d_mid = 0.5 * w.thickness_m
            d0, d1 = d_mid - 0.5 * fd, d_mid + 0.5 * fd

            for name, za, zb in (("top", z1 - fw, z1), ("bottom", z0, z0 + fw)):
                v, t = _box_in_wall_frame(fc, w, s0, s1, d0, d1, za, zb)
                parts.append(MeshPart(f"{op.opening_id}_frame_{name}", 5, "window_frame", v, t, "measured"))
            for name, sa, sb in (("left", s0, s0 + fw), ("right", s1 - fw, s1)):
                v, t = _box_in_wall_frame(fc, w, sa, sb, d0, d1, z0 + fw + gap, z1 - fw - gap)
                parts.append(MeshPart(f"{op.opening_id}_frame_{name}", 5, "window_frame", v, t, "measured"))
            gt = min(cfg.window_glass_thickness_m, fd * 0.6)
            v, t = _box_in_wall_frame(
                fc, w, s0 + fw + gap, s1 - fw - gap, d_mid - 0.5 * gt, d_mid + 0.5 * gt, z0 + fw + gap, z1 - fw - gap
            )
            parts.append(MeshPart(f"{op.opening_id}_glass", 5, "window_glass", v, t, "measured"))

        elif op.kind == "door":
            fw = min(cfg.door_frame_width_m, 0.15 * (s1 - s0), 0.12 * (z1 - z0))
            fd = min(cfg.door_frame_depth_m, w.thickness_m - 0.004)
            d_mid = 0.5 * w.thickness_m
            d0, d1 = d_mid - 0.5 * fd, d_mid + 0.5 * fd

            v, t = _box_in_wall_frame(fc, w, s0, s1, d0, d1, z1 - fw, z1)
            parts.append(MeshPart(f"{op.opening_id}_frame_top", 6, "door_frame", v, t, "measured"))
            for name, sa, sb in (("left", s0, s0 + fw), ("right", s1 - fw, s1)):
                v, t = _box_in_wall_frame(fc, w, sa, sb, d0, d1, z0, z1 - fw - gap)
                parts.append(MeshPart(f"{op.opening_id}_frame_{name}", 6, "door_frame", v, t, "measured"))
            lt = min(cfg.door_leaf_thickness_m, fd * 0.7)
            v, t = _box_in_wall_frame(
                fc, w, s0 + fw + gap, s1 - fw - gap, d_mid - 0.5 * lt, d_mid + 0.5 * lt, z0, z1 - fw - gap
            )
            parts.append(MeshPart(f"{op.opening_id}_leaf", 6, "door_leaf", v, t, "measured"))

    return parts
