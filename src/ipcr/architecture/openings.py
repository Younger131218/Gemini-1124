"""B14 Door & window identity verification, open-swing hinge pairing, wall-void evidence, and closed Option-A solids."""

from __future__ import annotations

from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from ipcr.architecture.frame_floors import FloorCeilingResult
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import ArchitectureConfig
from ipcr.geometry_primitives import make_oriented_box
from ipcr.types import MeshPart, OpeningSpec, WallSegment
from ipcr.working_labels import _voxel_connected_components_3d


def _safe_opening_s_bounds(
    host: WallSegment,
    wpoly: Polygon,
    s_lo: float,
    s_hi: float,
) -> tuple[float, float] | None:
    """Compute the largest [s_lo, s_hi] whose full wall-depth rectangle stays strictly inside wpoly.buffer(-0.005)."""
    safe_host = wpoly.buffer(-0.005)
    if safe_host.is_empty:
        return None
    p0, t_dir, n_dir = host.p0_xy, host.tangent_xy, host.normal_xy
    th = host.thickness_m

    # Sample along [s_lo, s_hi] at 2cm resolution to verify full-thickness containment (immune to oblique/concave corners!)
    if s_hi - s_lo < 0.25:
        return None
    n_samples = max(12, int(np.ceil((s_hi - s_lo) / 0.02)))
    s_grid = np.linspace(s_lo, s_hi, n_samples)
    from shapely import contains_xy

    # Check 3 depths across wall thickness: near inner face, mid, near outer face
    ok = np.ones(n_samples, dtype=bool)
    for frac in (0.08, 0.50, 0.92):
        pts_uv = p0[None, :] + s_grid[:, None] * t_dir[None, :] + (frac * th) * n_dir[None, :]
        ok &= contains_xy(safe_host, pts_uv[:, 0], pts_uv[:, 1])

    if not np.any(ok):
        return None
    # Pick longest contiguous valid s-run
    padded = np.concatenate(([False], ok, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diffs == 1)
    ends = np.flatnonzero(diffs == -1)
    best_i = int(np.argmax(ends - starts))
    sa = float(s_grid[starts[best_i]]) + 0.004
    sb = float(s_grid[ends[best_i] - 1]) - 0.004
    if sb - sa < 0.25:
        return None
    return sa, sb


def _match_cluster_to_host_wall(
    pts_uvz: np.ndarray,
    walls: list[WallSegment],
    max_dist_m: float,
) -> tuple[WallSegment | None, float, float, float, float]:
    """Find best wall-aligned host wall; returns (wall, s_lo, s_hi, median_perp_dist, inlier_ratio)."""
    best_wall: WallSegment | None = None
    best_score = -1.0
    best_info = (0.0, 0.0, 999.0, 0.0)

    for w in walls:
        if w.length_m < 0.55:
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
        s_lo = float(np.clip(np.percentile(s_in, 2), 0.05, w.length_m - 0.05))
        s_hi = float(np.clip(np.percentile(s_in, 98), 0.05, w.length_m - 0.05))
        if s_hi - s_lo < 0.28:
            continue
        med_d = float(np.median(np.abs(d[m] - 0.5 * w.thickness_m)))
        # Penalize clusters whose thickness perpendicular to this wall exceeds their width along it
        # (a 90-deg open door leaf sticks ~0.9m perpendicular to its true host wall!)
        d_span = float(np.percentile(d[m], 95) - np.percentile(d[m], 5))
        if d_span > (s_hi - s_lo) * 1.4 and d_span > 0.45:
            continue
        score = (inlier_ratio * min(w.length_m, 4.0)) / (0.05 + med_d)
        if score > best_score:
            best_score = score
            best_wall = w
            best_info = (s_lo, s_hi, med_d, inlier_ratio)

    return best_wall, best_info[0], best_info[1], best_info[2], best_info[3]


def _wall_void_ratio_below(
    wall_uvz: np.ndarray,
    host: WallSegment,
    s_lo: float,
    s_hi: float,
    z_floor_local: float,
    z_top_check: float,
) -> float:
    """Measure 10cm x 10cm (s, z) wall point occupancy directly beneath an upper door frame (B14)."""
    if z_top_check - z_floor_local < 0.40 or s_hi - s_lo < 0.30:
        return 1.0
    rel = wall_uvz[:, :2] - host.p0_xy[None, :]
    s = rel @ host.tangent_xy
    d = rel @ host.normal_xy
    z = wall_uvz[:, 2]

    sa, sb = s_lo + 0.08, s_hi - 0.08
    za, zb = z_floor_local + 0.25, z_top_check - 0.08
    if sb <= sa + 0.15 or zb <= za + 0.20:
        return 1.0

    m = (s >= sa) & (s <= sb) & (z >= za) & (z <= zb) & (d >= -0.18) & (d <= host.thickness_m + 0.18)
    cell = 0.10
    ns = max(1, int(np.ceil((sb - sa) / cell)))
    nz = max(1, int(np.ceil((zb - za) / cell)))
    total_cells = ns * nz
    if not np.any(m):
        return 0.0
    is_idx = np.clip(np.floor((s[m] - sa) / cell).astype(np.int64), 0, ns - 1)
    iz_idx = np.clip(np.floor((z[m] - za) / cell).astype(np.int64), 0, nz - 1)
    occupied = len(np.unique(is_idx * nz + iz_idx))
    return float(occupied / max(total_cells, 1))


def _pair_open_door_leaves(
    door_clusters: list[dict[str, Any]],
    walls: list[WallSegment],
) -> list[dict[str, Any]]:
    """Rule D1: Attach 90-deg open inward-swinging door leaves to their hinge-connected wall frame/transom."""
    # Classify each cluster: wall-aligned frame/transom vs inward-protruding leaf
    wall_aligned: list[dict[str, Any]] = []
    swing_candidates: list[dict[str, Any]] = []

    for cl in door_clusters:
        sub = cl["pts"]
        # Check if cluster is tightly aligned to ANY wall (median |d| <= 0.16 and depth span <= 0.42m)
        best_wall, s_lo, s_hi, med_d, _ir = _match_cluster_to_host_wall(sub, walls, max_dist_m=0.35)
        if best_wall is not None and med_d <= 0.18:
            cl["host"] = best_wall
            cl["s_lo"] = s_lo
            cl["s_hi"] = s_hi
            cl["med_d"] = med_d
            wall_aligned.append(cl)
        else:
            swing_candidates.append(cl)

    # Try hinge-pairing each swing candidate to a wall-aligned door frame/transom!
    unpaired_swings: list[dict[str, Any]] = []
    for sw in swing_candidates:
        sw_pts = sw["pts"]
        sw_min = sw_pts.min(axis=0)
        sw_max = sw_pts.max(axis=0)
        matched_frame = None
        best_dist = 999.0
        for fr in wall_aligned:
            fr_pts = fr["pts"]
            fr_min = fr_pts.min(axis=0)
            fr_max = fr_pts.max(axis=0)
            # 2D XY gap between bounding boxes + Z gap (hinge touches transom/jamb)
            dx = max(0.0, max(sw_min[0], fr_min[0]) - min(sw_max[0], fr_max[0]))
            dy = max(0.0, max(sw_min[1], fr_min[1]) - min(sw_max[1], fr_max[1]))
            dz = max(0.0, max(sw_min[2], fr_min[2]) - min(sw_max[2], fr_max[2]))
            dxy = float(np.hypot(dx, dy))
            if dxy <= 0.30 and dz <= 0.40:
                if dxy < best_dist:
                    best_dist = dxy
                    matched_frame = fr
        if matched_frame is not None:
            # Merge open leaf's Z extent & point count into the wall-aligned frame (Option A: keep wall-plane s_lo..s_hi!)
            matched_frame["z_lo"] = min(matched_frame["z_lo"], sw["z_lo"])
            matched_frame["z_hi"] = max(matched_frame["z_hi"], sw["z_hi"])
            matched_frame["point_count"] += sw["point_count"]
            matched_frame["open_leaf_paired"] = True
        else:
            unpaired_swings.append(sw)

    # Keep unpaired clusters for standard fallback matching
    return wall_aligned + unpaired_swings


def _merge_wall_openings(openings: list[OpeningSpec], merge_s_gap_m: float = 0.22, merge_z_gap_m: float = 0.45) -> list[OpeningSpec]:
    """Merge horizontally or vertically adjacent same-class opening fragments on the same host wall."""
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
    """Detect doors (6) and windows (5), handle open-swing leaves & wall-void doors, and bind safely."""
    uvz = fc.frame.pts_to_uvz(points_local)
    wall_uvz = uvz[work_labels == 2]
    wall_by_id = {w.wall_id: w for w in shell.walls}
    raw_openings: list[OpeningSpec] = []
    rejected: list[dict[str, Any]] = []

    # 1. Windows (class 5)
    win_pts = uvz[work_labels == 5]
    if len(win_pts) >= cfg.opening_window_min_points:
        for idx_c, comp in enumerate(_voxel_connected_components_3d(win_pts, voxel_size=0.14)):
            if len(comp) < cfg.opening_window_min_points:
                continue
            sub = win_pts[comp]
            host, s_lo, s_hi, med_d, _ir = _match_cluster_to_host_wall(
                sub, shell.walls, max_dist_m=max(cfg.opening_host_max_dist_m, 0.45)
            )
            if host is None:
                rejected.append({"kind": "window", "class_id": 5, "point_count": int(len(comp)), "reason": "no_compatible_host_wall"})
                continue
            bounds = _safe_opening_s_bounds(host, shell.wall_polys_uv[host.wall_id], s_lo, s_hi)
            if bounds is None:
                rejected.append({"kind": "window", "class_id": 5, "host_wall_id": host.wall_id, "point_count": int(len(comp)), "reason": "no_safe_span_inside_host_wall"})
                continue
            s_lo, s_hi = bounds
            z_lo = max(float(np.percentile(sub[:, 2], 2)), host.z_min)
            z_hi = min(float(np.percentile(sub[:, 2], 98)), host.z_max - 0.05)
            if s_hi - s_lo < cfg.opening_min_width_m or z_hi - z_lo < 0.25:
                rejected.append({"kind": "window", "class_id": 5, "host_wall_id": host.wall_id, "point_count": int(len(comp)), "reason": "below_min_fragment_size"})
                continue
            raw_openings.append(
                OpeningSpec(
                    opening_id=f"window_raw_{len(raw_openings):02d}",
                    kind="window",
                    class_id=5,
                    host_wall_id=host.wall_id,
                    s_min=s_lo,
                    s_max=s_hi,
                    z_min=z_lo,
                    z_max=z_hi,
                    point_count=int(len(comp)),
                    metadata={"median_host_dist_m": round(med_d, 4), "raw_component_idx": idx_c},
                )
            )

    # 2. Doors (class 6) with Rule D1 open-leaf hinge pairing
    door_pts = uvz[work_labels == 6]
    if len(door_pts) >= cfg.opening_door_min_points:
        door_clusters: list[dict[str, Any]] = []
        for idx_c, comp in enumerate(_voxel_connected_components_3d(door_pts, voxel_size=0.14)):
            if len(comp) < cfg.opening_door_min_points:
                continue
            sub = door_pts[comp]
            door_clusters.append({
                "pts": sub,
                "point_count": int(len(comp)),
                "z_lo": float(np.percentile(sub[:, 2], 2)),
                "z_hi": float(np.percentile(sub[:, 2], 98)),
                "raw_idx": idx_c,
                "open_leaf_paired": False,
            })

        paired_doors = _pair_open_door_leaves(door_clusters, shell.walls)
        for cl in paired_doors:
            host = cl.get("host")
            if host is None:
                host, s_lo, s_hi, med_d, _ir = _match_cluster_to_host_wall(
                    cl["pts"], shell.walls, max_dist_m=max(cfg.opening_host_max_dist_m, 0.45)
                )
            else:
                s_lo, s_hi, med_d = cl["s_lo"], cl["s_hi"], cl["med_d"]
            if host is None:
                rejected.append({"kind": "door", "class_id": 6, "point_count": cl["point_count"], "reason": "no_compatible_host_wall"})
                continue
            bounds = _safe_opening_s_bounds(host, shell.wall_polys_uv[host.wall_id], s_lo, s_hi)
            if bounds is None:
                rejected.append({"kind": "door", "class_id": 6, "host_wall_id": host.wall_id, "point_count": cl["point_count"], "reason": "no_safe_span_inside_host_wall"})
                continue
            s_lo, s_hi = bounds
            z_lo = max(cl["z_lo"], host.z_min)
            z_hi = min(cl["z_hi"], host.z_max - 0.05)
            if s_hi - s_lo < cfg.opening_min_width_m or z_hi - z_lo < 0.25:
                rejected.append({"kind": "door", "class_id": 6, "host_wall_id": host.wall_id, "point_count": cl["point_count"], "reason": "below_min_fragment_size"})
                continue
            raw_openings.append(
                OpeningSpec(
                    opening_id=f"door_raw_{len(raw_openings):02d}",
                    kind="door",
                    class_id=6,
                    host_wall_id=host.wall_id,
                    s_min=s_lo,
                    s_max=s_hi,
                    z_min=z_lo,
                    z_max=z_hi,
                    point_count=cl["point_count"],
                    metadata={
                        "median_host_dist_m": round(med_d, 4),
                        "open_leaf_paired": bool(cl.get("open_leaf_paired", False)),
                    },
                )
            )

    merged = _merge_wall_openings(raw_openings)

    # 3. Apply B14 physical opening identity gate + Rule D2 wall-void evidence for upper-only door frames
    accepted: list[OpeningSpec] = []
    floor_levels = [lvl.z_top for lvl in fc.floor_levels] if fc.floor_levels else [fc.floor_z]

    for o in merged:
        host = wall_by_id[o.host_wall_id]
        # Re-verify safe bounds after merge
        bounds = _safe_opening_s_bounds(host, shell.wall_polys_uv[host.wall_id], o.s_min, o.s_max)
        if bounds is None:
            rejected.append({"kind": o.kind, "host_wall_id": o.host_wall_id, "point_count": o.point_count, "reason": "no_safe_span_after_merge"})
            continue
        o.s_min, o.s_max = bounds
        width = o.s_max - o.s_min

        if o.kind == "door":
            # Find nearest floor level at or below door top
            valid_floors = [fz for fz in floor_levels if fz <= o.z_max - 1.30]
            local_floor = max(valid_floors) if valid_floors else fc.floor_z

            if o.z_min <= local_floor + 0.65:
                o.z_min = local_floor
                o.metadata["door_evidence"] = "open_leaf_paired_to_floor" if o.metadata.get("open_leaf_paired") else "direct_floor_reaching"
            else:
                # Rule D2: Check wall-void evidence below upper door transom (valid head height 1.85..2.85m above floor)
                head_above_floor = o.z_max - local_floor
                if 1.85 <= head_above_floor <= 2.85 and 0.55 <= width <= 2.60:
                    void_occ = _wall_void_ratio_below(wall_uvz, host, o.s_min, o.s_max, local_floor, o.z_min)
                    if void_occ < 0.22:
                        o.z_min = local_floor
                        o.metadata["door_evidence"] = "upper_frame_plus_wall_void"
                        o.metadata["wall_occupancy_below"] = round(void_occ, 4)
                    else:
                        rejected.append({
                            "kind": "door",
                            "class_id": 6,
                            "host_wall_id": o.host_wall_id,
                            "point_count": o.point_count,
                            "width_m": round(width, 3),
                            "wall_occupancy_below": round(void_occ, 4),
                            "reason": "failed_b14_door_solid_wall_below",
                        })
                        continue
                else:
                    rejected.append({
                        "kind": "door",
                        "class_id": 6,
                        "host_wall_id": o.host_wall_id,
                        "point_count": o.point_count,
                        "width_m": round(width, 3),
                        "height_m": round(o.z_max - o.z_min, 3),
                        "z_min_above_floor_m": round(o.z_min - local_floor, 3),
                        "reason": "failed_b14_door_physical_identity",
                    })
                    continue

            height = o.z_max - o.z_min
            if height < 1.50 or width < 0.50:
                rejected.append({
                    "kind": "door",
                    "class_id": 6,
                    "host_wall_id": o.host_wall_id,
                    "point_count": o.point_count,
                    "width_m": round(width, 3),
                    "height_m": round(height, 3),
                    "reason": "failed_b14_door_size",
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
                "evidence": o.metadata.get("door_evidence", "direct"),
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
    """Build closed Option-A frame & glass/leaf solids strictly inside each host opening."""
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
            fd = min(cfg.window_frame_depth_m, w.thickness_m - 0.006)
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
            # Option A: Closed door leaf mounted cleanly inside its wall-aligned door frame
            fw = min(cfg.door_frame_width_m, 0.15 * (s1 - s0), 0.12 * (z1 - z0))
            fd = min(cfg.door_frame_depth_m, w.thickness_m - 0.006)
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
