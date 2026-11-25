"""Optimized, auditable, conservative Working Label Layer (B09).

Original LAS labels (`raw_labels`) are NEVER mutated. All adjustments are stored
in a parallel `work_labels` array with per-point `rule_ids` and structured audit
records. Ambiguous clusters are conservatively preserved and flagged.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from ipcr.config import WorkingLabelConfig
from ipcr.types import SceneCloud, WorkingLabelAdjustment, WorkingLabelState


RULE_UNTOUCHED = 0
RULE_WL_A_CEILING_0_TO_11 = 1
RULE_WL_B_ISOLATED_NOISE = 2
RULE_WL_C1_FLUSH_COLUMN_TO_WALL = 3
RULE_WL_C2_MIXED_DOMINANCE = 4
RULE_WL_D_STAIR_LANDING_1_TO_13 = 5


def _bbox_summary(pts: np.ndarray) -> tuple[list[float], list[float]]:
    mn = np.round(pts.min(axis=0), 4).tolist()
    mx = np.round(pts.max(axis=0), 4).tolist()
    return mn, mx


def _estimate_primary_floor_z(points: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Return (primary_floor_z, approx_floor_area_m2) from class 1 points."""
    floor_pts = points[labels == 1]
    if len(floor_pts) == 0:
        return float(np.percentile(points[:, 2], 5)), 0.0
    # Use 3cm Z histogram peak to resist multi-level floors or low noise
    z = floor_pts[:, 2]
    zmin, zmax = float(z.min()), float(z.max())
    if zmax - zmin < 0.06:
        z_floor = float(np.median(z))
    else:
        bins = np.arange(zmin - 0.015, zmax + 0.045, 0.03)
        hist, edges = np.histogram(z, bins=bins)
        peak_idx = int(np.argmax(hist))
        z_center = 0.5 * (edges[peak_idx] + edges[peak_idx + 1])
        band = z[np.abs(z - z_center) <= 0.08]
        z_floor = float(np.median(band)) if len(band) else z_center

    in_floor = floor_pts[np.abs(floor_pts[:, 2] - z_floor) <= 0.10]
    if len(in_floor) == 0:
        return z_floor, 0.0
    vox = 0.10
    cells = np.unique(np.floor(in_floor[:, :2] / vox).astype(np.int64), axis=0)
    area = float(len(cells)) * (vox * vox)
    return z_floor, area


def _voxel_connected_components_3d(points: np.ndarray, voxel_size: float) -> list[np.ndarray]:
    """Fast vectorized 26-connected voxel clustering returning arrays of point indices."""
    n_pts = len(points)
    if n_pts == 0:
        return []
    coords = np.floor(points / voxel_size).astype(np.int64)
    # Shift to non-negative ints and pack into single int64 key for O(N log N) C-level grouping
    cmin = coords.min(axis=0)
    shifted = coords - cmin
    spans = shifted.max(axis=0) + 2
    stride_y = int(spans[2])
    stride_x = int(spans[1]) * stride_y

    if stride_x * int(spans[0]) < (1 << 60):
        keys = shifted[:, 0] * stride_x + shifted[:, 1] * stride_y + shifted[:, 2]
        order = np.argsort(keys, kind="mergesort")
        sorted_keys = keys[order]
        change = np.empty(n_pts, dtype=bool)
        change[0] = True
        change[1:] = sorted_keys[1:] != sorted_keys[:-1]
        group_starts = np.flatnonzero(change)
        uniq_keys = sorted_keys[group_starts]
        # Decode unique voxel coords
        uz = uniq_keys % stride_y
        rem = uniq_keys // stride_y
        uy = rem % int(spans[1])
        ux = rem // int(spans[1])
        uniq_coords = np.column_stack([ux, uy, uz])
        key_to_gid = {int(k): g for g, k in enumerate(uniq_keys.tolist())}

        # Precompute neighbor key offsets
        nb_offsets = [
            dx * stride_x + dy * stride_y + dz
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        n_vox = len(uniq_keys)
        visited = np.zeros(n_vox, dtype=bool)
        group_ends = np.append(group_starts[1:], n_pts)
        components: list[np.ndarray] = []

        for start_g in range(n_vox):
            if visited[start_g]:
                continue
            q: deque[int] = deque([start_g])
            visited[start_g] = True
            comp_groups: list[int] = []
            while q:
                g = q.popleft()
                comp_groups.append(g)
                k0 = int(uniq_keys[g])
                x0, y0, z0 = int(uniq_coords[g, 0]), int(uniq_coords[g, 1]), int(uniq_coords[g, 2])
                for off in nb_offsets:
                    nk = k0 + off
                    ng = key_to_gid.get(nk)
                    if ng is not None and not visited[ng]:
                        # Guard wrap-around across axis boundary
                        if (
                            abs(int(uniq_coords[ng, 0]) - x0) <= 1
                            and abs(int(uniq_coords[ng, 1]) - y0) <= 1
                            and abs(int(uniq_coords[ng, 2]) - z0) <= 1
                        ):
                            visited[ng] = True
                            q.append(ng)
            slices = [order[group_starts[g] : group_ends[g]] for g in comp_groups]
            components.append(np.concatenate(slices).astype(np.int64))
        components.sort(key=len, reverse=True)
        return components

    # Fallback for astronomical coordinate spans
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    order = np.argsort(inv, kind="mergesort")
    counts = np.bincount(inv)
    starts = np.cumsum(np.concatenate(([0], counts[:-1])))
    ends = starts + counts
    coord_to_gid = {(int(c[0]), int(c[1]), int(c[2])): i for i, c in enumerate(uniq)}
    visited = np.zeros(len(uniq), dtype=bool)
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if dx or dy or dz]
    components = []
    for sg in range(len(uniq)):
        if visited[sg]:
            continue
        q = deque([sg])
        visited[sg] = True
        cgroups: list[int] = []
        while q:
            g = q.popleft()
            cgroups.append(g)
            cx, cy, cz = int(uniq[g, 0]), int(uniq[g, 1]), int(uniq[g, 2])
            for dx, dy, dz in offsets:
                ng = coord_to_gid.get((cx + dx, cy + dy, cz + dz))
                if ng is not None and not visited[ng]:
                    visited[ng] = True
                    q.append(ng)
        components.append(np.concatenate([order[starts[g] : ends[g]] for g in cgroups]).astype(np.int64))
    components.sort(key=len, reverse=True)
    return components


def _fit_plane_svd(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid, unit_normal) from SVD plane fit."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    return centroid, normal


def _extract_dominant_vertical_wall_planes(
    wall_pts: np.ndarray,
    *,
    voxel_size: float = 0.08,
    plane_dist: float = 0.05,
    min_plane_voxels: int = 80,
    max_planes: int = 16,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Extract major vertical wall planes (centroid, normal_3d, bbox_min, bbox_max)."""
    if len(wall_pts) < 200:
        return []
    # Deterministic voxel downsampling
    q = np.floor(wall_pts / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(q, axis=0, return_index=True)
    pts = wall_pts[np.sort(unique_idx)]
    if len(pts) < min_plane_voxels:
        return []

    import open3d as o3d

    planes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    remaining = pts.copy()
    o3d.utility.random.seed(20261124)
    for _ in range(max_planes):
        if len(remaining) < min_plane_voxels:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(remaining)
        model, inliers = pcd.segment_plane(distance_threshold=plane_dist, ransac_n=3, num_iterations=600)
        if len(inliers) < min_plane_voxels:
            break
        inlier_pts = remaining[inliers]
        n = np.asarray(model[:3], dtype=np.float64)
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        if abs(float(n[2])) <= 0.25:
            # Re-fit cleanly with SVD
            c, n_svd = _fit_plane_svd(inlier_pts)
            if abs(float(n_svd[2])) <= 0.25:
                planes.append((c, n_svd, inlier_pts.min(axis=0), inlier_pts.max(axis=0)))
        mask = np.ones(len(remaining), dtype=bool)
        mask[inliers] = False
        remaining = remaining[mask]
    return planes


def _apply_wl_a_ceiling(
    points: np.ndarray,
    raw_labels: np.ndarray,
    work_labels: np.ndarray,
    rule_ids: np.ndarray,
    adjustments: list[WorkingLabelAdjustment],
    z_floor: float,
    floor_area: float,
    cfg: WorkingLabelConfig,
) -> None:
    """WL-A: Promote ONLY high-elevation continuous planar class-0 slabs to class 11."""
    idx0 = np.flatnonzero(work_labels == 0)
    if len(idx0) < 200:
        return
    z0 = points[idx0, 2]
    high_mask = z0 >= (z_floor + cfg.ceiling_min_height_above_floor_m)
    high_idx = idx0[high_mask]
    if len(high_idx) < 200:
        return

    # Find horizontal elevation peaks via 3cm Z histogram (supports multi-height ceilings!)
    zh = points[high_idx, 2]
    zmin, zmax = float(zh.min()), float(zh.max())
    bins = np.arange(zmin - 0.015, zmax + 0.045, 0.03)
    if len(bins) < 2:
        bins = np.array([zmin - 0.03, zmax + 0.03])

    remaining_mask = np.ones(len(high_idx), dtype=bool)
    # Primary ceiling uses cfg.ceiling_min_floor_area_ratio; secondary wing ceilings (e.g. lower entrance)
    # require >= 2.0 m^2 and >= 4% of floor area so low-ceiling corridors are also cleanly promoted.
    primary_min_area = max(cfg.ceiling_min_area_m2, floor_area * cfg.ceiling_min_floor_area_ratio)
    secondary_min_area = max(cfg.ceiling_min_area_m2, floor_area * 0.04)
    promoted_slabs = 0
    primary_xy_cells: set[tuple[int, int]] = set()
    primary_z: float | None = None
    promoted_planes: list[tuple[float, set[tuple[int, int]]]] = []
    # Precompute 0.50m XY interior coverage of floor (1) + stairs (13) so higher secondary ceilings
    # must sit over genuine interior floor/stair space (blocks outdoor/rooftop overhangs outside the room!).
    int_pts = points[(work_labels == 1) | (work_labels == 13)]
    interior_cells_50: set[tuple[int, int]] = set()
    if len(int_pts) > 0:
        base_ic = np.unique(np.floor(int_pts[:, :2] / 0.50).astype(np.int64), axis=0)
        interior_cells_50 = {
            (int(c[0]) + dx, int(c[1]) + dy)
            for c in base_ic
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        }

    for _ in range(5):
        cand_local = np.flatnonzero(remaining_mask)
        if len(cand_local) < 200:
            break
        zc = points[high_idx[cand_local], 2]
        hist, edges = np.histogram(zc, bins=bins)
        peak_bin = int(np.argmax(hist))
        if int(hist[peak_bin]) < 100:
            break
        z_peak = 0.5 * (edges[peak_bin] + edges[peak_bin + 1])
        slab_local = cand_local[np.abs(zc - z_peak) <= 0.10]
        if len(slab_local) < 200:
            remaining_mask[slab_local] = False
            continue

        slab_pts = points[high_idx[slab_local]]
        centroid, normal = _fit_plane_svd(slab_pts)
        if abs(float(normal[2])) < cfg.ceiling_normal_min_abs_z:
            remaining_mask[slab_local] = False
            continue

        # Refine to exact planar inliers within ceiling_plane_distance_m
        all_cand_pts = points[high_idx[cand_local]]
        dists = np.abs((all_cand_pts - centroid) @ normal)
        inlier_local = cand_local[dists <= cfg.ceiling_plane_distance_m]
        inlier_global = high_idx[inlier_local]
        inlier_pts = points[inlier_global]

        vox = cfg.ceiling_voxel_size_m
        cells = np.unique(np.floor(inlier_pts[:, :2] / vox).astype(np.int64), axis=0)
        area = float(len(cells)) * (vox * vox)
        remaining_mask[inlier_local] = False

        min_area = primary_min_area if promoted_slabs == 0 else secondary_min_area
        if area < min_area:
            continue

        coarse_cells = {
            (int(c[0]), int(c[1]))
            for c in np.unique(np.floor(inlier_pts[:, :2] / 0.25).astype(np.int64), axis=0)
        }
        if promoted_slabs == 0:
            # Dilate primary ceiling XY footprint by 1 coarse cell (0.25m) to close narrow beam/pipe scan gaps.
            primary_xy_cells = {
                (cx + dx, cy + dy)
                for (cx, cy) in coarse_cells
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
            }
            primary_z = float(centroid[2])
        else:
            # Secondary wing/stairwell ceiling (lower OR higher than primary!) must differ by >=0.65m,
            # sit >=50% outside the primary ceiling footprint, AND (when higher) cover >=55% interior floor/stair space!
            outside_ratio = len(coarse_cells - primary_xy_cells) / max(len(coarse_cells), 1)
            if primary_z is not None and abs(float(centroid[2]) - primary_z) < 0.65:
                continue
            if outside_ratio < 0.50:
                continue
            is_higher = primary_z is not None and float(centroid[2]) > primary_z + 0.65
            if is_higher:
                if area < 4.0:
                    continue
                if interior_cells_50:
                    cand_cells_50 = {
                        (int(c[0]), int(c[1]))
                        for c in np.unique(np.floor(inlier_pts[:, :2] / 0.50).astype(np.int64), axis=0)
                    }
                    in_room_ratio = len(cand_cells_50 & interior_cells_50) / max(len(cand_cells_50), 1)
                    if in_room_ratio < 0.55:
                        continue

        work_labels[inlier_global] = 11
        rule_ids[inlier_global] = RULE_WL_A_CEILING_0_TO_11
        promoted_slabs += 1
        promoted_planes.append((float(centroid[2]), coarse_cells))
        mn, mx = _bbox_summary(inlier_pts)
        adjustments.append(
            WorkingLabelAdjustment(
                rule_id=RULE_WL_A_CEILING_0_TO_11,
                rule_name="WL-A_high_plane_ceiling_promotion",
                raw_class=0,
                work_class=11,
                point_count=int(len(inlier_global)),
                bbox_min_local=mn,
                bbox_max_local=mx,
                evidence={
                    "plane_z_mean_local": round(float(centroid[2]), 4),
                    "normal_abs_z": round(abs(float(normal[2])), 5),
                    "occupied_area_m2": round(area, 3),
                    "min_required_area_m2": round(min_area, 3),
                    "height_above_floor_m": round(float(centroid[2] - z_floor), 3),
                },
            )
        )

    # WL-A2: Recover AI-mislabeled ceiling patches tagged as class 1 (floor) that are coplanar
    # with a verified ceiling plane (|z - cz| <= 0.12m) AND have no upper room/ceiling above them.
    # Real upper-storey floors always have walls/furniture/ceiling >= 0.35m above them, so they are untouched.
    if not promoted_planes:
        pre_c = points[work_labels == 11]
        if len(pre_c) >= 500:
            cz_pre = float(np.median(pre_c[:, 2]))
            if cz_pre >= z_floor + cfg.ceiling_min_height_above_floor_m:
                c_cells_pre = {
                    (int(c[0]), int(c[1]))
                    for c in np.unique(np.floor(pre_c[:, :2] / 0.25).astype(np.int64), axis=0)
                }
                promoted_planes.append((cz_pre, c_cells_pre))

    floor_idx = np.flatnonzero(work_labels == 1)
    if len(floor_idx) > 0 and promoted_planes:
        f_pts = points[floor_idx]
        col_vox = 0.50
        all_col_xy = np.floor(points[:, :2] / col_vox).astype(np.int64)
        for cz, _c_cells in promoted_planes:
            if cz < z_floor + cfg.ceiling_min_height_above_floor_m:
                continue
            coplanar_local = np.flatnonzero(np.abs(f_pts[:, 2] - cz) <= 0.12)
            if len(coplanar_local) < 300:
                continue
            cand_pts = f_pts[coplanar_local]
            cand_cells_xy = np.floor(cand_pts[:, :2] / col_vox).astype(np.int64)
            uniq_cand_cells = {
                (int(c[0]), int(c[1])) for c in np.unique(cand_cells_xy, axis=0)
            }
            if len(uniq_cand_cells) * (col_vox * col_vox) < 1.0:
                continue

            # Check whether a real upper room exists above these XY columns (points > cz + 0.35m)
            above_mask = points[:, 2] > cz + 0.35
            occupied_above_cells: set[tuple[int, int]] = set()
            if np.any(above_mask):
                ab_xy = all_col_xy[above_mask]
                uniq_ab, counts_ab = np.unique(ab_xy, axis=0, return_counts=True)
                # Require >= 25 points in a 0.5m column above cz + 0.35m to count as a genuine upper storey
                for cell_xy, cnt in zip(uniq_ab, counts_ab):
                    if int(cnt) >= 25:
                        occupied_above_cells.add((int(cell_xy[0]), int(cell_xy[1])))

            keep_mask = np.array(
                [(int(c[0]), int(c[1])) not in occupied_above_cells for c in cand_cells_xy],
                dtype=bool,
            )
            promote_local = coplanar_local[keep_mask]
            if len(promote_local) < 300:
                continue
            promote_global = floor_idx[promote_local]
            p_pts = points[promote_global]
            p_vox = cfg.ceiling_voxel_size_m
            p_area = float(len(np.unique(np.floor(p_pts[:, :2] / p_vox).astype(np.int64), axis=0))) * (p_vox * p_vox)
            if p_area < 1.0 or abs(float(np.median(p_pts[:, 2])) - cz) > 0.08:
                continue

            work_labels[promote_global] = 11
            rule_ids[promote_global] = RULE_WL_A_CEILING_0_TO_11
            mn, mx = _bbox_summary(p_pts)
            adjustments.append(
                WorkingLabelAdjustment(
                    rule_id=RULE_WL_A_CEILING_0_TO_11,
                    rule_name="WL-A2_ceiling_mislabeled_as_floor",
                    raw_class=1,
                    work_class=11,
                    point_count=int(len(promote_global)),
                    bbox_min_local=mn,
                    bbox_max_local=mx,
                    evidence={
                        "matched_ceiling_z_local": round(float(cz), 4),
                        "promoted_z_median_local": round(float(np.median(p_pts[:, 2])), 4),
                        "occupied_area_m2": round(p_area, 3),
                    },
                )
            )
            # Refresh remaining floor_idx for any subsequent ceiling plane
            floor_idx = np.flatnonzero(work_labels == 1)
            if len(floor_idx) == 0:
                break
            f_pts = points[floor_idx]


def _apply_wl_b_isolated_noise(
    points: np.ndarray,
    work_labels: np.ndarray,
    rule_ids: np.ndarray,
    adjustments: list[WorkingLabelAdjustment],
    z_floor: float,
    cfg: WorkingLabelConfig,
) -> None:
    """WL-B: Quarantine physically impossible micro-specks in structural classes."""
    # 1. Floating tiny stair (13) or railing (14) specks
    for cid, label_name in ((13, "stairs"), (14, "railing")):
        idx = np.flatnonzero(work_labels == cid)
        if len(idx) == 0:
            continue
        pts = points[idx]
        z_span = float(pts[:, 2].max() - pts[:, 2].min())
        z_min = float(pts[:, 2].min())
        is_tiny = len(idx) < cfg.stair_railing_min_points
        is_floating_flat = (z_span < cfg.stair_railing_min_vertical_span_m) and (z_min > z_floor + 0.50)
        if is_tiny or is_floating_flat:
            work_labels[idx] = 0
            rule_ids[idx] = RULE_WL_B_ISOLATED_NOISE
            mn, mx = _bbox_summary(pts)
            adjustments.append(
                WorkingLabelAdjustment(
                    rule_id=RULE_WL_B_ISOLATED_NOISE,
                    rule_name=f"WL-B_quarantine_impossible_{label_name}",
                    raw_class=cid,
                    work_class=0,
                    point_count=int(len(idx)),
                    bbox_min_local=mn,
                    bbox_max_local=mx,
                    evidence={
                        "point_count": int(len(idx)),
                        "min_points_threshold": cfg.stair_railing_min_points,
                        "z_span_m": round(z_span, 4),
                        "z_min_above_floor_m": round(z_min - z_floor, 4),
                    },
                )
            )

    # 2. Isolated below-floor noise in class 1 (preserving real lower storeys!)
    idx1 = np.flatnonzero(work_labels == 1)
    if len(idx1) > 0:
        below_mask = points[idx1, 2] < (z_floor - cfg.floor_outlier_below_m)
        below_idx = idx1[below_mask]
        if 0 < len(below_idx) < 0.02 * len(idx1):
            bpts = points[below_idx]
            cells = np.unique(np.floor(bpts[:, :2] / 0.10).astype(np.int64), axis=0)
            b_area = float(len(cells)) * 0.01
            if b_area < 2.0:
                work_labels[below_idx] = 0
                rule_ids[below_idx] = RULE_WL_B_ISOLATED_NOISE
                mn, mx = _bbox_summary(bpts)
                adjustments.append(
                    WorkingLabelAdjustment(
                        rule_id=RULE_WL_B_ISOLATED_NOISE,
                        rule_name="WL-B_quarantine_below_floor_noise",
                        raw_class=1,
                        work_class=0,
                        point_count=int(len(below_idx)),
                        bbox_min_local=mn,
                        bbox_max_local=mx,
                        evidence={"area_m2": round(b_area, 3), "ratio_of_floor": round(len(below_idx) / len(idx1), 6)},
                    )
                )

    # 3. Isolated low-elevation noise in class 3 (beams should be overhead, not on floor)
    idx3 = np.flatnonzero(work_labels == 3)
    if len(idx3) > 0:
        low_mask = points[idx3, 2] < (z_floor + cfg.beam_outlier_below_floor_clearance_m)
        low_idx = idx3[low_mask]
        if 0 < len(low_idx) < 0.05 * len(idx3):
            lpts = points[low_idx]
            work_labels[low_idx] = 0
            rule_ids[low_idx] = RULE_WL_B_ISOLATED_NOISE
            mn, mx = _bbox_summary(lpts)
            adjustments.append(
                WorkingLabelAdjustment(
                    rule_id=RULE_WL_B_ISOLATED_NOISE,
                    rule_name="WL-B_quarantine_low_beam_noise",
                    raw_class=3,
                    work_class=0,
                    point_count=int(len(low_idx)),
                    bbox_min_local=mn,
                    bbox_max_local=mx,
                    evidence={
                        "ratio_of_beam": round(len(low_idx) / len(idx3), 6),
                        "clearance_threshold_m": cfg.beam_outlier_below_floor_clearance_m,
                    },
                )
            )


def _apply_wl_c1_wall_column_mix(
    points: np.ndarray,
    work_labels: np.ndarray,
    rule_ids: np.ndarray,
    adjustments: list[WorkingLabelAdjustment],
    ambiguous: list[dict[str, Any]],
    cfg: WorkingLabelConfig,
) -> None:
    """WL-C1: Reassign flush thin wall-bleed patches mislabeled as class 4 back to wall (2)."""
    idx4 = np.flatnonzero(work_labels == 4)
    idx2 = np.flatnonzero(work_labels == 2)
    if len(idx4) == 0 or len(idx2) < 200:
        return

    wall_planes = _extract_dominant_vertical_wall_planes(points[idx2])
    if not wall_planes:
        return

    col_pts = points[idx4]
    comps = _voxel_connected_components_3d(col_pts, voxel_size=0.08)
    for comp_id, local_indices in enumerate(comps):
        global_indices = idx4[local_indices]
        cpts = points[global_indices]
        cmin, cmax = cpts.min(axis=0), cpts.max(axis=0)

        best_flush = 0.0
        best_protrusion = 999.0
        matched_plane_idx = -1

        for pi, (wc, wn, wmin, wmax) in enumerate(wall_planes):
            # Must overlap the wall's XY bounding extent within 0.25m
            if np.any(cmax[:2] < wmin[:2] - 0.25) or np.any(cmin[:2] > wmax[:2] + 0.25):
                continue
            dists = np.abs((cpts - wc) @ wn)
            flush_ratio = float(np.mean(dists <= cfg.column_wall_flush_threshold_m))
            protrusion = float(np.percentile(dists, 95))
            if flush_ratio > best_flush:
                best_flush = flush_ratio
                best_protrusion = protrusion
                matched_plane_idx = pi

        if matched_plane_idx < 0:
            continue

        if best_flush >= cfg.column_wall_flush_ratio and best_protrusion < cfg.column_min_protrusion_m:
            work_labels[global_indices] = 2
            rule_ids[global_indices] = RULE_WL_C1_FLUSH_COLUMN_TO_WALL
            mn, mx = _bbox_summary(cpts)
            adjustments.append(
                WorkingLabelAdjustment(
                    rule_id=RULE_WL_C1_FLUSH_COLUMN_TO_WALL,
                    rule_name="WL-C1_flush_column_patch_to_wall",
                    raw_class=4,
                    work_class=2,
                    point_count=int(len(global_indices)),
                    bbox_min_local=mn,
                    bbox_max_local=mx,
                    evidence={
                        "component_id": comp_id,
                        "flush_ratio": round(best_flush, 4),
                        "p95_protrusion_m": round(best_protrusion, 4),
                        "matched_wall_plane_index": matched_plane_idx,
                    },
                )
            )
        elif best_flush >= 0.78 and best_protrusion < cfg.column_min_protrusion_m * 1.5:
            mn, mx = _bbox_summary(cpts)
            ambiguous.append(
                {
                    "rule_name": "WL-C1_wall_column_ambiguous",
                    "raw_class": 4,
                    "kept_class": 4,
                    "point_count": int(len(global_indices)),
                    "bbox_min_local": mn,
                    "bbox_max_local": mx,
                    "flush_ratio": round(best_flush, 4),
                    "p95_protrusion_m": round(best_protrusion, 4),
                }
            )


def resolve_mixed_component_dominance(
    labels_in_component: np.ndarray,
    *,
    dominance_ratio: float = 0.90,
) -> dict[str, Any]:
    """WL-C2 helper: conservatively resolve minor boundary bleed in a physical object cluster.

    Returns decision='relabel' only when one class strictly dominates (>= dominance_ratio).
    Returns decision='ambiguous_keep' when mixed between (1 - dominance_ratio) and dominance_ratio.
    """
    u, c = np.unique(labels_in_component, return_counts=True)
    if len(u) <= 1:
        return {"decision": "pure", "dominant_class": int(u[0]) if len(u) else 0, "dominant_ratio": 1.0}
    order = np.argsort(c)[::-1]
    dom_cls = int(u[order[0]])
    dom_ratio = float(c[order[0]] / len(labels_in_component))
    counts = {int(k): int(v) for k, v in zip(u, c)}
    if dom_ratio >= dominance_ratio:
        return {
            "decision": "relabel_minor_bleed",
            "dominant_class": dom_cls,
            "dominant_ratio": round(dom_ratio, 4),
            "counts": counts,
        }
    return {
        "decision": "ambiguous_keep",
        "dominant_class": dom_cls,
        "dominant_ratio": round(dom_ratio, 4),
        "counts": counts,
    }


def _apply_wl_d_stair_landing_merge(
    points: np.ndarray,
    work_labels: np.ndarray,
    rule_ids: np.ndarray,
    adjustments: list[WorkingLabelAdjustment],
    z_floor: float,
) -> None:
    """WL-D: Merge elevated compact horizontal floor patches sandwiched between stair flights into class 13.

    AI segmenters frequently label flat turn landings between U/L-turn stair flights as class 1 (floor),
    causing 30cm step discontinuities or red-floor vs orange-stair splits. Merging ONLY compact (<= 8 m^2),
    flat, elevated clusters strictly inside the staircase Z-span and touching class 13 (< 0.35 m in 3D)
    keeps real upper-storey room floors (large area or at the top of stairs, e.g. DCJS 2F) 100% untouched.
    """
    stair_idx = np.flatnonzero(work_labels == 13)
    if len(stair_idx) < 1000:
        return
    stair_pts = points[stair_idx]
    z_st_lo = float(np.percentile(stair_pts[:, 2], 2))
    z_st_hi = float(np.percentile(stair_pts[:, 2], 98))
    if z_st_hi - z_st_lo < 1.10:
        return

    elev_f_idx = np.flatnonzero((work_labels == 1) & (points[:, 2] > z_floor + 0.40))
    if len(elev_f_idx) < 300:
        return
    elev_pts = points[elev_f_idx]

    from scipy.spatial import cKDTree

    st_tree = cKDTree(stair_pts)
    comps = _voxel_connected_components_3d(elev_pts, voxel_size=0.15)
    for comp in comps:
        if len(comp) < 300:
            continue
        sub = elev_pts[comp]
        z_med = float(np.median(sub[:, 2]))
        z_span = float(np.percentile(sub[:, 2], 95) - np.percentile(sub[:, 2], 5))
        if z_span > 0.38:
            continue
        # Must sit strictly BETWEEN stair flights (NOT the primary floor and NOT the top 2F room floor!)
        if not (z_st_lo + 0.30 <= z_med <= z_st_hi - 0.25):
            continue
        vox_cells = len(np.unique(np.floor(sub[:, :2] / 0.10).astype(np.int64), axis=0))
        area = float(vox_cells) * 0.01
        if not (0.40 <= area <= 8.0):
            continue
        dists, _ = st_tree.query(sub, k=1)
        near_cnt = int(np.sum(dists <= 0.35))
        if near_cnt < 200:
            continue

        global_idx = elev_f_idx[comp]
        work_labels[global_idx] = 13
        rule_ids[global_idx] = RULE_WL_D_STAIR_LANDING_1_TO_13
        mn, mx = _bbox_summary(sub)
        adjustments.append(
            WorkingLabelAdjustment(
                rule_id=RULE_WL_D_STAIR_LANDING_1_TO_13,
                rule_name="WL-D_stair_landing_floor_to_stairs",
                raw_class=1,
                work_class=13,
                point_count=int(len(global_idx)),
                bbox_min_local=mn,
                bbox_max_local=mx,
                evidence={
                    "z_median_local": round(z_med, 4),
                    "z_span_p5_p95_m": round(z_span, 4),
                    "occupied_area_m2": round(area, 3),
                    "near_stair_points_35cm": near_cnt,
                },
            )
        )


def build_working_labels(
    cloud: SceneCloud,
    cfg: WorkingLabelConfig | None = None,
) -> WorkingLabelState:
    """Compute auditable, non-destructive working labels from a read-only SceneCloud."""
    wcfg = cfg or WorkingLabelConfig()
    raw = cloud.raw_labels
    work = raw.copy()
    rule_ids = np.zeros(len(raw), dtype=np.uint8)
    adjustments: list[WorkingLabelAdjustment] = []
    ambiguous: list[dict[str, Any]] = []

    z_floor, floor_area = _estimate_primary_floor_z(cloud.points_local, work)

    _apply_wl_a_ceiling(cloud.points_local, raw, work, rule_ids, adjustments, z_floor, floor_area, wcfg)
    _apply_wl_b_isolated_noise(cloud.points_local, work, rule_ids, adjustments, z_floor, wcfg)
    _apply_wl_c1_wall_column_mix(cloud.points_local, work, rule_ids, adjustments, ambiguous, wcfg)
    _apply_wl_d_stair_landing_merge(cloud.points_local, work, rule_ids, adjustments, z_floor)

    return WorkingLabelState(
        raw_labels=raw,
        work_labels=work,
        rule_ids=rule_ids,
        adjustments=adjustments,
        ambiguous_clusters=ambiguous,
    )
