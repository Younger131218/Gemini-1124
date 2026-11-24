"""Multi-view diagnostic overlay visualizer (local review PNGs, never committed to Git)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

from ipcr.architecture.frame_floors import FloorCeilingResult
from ipcr.architecture.walls_shell import ExteriorShellPlan
from ipcr.config import load_r23_palette
from ipcr.types import MeshPart, SceneCloud, WorkingLabelState


def _mesh_wire_segments(parts: list[MeshPart], class_ids: set[int]) -> list[tuple[int, np.ndarray]]:
    out: list[tuple[int, np.ndarray]] = []
    for p in parts:
        if p.class_id not in class_ids:
            continue
        out.append((p.class_id, p.vertices))
    return out


def render_arch_review_png(
    cloud: SceneCloud,
    wl_state: WorkingLabelState,
    fc: FloorCeilingResult,
    shell: ExteriorShellPlan,
    parts: list[MeshPart],
    output_path: str | Path,
) -> str:
    """Render a 4-panel local diagnostic overlay (XY plan, UV plan, XZ/YZ elevation, WL changes)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    palette = load_r23_palette()["classes"]
    rng = np.random.default_rng(20261124)

    pts = cloud.points_local
    wl = wl_state.work_labels
    uvz = fc.frame.pts_to_uvz(pts)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=130)
    fig.suptitle(f"IPCR v2 Stage-1 Architecture Review ({Path(cloud.source_path).name})", fontsize=13)

    # Panel 1: Room-aligned UV Top-Down Plan (structural point cloud vs outer/inner shell + openings + columns)
    ax = axes[0, 0]
    ax.set_title("1. Top-Down Plan (Room UV Frame): Cloud vs Shell / Columns / Openings")
    for cid in (1, 2, 4, 5, 6):
        idx = np.flatnonzero(wl == cid)
        if len(idx) == 0:
            continue
        if len(idx) > 8000:
            idx = rng.choice(idx, size=8000, replace=False)
        rgba = palette.get(str(cid), {}).get("rgba", [0.5, 0.5, 0.5, 1.0])
        ax.scatter(uvz[idx, 0], uvz[idx, 1], s=0.6, alpha=0.35, color=rgba[:3], label=f"cls {cid}")

    ix, iy = shell.inner_uv.exterior.xy
    ox, oy = shell.outer_uv.exterior.xy
    ax.plot(ix, iy, color="black", lw=2.0, label="inner shell")
    ax.plot(ox, oy, color="navy", lw=1.6, ls="--", label="outer shell")

    # Draw openings & columns in UV
    for w in shell.walls:
        for op in w.openings:
            p0 = w.p0_xy + w.tangent_xy * op.s_min
            p1 = w.p0_xy + w.tangent_xy * op.s_max
            col = "cyan" if op.kind == "window" else "orange"
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=4.5, solid_capstyle="butt")

    for p in parts:
        if p.class_id == 4:
            puv = fc.frame.pts_to_uvz(p.vertices)
            u0, v0 = puv[:, 0].min(), puv[:, 1].min()
            u1, v1 = puv[:, 0].max(), puv[:, 1].max()
            ax.plot([u0, u1, u1, u0, u0], [v0, v0, v1, v1, v0], color="magenta", lw=1.8)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m)")
    ax.legend(loc="upper right", fontsize=7, markerscale=6)
    ax.grid(True, alpha=0.25)

    # Panel 2: Side Elevation U-Z (floor, ceiling, walls, windows, doors, beams)
    ax = axes[0, 1]
    ax.set_title("2. Elevation U-Z: Cloud vs Reconstructed Structural Z-Bands")
    for cid in (1, 2, 3, 5, 6, 11):
        idx = np.flatnonzero(wl == cid)
        if len(idx) == 0:
            continue
        if len(idx) > 6000:
            idx = rng.choice(idx, size=6000, replace=False)
        rgba = palette.get(str(cid), {}).get("rgba", [0.5, 0.5, 0.5, 1.0])
        ax.scatter(uvz[idx, 0], uvz[idx, 2], s=0.5, alpha=0.28, color=rgba[:3], label=f"cls {cid}")
    ax.axhline(fc.floor_z, color="red", lw=1.5, ls="-", label=f"floor_z={fc.floor_z:.2f}")
    if fc.ceiling_z is not None:
        ax.axhline(fc.ceiling_z, color="blue", lw=1.5, ls="-", label=f"ceil_z={fc.ceiling_z:.2f}")
    ax.set_xlabel("u (m)")
    ax.set_ylabel("z (m)")
    ax.legend(loc="lower right", fontsize=7, markerscale=6)
    ax.grid(True, alpha=0.25)

    # Panel 3: Beams & Columns UV Plan
    ax = axes[1, 0]
    ax.set_title("3. Overhead Beams (cls 3) & Columns (cls 4) Clipped Inside Shell")
    idx3 = np.flatnonzero(wl == 3)
    if len(idx3) > 12000:
        idx3 = rng.choice(idx3, size=12000, replace=False)
    if len(idx3):
        ax.scatter(uvz[idx3, 0], uvz[idx3, 1], s=0.7, alpha=0.35, color="#8bc34a", label="beam cloud")
    ax.plot(ix, iy, color="black", lw=1.8)
    for p in parts:
        if p.class_id in (3, 4):
            puv = fc.frame.pts_to_uvz(p.vertices)
            # Plot triangle edges lightly
            tris = p.triangles
            color = "#2e7d32" if p.class_id == 3 else "#c2185b"
            if len(tris) <= 400:
                for a, b, c in tris:
                    ax.plot(
                        [puv[a, 0], puv[b, 0], puv[c, 0], puv[a, 0]],
                        [puv[a, 1], puv[b, 1], puv[c, 1], puv[a, 1]],
                        color=color,
                        lw=0.4,
                        alpha=0.5,
                    )
    ax.set_aspect("equal", "box")
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m)")
    ax.grid(True, alpha=0.25)

    # Panel 4: Working Label Adjustments Audit Map
    ax = axes[1, 1]
    ax.set_title(f"4. Working Labels Audit (changed={int(np.sum(wl_state.raw_labels != wl)):,} pts)")
    unchanged = np.flatnonzero(wl_state.rule_ids == 0)
    if len(unchanged) > 12000:
        unchanged = rng.choice(unchanged, size=12000, replace=False)
    ax.scatter(uvz[unchanged, 0], uvz[unchanged, 1], s=0.4, alpha=0.15, color="gray", label="untouched")
    rule_colors = {1: ("#1e88e5", "WL-A 0->11 ceil"), 2: ("#e53935", "WL-B noise->0"), 3: ("#fb8c00", "WL-C1 col->wall")}
    for rid, (col, lbl) in rule_colors.items():
        ridx = np.flatnonzero(wl_state.rule_ids == rid)
        if len(ridx) == 0:
            continue
        if len(ridx) > 8000:
            ridx = rng.choice(ridx, size=8000, replace=False)
        ax.scatter(uvz[ridx, 0], uvz[ridx, 1], s=2.0, alpha=0.7, color=col, label=f"{lbl} ({int(np.sum(wl_state.rule_ids == rid))})")
    ax.set_aspect("equal", "box")
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m)")
    ax.legend(loc="upper right", fontsize=7, markerscale=4)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    fig.savefig(str(out), dpi=130)
    plt.close(fig)
    return str(out)
