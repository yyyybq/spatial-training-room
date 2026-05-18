"""
visualize_tasks.py  —  Render top-down floor-plan visualisations for generated tasks.

Usage:
    python testbench/visualize_tasks.py \
        --scene  C:/Users/user/Desktop/0267_840790 \
        --jsonl  out/batch/T01.jsonl \
        --out    out/vis/T01 \
        [--max   5]          # max tasks to render (default: all)

Outputs:
    out/vis/T01/task_000.png
    out/vis/T01/task_001.png
    ...
    out/vis/T01/index.html  (simple gallery)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap package
# ---------------------------------------------------------------------------
THIS = Path(__file__).resolve().parent
PKG  = THIS.parent
PARENT = PKG.parent
if "spatial_training_room" not in sys.modules:
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    spec = importlib.util.spec_from_file_location(
        "spatial_training_room",
        PKG / "__init__.py",
        submodule_search_locations=[str(PKG)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["spatial_training_room"] = mod
    spec.loader.exec_module(mod)

from spatial_training_room.core.scene_context import SceneContext  # noqa: E402
import spatial_training_room.core.scene_context_ext  # noqa: E402, F401  (patches SceneContext)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fov_polygon(pos, forward, hfov_deg=60.0, length=1.5, n=15):
    """Return (x, y) arrays for a FOV cone polygon (to fill)."""
    half = math.radians(hfov_deg / 2.0)
    base_ang = math.atan2(forward[1], forward[0])
    angs = np.linspace(base_ang - half, base_ang + half, n)
    xs = [pos[0]] + [pos[0] + length * math.cos(a) for a in angs] + [pos[0]]
    ys = [pos[1]] + [pos[1] + length * math.sin(a) for a in angs] + [pos[1]]
    return xs, ys


def _draw_arrow(ax, pos, forward, color, scale=0.4, lw=1.5):
    ax.annotate(
        "",
        xy=(pos[0] + forward[0] * scale, pos[1] + forward[1] * scale),
        xytext=(pos[0], pos[1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
    )


def visualise_task(task: dict, scene_ctx: SceneContext, ax: plt.Axes):
    """Draw a single task onto ax."""

    # ---- 1. Room polygons ------------------------------------------------
    scene_ctx._ensure_room_index()
    polys = getattr(scene_ctx, "_room_polygons_by_id", {})
    room_ids = list(polys.keys()) if polys else scene_ctx.room_ids()
    colors_room = plt.cm.Pastel1(np.linspace(0, 0.9, max(len(room_ids), 1)))
    for idx, rid in enumerate(room_ids):
        if polys:
            raw = polys[rid]
            # may be numpy array or shapely polygon
            if hasattr(raw, "exterior"):
                poly = np.array(raw.exterior.coords)
            else:
                poly = np.asarray(raw)
        else:
            continue
        patch = mpatches.Polygon(poly[:, :2], closed=True,
                                 facecolor=colors_room[idx], edgecolor="gray",
                                 linewidth=0.8, alpha=0.55, zorder=1)
        ax.add_patch(patch)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        ax.text(cx, cy, str(rid), ha="center", va="center",
                fontsize=6, color="gray", zorder=5)

    # ---- 2. Object AABBs -------------------------------------------------
    for obj in scene_ctx.valid_objects:
        center = 0.5 * (np.asarray(obj.bmin) + np.asarray(obj.bmax))
        size = np.asarray(obj.bmax) - np.asarray(obj.bmin)
        w, h = float(size[0]), float(size[1])
        rect = mpatches.FancyBboxPatch(
            (center[0] - w / 2, center[1] - h / 2), w, h,
            boxstyle="round,pad=0.02",
            facecolor="lightyellow", edgecolor="#bbb", linewidth=0.5,
            alpha=0.7, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(center[0], center[1], obj.label, ha="center", va="center",
                fontsize=4.5, color="#555", zorder=6)

    # ---- 3. Expert trajectory --------------------------------------------
    traj = task.get("expert_trajectory", [])
    if traj:
        xs = [v["position"][0] for v in traj]
        ys = [v["position"][1] for v in traj]
        ax.plot(xs, ys, "-", color="royalblue", linewidth=1.2, zorder=8,
                alpha=0.7, label="trajectory")
        for i, v in enumerate(traj):
            pos = np.array(v["position"])
            tgt = np.array(v["target"])
            fwd = tgt - pos
            fn  = np.linalg.norm(fwd[:2])
            if fn > 1e-6:
                fwd = fwd[:2] / fn
            else:
                fwd = np.array([1.0, 0.0])
            alpha = 0.3 + 0.7 * (i / max(len(traj) - 1, 1))
            ax.scatter(pos[0], pos[1], s=14, color="royalblue",
                       alpha=alpha, zorder=9)

    # ---- 4. Init view ----------------------------------------------------
    iv = task.get("init_view", {})
    pos_i = np.array(iv.get("position", [0, 0, 0]))
    tgt_i = np.array(iv.get("target", pos_i + [1, 0, 0]))
    fwd_i = tgt_i[:2] - pos_i[:2]
    fn = np.linalg.norm(fwd_i)
    fwd_i = fwd_i / fn if fn > 1e-6 else np.array([1.0, 0.0])
    fx, fy = _fov_polygon(pos_i[:2], fwd_i, hfov_deg=60.0, length=1.2)
    ax.fill(fx, fy, color="green", alpha=0.18, zorder=7)
    ax.scatter(pos_i[0], pos_i[1], s=60, color="green",
               marker="o", zorder=10, label="init")
    _draw_arrow(ax, pos_i[:2], fwd_i, "green", scale=0.5)

    # ---- 5. Target view (first slot) ------------------------------------
    tv = task.get("target_view", {})
    if tv:
        pos_t = np.array(tv.get("position", [0, 0, 0]))
        tgt_t = np.array(tv.get("target", pos_t + [1, 0, 0]))
        fwd_t = tgt_t[:2] - pos_t[:2]
        fn = np.linalg.norm(fwd_t)
        fwd_t = fwd_t / fn if fn > 1e-6 else np.array([1.0, 0.0])
        fx2, fy2 = _fov_polygon(pos_t[:2], fwd_t, hfov_deg=60.0, length=1.2)
        ax.fill(fx2, fy2, color="red", alpha=0.15, zorder=7)
        ax.scatter(pos_t[0], pos_t[1], s=60, color="red",
                   marker="*", zorder=10, label="target")
        _draw_arrow(ax, pos_t[:2], fwd_t, "red", scale=0.5)

    # ---- 6. Aesthetics ---------------------------------------------------
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.legend(loc="upper right", fontsize=6)
    ax.set_axis_off()


def render_task_page(task: dict, scene_ctx: SceneContext, out_path: Path,
                     task_idx: int):
    """Render one task to a PNG file."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={"width_ratios": [3, 1]})
    ax_map, ax_info = axes

    visualise_task(task, scene_ctx, ax_map)

    # Info panel
    ax_info.axis("off")
    lines = [
        f"Task #{task_idx}",
        f"Template: {task.get('template_id','?')}  ({task.get('subclass','?')})",
        "",
        f"Q: {task.get('question','?')}",
        "",
        f"A: {task.get('answer','?')}",
        "",
        f"Steps : {task.get('num_steps','?')}",
        f"Score : {task.get('score','?'):.3f}" if isinstance(task.get('score'), float) else f"Score : {task.get('score','?')}",
        f"Cov   : {task.get('coverage','?'):.3f}" if isinstance(task.get('coverage'), float) else f"Cov   : {task.get('coverage','?')}",
    ]
    # Choices
    choices = task.get("choices", [])
    if choices:
        lines.append("")
        lines.append("Choices:")
        for c in choices:
            marker = "✓" if str(c) == str(task.get("answer", "")) else " "
            lines.append(f"  [{marker}] {c}")
    # Actions
    acts = task.get("action_descriptions", task.get("action_sequence", []))
    if acts:
        lines.append("")
        lines.append("Actions:")
        for a in acts[:20]:
            lines.append(f"  {a}")
        if len(acts) > 20:
            lines.append(f"  ... (+{len(acts)-20} more)")

    text = "\n".join(lines)
    ax_info.text(
        0.02, 0.98, text,
        transform=ax_info.transAxes,
        va="top", ha="left",
        fontsize=8,
        fontfamily="monospace",
        wrap=True,
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    fig.suptitle(task.get("question", ""), fontsize=10, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Visualise generated task JSONL as top-down PNGs + HTML gallery")
    ap.add_argument("--scene",  required=True,  help="Path to scene folder")
    ap.add_argument("--jsonl",  required=True,  help="Path to task JSONL file (or glob like out/batch/*.jsonl)")
    ap.add_argument("--out",    required=True,  help="Output directory")
    ap.add_argument("--max",    type=int, default=999, help="Max tasks per file")
    args = ap.parse_args()

    scene_ctx = SceneContext.load(args.scene)

    jsonl_paths = sorted(Path(".").glob(args.jsonl)) if "*" in args.jsonl else [Path(args.jsonl)]
    if not jsonl_paths:
        print(f"No files matched: {args.jsonl}")
        return

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_entries = []  # (template_id, png_filename, question, answer, score)

    for jpath in jsonl_paths:
        tid = jpath.stem  # e.g. T01
        out_dir = out_root / tid
        out_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        with jpath.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))

        for i, task in enumerate(tasks[: args.max]):
            png_name = f"{tid}_task_{i:03d}.png"
            png_path = out_dir / png_name
            render_task_page(task, scene_ctx, png_path, i)
            print(f"  saved {png_path}")
            all_entries.append((
                tid, f"{tid}/{png_name}",
                task.get("question", ""),
                task.get("answer", ""),
                task.get("score", 0.0),
                task.get("num_steps", 0),
            ))

    # Write HTML gallery
    html_path = out_root / "index.html"
    with html_path.open("w", encoding="utf-8") as fh:
        fh.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        fh.write("<title>Task Visualisations</title>")
        fh.write("""<style>
body { font-family: sans-serif; background: #f5f5f5; margin: 0; padding: 16px; }
h1 { color: #333; }
.grid { display: flex; flex-wrap: wrap; gap: 12px; }
.card { background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,.2);
        padding: 8px; max-width: 700px; }
.card img { width: 100%; border-radius: 4px; cursor: pointer; }
.card img:hover { opacity: 0.9; }
.meta { font-size: 12px; color: #555; margin-top: 6px; }
.q { font-weight: bold; color: #222; }
.badge { display:inline-block; background:#4a90e2; color:white; border-radius:4px;
         padding:1px 6px; font-size:11px; margin-right:4px; }
</style></head><body>""")
        fh.write(f"<h1>Task Visualisations — {len(all_entries)} tasks</h1>")
        fh.write('<div class="grid">')
        for (tid, rel_png, q, ans, score, steps) in all_entries:
            score_s = f"{score:.2f}" if isinstance(score, float) else str(score)
            fh.write(f"""<div class="card">
  <a href="{rel_png}" target="_blank"><img src="{rel_png}" loading="lazy" alt="{tid}"></a>
  <div class="meta">
    <span class="badge">{tid}</span>
    <span class="q">{q}</span><br>
    Answer: <b>{ans}</b> &nbsp;|&nbsp; Score: {score_s} &nbsp;|&nbsp; Steps: {steps}
  </div>
</div>""")
        fh.write("</div></body></html>")

    print(f"\nGallery: {html_path}")
    print(f"Total images: {len(all_entries)}")


if __name__ == "__main__":
    main()
