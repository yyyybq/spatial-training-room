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
import re
import textwrap
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
# Highlight colour palette (for objects mentioned in the question)
# ---------------------------------------------------------------------------
_HIGHLIGHT_COLORS = [
    "#e05c00",   # deep orange  (subject / first object)
    "#0080e0",   # vivid blue   (ref_b / second object)
    "#00aa44",   # vivid green  (ref_c / third object)
    "#cc00cc",   # magenta      (fourth object if any)
    "#aaaa00",   # olive        (fifth object if any)
]


def _extract_highlighted_objects(task: dict, scene_ctx: SceneContext):
    """
    Return a list of (color, label, AABB_obj) for objects mentioned in the task.

    Reads `metadata.task_instance` to find all *_id / *_label pairs,
    then maps them to actual AABB objects in the scene.
    """
    instance = task.get("metadata", {}).get("task_instance", {})
    if not instance:
        return []

    # Build id→obj lookup
    obj_by_id = {o.id: o for o in scene_ctx.valid_objects}

    # Collect (role, obj_id, label) triples in deterministic order
    result = []
    seen_ids: set[str] = set()

    # Iterate keys in sorted order so roles come out consistently
    id_keys = sorted(k for k in instance if k.endswith("_id"))
    for id_key in id_keys:
        obj_id = str(instance[id_key]) if instance[id_key] is not None else None
        if obj_id is None or obj_id in seen_ids:
            continue
        # Find a label key: strip _id suffix, try *_label sibling
        base = id_key[:-3]  # e.g. "subject", "ref_b", "obj_a"
        label_key = base + "_label"
        label = instance.get(label_key, obj_id)
        obj = obj_by_id.get(obj_id)
        if obj is None:
            continue
        seen_ids.add(obj_id)
        color = _HIGHLIGHT_COLORS[len(result) % len(_HIGHLIGHT_COLORS)]
        result.append((color, label, obj))

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fov_polygon(pos, forward, hfov_deg=60.0, length=1.8, n=20):
    """Return (x, y) arrays for a FOV cone polygon (to fill)."""
    half = math.radians(hfov_deg / 2.0)
    base_ang = math.atan2(forward[1], forward[0])
    angs = np.linspace(base_ang - half, base_ang + half, n)
    xs = [pos[0]] + [pos[0] + length * math.cos(a) for a in angs] + [pos[0]]
    ys = [pos[1]] + [pos[1] + length * math.sin(a) for a in angs] + [pos[1]]
    return xs, ys


def _draw_arrow(ax, pos, forward, color, scale=0.55, lw=2.0):
    ax.annotate(
        "",
        xy=(pos[0] + forward[0] * scale, pos[1] + forward[1] * scale),
        xytext=(pos[0], pos[1]),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw),
    )


def visualise_task(task: dict, scene_ctx: SceneContext, ax: plt.Axes,
                   highlight_objs: list):
    """Draw a single task onto ax.  highlight_objs: list of (color, label, AABB)."""

    # Build set of highlighted object ids for quick lookup
    highlight_ids = {str(o.id) for _, _, o in highlight_objs}

    # ---- 1. Room polygons ------------------------------------------------
    scene_ctx._ensure_room_index()
    polys = getattr(scene_ctx, "_room_polygons_by_id", {})
    room_ids = list(polys.keys()) if polys else scene_ctx.room_ids()
    colors_room = plt.cm.Pastel1(np.linspace(0, 0.9, max(len(room_ids), 1)))
    for idx, rid in enumerate(room_ids):
        if polys:
            raw = polys[rid]
            if hasattr(raw, "exterior"):
                poly = np.array(raw.exterior.coords)
            else:
                poly = np.asarray(raw)
        else:
            continue
        patch = mpatches.Polygon(poly[:, :2], closed=True,
                                 facecolor=colors_room[idx], edgecolor="gray",
                                 linewidth=1.0, alpha=0.55, zorder=1)
        ax.add_patch(patch)
        cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
        ax.text(cx, cy, str(rid), ha="center", va="center",
                fontsize=9, color="#666", fontweight="bold", zorder=5)

    # ---- 2a. Normal (non-highlighted) object AABBs -----------------------
    for obj in scene_ctx.valid_objects:
        if str(obj.id) in highlight_ids:
            continue
        center = 0.5 * (np.asarray(obj.bmin) + np.asarray(obj.bmax))
        size   = np.asarray(obj.bmax) - np.asarray(obj.bmin)
        w, h   = float(size[0]), float(size[1])
        rect = mpatches.FancyBboxPatch(
            (center[0] - w / 2, center[1] - h / 2), w, h,
            boxstyle="round,pad=0.02",
            facecolor="lightyellow", edgecolor="#ccc", linewidth=0.6,
            alpha=0.65, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(center[0], center[1], obj.label, ha="center", va="center",
                fontsize=6.5, color="#777", zorder=6)

    # ---- 2b. Highlighted objects (drawn on top with bold borders) --------
    for color, label, obj in highlight_objs:
        center = 0.5 * (np.asarray(obj.bmin) + np.asarray(obj.bmax))
        size   = np.asarray(obj.bmax) - np.asarray(obj.bmin)
        w, h   = float(size[0]), float(size[1])
        # filled box
        rect = mpatches.FancyBboxPatch(
            (center[0] - w / 2, center[1] - h / 2), w, h,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor=color, linewidth=2.5,
            alpha=0.30, zorder=3,
        )
        ax.add_patch(rect)
        # border ring (solid, opaque)
        border = mpatches.FancyBboxPatch(
            (center[0] - w / 2, center[1] - h / 2), w, h,
            boxstyle="round,pad=0.04",
            facecolor="none", edgecolor=color, linewidth=2.5,
            alpha=1.0, zorder=4,
        )
        ax.add_patch(border)
        # label above the box
        ax.text(center[0], center[1] + h / 2 + 0.12, label,
                ha="center", va="bottom",
                fontsize=8.5, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor=color, linewidth=1.2, alpha=0.85),
                zorder=11)

    # ---- 3. Expert trajectory --------------------------------------------
    traj = task.get("expert_trajectory", [])
    if traj:
        xs = [v["position"][0] for v in traj]
        ys = [v["position"][1] for v in traj]
        ax.plot(xs, ys, "-", color="royalblue", linewidth=1.6, zorder=8,
                alpha=0.75, label="trajectory")
        for i, v in enumerate(traj):
            pos = np.array(v["position"])
            alpha = 0.35 + 0.65 * (i / max(len(traj) - 1, 1))
            ax.scatter(pos[0], pos[1], s=22, color="royalblue",
                       alpha=alpha, zorder=9)

    # ---- 4. Init view ----------------------------------------------------
    iv = task.get("init_view", {})
    pos_i = np.array(iv.get("position", [0, 0, 0]))
    tgt_i = np.array(iv.get("target", pos_i + [1, 0, 0]))
    fwd_i = tgt_i[:2] - pos_i[:2]
    fn = np.linalg.norm(fwd_i)
    fwd_i = fwd_i / fn if fn > 1e-6 else np.array([1.0, 0.0])
    fx, fy = _fov_polygon(pos_i[:2], fwd_i, hfov_deg=60.0, length=1.5)
    ax.fill(fx, fy, color="green", alpha=0.20, zorder=7)
    ax.scatter(pos_i[0], pos_i[1], s=90, color="green",
               marker="o", zorder=10, label="init")
    _draw_arrow(ax, pos_i[:2], fwd_i, "green")

    # ---- 5. Target view --------------------------------------------------
    tv = task.get("target_view", {})
    if tv:
        pos_t = np.array(tv.get("position", [0, 0, 0]))
        tgt_t = np.array(tv.get("target", pos_t + [1, 0, 0]))
        fwd_t = tgt_t[:2] - pos_t[:2]
        fn = np.linalg.norm(fwd_t)
        fwd_t = fwd_t / fn if fn > 1e-6 else np.array([1.0, 0.0])
        fx2, fy2 = _fov_polygon(pos_t[:2], fwd_t, hfov_deg=60.0, length=1.5)
        ax.fill(fx2, fy2, color="red", alpha=0.18, zorder=7)
        ax.scatter(pos_t[0], pos_t[1], s=100, color="red",
                   marker="*", zorder=10, label="target")
        _draw_arrow(ax, pos_t[:2], fwd_t, "red")

    # ---- 6. Legend + aesthetics ------------------------------------------
    legend_handles = [
        mpatches.Patch(color="royalblue", alpha=0.7, label="trajectory"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
                   markersize=10, label="init"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
                   markersize=12, label="target"),
    ]
    for color, label, _ in highlight_objs:
        legend_handles.append(
            mpatches.Patch(facecolor=color, edgecolor=color, alpha=0.6, label=label)
        )
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              framealpha=0.85)
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_axis_off()


def render_task_page(task: dict, scene_ctx: SceneContext, out_path: Path,
                     task_idx: int):
    """Render one task to a PNG file."""
    highlight_objs = _extract_highlighted_objects(task, scene_ctx)

    # width_ratios [3,2]: map gets 60%, info panel gets 40% of the figure width
    fig, axes = plt.subplots(1, 2, figsize=(22, 11),
                             gridspec_kw={"width_ratios": [3, 2]})
    ax_map, ax_info = axes

    visualise_task(task, scene_ctx, ax_map, highlight_objs)

    # ---- Info panel: fill the whole axes with a coloured background ------
    ax_info.set_facecolor("#fefce8")          # light-yellow fill for whole panel
    for sp in ax_info.spines.values():
        sp.set_edgecolor("#bbb")
        sp.set_linewidth(1.5)
    ax_info.set_xticks([])
    ax_info.set_yticks([])

    score_s = f"{task['score']:.3f}" if isinstance(task.get("score"), float) else str(task.get("score", "?"))
    cov_s   = f"{task['coverage']:.3f}" if isinstance(task.get("coverage"), float) else str(task.get("coverage", "?"))

    # Build (text, fontsize, weight, color) rows
    q_lines = textwrap.wrap(task.get("question", "?"), width=32)
    rows = [
        (f"Task #{task_idx}", 15, "bold", "#222"),
        (f"Template: {task.get('template_id','?')}  ({task.get('subclass','?')})",
         13, "normal", "#555"),
        ("─" * 36, 11, "normal", "#bbb"),
    ]
    rows.append(("Q:", 13, "bold", "#111"))
    for ql in q_lines:
        rows.append(("  " + ql, 13, "normal", "#111"))
    rows.append(("", 6, "normal", "#000"))
    rows.append((f"A:  {task.get('answer','?')}", 14, "bold", "#b00000"))
    rows.append(("─" * 36, 11, "normal", "#bbb"))
    rows.append((f"Steps : {task.get('num_steps','?')}", 13, "normal", "#333"))
    rows.append((f"Score : {score_s}", 13, "normal", "#333"))
    rows.append((f"Cov   : {cov_s}", 13, "normal", "#333"))

    choices = task.get("choices", [])
    if choices:
        rows.append(("─" * 36, 11, "normal", "#bbb"))
        rows.append(("Choices:", 13, "bold", "#333"))
        for c in choices:
            is_ans = str(c) == str(task.get("answer", ""))
            marker = "✓" if is_ans else " "
            col = "#007700" if is_ans else "#555"
            rows.append((f"  [{marker}] {c}", 13, "normal", col))

    acts = task.get("action_descriptions", task.get("action_sequence", []))
    if acts:
        rows.append(("─" * 36, 11, "normal", "#bbb"))
        rows.append(("Actions:", 13, "bold", "#333"))
        for a in acts[:22]:
            rows.append((f"  {a}", 12, "normal", "#444"))
        if len(acts) > 22:
            rows.append((f"  … +{len(acts)-22} more", 11, "normal", "#888"))

    # Draw lines top→bottom, estimating y-step from font size.
    # axes height ≈ 10 inches (figsize=22×11 minus padding); 72 pt/inch
    # one line ≈ fs * 1.35 pt; normalised: fs*1.35 / (10*72) = fs/533
    AXES_H_PT = 10.0 * 72   # approximate
    y = 0.975
    for (txt, fs, weight, color) in rows:
        if txt == "":
            y -= fs / AXES_H_PT          # blank gap proportional to requested fs
            continue
        ax_info.text(0.04, y, txt,
                     transform=ax_info.transAxes,
                     va="top", ha="left",
                     fontsize=fs, fontfamily="monospace",
                     fontweight=weight, color=color,
                     clip_on=True)
        n_sub = txt.count("\n") + 1
        y -= fs * 1.45 * n_sub / AXES_H_PT
        if y < 0.01:
            break

    q_short = textwrap.shorten(task.get("question", ""), width=95, placeholder="…")
    fig.suptitle(q_short, fontsize=14, y=1.002, fontweight="bold")
    fig.tight_layout(pad=0.8, rect=[0, 0, 1, 0.99])
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
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
