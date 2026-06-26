"""
trace_T05.py  -  End-to-end visual trace of one T05 task.

Walks through:
    1. trigger check       (distance ratio, label-distinct, same-room)
    2. region sampling     (around_pair for AB_view and AC_view)
    3. init vs target view (on top-down map with FOV cones)
    4. expert trajectory   (per-step Phi(s_t) and slot-satisfied flags)

Usage:
    python testbench/trace_T05.py \\
        --scene C:/Users/user/Desktop/0267_840790 \\
        --jsonl out/batch/T05.jsonl \\
        --task-index 0 \\
        --out    out/trace

Outputs (under out/trace/T05_task_XXX/):
    summary.md
    01_regions.png
    02_init_target.png
    03_trajectory_phi.png
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
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
PKG = THIS.parent
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
import spatial_training_room.core.scene_context_ext  # noqa: E402, F401
from spatial_training_room.evaluation.template_spec import (  # noqa: E402
    load_template,
    expand_evidence_slots,
)
from spatial_training_room.evaluation.coverage import (  # noqa: E402
    resolve_slot,
    slot_satisfied_at,
)
from spatial_training_room.evaluation.region_generators import sample_region  # noqa: E402
from spatial_training_room.evaluation.potential import potential_at  # noqa: E402


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _fov_polygon(pos, forward, hfov_deg=60.0, length=1.5, n=18):
    half = math.radians(hfov_deg / 2.0)
    base = math.atan2(forward[1], forward[0])
    angs = np.linspace(base - half, base + half, n)
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


def _draw_floorplan(ax, scene_ctx, highlight):
    scene_ctx._ensure_room_index()
    polys = getattr(scene_ctx, "_room_polygons_by_id", {})
    room_ids = list(polys.keys()) if polys else scene_ctx.room_ids()
    colors_room = plt.cm.Pastel1(np.linspace(0, 0.9, max(len(room_ids), 1)))
    for idx, rid in enumerate(room_ids):
        if not polys:
            continue
        raw = polys[rid]
        poly = np.array(raw.exterior.coords) if hasattr(raw, "exterior") else np.asarray(raw)
        ax.add_patch(mpatches.Polygon(
            poly[:, :2], closed=True,
            facecolor=colors_room[idx], edgecolor="gray",
            linewidth=1.0, alpha=0.45, zorder=1,
        ))

    for obj in scene_ctx.valid_objects:
        c = 0.5 * (np.asarray(obj.bmin) + np.asarray(obj.bmax))
        sz = np.asarray(obj.bmax) - np.asarray(obj.bmin)
        w, h = float(sz[0]), float(sz[1])
        col = highlight.get(str(obj.id))
        if col is None:
            ax.add_patch(mpatches.Rectangle(
                (c[0] - w / 2, c[1] - h / 2), w, h,
                facecolor="lightyellow", edgecolor="#ccc",
                linewidth=0.5, alpha=0.55, zorder=2,
            ))
            ax.text(c[0], c[1], obj.label, ha="center", va="center",
                    fontsize=5.5, color="#888", zorder=6)
        else:
            ax.add_patch(mpatches.Rectangle(
                (c[0] - w / 2, c[1] - h / 2), w, h,
                facecolor=col, edgecolor=col,
                linewidth=2.0, alpha=0.30, zorder=3,
            ))
            ax.add_patch(mpatches.Rectangle(
                (c[0] - w / 2, c[1] - h / 2), w, h,
                facecolor="none", edgecolor=col,
                linewidth=2.0, alpha=1.0, zorder=4,
            ))
            ax.text(c[0], c[1] + h / 2 + 0.10, obj.label,
                    ha="center", va="bottom",
                    fontsize=8.5, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                              edgecolor=col, linewidth=1.0, alpha=0.9),
                    zorder=11)


# ---------------------------------------------------------------------------
# Trigger check
# ---------------------------------------------------------------------------
def trigger_check(scene_ctx, task, spec):
    inst = task["metadata"]["task_instance"]
    a, b, c = inst["subject_id"], inst["ref_b_id"], inst["ref_c_id"]
    ca = scene_ctx.get_object_centre(a)
    cb = scene_ctx.get_object_centre(b)
    cc = scene_ctx.get_object_centre(c)
    d_ab = float(np.linalg.norm(ca - cb))
    d_ac = float(np.linalg.norm(ca - cc))
    ratio = max(d_ab, d_ac) / max(min(d_ab, d_ac), 1e-9)
    min_ratio = 1.3
    raw_trigger = getattr(spec, "trigger", None)
    if isinstance(raw_trigger, dict):
        min_ratio = float(raw_trigger.get("min_dist_ratio", 1.3))
    obj_by_id = {o.id: o for o in scene_ctx.valid_objects}
    rooms = {x: getattr(obj_by_id.get(x), "room_id", None) for x in (a, b, c)}
    labels_distinct = len({inst["subject_label"], inst["ref_b_label"],
                           inst["ref_c_label"]}) == 3
    return {
        "subject": (a, inst["subject_label"]),
        "ref_b":   (b, inst["ref_b_label"]),
        "ref_c":   (c, inst["ref_c_label"]),
        "d_AB": d_ab,
        "d_AC": d_ac,
        "ratio": ratio,
        "min_ratio": min_ratio,
        "ratio_ok": ratio >= min_ratio,
        "labels_distinct": labels_distinct,
        "rooms": rooms,
        "same_room": len(set(rooms.values())) == 1,
        "gt": inst["gt_answer"],
        "closer_to": "ref_b" if d_ab < d_ac else "ref_c",
    }


# ---------------------------------------------------------------------------
# Region sampling figure
# ---------------------------------------------------------------------------
def fig_regions(task, scene_ctx, slots, out_path):
    inst = task["metadata"]["task_instance"]
    highlight = {
        inst["subject_id"]: "#e05c00",
        inst["ref_b_id"]:   "#0080e0",
        inst["ref_c_id"]:   "#00aa44",
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    rng = random.Random(0)
    hfov = scene_ctx.default_hfov_deg()

    for ax, slot in zip(axes, slots):
        resolved = resolve_slot(slot, inst)
        _draw_floorplan(ax, scene_ctx, highlight)
        args = dict(resolved.region_args)
        args["n"] = 80
        try:
            samples = sample_region(resolved.region_generator, scene_ctx, rng, **args)
        except Exception as exc:
            samples = []
            ax.text(0.02, 0.98, "sample_region error: " + str(exc),
                    transform=ax.transAxes, va="top", fontsize=8, color="red")
        n_pass = 0
        for cam_pos, cam_tgt in samples:
            ok = slot_satisfied_at(resolved, np.asarray(cam_pos),
                                   np.asarray(cam_tgt), hfov, scene_ctx)
            col = "#00aa00" if ok else "#cc2222"
            ax.scatter(cam_pos[0], cam_pos[1], s=14, color=col,
                       alpha=0.75, zorder=8, edgecolor="black", linewidth=0.3)
            if ok:
                n_pass += 1
        ax.set_aspect("equal")
        ax.set_axis_off()
        ax.set_title(
            "slot=" + slot.slot_id + "  region=" + slot.region_generator + "\n"
            "green=passes predicates, red=fails   ("
            + str(n_pass) + "/" + str(len(samples)) + " pass)",
            fontsize=11,
        )
        ax.legend(handles=[
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#00aa00", markersize=8, label="pass"),
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#cc2222", markersize=8, label="fail"),
        ], loc="upper right", fontsize=8, framealpha=0.85)

    plt.suptitle("Step 2 - region_generator samples vs. predicate satisfaction",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Init / target figure
# ---------------------------------------------------------------------------
def fig_init_target(task, scene_ctx, out_path):
    inst = task["metadata"]["task_instance"]
    highlight = {
        inst["subject_id"]: "#e05c00",
        inst["ref_b_id"]:   "#0080e0",
        inst["ref_c_id"]:   "#00aa44",
    }
    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_floorplan(ax, scene_ctx, highlight)

    iv, tv = task["init_view"], task["target_view"]
    pi = np.asarray(iv["position"][:2])
    fi = np.asarray(iv["forward"][:2])
    pt = np.asarray(tv["position"][:2])
    ft = np.asarray(tv["forward"][:2])

    fx, fy = _fov_polygon(pi, fi, length=1.5)
    ax.fill(fx, fy, color="green", alpha=0.22, zorder=7)
    ax.scatter(pi[0], pi[1], s=180, color="green", marker="o",
               zorder=10, edgecolor="white", linewidth=1.5)
    _draw_arrow(ax, pi, fi, "green")
    ax.text(pi[0], pi[1] - 0.20, "INIT", ha="center", va="top",
            color="green", fontsize=11, fontweight="bold", zorder=12)

    fx2, fy2 = _fov_polygon(pt, ft, length=1.5)
    ax.fill(fx2, fy2, color="red", alpha=0.20, zorder=7)
    ax.scatter(pt[0], pt[1], s=220, color="red", marker="*",
               zorder=10, edgecolor="white", linewidth=1.5)
    _draw_arrow(ax, pt, ft, "red")
    ax.text(pt[0], pt[1] - 0.20, "TARGET", ha="center", va="top",
            color="red", fontsize=11, fontweight="bold", zorder=12)

    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Step 3 - init view (green) deliberately fails predicates;\n"
                 "target view (red) satisfies all evidence slots",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Trajectory + Phi figure
# ---------------------------------------------------------------------------
def fig_trajectory_phi(task, scene_ctx, spec, slots, out_path):
    inst = task["metadata"]["task_instance"]
    highlight = {
        inst["subject_id"]: "#e05c00",
        inst["ref_b_id"]:   "#0080e0",
        inst["ref_c_id"]:   "#00aa44",
    }
    traj = task["expert_trajectory"]
    actions = task.get("action_sequence", [])
    hfov = scene_ctx.default_hfov_deg()

    phis = []
    slot_flags = []
    resolved_slots = [resolve_slot(s, inst) for s in slots]
    for v in traj:
        cp = np.asarray(v["position"])
        ct = np.asarray(v["target"])
        phi = potential_at(spec, cp, ct, inst, scene_ctx)
        phis.append(phi)
        flags = {
            rs.slot_id: slot_satisfied_at(rs, cp, ct, hfov, scene_ctx)
            for rs in resolved_slots
        }
        slot_flags.append(flags)

    fig, (ax_map, ax_phi) = plt.subplots(
        1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [3, 2]},
    )

    # LEFT: top-down with trajectory
    _draw_floorplan(ax_map, scene_ctx, highlight)
    xs = [v["position"][0] for v in traj]
    ys = [v["position"][1] for v in traj]
    ax_map.plot(xs, ys, "-", color="royalblue", linewidth=2.0,
                alpha=0.85, zorder=8)
    for t, v in enumerate(traj):
        cp = np.asarray(v["position"])
        fwd = np.asarray(v["forward"][:2])
        alpha = 0.35 + 0.65 * (t / max(len(traj) - 1, 1))
        ax_map.scatter(cp[0], cp[1], s=55, color="royalblue",
                       alpha=alpha, zorder=9, edgecolor="white", linewidth=0.6)
        _draw_arrow(ax_map, cp[:2], fwd, "royalblue", scale=0.30, lw=1.2)
        ax_map.text(cp[0] + 0.08, cp[1] + 0.08, str(t),
                    fontsize=7.5, color="darkblue", zorder=12)
    ax_map.scatter(xs[0], ys[0], s=200, color="green", marker="o",
                   zorder=11, edgecolor="white", linewidth=1.5,
                   label="init (t=0)")
    ax_map.scatter(xs[-1], ys[-1], s=240, color="red", marker="*",
                   zorder=11, edgecolor="white", linewidth=1.5,
                   label="final (t=" + str(len(traj) - 1) + ")")
    ax_map.legend(loc="upper right", fontsize=9, framealpha=0.85)
    ax_map.set_aspect("equal")
    ax_map.set_axis_off()
    ax_map.set_title("Step 4 - expert trajectory (step numbers)",
                     fontsize=12, fontweight="bold")

    # RIGHT: Phi curve + slot bands
    ts = list(range(len(traj)))
    ax_phi.plot(ts, phis, "-o", color="#3a52d4", linewidth=2.0,
                markersize=6, label="Phi(s_t)")
    ax_phi.fill_between(ts, 0, phis, color="#3a52d4", alpha=0.10)
    band_y = -0.07
    band_h = 0.05
    band_colors = ["#e05c00", "#0080e0", "#00aa44"]
    for i, rs in enumerate(resolved_slots):
        y = band_y - i * (band_h + 0.02)
        col = band_colors[i % len(band_colors)]
        for t in ts:
            ok = slot_flags[t][rs.slot_id]
            ax_phi.add_patch(mpatches.Rectangle(
                (t - 0.4, y), 0.8, band_h,
                facecolor=(col if ok else "#dddddd"),
                edgecolor="white", linewidth=0.3,
            ))
        ax_phi.text(-0.6, y + band_h / 2, rs.slot_id,
                    ha="right", va="center", fontsize=9,
                    color=col, fontweight="bold")

    ax_phi.set_xlim(-1.5, len(traj) - 0.5)
    ax_phi.set_ylim(
        band_y - len(resolved_slots) * (band_h + 0.02) - 0.03, 1.10,
    )
    ax_phi.axhline(1.0, color="#999", linestyle=":", linewidth=1.0)
    ax_phi.axhline(0.0, color="#999", linestyle="-", linewidth=0.6)
    ax_phi.set_xlabel("step  t", fontsize=11)
    ax_phi.set_ylabel("Phi(s_t)  (mean fraction-passed across slots)",
                      fontsize=11)
    ax_phi.set_xticks(ts)
    ax_phi.grid(True, axis="y", alpha=0.3)
    ax_phi.legend(loc="lower right", fontsize=10)
    for t in ts:
        a = actions[t] if t < len(actions) else ""
        ax_phi.text(t, 1.04, a, rotation=45, ha="left", va="bottom",
                    fontsize=7, color="#555")
    ax_phi.set_title("Potential Phi and per-slot satisfaction",
                     fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    return phis, slot_flags, resolved_slots


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def write_summary(path, task, trig, phis, slot_flags, resolved_slots):
    actions = task.get("action_sequence", [])
    descs = task.get("action_descriptions", [])

    PASS = "PASS"
    FAIL = "FAIL"
    YES = "[Y]"
    NO = "[ ]"

    lines = []
    a = lines.append
    a("# T05 trace - `" + task["task_id"] + "`")
    a("")
    a("**Question:** " + task["question"])
    a("**Choices:** " + str(task["choices"]) + "  ")
    a("**Answer:** **" + str(task["answer"]) + "**")
    a("**Score:** " + format(task.get("score", 0), ".4f")
      + "    **Coverage:** " + format(task.get("coverage", 0), ".2f")
      + "    **Steps:** " + str(task.get("num_steps", len(phis) - 1)))
    a("")
    a("## 1. Trigger check")
    a("")
    a("| field | value | check |")
    a("|---|---|---|")
    a("| subject | " + trig["subject"][1]
      + " (id=" + str(trig["subject"][0]) + ") | - |")
    a("| ref_b   | " + trig["ref_b"][1]
      + " (id=" + str(trig["ref_b"][0]) + ") | - |")
    a("| ref_c   | " + trig["ref_c"][1]
      + " (id=" + str(trig["ref_c"][0]) + ") | - |")
    a("| d(A,B)  | " + format(trig["d_AB"], ".3f") + " m | - |")
    a("| d(A,C)  | " + format(trig["d_AC"], ".3f") + " m | - |")
    a("| ratio   | " + format(trig["ratio"], ".3f") + " | "
      + (PASS if trig["ratio_ok"] else FAIL)
      + " (need >= " + format(trig["min_ratio"], ".2f") + ") |")
    a("| labels distinct | " + str(trig["labels_distinct"]) + " | "
      + (PASS if trig["labels_distinct"] else FAIL) + " |")
    a("| rooms   | " + str(trig["rooms"]) + " | "
      + ("same" if trig["same_room"] else "different") + " |")
    a("| closer  | A is closer to **" + trig["closer_to"] + "** | "
      + "gt = " + str(trig["gt"]) + " |")
    a("")
    a("## 2. Per-step trace")
    a("")
    header_slots = " | ".join(rs.slot_id for rs in resolved_slots)
    a("| t | action | description | Phi(s_t) | dPhi | " + header_slots + " |")
    sep = "|---|---|---|---|---|" + "|".join(["---"] * len(resolved_slots)) + "|"
    a(sep)
    for t, phi in enumerate(phis):
        act = actions[t] if t < len(actions) else ""
        dsc = descs[t] if t < len(descs) else ""
        d_phi = phi - phis[t - 1] if t > 0 else 0.0
        cells = []
        for rs in resolved_slots:
            cells.append(YES if slot_flags[t][rs.slot_id] else NO)
        flags = " | ".join(cells)
        a("| " + str(t) + " | `" + act + "` | " + dsc + " | "
          + format(phi, ".3f") + " | " + format(d_phi, "+.3f")
          + " | " + flags + " |")
    a("")
    a("Submission is judged on the **last row only**: all slots must show [Y] "
      "and Phi(s_T) = 1.0 for full coverage credit.")
    a("")
    a("Score = 1[answer correct] * min(1, Cov / min_cov) * gamma^T")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, type=Path)
    ap.add_argument("--jsonl", required=True, type=Path,
                    help="path to T05.jsonl produced by sweep")
    ap.add_argument("--task-index", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    scene_ctx = SceneContext.load(args.scene)

    with args.jsonl.open(encoding="utf-8") as fh:
        tasks = [json.loads(line) for line in fh if line.strip()]
    if not tasks:
        print("No tasks in " + str(args.jsonl), file=sys.stderr)
        return 1
    if args.task_index >= len(tasks):
        print("--task-index " + str(args.task_index)
              + " out of range (have " + str(len(tasks)) + ")", file=sys.stderr)
        return 1
    task = tasks[args.task_index]
    if task.get("template_id") != "T05":
        print("warning: task template_id=" + str(task.get("template_id"))
              + ", expected T05", file=sys.stderr)

    spec = load_template("T05")
    slots = expand_evidence_slots(spec, task["metadata"]["task_instance"])

    out_dir = args.out / ("T05_task_" + format(args.task_index, "03d"))
    out_dir.mkdir(parents=True, exist_ok=True)

    trig = trigger_check(scene_ctx, task, spec)

    print("[trace] writing 01_regions.png")
    fig_regions(task, scene_ctx, slots, out_dir / "01_regions.png")

    print("[trace] writing 02_init_target.png")
    fig_init_target(task, scene_ctx, out_dir / "02_init_target.png")

    print("[trace] writing 03_trajectory_phi.png")
    phis, slot_flags, resolved_slots = fig_trajectory_phi(
        task, scene_ctx, spec, slots, out_dir / "03_trajectory_phi.png"
    )

    print("[trace] writing summary.md")
    write_summary(out_dir / "summary.md", task, trig, phis, slot_flags,
                  resolved_slots)

    print("[trace] done -> " + str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
