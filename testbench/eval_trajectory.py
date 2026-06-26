"""Evaluate trajectory quality metrics for generated/model rollouts.

Outputs:
  - per_task_metrics.jsonl
  - summary.json
  - summary_by_template.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent
PARENT = PKG_ROOT.parent

_PKG_ALIAS = "spatial_training_room"
if _PKG_ALIAS not in sys.modules:
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    spec = importlib.util.spec_from_file_location(
        _PKG_ALIAS,
        PKG_ROOT / "__init__.py",
        submodule_search_locations=[str(PKG_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_ALIAS] = module
    spec.loader.exec_module(module)


from spatial_training_room.core.scene_context import SceneContext
from spatial_training_room.evaluation.template_spec import load_template
from spatial_training_room.evaluation.trajectory_metrics import (
    belief_score,
    counterfactual_regret,
    information_gain_per_step,
    normalize_trajectory,
    spl,
    steps_to_success,
    trajectory_path_length,
)


def _load_jsonl(path: Path, max_n: int) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if max_n > 0 and len(out) >= max_n:
                break
    return out


def _iter_jsonl_inputs(args) -> List[Path]:
    if args.jsonl:
        return [Path(args.jsonl)]
    if args.jsonl_dir:
        root = Path(args.jsonl_dir)
        return sorted(root.glob(args.jsonl_glob))
    raise ValueError("Provide either --jsonl or --jsonl-dir")


_MISSING_TI_WARNED: set = set()


def _extract_task_instance(task: Dict) -> Dict:
    md = task.get("metadata") or {}
    ti = md.get("task_instance")
    if isinstance(ti, dict):
        return ti
    tid = task.get("task_id") or task.get("template_id") or "unknown"
    if tid not in _MISSING_TI_WARNED:
        import logging
        logging.getLogger(__name__).warning(
            "task %s has no metadata.task_instance; "
            "belief/counterfactual metrics will be unreliable.",
            tid,
        )
        _MISSING_TI_WARNED.add(tid)
    return {}


def _extract_agent_trajectory(task: Dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    # Preferred explicit rollout keys (for closed-loop model evals)
    for k in ("model_trajectory", "trajectory", "agent_trajectory"):
        if k in task and isinstance(task[k], list):
            tr = normalize_trajectory(task[k])
            if tr:
                return tr

    # Fallback to expert trajectory from generated dataset
    tr = normalize_trajectory(task.get("expert_trajectory") or [])
    if tr:
        return tr

    # Last fallback to [init, target]
    fallback = normalize_trajectory([task.get("init_view") or {}, task.get("target_view") or {}])
    return fallback


def _extract_expert_trajectory(task: Dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    tr = normalize_trajectory(task.get("expert_trajectory") or [])
    if tr:
        return tr
    return _extract_agent_trajectory(task)


def _compute_task_metrics(task: Dict, scene_ctx: SceneContext) -> Optional[Dict]:
    template_id = task.get("template_id")
    if not template_id:
        return None

    spec = load_template(template_id)
    ti = _extract_task_instance(task)
    agent_traj = _extract_agent_trajectory(task)
    expert_traj = _extract_expert_trajectory(task)
    if len(agent_traj) < 2:
        return None

    action_sequence = task.get("action_sequence") or task.get("actions") or []

    agent_len = trajectory_path_length(agent_traj)
    expert_len = trajectory_path_length(expert_traj)

    b0 = belief_score(spec, agent_traj[0][0], agent_traj[0][1], ti, scene_ctx)
    bT = belief_score(spec, agent_traj[-1][0], agent_traj[-1][1], ti, scene_ctx)
    ig = information_gain_per_step(spec, agent_traj, ti, scene_ctx)
    cf = counterfactual_regret(
        spec,
        agent_traj,
        ti,
        scene_ctx,
        action_sequence=action_sequence,
    )

    # Success proxy: generated tasks carry a pre-computed score; for model
    # rollouts use a relaxed belief threshold (0.90) to account for Tier-2
    # overrides not being fully reflected in the per-frame BRS.
    success = bool(float(task.get("score", 0.0)) > 0.0 or bT >= 0.90)
    sts = steps_to_success(agent_traj, success)
    spl_value = spl(success, agent_len, expert_len)

    return {
        "task_id": task.get("task_id"),
        "template_id": template_id,
        "success": success,
        "steps_to_success": sts,
        "path_length_m": float(agent_len),
        "shortest_path_m": float(expert_len),
        "spl": float(spl_value),
        "belief_start": float(b0),
        "belief_end": float(bT),
        "belief_gain": float(bT - b0),
        "mean_info_gain_per_step": float(np.mean(ig)) if ig else 0.0,
        "min_info_gain_per_step": float(np.min(ig)) if ig else 0.0,
        "max_info_gain_per_step": float(np.max(ig)) if ig else 0.0,
        "counterfactual": cf,
    }


def _write_jsonl(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _aggregate(rows: Sequence[Dict]) -> Dict:
    by_template: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_template[r.get("template_id", "unknown")].append(r)

    def _agg(bucket: Sequence[Dict]) -> Dict:
        if not bucket:
            return {
                "n": 0,
                "success_rate": 0.0,
                "mean_spl": 0.0,
                "mean_steps_to_success": 0.0,
                "mean_path_length_m": 0.0,
                "mean_belief_gain": 0.0,
                "mean_info_gain_per_step": 0.0,
                "mean_regret": 0.0,
                "mean_redundant_step_ratio": 0.0,
            }

        success = [1.0 if r["success"] else 0.0 for r in bucket]
        spls = [float(r["spl"]) for r in bucket]
        sts = [float(r["steps_to_success"]) for r in bucket if r.get("steps_to_success") is not None]
        plen = [float(r["path_length_m"]) for r in bucket]
        b_gain = [float(r["belief_gain"]) for r in bucket]
        ig = [float(r["mean_info_gain_per_step"]) for r in bucket]
        reg = [float(r["counterfactual"]["mean_regret"]) for r in bucket]
        red = [float(r["counterfactual"]["redundant_step_ratio"]) for r in bucket]

        return {
            "n": len(bucket),
            "success_rate": float(np.mean(success)),
            "mean_spl": float(np.mean(spls)),
            "mean_steps_to_success": float(np.mean(sts)) if sts else 0.0,
            "mean_path_length_m": float(np.mean(plen)),
            "mean_belief_gain": float(np.mean(b_gain)),
            "mean_info_gain_per_step": float(np.mean(ig)),
            "mean_regret": float(np.mean(reg)),
            "mean_redundant_step_ratio": float(np.mean(red)),
        }

    by_template_summary = {k: _agg(v) for k, v in sorted(by_template.items())}
    return {
        "overall": _agg(rows),
        "by_template": by_template_summary,
    }


def _write_template_csv(path: Path, by_template: Dict[str, Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "template_id",
        "n",
        "success_rate",
        "mean_spl",
        "mean_steps_to_success",
        "mean_path_length_m",
        "mean_belief_gain",
        "mean_info_gain_per_step",
        "mean_regret",
        "mean_redundant_step_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for tid, stats in sorted(by_template.items()):
            row = {"template_id": tid}
            row.update(stats)
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="Scene directory path")
    ap.add_argument("--jsonl", help="Single JSONL file to evaluate")
    ap.add_argument("--jsonl-dir", help="Directory containing JSONLs to evaluate")
    ap.add_argument("--jsonl-glob", default="*.jsonl", help="Glob under --jsonl-dir")
    ap.add_argument("--max", type=int, default=0, help="Max tasks per JSONL (0 means all)")
    ap.add_argument("--out", default="out/metrics", help="Output directory")
    args = ap.parse_args()

    scene_ctx = SceneContext.load(args.scene)
    jsonl_paths = _iter_jsonl_inputs(args)
    if not jsonl_paths:
        raise RuntimeError("No JSONL inputs found")

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict] = []
    for jsonl_path in jsonl_paths:
        tasks = _load_jsonl(jsonl_path, max_n=args.max)
        rows: List[Dict] = []
        for t in tasks:
            m = _compute_task_metrics(t, scene_ctx)
            if m is not None:
                m["source_jsonl"] = str(jsonl_path)
                rows.append(m)
                all_rows.append(m)

        per_file_dir = out_root / jsonl_path.stem
        _write_jsonl(per_file_dir / "per_task_metrics.jsonl", rows)
        summary = _aggregate(rows)
        with (per_file_dir / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        _write_template_csv(per_file_dir / "summary_by_template.csv", summary["by_template"])
        print(f"[done] {jsonl_path} -> {per_file_dir}")

    overall = _aggregate(all_rows)
    with (out_root / "summary_all.json").open("w", encoding="utf-8") as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    _write_template_csv(out_root / "summary_all_by_template.csv", overall["by_template"])
    _write_jsonl(out_root / "per_task_metrics_all.jsonl", all_rows)
    print(f"[done] overall -> {out_root}")


if __name__ == "__main__":
    main()
