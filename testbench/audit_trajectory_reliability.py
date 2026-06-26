"""
Audit generated APL jsonl files for trajectory reliability signals.

This is a lightweight data audit: it only reads generated jsonl and the
diagnostics written by template_active_generator. It does not need to load a
scene or render images.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STRAFE_OR_BACK = {"move_left", "move_right", "move_backward"}


def _iter_items(paths: Iterable[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                yield path, line_no, item


def _action_stats(actions: list[str]) -> dict[str, Any]:
    move_actions = [a for a in actions if a != "stop"]
    odd = [a for a in move_actions if a in STRAFE_OR_BACK]
    return {
        "steps": len(move_actions),
        "odd_motion": len(odd),
        "odd_motion_frac": len(odd) / max(1, len(move_actions)),
    }


def audit(paths: list[Path], odd_motion_warn: float = 0.5) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "num_items": 0,
        "by_template": defaultdict(int),
        "coverage_modes": Counter(),
        "warnings": [],
    }

    for path, line_no, item in _iter_items(paths):
        summary["num_items"] += 1
        tid = item.get("template_id") or path.stem
        summary["by_template"][tid] += 1

        qspec = item.get("quality_spec") or {}
        mode = qspec.get("coverage_mode", "submit")
        summary["coverage_modes"][mode] += 1
        min_cov = float(qspec.get("min_coverage_for_credit", 1.0))
        submit_cov = float(item.get("submit_view_coverage", item.get("coverage", 0.0)))
        traj_cov = float(item.get("trajectory_evidence_coverage", item.get("coverage", 0.0)))

        where = f"{path.name}:{line_no}:{tid}"
        if mode == "submit" and submit_cov + 1e-9 < min_cov:
            summary["warnings"].append(
                f"{where} submit mode but final coverage {submit_cov:.3f} < {min_cov:.3f}"
            )
        if mode == "trajectory" and traj_cov + 1e-9 < min_cov:
            summary["warnings"].append(
                f"{where} trajectory mode but evidence coverage {traj_cov:.3f} < {min_cov:.3f}"
            )
        if mode == "trajectory" and submit_cov + 1e-9 < min_cov:
            summary["warnings"].append(
                f"{where} trajectory-memory item: final frame alone is not answerable "
                f"({submit_cov:.3f} < {min_cov:.3f})"
            )

        reliability = item.get("trajectory_reliability") or {}
        if reliability and not reliability.get("stable_full_coverage", False):
            summary["warnings"].append(
                f"{where} beam sensitivity not stable: "
                f"coverage_range={reliability.get('coverage_range')} "
                f"step_range={reliability.get('step_range_full_coverage')}"
            )

        astats = _action_stats(list(item.get("action_sequence") or []))
        if astats["steps"] >= 4 and astats["odd_motion_frac"] >= odd_motion_warn:
            summary["warnings"].append(
                f"{where} high strafe/backward fraction "
                f"{astats['odd_motion_frac']:.2f} over {astats['steps']} steps"
            )

    summary["by_template"] = dict(summary["by_template"])
    summary["coverage_modes"] = dict(summary["coverage_modes"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", help="jsonl files or directories")
    parser.add_argument("--odd-motion-warn", type=float, default=0.5)
    args = parser.parse_args()

    paths: list[Path] = []
    for raw in args.jsonl:
        p = Path(raw)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.jsonl")))
        else:
            paths.append(p)
    result = audit(paths, odd_motion_warn=args.odd_motion_warn)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
