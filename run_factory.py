#!/usr/bin/env python3
"""
run_factory.py — Unified entry point for the dual-track data factory.

Generates QA and/or APL tasks for one or more scenes and writes JSONL output.

Usage examples:

    # Generate APL tasks for a single scene (passive + active)
    python run_factory.py \
        --scenes /data/.../0013_840910 \
        --mode apl \
        --config configs/apl_config.yaml \
        --out-dir ./out

    # Generate QA tasks
    python run_factory.py \
        --scenes /data/.../scenes_root \
        --mode qa \
        --config configs/qa_config.yaml \
        --out-dir ./out

    # Generate both
    python run_factory.py \
        --scenes /data/.../0013_840910 /data/.../0014_... \
        --mode all \
        --out-dir ./out \
        --max-items 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Optional yaml support
# ---------------------------------------------------------------------------
try:
    import yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


def _load_config(path: str | None) -> Dict[str, Any]:
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[warn] config file not found: {path}")
        return {}
    if _YAML_OK:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    else:
        print("[warn] PyYAML not installed — ignoring config file")
        return {}


def _flatten_apl_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Merge top-level APL keys + passive/active sub-dicts into a flat dict."""
    flat = {k: v for k, v in cfg.items() if k not in ("passive", "active", "output")}
    passive_cfg = cfg.get("passive", {})
    active_cfg  = cfg.get("active", {})
    flat.update(passive_cfg)
    flat.update(active_cfg)
    return flat


# ---------------------------------------------------------------------------
# Per-scene generation
# ---------------------------------------------------------------------------

def _generate_for_scene(
    scene_path: str,
    mode: str,
    cfg: Dict[str, Any],
    max_items: int,
    out_dir: Path,
    template_id: Optional[str] = None,
) -> Dict[str, int]:
    """Run generation for a single scene, return counts per task type."""
    try:
        from .task_generation.apl_tasks.passive_generator import APLPassiveGenerator
        from .task_generation.apl_tasks.active_generator import APLActiveGenerator
        from .task_generation.apl_tasks.template_active_generator import TemplateActiveGenerator
        from .task_generation.qa_tasks.qa_generator import QAGenerator
    except ImportError:
        # Support direct script execution from the repository root:
        # ``python run_factory.py ...``.
        from task_generation.apl_tasks.passive_generator import APLPassiveGenerator
        from task_generation.apl_tasks.active_generator import APLActiveGenerator
        from task_generation.apl_tasks.template_active_generator import TemplateActiveGenerator
        from task_generation.qa_tasks.qa_generator import QAGenerator

    scene_name = Path(scene_path).name
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}

    # Template-driven branch: short-circuits the standard active generator.
    if mode == "template":
        if template_id is None:
            raise ValueError("--template-id is required when --mode=template")
        apl_cfg = _flatten_apl_config(cfg)
        print(f"  [template:{template_id}] {scene_name} ...")
        gen = TemplateActiveGenerator(scene_path, apl_cfg)
        tasks = gen.generate_for_template(template_id, n=max_items)
        out_t = out_dir / f"{scene_name}_template_{template_id}.jsonl"
        gen.save_batch_to_jsonl(tasks, str(out_t))
        counts[f"template_{template_id}"] = len(tasks)
        print(f"    \u2192 {len(tasks)} tasks \u2192 {out_t}")
        return counts

    if mode in ("apl", "all"):
        apl_cfg = _flatten_apl_config(cfg)

        # Passive
        print(f"  [passive] {scene_name} ...")
        passive_gen = APLPassiveGenerator(scene_path, apl_cfg)
        passive_tasks = passive_gen.generate_batch(max_items)
        out_p = out_dir / f"{scene_name}_apl_passive.jsonl"
        passive_gen.save_batch_to_jsonl(passive_tasks, str(out_p))
        counts["apl_passive"] = len(passive_tasks)
        print(f"    → {len(passive_tasks)} passive tasks → {out_p}")

        # Active
        print(f"  [active]  {scene_name} ...")
        active_gen = APLActiveGenerator(scene_path, apl_cfg)
        active_tasks = active_gen.generate_batch(max_items)
        out_a = out_dir / f"{scene_name}_apl_active.jsonl"
        active_gen.save_batch_to_jsonl(active_tasks, str(out_a))
        counts["apl_active"] = len(active_tasks)
        print(f"    → {len(active_tasks)} active tasks → {out_a}")

    if mode in ("qa", "all"):
        qa_cfg = {k: v for k, v in cfg.items() if k not in ("passive", "active")}
        print(f"  [qa]      {scene_name} ...")
        qa_gen = QAGenerator(scene_path, qa_cfg)
        qa_tasks = qa_gen.generate_batch(max_items)
        out_q = out_dir / f"{scene_name}_qa.jsonl"
        qa_gen.save_batch_to_jsonl(qa_tasks, str(out_q))
        counts["qa"] = len(qa_tasks)
        print(f"    → {len(qa_tasks)} QA tasks → {out_q}")

    return counts


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------

def _resolve_scenes(scene_args: List[str]) -> List[str]:
    """
    Accept either:
    - explicit scene directories (that contain labels.json)
    - a parent directory of multiple scene folders
    """
    scenes: List[str] = []
    for arg in scene_args:
        p = Path(arg)
        if not p.exists():
            print(f"[warn] path not found: {arg}")
            continue
        if (p / "labels.json").exists():
            scenes.append(str(p))
        else:
            # Treat as parent — iterate children
            children = sorted(p.iterdir())
            for child in children:
                if child.is_dir() and (child / "labels.json").exists():
                    scenes.append(str(child))
    return scenes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-track data factory: generate QA and/or APL tasks.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--scenes", nargs="+", required=True,
        help="Scene directories or parent directory containing multiple scenes.",
    )
    parser.add_argument(
        "--mode", choices=["qa", "apl", "all", "template"], default="apl",
        help="Which task track to generate (default: apl). Use 'template' with --template-id.",
    )
    parser.add_argument(
        "--template-id", type=str, default=None,
        help="Template id (e.g. T01) when --mode=template.",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (optional).",
    )
    parser.add_argument(
        "--out-dir", type=str, default="./out",
        help="Output directory for JSONL files (default: ./out).",
    )
    parser.add_argument(
        "--max-items", type=int, default=100,
        help="Maximum items to generate per scene per track (default: 100).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: 42).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    cfg.setdefault("seed", args.seed)

    scenes = _resolve_scenes(args.scenes)
    if not scenes:
        print("[error] No valid scenes found. Exiting.")
        sys.exit(1)

    print(f"Found {len(scenes)} scene(s). Mode: {args.mode}. Max items: {args.max_items}")

    total: Dict[str, int] = {}
    for scene_path in scenes:
        print(f"\n=== Scene: {Path(scene_path).name} ===")
        try:
            counts = _generate_for_scene(
                scene_path, args.mode, cfg, args.max_items, Path(args.out_dir),
                template_id=args.template_id,
            )
            for k, v in counts.items():
                total[k] = total.get(k, 0) + v
        except Exception as exc:
            print(f"  [error] Scene {scene_path} failed: {exc}")
            import traceback
            traceback.print_exc()

    print("\n=== Summary ===")
    for k, v in total.items():
        print(f"  {k}: {v} items")
    print("Done.")


if __name__ == "__main__":
    main()
