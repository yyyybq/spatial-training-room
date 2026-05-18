"""
testbench/smoke_test_template.py

Minimal end-to-end smoke test for the template-driven APL generator.
Usage::

    python -m spatial_training_room.testbench.smoke_test_template \\
        --scene /path/to/scene_dir --template T01 --n 3

Or directly::

    python testbench/smoke_test_template.py --scene <path> --template T01

Pass `--list` to enumerate available templates that have an instantiator.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import traceback
from pathlib import Path

# The package directory has a hyphen ("spatial-training-room") which is NOT a
# legal Python identifier.  Because sub-packages use relative imports such as
# `from ..core.task_base import ...`, we MUST expose the package under a legal
# importable alias.  We therefore register this directory as
# `spatial_training_room` in sys.modules before any sub-package import.
THIS_DIR = Path(__file__).resolve().parent
PKG_ROOT = THIS_DIR.parent          # spatial-training-room/
PARENT   = PKG_ROOT.parent

_PKG_ALIAS = "spatial_training_room"
if _PKG_ALIAS not in sys.modules:
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    spec = importlib.util.spec_from_file_location(
        _PKG_ALIAS,
        PKG_ROOT / "__init__.py",
        submodule_search_locations=[str(PKG_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create spec for {PKG_ROOT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_ALIAS] = module
    spec.loader.exec_module(module)


def _print_item(i: int, item) -> None:
    d = item.to_jsonl_dict()
    print(f"--- task {i} ---")
    print(f"  task_id        : {d['task_id']}")
    print(f"  template/sub   : {d['template_id']} / {d['subclass']}")
    print(f"  question       : {d['question']}")
    print(f"  answer/choices : {d['answer']}  in  {d.get('choices')}")
    print(f"  steps          : {len(d['action_sequence'])}  ({d['action_sequence']})")
    print(f"  coverage/score : {d['coverage']:.3f} / {d['score']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=False, help="Path to scene directory.")
    ap.add_argument("--template", default="T01", help="Template id (e.g. T01).")
    ap.add_argument("--n", type=int, default=3, help="Number of tasks to attempt.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                    help="Optional JSONL output path.")
    ap.add_argument("--list", action="store_true",
                    help="List registered template instantiators and exit.")
    args = ap.parse_args()

    # Local imports (after sys.path tweak)
    from spatial_training_room.task_generation.apl_tasks.template_active_generator import (
        INSTANTIATORS,
        TemplateActiveGenerator,
    )

    if args.list:
        print("Registered template instantiators:")
        for tid in sorted(INSTANTIATORS):
            fn = INSTANTIATORS[tid]
            print(f"  {tid:<6}  {fn.__doc__ or ''}")
        return

    if not args.scene:
        ap.error("--scene is required (or pass --list).")

    cfg = {"seed": args.seed}
    print(f"[smoke] loading scene: {args.scene}")
    gen = TemplateActiveGenerator(args.scene, cfg)
    print(f"[smoke] generating {args.n} task(s) for template {args.template} ...")
    try:
        tasks = gen.generate_for_template(args.template, n=args.n)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    print(f"[smoke] produced {len(tasks)} task(s).")
    for i, t in enumerate(tasks):
        _print_item(i, t)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as fh:
            for t in tasks:
                fh.write(json.dumps(t.to_jsonl_dict(), ensure_ascii=False) + "\n")
        print(f"[smoke] wrote {outp}")


if __name__ == "__main__":
    main()
