"""Sweep every registered template against one scene and report success."""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

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


from spatial_training_room.task_generation.apl_tasks.template_active_generator import (
    INSTANTIATORS, TemplateActiveGenerator, load_template,
)


def sweep(scene_path: str, n_per_template: int = 2, out_dir: str | None = None,
          seed: int = 0, verbose: bool = True, templates: list | None = None):
    print(f"[sweep] scene = {scene_path}")
    gen = TemplateActiveGenerator(scene_path, {"seed": seed})
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)

    summary = []
    tids = sorted(INSTANTIATORS)
    if templates:
        tids = [t for t in tids if t in templates]
    for tid in tids:
        t0 = time.time()
        status, n_ok, err = "ok", 0, None
        try:
            spec = load_template(tid)
            tasks = gen.generate_for_template(tid, n=n_per_template)
            n_ok = len(tasks)
            if n_ok == 0:
                status = "empty"
            if out_path and tasks:
                f = out_path / f"{tid}.jsonl"
                with f.open("w", encoding="utf-8") as fh:
                    for t in tasks:
                        fh.write(json.dumps(t.to_jsonl_dict(), ensure_ascii=False) + "\n")
        except NotImplementedError as e:
            status, err = "stub", str(e)[:80]
        except Exception as e:
            status, err = "error", f"{type(e).__name__}: {e}"[:120]
            if verbose:
                traceback.print_exc()
        dt = time.time() - t0
        sample_q = ""
        sample_cov = ""
        if status == "ok" and tasks:
            d0 = tasks[0].to_jsonl_dict()
            sample_q = d0["question"][:60]
            sample_cov = f"cov={d0['coverage']:.2f}/score={d0['score']:.2f}/steps={len(d0['action_sequence'])}"
        summary.append((tid, status, n_ok, n_per_template, dt, err, sample_q, sample_cov))
        print(f"  {tid:<6} {status:<6} {n_ok}/{n_per_template}  {dt:5.1f}s  "
              f"{(err or sample_cov):<55} {sample_q}")

    print("\n=== SUMMARY ===")
    n_total = len(summary)
    n_ok = sum(1 for s in summary if s[1] == "ok")
    n_empty = sum(1 for s in summary if s[1] == "empty")
    n_err = sum(1 for s in summary if s[1] in ("error", "stub"))
    print(f"  total templates    : {n_total}")
    print(f"  produced ≥1 task   : {n_ok}")
    print(f"  produced 0 tasks   : {n_empty}")
    print(f"  errored / stubs    : {n_err}")
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--templates", nargs="*", default=None,
                    help="Only run these template IDs (e.g. T05 T27)")
    args = ap.parse_args()
    sweep(args.scene, args.n, args.out, args.seed, verbose=not args.quiet,
          templates=args.templates)
