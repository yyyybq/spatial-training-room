"""Sweep every registered template against one scene and report success."""
from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
import queue
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

RELEASE_EXCLUDED_TEMPLATES = {"T13", "T19", "T23", "T24", "T27", "T32"}


def _write_tasks(out_path: Path | None, tid: str, tasks: list) -> None:
    if out_path and tasks:
        f = out_path / f"{tid}.jsonl"
        with f.open("w", encoding="utf-8") as fh:
            for t in tasks:
                fh.write(json.dumps(t.to_jsonl_dict(), ensure_ascii=False) + "\n")


def _summarize_sample(status: str, tasks: list) -> tuple[str, str]:
    sample_q = ""
    sample_cov = ""
    if status == "ok" and tasks:
        d0 = tasks[0].to_jsonl_dict()
        sample_q = d0["question"][:60]
        sample_cov = (
            f"cov={d0['coverage']:.2f}/score={d0['score']:.2f}/"
            f"steps={len(d0['action_sequence'])}"
        )
    return sample_q, sample_cov


def _run_one_template(
    scene_path: str,
    seed: int,
    tid: str,
    n_per_template: int,
    out_dir: str | None,
) -> tuple[str, str, int, str | None, str, str]:
    status, n_ok, err = "ok", 0, None
    tasks = []
    try:
        load_template(tid)
        gen = TemplateActiveGenerator(scene_path, {"seed": seed})
        tasks = gen.generate_for_template(tid, n=n_per_template)
        n_ok = len(tasks)
        if n_ok == 0:
            status = "empty"
        _write_tasks(Path(out_dir) if out_dir else None, tid, tasks)
    except NotImplementedError as e:
        status, err = "stub", str(e)[:80]
    except Exception:
        raise
    sample_q, sample_cov = _summarize_sample(status, tasks)
    return tid, status, n_ok, err, sample_q, sample_cov


def _template_worker(
    result_queue,
    scene_path: str,
    seed: int,
    tid: str,
    n_per_template: int,
    out_dir: str | None,
) -> None:
    try:
        result_queue.put(
            {
                "result": _run_one_template(
                    scene_path, seed, tid, n_per_template, out_dir
                )
            }
        )
    except NotImplementedError as e:
        result_queue.put({"result": (tid, "stub", 0, str(e)[:80], "", "")})
    except Exception as e:
        result_queue.put(
            {
                "result": (
                    tid,
                    "error",
                    0,
                    f"{type(e).__name__}: {e}"[:120],
                    "",
                    "",
                ),
                "traceback": traceback.format_exc(),
            }
        )


def _run_one_template_with_timeout(
    scene_path: str,
    seed: int,
    tid: str,
    n_per_template: int,
    out_dir: str | None,
    timeout_per_template: float | None,
    verbose: bool,
) -> tuple[str, str, int, str | None, str, str]:
    if not timeout_per_template or timeout_per_template <= 0:
        return _run_one_template(scene_path, seed, tid, n_per_template, out_dir)

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_template_worker,
        args=(result_queue, scene_path, seed, tid, n_per_template, out_dir),
    )
    proc.start()
    proc.join(timeout_per_template)

    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return tid, "timeout", 0, f">{timeout_per_template:.1f}s", "", ""

    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        if proc.exitcode == 0:
            return tid, "error", 0, "worker exited without result", "", ""
        return tid, "error", 0, f"worker exitcode {proc.exitcode}", "", ""

    if verbose and payload.get("traceback"):
        print(payload["traceback"])
    return payload["result"]


def sweep(
    scene_path: str,
    n_per_template: int = 2,
    out_dir: str | None = None,
    seed: int = 0,
    verbose: bool = True,
    templates: list | None = None,
    timeout_per_template: float | None = None,
    include_excluded: bool = False,
):
    print(f"[sweep] scene = {scene_path}", flush=True)
    out_path = Path(out_dir) if out_dir else None
    if out_path:
        out_path.mkdir(parents=True, exist_ok=True)
    if timeout_per_template:
        print(f"[sweep] timeout_per_template = {timeout_per_template:.1f}s", flush=True)

    summary = []
    tids = sorted(INSTANTIATORS)
    if not include_excluded:
        tids = [t for t in tids if t not in RELEASE_EXCLUDED_TEMPLATES]
    if templates:
        tids = [t for t in tids if t in templates]
    for tid in tids:
        t0 = time.time()
        try:
            tid, status, n_ok, err, sample_q, sample_cov = _run_one_template_with_timeout(
                scene_path,
                seed,
                tid,
                n_per_template,
                str(out_path) if out_path else None,
                timeout_per_template,
                verbose,
            )
        except Exception as e:
            status, n_ok, err = "error", 0, f"{type(e).__name__}: {e}"[:120]
            sample_q, sample_cov = "", ""
            if verbose:
                traceback.print_exc()
        dt = time.time() - t0
        summary.append((tid, status, n_ok, n_per_template, dt, err, sample_q, sample_cov))
        print(f"  {tid:<6} {status:<8} {n_ok}/{n_per_template}  {dt:5.1f}s  "
              f"{(err or sample_cov):<55} {sample_q}", flush=True)

    print("\n=== SUMMARY ===")
    n_total = len(summary)
    n_ok = sum(1 for s in summary if s[1] == "ok")
    n_empty = sum(1 for s in summary if s[1] == "empty")
    n_timeout = sum(1 for s in summary if s[1] == "timeout")
    n_err = sum(1 for s in summary if s[1] in ("error", "stub"))
    print(f"  total templates    : {n_total}")
    print(f"  produced >=1 task  : {n_ok}")
    print(f"  produced 0 tasks   : {n_empty}")
    print(f"  timed out          : {n_timeout}")
    print(f"  errored / stubs    : {n_err}")
    if out_path:
        rows = [
            {
                "template": tid,
                "status": status,
                "n_ok": n_ok,
                "n_requested": n_req,
                "seconds": round(dt, 3),
                "error": err,
                "sample_question": sample_q,
                "sample_coverage": sample_cov,
            }
            for tid, status, n_ok, n_req, dt, err, sample_q, sample_cov in summary
        ]
        summary_path = out_path / "sweep_summary.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
        print(f"  summary json       : {summary_path}")
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
    ap.add_argument("--timeout-per-template", type=float, default=None,
                    help="Seconds before skipping a template; <=0 disables.")
    ap.add_argument("--include-excluded", action="store_true",
                    help="Also run templates excluded from the current release mix.")
    args = ap.parse_args()
    sweep(args.scene, args.n, args.out, args.seed, verbose=not args.quiet,
          templates=args.templates, timeout_per_template=args.timeout_per_template,
          include_excluded=args.include_excluded)
