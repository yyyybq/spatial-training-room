"""Render top-3 examples for all 19 fixed_v7 templates."""
import subprocess, sys, pathlib

PYTHON = sys.executable
SCENE = r"C:\Users\user\Desktop\0267_840790"
JSONL_ROOT = pathlib.Path(r"C:\Users\user\Desktop\code\spatial-training-room\out\fixed_v7")
OUT_ROOT = pathlib.Path(r"C:\Users\user\Desktop\code\spatial-training-room\out\fixed_v7_rendered")
RENDER_SCRIPT = pathlib.Path(r"C:\Users\user\Desktop\code\spatial-training-room\testbench\render_real_tasks.py")
TEMPLATES = ["T01","T04","T05","T06","T08","T11","T13","T17","T18","T19",
             "T20","T21","T23","T24","T26","T27","T29","T32","T33"]

for tid in TEMPLATES:
    jsonl = JSONL_ROOT / f"{tid}.jsonl"
    out = OUT_ROOT / tid
    if not jsonl.exists():
        print(f"  {tid}: MISSING jsonl, skip")
        continue
    print(f"  Rendering {tid} ...", flush=True)
    result = subprocess.run(
        [PYTHON, str(RENDER_SCRIPT), "--scene", SCENE,
         "--jsonl", str(jsonl), "--out", str(out), "--max", "3"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        manifest = out / "manifest.json"
        print(f"  {tid}: OK (manifest={manifest.exists()})")
    else:
        print(f"  {tid}: FAILED")
        print(result.stderr[-500:] if result.stderr else "")

print("\nDone!")
