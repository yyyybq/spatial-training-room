#!/usr/bin/env python3
"""Convenience script: render all fixed_v3 templates into fixed_v5_rendered."""
import subprocess
import sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
SCENE   = r"C:\Users\user\Desktop\0267_840790"
JSONL   = ROOT / "out" / "fixed_v3"
OUT     = ROOT / "out" / "fixed_v5_rendered"
SCRIPT  = ROOT / "testbench" / "render_real_tasks.py"
MAX     = 3

TEMPLATES = [
    "T01","T04","T05","T06","T08","T11","T13",
    "T17","T18","T19","T20","T21","T23","T24",
    "T26","T27","T29","T32","T33",
]

for t in TEMPLATES:
    jsonl_path = JSONL / f"{t}.jsonl"
    out_path   = OUT / t
    if not jsonl_path.exists():
        print(f"[skip] {jsonl_path} not found")
        continue
    cmd = [
        sys.executable, str(SCRIPT),
        "--scene", SCENE,
        "--jsonl", str(jsonl_path),
        "--out",   str(out_path),
        "--max",   str(MAX),
    ]
    print(f"\n=== {t} ===")
    subprocess.run(cmd, check=False)

print("\n=== ALL DONE ===")
