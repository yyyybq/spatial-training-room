"""Quick test for T08 same-room fix."""
from __future__ import annotations
import sys, time, importlib.util
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent
PARENT = PKG.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
s = importlib.util.spec_from_file_location(
    "spatial_training_room", PKG / "__init__.py",
    submodule_search_locations=[str(PKG)])
m = importlib.util.module_from_spec(s)
sys.modules["spatial_training_room"] = m
s.loader.exec_module(m)

from spatial_training_room.task_generation.apl_tasks.template_active_generator import (
    INSTANTIATORS, TemplateActiveGenerator,
)

SCENE = r"C:\Users\user\Desktop\0267_840790"
gen = TemplateActiveGenerator(SCENE, {"seed": 0})

for tid in ["T08"]:
    t0 = time.time()
    try:
        tasks = gen.generate_for_template(tid, n=1)
        dt = time.time() - t0
        if tasks:
            d = tasks[0].to_jsonl_dict()
            print(
                f"{tid} OK  {dt:.1f}s  cov={d['coverage']:.2f} "
                f"steps={len(d['action_sequence'])}  q={d['question'][:55]}"
            )
        else:
            print(f"{tid} EMPTY  {dt:.1f}s")
    except Exception as e:
        dt = time.time() - t0
        import traceback; traceback.print_exc()
        print(f"{tid} ERROR  {dt:.1f}s  {e}")
