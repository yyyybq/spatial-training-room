"""Sanity test for Items 6 (checks: schema) and 7 (TemplateHandler)."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent.parent
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


from spatial_training_room.evaluation.template_spec import (
    TemplateSpec, load_all_templates,
)
from spatial_training_room.task_generation.apl_tasks.template_handler import (
    TemplateHandler, get_template_handler, all_template_handlers,
)


# --- Item 7: TemplateHandler ---
print("--- Item 7: TemplateHandler ---")
handlers = all_template_handlers()
print(f"loaded {len(handlers)} handlers")
h = handlers.get("T05")
assert isinstance(h, TemplateHandler)
assert h.template_id == "T05"
assert h.spec.template_id == "T05"
assert callable(h.instantiator)
print(f"T05 handler OK: tid={h.template_id} name={h.spec.name}")
h2 = get_template_handler("T20")
assert h2.template_id == "T20" and callable(h2.instantiator)
print(f"T20 handler OK via get_template_handler")


# --- Item 6: checks: schema ---
print("--- Item 6: checks projection (pure) ---")
synth = {
    "template_id": "TX0",
    "name": "synthetic",
    "subclass": "C9.9",
    "evidence_slots": [{
        "slot_id": "main",
        "region_generator": "around_object",
        "region_args": {},
        "predicates": [],
    }],
    "checks": [
        {"when": "scene", "name": "min_objects", "args": 3},
        {"when": "init",  "name": "target_invisible_at_init", "args": True},
        {"when": "target", "slot": "main",
         "name": "PairVisible", "args": {"obj_a": "a", "obj_b": "b"}},
    ],
}
spec = TemplateSpec.from_dict(synth)
assert spec.scene_requirements == {"min_objects": 3}, spec.scene_requirements
assert spec.trigger == {"target_invisible_at_init": True}, spec.trigger
assert len(spec.evidence_slots[0].predicates) == 1
p = spec.evidence_slots[0].predicates[0]
assert p.name == "PairVisible"
assert p.args == {"obj_a": "a", "obj_b": "b"}
print("checks projection OK (scene/init/target)")

# --- Item 6: mixed legacy + checks (legacy wins) ---
synth2 = {
    "template_id": "TX1",
    "name": "synth_mixed",
    "subclass": "C9.9",
    "scene_requirements": {"min_objects": 99},  # legacy wins
    "trigger": {"existing_field": "leg"},
    "evidence_slots": [{
        "slot_id": "s1", "region_generator": "r", "region_args": {},
        "predicates": [{"name": "Visible", "args": {"obj": "x"}}],
    }],
    "checks": [
        {"when": "scene", "name": "min_objects", "args": 3},  # should be ignored
        {"when": "init",  "name": "new_field", "args": 1.5},
        {"when": "target", "name": "Centered", "args": {"obj": "y"}},
    ],
}
spec2 = TemplateSpec.from_dict(synth2)
assert spec2.scene_requirements["min_objects"] == 99, "legacy must win"
assert spec2.trigger["existing_field"] == "leg"
assert spec2.trigger["new_field"] == 1.5
assert len(spec2.evidence_slots[0].predicates) == 2
assert spec2.evidence_slots[0].predicates[1].name == "Centered"
print("mixed legacy + checks OK (legacy precedence honoured)")

# --- Item 6: bad inputs raise cleanly ---
print("--- Item 6: error paths ---")


def _expect_err(d, fragment):
    try:
        TemplateSpec.from_dict(d)
    except ValueError as e:
        assert fragment in str(e), f"expected {fragment!r} in {e!r}"
        print(f"  OK raised: {str(e)[:80]}")
        return
    raise AssertionError(f"expected ValueError containing {fragment!r}")


base = {"template_id": "TXE", "name": "e", "subclass": "C", "evidence_slots": [{"slot_id": "m", "region_generator": "r", "region_args": {}, "predicates": []}]}
_expect_err({**base, "checks": [{"when": "weird", "name": "x"}]}, "scene|init|target")
_expect_err({**base, "checks": [{"when": "scene"}]}, "missing 'name'")
_expect_err({**base, "checks": [{"when": "target", "slot": "nonexistent", "name": "X"}]}, "unknown evidence slot_id")
_expect_err({**base, "checks": "not-a-list"}, "must be a list")


# --- Item 6: all 19 legacy YAMLs still load ---
print("--- Item 6: legacy YAMLs ---")
all_specs = load_all_templates("p0")
print(f"loaded {len(all_specs)} p0 templates")
assert "T05" in all_specs
assert all_specs["T05"].trigger.get("min_dist_ratio") == 1.3

print("ALL CHECKS PASSED")
