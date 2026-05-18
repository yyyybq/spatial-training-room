"""Probe: time each stage to find what hangs."""
from __future__ import annotations

import importlib.util
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


def t(label, fn):
    t0 = time.time()
    try:
        out = fn()
        print(f"[probe] {label}: OK in {time.time()-t0:.2f}s -> {type(out).__name__}")
        return out
    except Exception as e:
        print(f"[probe] {label}: FAIL in {time.time()-t0:.2f}s -> {type(e).__name__}: {e}")
        traceback.print_exc()
        return None


SCENE = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\user\Desktop\0267_840790"

from spatial_training_room.task_generation.apl_tasks.template_active_generator import (
    INSTANTIATORS, TemplateActiveGenerator, load_template,
)

print(f"[probe] scene = {SCENE}")
gen = t("TemplateActiveGenerator(...) load", lambda: TemplateActiveGenerator(SCENE, {"seed": 0}))
if gen is None:
    sys.exit(1)

ctx = gen.scene_ctx
print(f"[probe] aabb count: {len(getattr(ctx, 'aabbs', []))}")
print(f"[probe] room_ids:   {list(getattr(ctx, 'room_ids', lambda: [])())[:5] if hasattr(ctx, 'room_ids') else 'n/a'}")

spec = t("load_template T01", lambda: load_template("T01"))
if spec is None:
    sys.exit(1)

inst_fn = INSTANTIATORS["T01"]
ti = t("instantiator T01", lambda: inst_fn(spec, ctx, gen._rng))
print(f"[probe] task_instance keys: {list(ti.keys()) if ti else 'None'}")
if ti is None:
    sys.exit(1)

# Try sampling target view directly
from spatial_training_room.evaluation.coverage import resolve_slot, slot_satisfied_at
from spatial_training_room.evaluation.region_generators import sample_region

slot = spec.evidence_slots[0]
resolved = resolve_slot(slot, ti)
print(f"[probe] slot.region_generator = {resolved.region_generator}, args = {resolved.region_args}")

samples = t(
    f"sample_region {resolved.region_generator}",
    lambda: sample_region(resolved.region_generator, ctx, gen._rng, **resolved.region_args),
)
print(f"[probe] # samples: {len(samples) if samples else 0}")

# Diagnose why _check_valid fails
import numpy as np
target_id = resolved.region_args.get("target")
centre = ctx.get_object_centre(target_id) if target_id else None
print(f"[probe] target {target_id} centre = {centre}")
print(f"[probe] floor_z / ceiling_z = {ctx.floor_z} / {ctx.ceiling_z}")
print(f"[probe] # objects = {len(ctx.objects)}; # room_polygons = {len(ctx.room_polygons)}")
if ctx.room_polygons:
    poly0 = ctx.room_polygons[0]
    arr = np.asarray(poly0)
    print(f"[probe] room0 poly shape = {arr.shape}, xy bounds = ({arr[:,0].min():.2f},{arr[:,0].max():.2f}) ({arr[:,1].min():.2f},{arr[:,1].max():.2f})")
if centre is not None:
    test_pos = np.array([centre[0] + 1.0, centre[1], 1.5])
    print(f"[probe] test_pos {test_pos} in_room? {ctx.is_position_in_room(test_pos)}")
    print(f"[probe] test_pos valid? {ctx.is_position_valid(test_pos)}")
    # Check distance to object centres
    near = [o for o in ctx.objects if np.linalg.norm(test_pos - 0.5*(o.bmin+o.bmax)) < 0.2]
    print(f"[probe] objects within 0.2m: {len(near)}")

# Walk a real attempt
print("\n[probe] === walking _build_task_item ===")
hfov = ctx.default_hfov_deg()
n_target_ok = 0
for cp, ct in samples:
    if slot_satisfied_at(resolved, cp, ct, hfov, ctx):
        n_target_ok += 1
print(f"[probe] # samples that satisfy slot: {n_target_ok}/{len(samples)}")

# Per-predicate diagnosis
from spatial_training_room.evaluation.predicates import PREDICATE_REGISTRY
print(f"[probe] hfov={hfov:.1f} deg")
print(f"[probe] predicates in slot: {[(p.name, p.args) for p in resolved.predicates]}")
fail_counts = {p.name: 0 for p in resolved.predicates}
for cp, ct in samples:
    for p in resolved.predicates:
        fn = PREDICATE_REGISTRY.get(p.name)
        try:
            ok = fn(cp, ct, hfov, ctx, **p.args)
        except Exception as e:
            ok = False
            fail_counts[p.name] = -1
        if not ok and fail_counts[p.name] >= 0:
            fail_counts[p.name] += 1
print(f"[probe] per-predicate fail counts (out of {len(samples)}): {fail_counts}")

# Deep dive into Visible: how many corners pass each sub-check?
print("\n[probe] === Visible deep dive on first 5 samples ===")
from spatial_training_room.utils.occlusion import is_box_occluded_by_any
target_id = resolved.region_args["target"]
box = ctx.get_object_by_id(target_id)
print(f"[probe] box bmin={box.bmin}, bmax={box.bmax}")
for k, (cp, ct) in enumerate(samples[:5]):
    mask = ctx.visible_corner_mask(target_id, cp, ct)
    occ = ctx.occlusion_fraction(target_id, cp, ct)
    # Per-corner reasons
    from spatial_training_room.utils.occlusion import camtoworld_from_pos_target, world_to_camera, project_point, aabb_corners
    import numpy as np
    c2w = camtoworld_from_pos_target(np.asarray(cp), np.asarray(ct))
    view = np.linalg.inv(c2w)
    K = ctx.intrinsics
    if isinstance(K, dict):
        K = np.asarray(K.get("K"), dtype=float)
    width, height = ctx._image_width(), ctx._image_height()
    corners = aabb_corners(box.bmin, box.bmax) if not callable(getattr(aabb_corners,'__call__',None)) else aabb_corners(box.bmin, box.bmax)
    in_front = in_frame = not_occ = 0
    for c in corners:
        pc = world_to_camera(view, c)
        if pc[2] <= 1e-6:
            continue
        in_front += 1
        u, v, _ = project_point(K, pc)
        if not (0 <= u < width and 0 <= v < height):
            continue
        in_frame += 1
        blockers = [b for b in ctx.all_blockers if b.id != box.id]
        tiny_min = c - 1e-3
        tiny_max = c + 1e-3
        if not is_box_occluded_by_any(np.asarray(cp), tiny_min, tiny_max, blockers, target_id=None):
            not_occ += 1
    print(f"  sample{k} cp={cp[:2].round(2)}: in_front={in_front}/8 in_frame={in_frame} not_occ={not_occ} mask={mask.sum()} occ_frac={occ:.2f}")

# Also try _build_task_item directly to see exception
import traceback as tb
try:
    item = gen._build_task_item(spec, ti)
    print(f"[probe] _build_task_item -> {item!r}")
    if item is not None:
        d = item.to_jsonl_dict()
        print(f"[probe] q={d['question']!r} ans={d['answer']!r} cov={d['coverage']:.3f} steps={len(d['action_sequence'])}")
except Exception:
    tb.print_exc()

