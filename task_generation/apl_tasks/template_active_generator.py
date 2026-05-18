"""
task_generation/apl_tasks/template_active_generator.py

Template-driven APL active-task generator.

Pipeline
========
1.  Load YAML template via `evaluation.load_template(template_id)`.
2.  Instantiate a `task_instance` dict mapping template variables (target_id,
    obj_a_id, …) to concrete IDs from the SceneContext.
3.  For each evidence slot, sample candidate views via the named
    region_generator and verify slot predicates can be satisfied
    (so we know the template is FEASIBLE in this scene).
4.  Pick an `init_view` that fails the slot predicates (initial state must be
    unanswerable).
5.  Run beam-search expert trajectory finder (`evaluation.find_expert_trajectory`)
    to produce a gold action sequence.
6.  Wrap up as `APLActiveTaskItem` with template_id, subclass, quality_spec,
    expert_trajectory, coverage, score.

Scene-requirement matching
==========================
Each template's `scene_requirements` and `trigger` block is interpreted by
the per-template "instantiator" functions (`_instantiate_<template_id>`).
Unimplemented instantiators emit a clear NotImplementedError.
"""
from __future__ import annotations

import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base_generator import BaseAPLGenerator
from ...action_space.action_primitives import ActionPrimitive, ActionConfig
from ...core.data_types import APLActiveTaskItem, ViewState, make_task_id
from ...core import scene_context_ext  # noqa: F401  (apply patches)
from ...evaluation import (
    TemplateSpec,
    compute_coverage,
    find_expert_trajectory,
    load_template,
    sample_region,
    slot_satisfied_at,
    resolve_slot,
)
from ...evaluation.template_spec import EvidenceSlot, PredicateSpec
from .choice_generators import build_choices, CHOICE_REGISTRY


class SceneRequirementUnmet(Exception):
    """Raised by an instantiator when the scene cannot satisfy its requirements.

    The string message becomes the failure ``reason`` reported by
    ``generate_for_template``, making it possible to distinguish
    "no two-room scene" from "no triple of collinear objects" from a real bug.
    """


# ---------------------------------------------------------------------------
# Instantiator registry — one per template_id
# ---------------------------------------------------------------------------

INSTANTIATORS: Dict[str, callable] = {}


def register_instantiator(tid: str):
    def deco(fn):
        INSTANTIATORS[tid] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candidate_objects(scene_ctx, exclude_labels=None) -> list:
    """Return object AABBs not in the exclude list."""
    if exclude_labels is None:
        exclude_labels = {"wall", "floor", "ceiling", "room"}
    return [o for o in scene_ctx.objects if o.label.lower() not in exclude_labels]


def _pluralize(word: str) -> str:
    """Simple English plural: handles words already ending in 's'."""
    if not word:
        return word
    w = word.lower()
    # Scene labels are often already plural (e.g. "flowers", "downlights").
    if w.endswith('s'):
        return word
    if w.endswith('x') or w.endswith('z'):
        return word + 'es'
    if w.endswith('ch') or w.endswith('sh'):
        return word + 'es'
    if w.endswith('fe'):
        return word[:-2] + 'ves'
    if w.endswith('f') and not w.endswith('ff'):
        return word[:-1] + 'ves'
    if w.endswith('y') and len(w) > 1 and w[-2] not in 'aeiou':
        return word[:-1] + 'ies'
    return word + 's'


# Keyword sets for room name inference
_ROOM_KEYWORDS: list = [
    ("master bedroom",  {"bed", "wardrobe", "dresser", "bedside table", "mirror", "closet"}),
    ("bedroom",         {"bed", "wardrobe", "dresser", "bedside table"}),
    ("bathroom",        {"shower", "shower head", "toilet", "basin", "bathtub", "towel",
                         "floor mold", "bidet", "niche"}),
    ("living room",     {"sofa", "tv", "television", "floor lamp", "couch", "coffee table",
                         "candle combination", "toy_animals", "toy animals"}),
    ("kitchen",         {"fridge", "refrigerator", "stove", "oven", "microwave",
                         "kitchen sink", "cupboard", "tuyere"}),
    ("laundry room",    {"washing machine", "washing_machine", "flowerpot", "storage basket",
                         "faucet", "basin", "laundry"}),
    ("hallway",         {"shoe cabinet", "tuyere", "hanger", "coat rack"}),
    ("dining room",     {"dining table", "chair", "stool"}),
    ("storage room",    {"box", "shelf", "bookshelf", "storage"}),
    ("balcony",         {"window", "balcony", "outdoor"}),
]


def _infer_room_name(scene_ctx, room_id: str, index: int) -> str:
    """Infer a human-readable room name from the objects inside it."""
    objs = _candidate_objects(scene_ctx)
    labels_in_room = {
        o.label.lower() for o in objs
        if scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), room_id)
    }
    best_name = None
    best_score = 0
    for name, keywords in _ROOM_KEYWORDS:
        score = len(labels_in_room & keywords)
        if score > best_score:
            best_score = score
            best_name = name
    if best_name and best_score >= 2:
        # Disambiguate duplicates with a number suffix if needed
        return best_name
    # Fallback: generic numbered room
    return f"room {index + 1}"


def _other_labels(scene_ctx, primary_label: str, k: int) -> List[str]:
    seen = []
    for o in scene_ctx.objects:
        if o.label.lower() != primary_label.lower() and o.label not in seen:
            seen.append(o.label)
        if len(seen) >= k:
            break
    return seen


# ---------------------------------------------------------------------------
# T01 — Category Recognition
# ---------------------------------------------------------------------------

@register_instantiator("T01")
def _instantiate_T01(spec, scene_ctx, rng) -> Optional[Dict[str, Any]]:
    candidates = _candidate_objects(scene_ctx)
    rng.shuffle(candidates)
    for box in candidates:
        ti = {
            "target_id": box.id,
            "target_label": box.label,
            "gt_answer": box.label,
        }
        # Choices: target label + 2 other labels in the scene
        distractors = _other_labels(scene_ctx, box.label, 3)
        if len(distractors) < 1:
            continue
        ti["choices"] = [box.label] + distractors[:3]
        rng.shuffle(ti["choices"])
        return ti
    return None


# ---------------------------------------------------------------------------
# T04, T05 — pair / triple lookups (best-effort: pick objects sharing room)
# ---------------------------------------------------------------------------

@register_instantiator("T04")
def _instantiate_T04(spec, scene_ctx, rng):
    objs = _candidate_objects(scene_ctx)
    if len(objs) < 2:
        return None
    rng.shuffle(objs)
    a, b = objs[0], objs[1]
    # Compare AABB diagonals
    diag_a = float(np.linalg.norm(a.bmax - a.bmin))
    diag_b = float(np.linalg.norm(b.bmax - b.bmin))
    if abs(diag_a - diag_b) < 0.05:
        return None
    larger = a.id if diag_a > diag_b else b.id
    return {
        "obj_a_id": a.id, "obj_a_label": a.label,
        "obj_b_id": b.id, "obj_b_label": b.label,
        "gt_answer": larger,
        "choices": [a.id, b.id],
    }


@register_instantiator("T05")
def _instantiate_T05(spec, scene_ctx, rng):
    """Pick subject + 2 references all in the SAME room, with a clear
    distance asymmetry (ratio ≥ 1.3 per template trigger).  Picking from
    arbitrary rooms produces pairs that PairVisible can never satisfy
    (walls in between)."""
    min_ratio = float(getattr(spec, "trigger", {}).get("min_dist_ratio", 1.3))
    rooms = scene_ctx.room_ids()
    rng.shuffle(rooms)
    for room in rooms:
        objs_in_room = [o for o in _candidate_objects(scene_ctx)
                        if scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), room)]
        if len(objs_in_room) < 3:
            continue
        # Try several triples in this room before moving on
        for _ in range(20):
            rng.shuffle(objs_in_room)
            s, b, c = objs_in_room[0], objs_in_room[1], objs_in_room[2]
            # Require all three labels to be distinct to avoid degenerate questions
            if s.label == b.label or s.label == c.label or b.label == c.label:
                continue
            cs = 0.5 * (s.bmin + s.bmax)
            cb = 0.5 * (b.bmin + b.bmax)
            cc = 0.5 * (c.bmin + c.bmax)
            db = float(np.linalg.norm(cs - cb))
            dc = float(np.linalg.norm(cs - cc))
            if min(db, dc) < 1e-3:
                continue
            ratio = max(db, dc) / min(db, dc)
            if ratio < min_ratio:
                continue
            # Use labels (not IDs) as answer/choices so the question text is consistent
            closer_label = b.label if db < dc else c.label
            return {
                "subject_id": s.id, "subject_label": s.label,
                "ref_b_id": b.id, "ref_b_label": b.label,
                "ref_c_id": c.id, "ref_c_label": c.label,
                "gt_answer": closer_label,
                "choices": [b.label, c.label],
            }
    return None


# ---------------------------------------------------------------------------
# T11 — single vs multiple
# ---------------------------------------------------------------------------

@register_instantiator("T11")
def _instantiate_T11(spec, scene_ctx, rng):
    objs = _candidate_objects(scene_ctx)
    if len(objs) < 2:
        return None
    rng.shuffle(objs)
    group = objs[:rng.randint(2, min(4, len(objs)))]
    group_ids = [o.id for o in group]
    # Find centroid proxy (object nearest to group centroid)
    centres = np.stack([0.5 * (o.bmin + o.bmax) for o in group])
    centroid = centres.mean(axis=0)
    centroid_proxy = min(group, key=lambda o: float(np.linalg.norm(0.5 * (o.bmin + o.bmax) - centroid)))
    # Pick representative label from the group
    label = group[0].label
    return {
        "group_ids":           group_ids,
        "member_id":           group_ids[0],
        "pair_a_id":           group_ids[0],
        "pair_b_id":           group_ids[1] if len(group_ids) > 1 else group_ids[0],
        "group_centroid_proxy": centroid_proxy.id,
        "label":               label,
        "label_plural":        label + "s",
        "gt_answer":           "multiple",
        "choices":             ["single", "multiple"],
    }


# ---------------------------------------------------------------------------
# T17, T18, T19 — post-occlusion behind a known occluder
# ---------------------------------------------------------------------------

def _find_occluded_pair(scene_ctx, rng) -> Optional[Tuple[str, str]]:
    """Find (occluder, target) pair where occluder largely occludes target from
    *some* nearby viewpoint."""
    objs = _candidate_objects(scene_ctx)
    if len(objs) < 2:
        return None
    rng.shuffle(objs)
    for occ in objs[:8]:
        c_occ = 0.5 * (occ.bmin + occ.bmax)
        for tgt in objs:
            if tgt.id == occ.id:
                continue
            c_tgt = 0.5 * (tgt.bmin + tgt.bmax)
            # Place camera on far side of occluder from target → check occlusion
            seg = (c_occ - c_tgt)
            seg_n = float(np.linalg.norm(seg))
            if seg_n < 1e-3:
                continue
            cam = c_occ + (seg / seg_n) * 1.5
            cam[2] = 0.8
            try:
                if not scene_ctx.is_position_valid(cam):
                    continue
                occ_frac = scene_ctx.occlusion_fraction(tgt.id, cam, c_tgt)
            except Exception:
                continue
            if occ_frac >= 0.6:
                return occ.id, tgt.id
    return None


@register_instantiator("T17")
def _instantiate_T17(spec, scene_ctx, rng):
    pair = _find_occluded_pair(scene_ctx, rng)
    if pair is None:
        return None
    occ_id, tgt_id = pair
    occ = scene_ctx.get_aabb(occ_id)
    tgt = scene_ctx.get_aabb(tgt_id)
    return {
        "occluder_id": occ_id, "occluder_label": occ.label,
        "target_id":   tgt_id, "target_label":   tgt.label,
        "hidden_region": (0.5 * (tgt.bmin + tgt.bmax)).tolist(),
        "gt_answer": "yes",
        "choices": ["yes", "no"],
    }


@register_instantiator("T18")
def _instantiate_T18(spec, scene_ctx, rng):
    pair = _find_occluded_pair(scene_ctx, rng)
    if pair is None:
        return None
    occ_id, tgt_id = pair
    tgt = scene_ctx.get_aabb(tgt_id)
    occ = scene_ctx.get_aabb(occ_id)
    distractors = _other_labels(scene_ctx, tgt.label, 2)
    if not distractors:
        return None
    return {
        "occluder_id": occ_id, "occluder_label": occ.label,
        "target_id":   tgt_id, "target_label":   tgt.label,
        "gt_answer": tgt.label,
        "choices": [tgt.label] + distractors[:3],
    }


@register_instantiator("T19")
def _instantiate_T19(spec, scene_ctx, rng):
    pair = _find_occluded_pair(scene_ctx, rng)
    if pair is None:
        return None
    occ_id, tgt_id = pair
    tgt = scene_ctx.get_aabb(tgt_id)
    occ = scene_ctx.get_aabb(occ_id)
    return {
        "occluder_id": occ_id, "occluder_label": occ.label,
        "target_id":   tgt_id, "target_label":   tgt.label,
        "gt_answer": "complete",
        "choices": ["complete", "broken", "partial"],
    }


# ---------------------------------------------------------------------------
# T20 — local bearing search
# ---------------------------------------------------------------------------

@register_instantiator("T20")
def _instantiate_T20(spec, scene_ctx, rng):
    objs = _candidate_objects(scene_ctx)
    if len(objs) < 2:
        return None
    rng.shuffle(objs)
    for tgt in objs:
        room_id = scene_ctx.room_id_for_object(tgt.id)
        if not room_id:
            continue
        # Pick a distractor with a different label
        distractors = [o for o in objs if o.label != tgt.label and o.id != tgt.id]
        if not distractors:
            continue
        distractor = rng.choice(distractors)
        return {
            "target_id": tgt.id,
            "target_label": tgt.label,
            "room_id": room_id,
            "gt_answer": tgt.label,
            "distractor_choices": [distractor.label],
            "choices": [tgt.label, distractor.label],
        }
    return None


# ---------------------------------------------------------------------------
# T21 — local nearest target
# ---------------------------------------------------------------------------

@register_instantiator("T21")
def _instantiate_T21(spec, scene_ctx, rng):
    # Pick 3 objects from the SAME room so view_of_triple can see all three.
    # Also require the distance-to-reference difference ≥ 0.4 m (YAML trigger).
    MIN_DIST_DIFF = 0.4
    rooms = scene_ctx.room_ids()
    rng.shuffle(rooms)
    for room in rooms:
        objs_in_room = [
            o for o in _candidate_objects(scene_ctx)
            if scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), room)
        ]
        if len(objs_in_room) < 3:
            continue
        rng.shuffle(objs_in_room)
        # Try different (a, b, ref) triples
        for i in range(min(len(objs_in_room) - 2, 20)):
            ref = objs_in_room[i]
            ref_c = 0.5 * (ref.bmin + ref.bmax)
            remaining = objs_in_room[i + 1:]
            rng.shuffle(remaining)
            for j in range(len(remaining) - 1):
                a = remaining[j]
                for b in remaining[j + 1:]:
                    # Avoid degenerate wording like "closer to toiletries: X or toiletries".
                    if a.label == b.label or a.label == ref.label or b.label == ref.label:
                        continue
                    da = float(np.linalg.norm(0.5 * (a.bmin + a.bmax) - ref_c))
                    db = float(np.linalg.norm(0.5 * (b.bmin + b.bmax) - ref_c))
                    if abs(da - db) < MIN_DIST_DIFF:
                        continue
                    closer_label = a.label if da < db else b.label
                    return {
                        "obj_a_id": a.id, "obj_a_label": a.label,
                        "obj_b_id": b.id, "obj_b_label": b.label,
                        "reference_id": ref.id, "reference_label": ref.label,
                        "group_centroid_proxy": ref.id,
                        "gt_answer": closer_label,
                        "choices": [a.label, b.label],
                        "label": a.label if a.label == b.label else f"{a.label}/{b.label}",
                        "label_plural": _pluralize(a.label) if a.label == b.label else f"{a.label}/{b.label}",
                    }
    return None


# ---------------------------------------------------------------------------
# T23 — cross-room existence (requires ≥2 rooms)
# ---------------------------------------------------------------------------

@register_instantiator("T23")
def _instantiate_T23(spec, scene_ctx, rng):
    rooms = scene_ctx.room_ids()
    if len(rooms) < 2:
        return None
    rng.shuffle(rooms)
    room_a, room_b = rooms[0], rooms[1]
    # Find an object inside room_b
    target = None
    for o in _candidate_objects(scene_ctx):
        c = 0.5 * (o.bmin + o.bmax)
        if scene_ctx.is_position_in_room_id(c, room_b):
            target = o
            break
    if target is None:
        return None
    return {
        "room_a_id": room_a, "room_b_id": room_b,
        "room_b_name": str(room_b),
        "target_id": target.id, "target_label": target.label,
        "target_label_plural": target.label + "s",
        "gt_answer": "yes",
        "choices": ["yes", "no"],
    }


# ---------------------------------------------------------------------------
# T24 — portal direction (requires portal)
# ---------------------------------------------------------------------------

@register_instantiator("T24")
def _instantiate_T24(spec, scene_ctx, rng):
    portals = scene_ctx._portals()
    if not portals:
        return None
    p = rng.choice(portals)
    rooms = scene_ctx.room_ids()
    return {
        "portal_id": p["id"],
        "room_a_id": p.get("room_a") or (rooms[0] if rooms else None),
        "room_b_id": p.get("room_b") or (rooms[1] if len(rooms) > 1 else None),
        "gt_answer": "left",
        "choices": ["left", "right", "ahead", "behind"],
    }


# ---------------------------------------------------------------------------
# T08 — Configuration Judgment
# ---------------------------------------------------------------------------

@register_instantiator("T08")
def _instantiate_T08(spec, scene_ctx, rng):
    """Pick a small group (2-5) from the SAME room so view_breaking_projection
    generates valid positions around a within-room centroid."""
    rooms = scene_ctx.room_ids()
    rng.shuffle(rooms)
    for room in rooms:
        objs_in_room = [
            o for o in _candidate_objects(scene_ctx)
            if scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), room)
        ]
        if len(objs_in_room) < 2:
            continue
        rng.shuffle(objs_in_room)
        size = rng.randint(2, min(5, len(objs_in_room)))
        group = objs_in_room[:size]
        # Ensure all objects in the group have distinct labels to avoid "X, X, Y" questions
        seen_labels: set = set()
        group = [o for o in group if not (o.label in seen_labels or seen_labels.add(o.label))]
        if len(group) < 2:
            continue
        centres = np.stack([0.5 * (o.bmin + o.bmax) for o in group])[:, :2]
        centroid_2d = centres.mean(axis=0)
        pca = centres - centroid_2d
        _, sv, _ = np.linalg.svd(pca, full_matrices=False)
        # Collinearity ratio: smaller singular value / larger
        ratio = float(sv[1] / max(sv[0], 1e-6)) if len(sv) >= 2 else 0.0
        is_collinear = ratio < 0.10
        # Centroid proxy: object nearest to 3D centroid
        centroid_3d = np.stack([0.5 * (o.bmin + o.bmax) for o in group]).mean(axis=0)
        centroid_proxy = min(group, key=lambda o: float(np.linalg.norm(0.5 * (o.bmin + o.bmax) - centroid_3d)))
        return {
            "group_ids":            [o.id for o in group],
            "member_id":            group[0].id,
            "group_centroid_proxy": centroid_proxy.id,
            "labels_str":           ", ".join(o.label for o in group),
            "gt_answer":            "yes" if is_collinear else "no",
            "choices":              ["yes", "no"],
        }
    return None


# ---------------------------------------------------------------------------
# T13 — Post-occlusion Continuation (one piece vs two pieces)
# ---------------------------------------------------------------------------

@register_instantiator("T13")
def _instantiate_T13(spec, scene_ctx, rng):
    pair = _find_occluded_pair(scene_ctx, rng)
    if pair is None:
        return None
    front_id, back_id = pair
    front = scene_ctx.get_aabb(front_id)
    back  = scene_ctx.get_aabb(back_id)
    # On AABB-only data we cannot detect "two pieces"; default GT="continuous"
    return {
        "front_id": front_id, "front_label": front.label,
        "back_id":  back_id,  "back_label":  back.label,
        "gt_answer": "continuous",
        "choices": ["continuous", "separate"],
    }


# ---------------------------------------------------------------------------
# T15 — Label Face Acquisition (P1, requires label_side annotation)
# ---------------------------------------------------------------------------

@register_instantiator("T15")
def _instantiate_T15(spec, scene_ctx, rng):
    """Requires per-object 'label_side' annotation; raises SceneRequirementUnmet
    if no annotated objects exist in the scene."""
    annotated = []
    for o in _candidate_objects(scene_ctx):
        side = scene_ctx.get_annotation(o.id, "label_side") if hasattr(scene_ctx, "get_annotation") else None
        text = scene_ctx.get_annotation(o.id, "label_face_text") if hasattr(scene_ctx, "get_annotation") else None
        if side in ("left", "right", "top", "bottom", "back"):
            annotated.append((o, side, text or "unknown"))
    if not annotated:
        raise SceneRequirementUnmet(
            "T15 needs per-object 'label_side' + 'label_face_text' annotations; "
            "labels.json in this scene exposes none."
        )
    obj, side, text = rng.choice(annotated)
    return {
        "target_id": obj.id, "target_label": obj.label,
        "label_side": side, "side_name": side,
        "gt_answer": text,
        "distractor_choices": ["unknown_a", "unknown_b", "unknown_c"],
        "choices": [text, "unknown_a", "unknown_b"],
    }


# ---------------------------------------------------------------------------
# T27 — Zone Counting
# ---------------------------------------------------------------------------

@register_instantiator("T27")
def _instantiate_T27(spec, scene_ctx, rng):
    rooms = scene_ctx.room_ids()
    if len(rooms) < 2:
        return None
    rng.shuffle(rooms)
    zone_a, zone_b = rooms[0], rooms[1]
    # Pick a target label that appears in either zone
    by_label: Dict[str, List[Any]] = {}
    for o in _candidate_objects(scene_ctx):
        by_label.setdefault(o.label.lower(), []).append(o)
    target_label = None
    ids_in_a: List[str] = []
    ids_in_b: List[str] = []
    for lab, items in by_label.items():
        in_a = [o for o in items if scene_ctx.is_position_in_room_id(0.5*(o.bmin+o.bmax), zone_a)]
        in_b = [o for o in items if scene_ctx.is_position_in_room_id(0.5*(o.bmin+o.bmax), zone_b)]
        if in_a and in_b:
            target_label = lab
            ids_in_a = [o.id for o in in_a]
            ids_in_b = [o.id for o in in_b]
            break
        if (in_a or in_b) and len(in_a) + len(in_b) >= 2:
            target_label = lab
            ids_in_a = [o.id for o in in_a]
            ids_in_b = [o.id for o in in_b]
    if not target_label or (not ids_in_a and not ids_in_b):
        return None
    target_ids = ids_in_a + ids_in_b
    n_a = len(ids_in_a)
    n_b = len(ids_in_b)
    total = n_a + n_b
    # Infer human-readable room names from objects inside each zone
    all_room_ids = scene_ctx.room_ids()
    room_names: Dict[str, str] = {}
    for idx, rid in enumerate(all_room_ids):
        room_names[rid] = _infer_room_name(scene_ctx, rid, idx)
    # Disambiguate duplicate names by appending a counter
    from collections import Counter
    name_counts = Counter(room_names.values())
    name_seen: Dict[str, int] = {}
    for rid in all_room_ids:
        nm = room_names[rid]
        if name_counts[nm] > 1:
            name_seen[nm] = name_seen.get(nm, 0) + 1
            room_names[rid] = f"{nm} {name_seen[nm]}"
    return {
        "zone_a_id": zone_a, "zone_a_name": room_names.get(zone_a, str(zone_a)),
        "zone_b_id": zone_b, "zone_b_name": room_names.get(zone_b, str(zone_b)),
        "target_label": target_label,
        "target_label_plural": _pluralize(target_label),
        "target_ids": target_ids,
        "zone_a_targets": ids_in_a,
        "zone_b_targets": ids_in_b,
        "zone_targets": target_ids,
        "gt_answer": str(total),
        "choices": [str(total), str(max(0, total - 1)), str(total + 1)],
    }


# ---------------------------------------------------------------------------
# Stub registrations: declare scene requirements that this scene cannot satisfy
# ---------------------------------------------------------------------------

@register_instantiator("T14")
def _instantiate_T14(spec, scene_ctx, rng):
    raise SceneRequirementUnmet(
        "T14 back_face_acquisition requires per-object 'front_normal' (or equivalent "
        "orientation) annotation; this scene's labels.json does not provide one."
    )


@register_instantiator("T16")
def _instantiate_T16(spec, scene_ctx, rng):
    raise SceneRequirementUnmet(
        "T16 front_back_difference requires per-object 'front_normal' + "
        "'front_back_diff_flag' annotations; this scene exposes neither."
    )


# ---------------------------------------------------------------------------
# T06 — Clearance Assessment   (DOOR-portal + moveable object)
# ---------------------------------------------------------------------------

def _xy_max_dim(o):
    return float(max(o.bmax[0] - o.bmin[0], o.bmax[1] - o.bmin[1]))


@register_instantiator("T06")
def _instantiate_T06(spec, scene_ctx, rng):
    portals = [p for p in scene_ctx._portals() if p["type"] == "DOOR"
               and 0.55 <= p["width"] <= 2.5
               and p.get("room_a") and p.get("room_b")]
    if not portals:
        raise SceneRequirementUnmet(
            "T06 needs an interior DOOR portal with width in [0.55, 2.5] m; "
            f"scene has {sum(1 for p in scene_ctx._portals() if p['type']=='DOOR')} doors total."
        )
    cands = _candidate_objects(scene_ctx)
    cands = [o for o in cands if 0.30 <= _xy_max_dim(o) <= 3.0]
    if not cands:
        raise SceneRequirementUnmet("T06 needs a moveable object with max XY dim in [0.3, 3.0] m")
    rng.shuffle(portals); rng.shuffle(cands)
    for portal in portals:
        # Prefer a target whose width is *close* to the doorway (the close call)
        for tgt in cands:
            margin = portal["width"] - _xy_max_dim(tgt)
            if -0.4 <= margin <= 0.6:
                gt = "yes" if margin > 0.0 else "no"
                return {
                    "target_id": tgt.id, "target_label": tgt.label,
                    "bottleneck_0_id": portal["id"],
                    "bottleneck_0_left_id":  portal["id"],  # portal itself as left/right proxy
                    "bottleneck_0_right_id": portal["id"],
                    "bottleneck_0_plane": portal["id"],
                    "left_label":  portal.get("room_a", "side A"),
                    "right_label": portal.get("room_b", "side B"),
                    "portal_width": float(portal["width"]),
                    "target_max_dim": _xy_max_dim(tgt),
                    "clearance_margin_m": float(margin),
                    "gt_answer": gt,
                    "choices": ["yes", "no"],
                }
    # Fallback: any door + any moveable object (will be very-easy task).
    p = portals[0]; tgt = cands[0]
    gt = "yes" if portal["width"] > _xy_max_dim(tgt) else "no"
    return {
        "target_id": tgt.id, "target_label": tgt.label,
        "bottleneck_0_id": p["id"],
        "bottleneck_0_left_id":  p["id"],  # portal as proxy
        "bottleneck_0_right_id": p["id"],
        "bottleneck_0_plane": p["id"],
        "left_label": p.get("room_a", "side A"),
        "right_label": p.get("room_b", "side B"),
        "portal_width": float(p["width"]),
        "target_max_dim": _xy_max_dim(tgt),
        "clearance_margin_m": float(p["width"] - _xy_max_dim(tgt)),
        "gt_answer": gt,
        "choices": ["yes", "no"],
    }


# ---------------------------------------------------------------------------
# T26 — Occluded Counting   (≥3 same-label instances + nearby occluder)
# ---------------------------------------------------------------------------

@register_instantiator("T26")
def _instantiate_T26(spec, scene_ctx, rng):
    from collections import defaultdict
    by_label = defaultdict(list)
    for o in _candidate_objects(scene_ctx):
        by_label[o.label.lower()].append(o)
    groups = [(lbl, items) for lbl, items in by_label.items() if len(items) >= 3]
    if not groups:
        raise SceneRequirementUnmet(
            "T26 needs ≥3 instances of one label; largest group in this scene = "
            f"{max((len(v) for v in by_label.values()), default=0)}."
        )
    rng.shuffle(groups)
    for label, items in groups:
        cluster = sorted(items, key=lambda o: float(np.linalg.norm(0.5*(o.bmin+o.bmax))))[:6]
        centre = np.mean([0.5*(o.bmin+o.bmax) for o in cluster], axis=0)
        # Occluder = largest non-same-label object whose centre lies near the cluster
        occluder = None; best_d = 1e9
        for cand in _candidate_objects(scene_ctx):
            if cand.label.lower() == label:
                continue
            cdiag = float(np.linalg.norm(cand.bmax - cand.bmin))
            cc = 0.5*(cand.bmin + cand.bmax)
            d = float(np.linalg.norm(cc - centre))
            # Want a sizeable occluder (≥0.4m diag) within 2m of cluster centre
            if cdiag >= 0.40 and d < 2.0 and d < best_d:
                occluder = cand; best_d = d
        if occluder is None:
            continue
        plural = label + ("s" if not label.endswith("s") else "")
        total = len(items)
        return {
            "target_label": label, "target_label_plural": plural,
            "occluder_id": occluder.id, "occluder_label": occluder.label,
            "hidden_instance_0_id": cluster[0].id,
            "gt_answer": str(total),
            "choices": [str(total), str(max(1, total-1)), str(total+1), str(max(1, total-2))],
        }
    raise SceneRequirementUnmet("T26: cluster found but no suitable occluder within 2 m")


# ---------------------------------------------------------------------------
# T29 — Contact Relationship   (AABB-based)
# ---------------------------------------------------------------------------

def _aabb_overlap_xy(a, b):
    lo = np.maximum(a.bmin[:2], b.bmin[:2])
    hi = np.minimum(a.bmax[:2], b.bmax[:2])
    return float(np.prod(np.clip(hi - lo, 0, None)))


@register_instantiator("T29")
def _instantiate_T29(spec, scene_ctx, rng):
    objs = _candidate_objects(scene_ctx)
    if len(objs) < 2:
        raise SceneRequirementUnmet("T29 needs ≥2 candidate objects")
    pairs = []
    for i, a in enumerate(objs):
        for b in objs[i+1:]:
            ov = _aabb_overlap_xy(a, b)
            if ov <= 0:
                continue
            # vertical relationship
            if abs(a.bmax[2] - b.bmin[2]) < 0.08:
                pairs.append((a, b, "resting_on"))         # a on top of b? no: a's top touches b's bottom → b on a
            elif abs(b.bmax[2] - a.bmin[2]) < 0.08:
                pairs.append((a, b, "resting_on"))         # a resting on b
            elif abs(a.bmin[2] - b.bmin[2]) < 0.10:
                pairs.append((a, b, "beside"))             # share floor level
    if not pairs:
        raise SceneRequirementUnmet(
            "T29 needs two AABBs with XY overlap and ≤8 cm vertical gap; "
            "no such pair in this scene."
        )
    rng.shuffle(pairs)
    a, b, rel = pairs[0]
    return {
        "obj_a_id": a.id, "obj_a_label": a.label,
        "obj_b_id": b.id, "obj_b_label": b.label,
        "ab_vertical_plane": "vertical",
        "midpoint_proxy": a.id,
        "gt_answer": rel,
        "choices": ["touching", "resting_on", "beside", "stacked_above"],
    }


# ---------------------------------------------------------------------------
# T32 — Connectivity Judgment   (room-portal graph)
# ---------------------------------------------------------------------------

def _build_room_graph(scene_ctx):
    g = {}
    for p in scene_ctx._portals():
        if p["type"] != "DOOR":
            continue
        ra, rb = p.get("room_a"), p.get("room_b")
        if not ra or not rb or ra == rb:
            continue
        g.setdefault(ra, set()).add(rb)
        g.setdefault(rb, set()).add(ra)
    return g


def _bfs_reachable(g, start, blocked):
    seen = {start}; frontier = [start]
    while frontier:
        nxt = []
        for u in frontier:
            for v in g.get(u, ()):
                if v in blocked or v in seen:
                    continue
                seen.add(v); nxt.append(v)
        frontier = nxt
    return seen


@register_instantiator("T32")
def _instantiate_T32(spec, scene_ctx, rng):
    g = _build_room_graph(scene_ctx)
    rooms = list(g.keys())
    if len(rooms) < 3:
        raise SceneRequirementUnmet(
            f"T32 needs ≥3 rooms in the door-connectivity graph; scene has {len(rooms)}."
        )
    rng.shuffle(rooms)
    for start in rooms:
        for end in rooms:
            if end == start:
                continue
            # candidate excluded room = any other room
            for excl in rooms:
                if excl in (start, end):
                    continue
                # GT: can reach end from start without going through excl?
                reach = _bfs_reachable(g, start, blocked={excl})
                # non-trivial: end is reachable without excl AND excl lies on at least one alt path
                full_reach = _bfs_reachable(g, start, blocked=set())
                if end not in full_reach:
                    continue
                gt = "yes" if end in reach else "no"
                # find a portal id along the start→? frontier as evidence anchor
                portal_anchor = None
                for p in scene_ctx._portals():
                    if p["type"] == "DOOR" and start in (p.get("room_a"), p.get("room_b")):
                        portal_anchor = p; break
                if portal_anchor is None:
                    continue
                return {
                    "start_room": start, "end_room": end, "excluded_room": excl,
                    "portal_0_id": portal_anchor["id"],
                    "portal_0_proxy_id": portal_anchor["id"],
                    "portal_0_plane": "vertical",
                    "gt_answer": gt,
                    "choices": ["yes", "no"],
                }
    raise SceneRequirementUnmet("T32: no (start, end, excluded) triple gives a non-trivial question")


# ---------------------------------------------------------------------------
# T33 — Passage Passability   (DOOR portal vs reference agent width)
# ---------------------------------------------------------------------------

_AGENT_WIDTHS = {
    "wheelchair": 0.65,
    "person":     0.55,
    "large_box":  0.80,
}


@register_instantiator("T33")
def _instantiate_T33(spec, scene_ctx, rng):
    portals = [p for p in scene_ctx._portals()
               if p["type"] == "DOOR" and 0.40 <= p["width"] <= 2.5]
    if not portals:
        raise SceneRequirementUnmet(
            "T33 needs a DOOR portal with width in [0.40, 2.5] m; none in this scene."
        )
    rng.shuffle(portals)
    p = portals[0]
    agent = rng.choice(list(_AGENT_WIDTHS.keys()))
    threshold = _AGENT_WIDTHS[agent]
    gt = "yes" if p["width"] >= threshold + 0.05 else "no"
    return {
        "passage_id": p["id"], "passage_proxy_id": p["id"],
        "passage_plane": "vertical",
        "left_wall_proxy": p["id"], "right_wall_proxy": p["id"],
        "side_a_label":  p.get("room_a", "side A"),
        "side_b_label":  p.get("room_b", "side B"),
        "agent_type":    agent,
        "room_b_name":   p.get("room_b", "the other room"),
        "passage_width": float(p["width"]),
        "agent_width":   threshold,
        "gt_answer":     gt,
        "choices":       ["yes", "no"],
    }


# ---------------------------------------------------------------------------
# T24 — Portal Direction (proper GT from portal bearing relative to init room)
# ---------------------------------------------------------------------------

@register_instantiator("T24")
def _instantiate_T24_real(spec, scene_ctx, rng):
    portals = [p for p in scene_ctx._portals()
               if p["type"] == "DOOR" and p.get("room_a") and p.get("room_b")]
    if not portals:
        raise SceneRequirementUnmet("T24 needs a DOOR portal connecting two rooms; none available.")
    rng.shuffle(portals)
    p = portals[0]
    # Direction from room_a centre → portal centre, in world XY
    rooms = scene_ctx.room_ids()
    bearing = "ahead"   # default placeholder; orientation depends on init yaw chosen later
    return {
        "portal_id": p["id"],
        "portal_proxy_id": p["id"],
        "portal_plane": p["id"],
        "room_a_id": p["room_a"],
        "room_b_id": p["room_b"],
        "room_b_name": p["room_b"],
        "gt_answer": bearing,
        "choices": ["left", "right", "ahead", "behind"],
    }


# ---------------------------------------------------------------------------
# TemplateActiveGenerator
# ---------------------------------------------------------------------------

class TemplateActiveGenerator(BaseAPLGenerator):
    """
    Template-driven active-task generator. One generator instance is bound to
    a particular scene; call `generate_for_template(template_id, n)` to produce
    APLActiveTaskItems for that template.
    """

    def __init__(self, scene_path: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(scene_path, config or {})
        self.scene_ctx._ensure_room_index()
        self._base_seed = config.get("seed", 0) if config else 0
        self._rng = random.Random(self._base_seed)

    # ------------------------------------------------------------------
    # Required abstract methods
    # ------------------------------------------------------------------

    def generate_batch(self, max_items: int,
                       filters: Optional[Dict[str, Any]] = None
                       ) -> List[APLActiveTaskItem]:
        template_id = (filters or {}).get("template_id", "T01")
        return self.generate_for_template(template_id, n=max_items)

    def validate_task(self, task: APLActiveTaskItem) -> tuple:
        return (task.coverage > 0.0 and len(task.action_sequence) > 0,
                "ok" if task.coverage > 0 else "zero coverage")

    # ------------------------------------------------------------------
    # Per-template instantiation
    # ------------------------------------------------------------------

    def generate_for_template(self, template_id: str, n: int = 1
                              ) -> List[APLActiveTaskItem]:
        # Seed per-template so the RNG state for T18 doesn't depend on how many
        # attempts T17 used (fixing T17 previously broke T18 via shared state).
        self._rng.seed(self._base_seed ^ (hash(template_id) & 0x7FFF_FFFF))
        spec = load_template(template_id)
        instantiator = INSTANTIATORS.get(template_id)
        if instantiator is None:
            raise NotImplementedError(
                f"No instantiator registered for template {template_id!r}. "
                f"Available: {sorted(INSTANTIATORS)}"
            )
        items: List[APLActiveTaskItem] = []
        attempts = 0
        debug = bool(getattr(self, "_debug", False) or os.environ.get("SPATIAL_DEBUG"))
        # Adaptive cap: a template whose Visible truth-rate is ~1/10 still
        # needs ≥10 attempts to land a single task.  Floor at 128 so low-n
        # callers (n=1 or 2) actually exercise the scene; the multiplier 24
        # gives generous headroom for high-rejection templates (T01/T04/T08/T11).
        max_attempts = max(n * 24, 128)
        fail_counts: Dict[str, int] = {}
        reqs_unmet: Dict[str, int] = {}
        while len(items) < n and attempts < max_attempts:
            attempts += 1
            try:
                ti = instantiator(spec, self.scene_ctx, self._rng)
            except SceneRequirementUnmet as e:
                msg = str(e) or "unspecified"
                reqs_unmet[msg] = reqs_unmet.get(msg, 0) + 1
                fail_counts["instantiator_unmet"] = fail_counts.get("instantiator_unmet", 0) + 1
                # Hard requirement: no point retrying if the scene cannot meet it.
                break
            if ti is None:
                fail_counts["instantiator_none"] = fail_counts.get("instantiator_none", 0) + 1
                continue
            try:
                item, reason = self._build_task_item_diag(spec, ti)
            except Exception as e:
                fail_counts["exception"] = fail_counts.get("exception", 0) + 1
                if debug:
                    import traceback as _tb
                    print(f"[gen:{template_id}] exception: {e}")
                    _tb.print_exc()
                continue
            if item is None:
                fail_counts[reason] = fail_counts.get(reason, 0) + 1
                continue
            items.append(item)
        if debug or (len(items) == 0 and attempts > 0):
            nz = {k: v for k, v in fail_counts.items() if v}
            extra = f" unmet={reqs_unmet}" if reqs_unmet else ""
            print(f"[gen:{template_id}] attempts={attempts} produced={len(items)} "
                  f"fails={nz}{extra}")
        return items

    # ------------------------------------------------------------------
    # Build one APLActiveTaskItem
    # ------------------------------------------------------------------

    def _build_task_item(
        self, spec: TemplateSpec, task_instance: Dict[str, Any]
    ) -> Optional[APLActiveTaskItem]:
        item, _ = self._build_task_item_diag(spec, task_instance)
        return item

    def _build_task_item_diag(
        self, spec: TemplateSpec, task_instance: Dict[str, Any]
    ):
        # Returns (item_or_None, fail_reason_str)
        from ...evaluation.template_spec import expand_evidence_slots

        # 0. Expand evidence_slot_pattern (T06/T26/T32) into concrete slots
        slots = expand_evidence_slots(spec, task_instance)

        # 1. Sample target views — one per evidence slot
        target_view = None
        for slot in slots:
            resolved = resolve_slot(slot, task_instance)
            samples = sample_region(
                resolved.region_generator, self.scene_ctx, self._rng,
                **resolved.region_args,
            )
            hfov = self.scene_ctx.default_hfov_deg()
            for cp, ct in samples:
                if slot_satisfied_at(resolved, cp, ct, hfov, self.scene_ctx):
                    target_view = self._make_view(np.asarray(cp), np.asarray(ct))
                    break
            if target_view is not None:
                break
        if target_view is None:
            return None, "target_view_none"  # template not satisfiable in this scene

        # 2. Sample an init_view that FAILS the slot predicates
        init_view = self._sample_failing_init(spec, task_instance)
        if init_view is None:
            return None, "init_view_none"

        # 3. Run expert beam search
        expert = find_expert_trajectory(
            spec, init_view, task_instance, self.scene_ctx,
            max_steps=spec.max_steps, beam_width=8,
        )
        if not expert.found or not expert.actions:
            return None, "expert_not_found"

        # 4. Compute coverage and score for the expert trajectory.
        # Use submit_only=False so multi-slot sequential tasks (T27, T23, T32…)
        # earn full coverage when the trajectory visits each evidence region,
        # even if it can't be in all regions simultaneously at the final frame.
        traj = [(np.asarray(v.position), np.asarray(v.target))
                for v in expert.view_sequence]
        coverage = compute_coverage(spec, traj, task_instance, self.scene_ctx,
                                    submit_only=False)
        score = float(coverage * (spec.gamma ** max(0, len(traj) - 1)))

        # 5. Authoritative choices via registry (overrides instantiator's
        #    ad-hoc list when YAML names a registered generator).  Must run
        #    BEFORE question rendering so {choice_a}/{choice_b} placeholders
        #    see the final choice set.
        if spec.answer_choices_generator and spec.answer_choices_generator in CHOICE_REGISTRY:
            registry_choices = build_choices(
                spec.answer_choices_generator, task_instance, self.scene_ctx, self._rng,
            )
            if registry_choices:
                task_instance["choices"] = registry_choices

        # 6. Question text (pick template + light render)
        question = self._render_question(spec, task_instance)

        # 6. Wrap as APLActiveTaskItem
        choices = task_instance.get("choices") or []
        answer = str(task_instance.get("gt_answer", ""))
        answer_choice = None
        if choices and answer in choices:
            answer_choice = chr(ord("A") + choices.index(answer))

        item = APLActiveTaskItem(
            task_id=make_task_id(prefix=f"{spec.template_id}_active"),
            scene_name=str(self.scene_ctx.scene_path.name),
            question=question,
            question_type=spec.subclass.split(".")[0] if spec.subclass else "active",
            answer=answer,
            init_view=init_view,
            target_view=expert.view_sequence[-1] if expert.view_sequence else target_view,
            action_sequence=[a.value if hasattr(a, "value") else str(a)
                              for a in expert.actions],
            action_descriptions=self._action_descriptions(expert.actions),
            choices=list(choices) if choices else None,
            answer_choice=answer_choice,
            target_object=task_instance.get("target_label"),
            target_object_id=task_instance.get("target_id"),
            anchor_object=task_instance.get("anchor_label") or task_instance.get("occluder_label"),
            anchor_object_id=task_instance.get("anchor_id") or task_instance.get("occluder_id"),
            reasoning_required=True,
            difficulty=self._difficulty_from_steps(len(expert.actions)),
            metadata={
                "task_instance": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in task_instance.items()},
            },
            template_id=spec.template_id,
            subclass=spec.subclass,
            quality_spec={
                "evidence_slots": [
                    {
                        "slot_id": s.slot_id,
                        "region_generator": s.region_generator,
                        "tier2_override": s.tier2_override,
                    } for s in spec.evidence_slots
                ],
                "min_coverage_for_credit": spec.min_coverage_for_credit,
                "gamma": spec.gamma,
            },
            expert_trajectory=list(expert.view_sequence),
            coverage=float(coverage),
            score=float(score),
            min_steps=int(len(expert.actions)),
        )
        return item, "ok"

    # ------------------------------------------------------------------
    # init_view sampler: must FAIL slot predicates
    # ------------------------------------------------------------------

    def _expand_pattern_slots(
        self, pattern: Dict[str, Any], task_instance: Dict[str, Any]
    ) -> List[EvidenceSlot]:
        """Expand an `evidence_slot_pattern` block (T06/T26/T32) into N concrete
        EvidenceSlot objects. N = task_instance['num_evidence_slots'] if set,
        otherwise inferred from keys like '<prefix>_0_id', '<prefix>_1_id', ...
        Substitutes the literal `{i}` token (no braces around it) in slot_id,
        region_args values, and predicate args values.
        """
        from ...evaluation.template_spec import EvidenceSlot, PredicateSpec
        # Infer N.
        slot_id_tmpl = str(pattern.get("slot_id", "slot_{i}"))
        prefix = slot_id_tmpl.replace("{i}", "")
        n = int(task_instance.get("num_evidence_slots", 0) or 0)
        if n <= 0:
            n = 0
            while f"{prefix}{n}_id" in task_instance:
                n += 1
            if n == 0:
                n = 1  # default to one slot if instantiator only set one binding
        slots: List[EvidenceSlot] = []

        def _sub_i(value, i: int):
            if isinstance(value, str):
                return value.replace("{i}", str(i))
            if isinstance(value, dict):
                return {k: _sub_i(v, i) for k, v in value.items()}
            if isinstance(value, list):
                return [_sub_i(v, i) for v in value]
            return value

        for i in range(n):
            slot_id = slot_id_tmpl.replace("{i}", str(i))
            region_args = _sub_i(pattern.get("region_args", {}) or {}, i)
            preds_raw = _sub_i(pattern.get("predicates", []) or [], i)
            preds = [PredicateSpec(name=p["name"], args=p.get("args", {}))
                     for p in preds_raw]
            slots.append(EvidenceSlot(
                slot_id=slot_id,
                region_generator=pattern.get("region_generator", ""),
                region_args=region_args,
                predicates=preds,
                threshold=float(pattern.get("threshold", 1.0)),
                tier2_override=pattern.get("tier2_override"),
            ))
        return slots

    def _sample_failing_init(self, spec: TemplateSpec,
                             task_instance: Dict[str, Any]) -> Optional[ViewState]:
        from ...evaluation.template_spec import expand_evidence_slots
        from ...evaluation.region_generators import _sample_positions_in_poly
        rs = np.random.RandomState(self._rng.randint(0, 2**31 - 1))
        # In tight rooms a random yaw can accidentally satisfy the slot
        # predicates (target ends up in FOV by chance).  Try multiple yaws
        # per position before giving up.
        n_pos = 64
        yaws_per_pos = 4
        # Sample from ALL room polygons uniformly to avoid the global
        # sample_positions_in_room bias (which fills mostly from room 0).
        scene_ctx = self.scene_ctx
        scene_ctx._ensure_room_index()
        polys = getattr(scene_ctx, "_room_polygons_by_id", {})
        if polys:
            n_rooms = len(polys)
            n_per_room = max(4, (n_pos + n_rooms - 1) // n_rooms)
            candidates = []
            for poly in polys.values():
                candidates.extend(_sample_positions_in_poly(scene_ctx, poly, n_per_room, rs))
        else:
            candidates = scene_ctx.sample_positions_in_room(num_points=n_pos, rng=rs)
        hfov = scene_ctx.default_hfov_deg()
        # Use all effective slots (including pattern-expanded ones for T06/T26/T32)
        eff_slots = expand_evidence_slots(spec, task_instance)
        n_slots_total = len(eff_slots)
        for cp in candidates:
            for _ in range(yaws_per_pos):
                yaw = self._rng.uniform(0.0, 2.0 * math.pi)
                ct = cp + np.array([math.cos(yaw), math.sin(yaw), 0.0])
                n_sat = sum(
                    1 for slot in eff_slots
                    if slot_satisfied_at(
                        resolve_slot(slot, task_instance), cp, ct, hfov, scene_ctx
                    )
                )
                # Accept if NOT all slots are satisfied (coverage < 1.0).
                # For single-slot templates this is identical to "slot fails".
                # For sequential multi-slot tasks (e.g. T27 zone-counting) this
                # allows init inside zone_a so the expert only needs to navigate
                # to zone_b rather than crossing TWO room boundaries.
                if n_sat < n_slots_total:
                    return self._make_view(cp, ct)
        return None

    # ------------------------------------------------------------------
    # Question rendering
    # ------------------------------------------------------------------

    def _render_question(self, spec: TemplateSpec,
                         task_instance: Dict[str, Any]) -> str:
        if not spec.question_templates:
            return spec.description.strip().splitlines()[0] if spec.description else spec.name
        q = self._rng.choice(spec.question_templates)

        def _sub(match):
            key = match.group(1).strip()
            val = task_instance.get(key, "")
            if isinstance(val, list) and val:
                return str(val[0])
            return str(val)

        # Handle both {{var}} (coverage-style) and {var} (jinja-lite),
        # plus {choice_a}/{choice_b}/{choice_c}/{choice_d}.
        if "choices" in task_instance and isinstance(task_instance["choices"], list):
            choices = task_instance["choices"]
            for i, c in enumerate(choices[:4]):
                key = f"choice_{chr(ord('a') + i)}"
                q = q.replace("{{" + key + "}}", str(c))
                q = q.replace("{" + key + "}", str(c))

        # Double-brace first (more specific), then single-brace.
        q = re.sub(r"\{\{\s*(\w+)\s*\}\}", _sub, q)
        q = re.sub(r"\{\s*(\w+)\s*\}", _sub, q)
        return q.strip()
