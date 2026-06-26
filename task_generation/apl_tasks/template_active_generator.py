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
import hashlib
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base_generator import BaseAPLGenerator
from ...action_space.action_primitives import ActionPrimitive, ActionConfig
from ...core.data_types import APLActiveTaskItem, ViewState, make_task_id
from ...core import scene_context_ext  # noqa: F401  (apply patches)
from ...evaluation import (
    TemplateSpec,
    compute_coverage,
    episode_score,
    find_expert_trajectory,
    find_robust_expert_trajectory,
    load_template,
    sample_region,
    slot_satisfied_at,
    slot_almost_satisfied_at,
    slot_satisfied,
    resolve_slot,
)
from ...evaluation.template_spec import EvidenceSlot, PredicateSpec
from .choice_generators import build_choices, CHOICE_REGISTRY
from .init_validators import score_init_view


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
    generic_labels = {
        "other", "others", "object", "objects", "unknown", "misc", "miscellaneous",
    }
    return [
        o for o in scene_ctx.objects
        if o.label.lower() not in exclude_labels and o.label.lower() not in generic_labels
    ]


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


def _looks_plural(noun: str) -> bool:
    """Heuristic: scene labels ending with 's' are usually plural."""
    n = (noun or "").strip().lower()
    if not n:
        return False
    if n.endswith(("ss", "us", "is")):
        return False
    return n.endswith("s")


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


def _dims(o) -> np.ndarray:
    return np.asarray(o.bmax - o.bmin, dtype=float)


def _center(o) -> np.ndarray:
    return 0.5 * (np.asarray(o.bmin, dtype=float) + np.asarray(o.bmax, dtype=float))


def _xy_distance(a, b) -> float:
    return float(np.linalg.norm((_center(a) - _center(b))[:2]))


def _volume(o) -> float:
    return float(np.prod(np.clip(_dims(o), 1e-6, None)))


def _shape_distance(a, b) -> float:
    """Scale-invariant AABB shape distance; lower means more visually confusable."""
    da = np.sort(np.clip(_dims(a), 1e-6, None))
    db = np.sort(np.clip(_dims(b), 1e-6, None))
    da = da / max(float(np.linalg.norm(da)), 1e-6)
    db = db / max(float(np.linalg.norm(db)), 1e-6)
    return float(np.linalg.norm(da - db))


def _same_room(scene_ctx, a, b) -> bool:
    try:
        ra = scene_ctx.room_id_for_object(a.id)
        rb = scene_ctx.room_id_for_object(b.id)
        return bool(ra and rb and ra == rb)
    except Exception:
        return False


_T01_CONFUSABLE_GROUPS = [
    {"cabinet", "wardrobe", "bookshelf", "bookcase", "sideboard", "dresser", "piano"},
    {"chair", "armchair", "stool", "bench", "ottoman"},
    {"table", "desk", "coffee table", "dining table", "side table", "nightstand", "bedside table"},
    {"sofa", "couch", "settee", "bed", "bench"},
    {"lamp", "floor lamp", "table lamp", "vase", "plant", "flowerpot"},
]


def _labels_in_same_confusion_group(a_label: str, b_label: str) -> bool:
    a = a_label.lower()
    b = b_label.lower()
    return any(a in group and b in group for group in _T01_CONFUSABLE_GROUPS)


def _shape_similar(a, b, max_shape_dist: float = 0.18, max_volume_ratio: float = 2.5) -> bool:
    va, vb = _volume(a), _volume(b)
    if min(va, vb) <= 1e-6:
        return False
    return (
        _shape_distance(a, b) <= max_shape_dist
        and max(va, vb) / min(va, vb) <= max_volume_ratio
    )


def _t01_confusing_distractors(scene_ctx, target, limit: int = 3) -> List[str]:
    distractors: List[tuple[float, str]] = []
    for other in _candidate_objects(scene_ctx):
        if other.id == target.id or other.label.lower() == target.label.lower():
            continue
        semantic = _labels_in_same_confusion_group(target.label, other.label)
        shape_ok = _shape_similar(target, other)
        if not (semantic or shape_ok):
            continue
        score = _shape_distance(target, other) + (0.0 if semantic else 0.15)
        distractors.append((score, other.label))
    out: List[str] = []
    for _, label in sorted(distractors, key=lambda x: x[0]):
        if label not in out:
            out.append(label)
        if len(out) >= limit:
            break
    return out


_FINE_OBJECT_LABELS = {
    "book", "bottle", "cup", "mug", "plate", "bowl", "pen", "remote", "phone",
    "candle", "clock", "ornament", "fruit", "glass",
}


def _is_meaningful_object(o, min_volume: float = 0.02) -> bool:
    label = o.label.lower()
    return label not in _FINE_OBJECT_LABELS and _volume(o) >= min_volume


def _labels_semantically_close(a_label: str, b_label: str) -> bool:
    return a_label.lower() == b_label.lower() or _labels_in_same_confusion_group(a_label, b_label)


# ---------------------------------------------------------------------------
# T01 — Category Recognition
# ---------------------------------------------------------------------------

@register_instantiator("T01")
def _instantiate_T01(spec, scene_ctx, rng) -> Optional[Dict[str, Any]]:
    candidates = _candidate_objects(scene_ctx)
    rng.shuffle(candidates)
    for box in candidates:
        distractors = _t01_confusing_distractors(scene_ctx, box, limit=3)
        if len(distractors) < 1:
            continue
        ti = {
            "target_id": box.id,
            "target_label": box.label,
            "gt_answer": box.label,
        }
        ti["choices"] = [box.label] + distractors[:3]
        rng.shuffle(ti["choices"])
        return ti
    return None


# ---------------------------------------------------------------------------
# T04, T05 — pair / triple lookups (best-effort: pick objects sharing room)
# ---------------------------------------------------------------------------

# Categories so obviously large/small that any cross-category pair is trivially answerable
_T04_OBVIOUSLY_LARGE = {
    "cabinet", "wardrobe", "sofa", "bed", "bookshelf", "table", "desk",
    "bathtub", "shower", "refrigerator", "washing machine", "dining table",
    "coffee table", "sideboard", "wardrobe", "chest of drawers", "dresser",
}
_T04_OBVIOUSLY_SMALL = {
    "book", "bottle", "cup", "mug", "phone", "remote", "pen", "vase",
    "ornament", "fruit", "toy", "bowl", "plate", "lamp", "candle",
    "glass", "kettle", "clock",
}


@register_instantiator("T04")
def _instantiate_T04(spec, scene_ctx, rng):
    objs = [o for o in _candidate_objects(scene_ctx) if _is_meaningful_object(o)]
    if len(objs) < 2:
        return None
    rng.shuffle(objs)
    min_vol_ratio = float(getattr(spec, "trigger", {}).get("min_true_volume_ratio", 2.0))
    max_vol_ratio = float(getattr(spec, "trigger", {}).get("max_true_volume_ratio", 4.0))
    pairs = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            if not _same_room(scene_ctx, a, b):
                continue
            if not (_labels_semantically_close(a.label, b.label) or _shape_similar(a, b, max_volume_ratio=max_vol_ratio)):
                continue
            pairs.append((a, b))
    rng.shuffle(pairs)
    # Prefer a same-room, same-family pair whose size difference is visible but
    # not so large that everyday priors alone answer the question.
    for a, b in pairs:
        if a.label == b.label:
            continue
        # Skip pairs where common knowledge makes the size difference trivially obvious
        a_lo, b_lo = a.label.lower(), b.label.lower()
        if ((a_lo in _T04_OBVIOUSLY_LARGE and b_lo in _T04_OBVIOUSLY_SMALL) or
                (a_lo in _T04_OBVIOUSLY_SMALL and b_lo in _T04_OBVIOUSLY_LARGE)):
            continue
        vol_a = _volume(a)
        vol_b = _volume(b)
        if min(vol_a, vol_b) < 1e-6:
            continue
        ratio = max(vol_a, vol_b) / min(vol_a, vol_b)
        if ratio < min_vol_ratio or ratio > max_vol_ratio:
            continue
        larger_label = a.label if vol_a > vol_b else b.label
        return {
            "obj_a_id": a.id, "obj_a_label": a.label,
            "obj_b_id": b.id, "obj_b_label": b.label,
            "gt_answer": larger_label,
            "choices": [a.label, b.label],
        }
    return None


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
    similar_pairs = []
    for i, a in enumerate(objs):
        for b in objs[i + 1:]:
            if not _same_room(scene_ctx, a, b):
                continue
            dist_xy = _xy_distance(a, b)
            if dist_xy > 1.2:
                continue
            same_label = a.label.lower() == b.label.lower()
            similar_shape = _shape_similar(a, b, max_shape_dist=0.16, max_volume_ratio=2.0)
            if not (same_label or similar_shape):
                continue
            if abs(_center(a)[2] - _center(b)[2]) > 0.8:
                continue
            similar_pairs.append((dist_xy, a, b))
    similar_pairs.sort(key=lambda x: x[0])
    # ~40 % of the time generate a "single" task so the dataset is balanced.
    # GT="single": pick one object; the agent approaches and confirms it is a
    # single piece.  SeparationOnImage(pair_a_id == pair_b_id) returns True
    # trivially, so the coverage slot is satisfied whenever the agent is close
    # enough to see the object clearly.
    if rng.random() < 0.40:
        solo_pool = []
        for o in objs:
            if any(o.id in (a.id, b.id) for _, a, b in similar_pairs):
                solo_pool.append(o)
        if not solo_pool:
            solo_pool = objs
        solo = rng.choice(solo_pool)
        return {
            "group_ids":            [solo.id],
            "member_id":            solo.id,
            "pair_a_id":            solo.id,
            "pair_b_id":            solo.id,
            "group_centroid_proxy": solo.id,
            "label":                solo.label,
            "label_plural":         _pluralize(solo.label),
            "gt_answer":            "single",
            "choices":              ["single", "multiple"],
        }
    if not similar_pairs:
        return None
    _, a, b = similar_pairs[0]
    group = [a, b]
    # Add at most one more nearby same/similar item, keeping the cluster tight.
    for o in objs:
        if o.id in {a.id, b.id}:
            continue
        if not _same_room(scene_ctx, a, o):
            continue
        if _xy_distance(o, a) > 1.4 and _xy_distance(o, b) > 1.4:
            continue
        if o.label.lower() == a.label.lower() or _shape_similar(o, a, max_shape_dist=0.18, max_volume_ratio=2.2):
            group.append(o)
            break
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
        "label_plural":        _pluralize(label),
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
    # ~40 % of the time: GT="no" — a plausible but absent target label.
    # hidden_region_visibility_check handles GT=="no" by evaluating the slot
    # predicates at submit (clear sightline past occluder) rather than
    # checking target visibility, so target_id="" is safe.
    if rng.random() < 0.40:
        decoy_labels = []
        target_room = scene_ctx.room_id_for_object(tgt.id)
        for o in _candidate_objects(scene_ctx):
            if o.id in {occ_id, tgt_id} or o.label == tgt.label:
                continue
            same_room = target_room and scene_ctx.room_id_for_object(o.id) == target_room
            plausible_shape = _shape_similar(tgt, o, max_shape_dist=0.25, max_volume_ratio=3.0)
            plausible_semantic = _labels_in_same_confusion_group(tgt.label, o.label)
            if same_room and (plausible_shape or plausible_semantic) and o.label not in decoy_labels:
                decoy_labels.append(o.label)
        if decoy_labels:
            decoy_label = rng.choice(decoy_labels)
            return {
                "occluder_id": occ_id, "occluder_label": occ.label,
                "target_id":   "",    "target_label":   decoy_label,
                "hidden_region": (0.5 * (tgt.bmin + tgt.bmax)).tolist(),
                "gt_answer": "no",
                "choices": ["yes", "no"],
            }
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
    distractors = _t01_confusing_distractors(scene_ctx, tgt, limit=3)
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
        room_objs = [o for o in objs if scene_ctx.room_id_for_object(o.id) == room_id]
        if len(room_objs) < 2:
            continue
        distractors = [
            o for o in room_objs
            if o.label != tgt.label
            and o.id != tgt.id
            and (_labels_semantically_close(tgt.label, o.label) or _shape_similar(tgt, o, max_volume_ratio=3.0))
        ]
        if not distractors:
            distractors = [o for o in room_objs if o.label != tgt.label and o.id != tgt.id]
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
    # Also require absolute and relative distance asymmetry.
    trig = getattr(spec, "trigger", {})
    MIN_DIST_DIFF = float(trig.get("min_dist_difference_m", 0.4))
    MIN_DIST_RATIO = float(trig.get("min_dist_ratio", 1.25))
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
                    if min(da, db) <= 1e-6 or max(da, db) / min(da, db) < MIN_DIST_RATIO:
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
    portals = [
        p for p in scene_ctx._portals()
        if p.get("type") == "DOOR" and p.get("room_a") and p.get("room_b")
    ]
    if not portals:
        return None
    rng.shuffle(portals)
    portal = portals[0]
    room_a, room_b = portal["room_a"], portal["room_b"]
    rooms = scene_ctx.room_ids()
    objs_a = [o for o in _candidate_objects(scene_ctx)
              if scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), room_a)]
    objs_b = [o for o in _candidate_objects(scene_ctx)
              if scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), room_b)]
    if not objs_b:
        return None
    # ~40 % of the time: GT="no" — pick a label present in room_a but absent
    # from room_b.  The agent must enter room_b to disprove its presence.
    # cross_room_existence_check handles GT=="no" by checking InRoom(room_b)
    # at submit, so target_id="" is safe.
    if rng.random() < 0.40 and objs_a:
        labels_b = {o.label.lower() for o in objs_b}
        absent = [o for o in objs_a if o.label.lower() not in labels_b]
        if absent:
            decoy = rng.choice(absent)
            all_rooms = rooms  # already the full list from scene_ctx.room_ids()
            return {
                "room_a_id": room_a, "room_b_id": room_b,
                "portal_0_id": portal["id"],
                "portal_0_proxy_id": portal["id"],
                "room_b_name": _infer_room_name(scene_ctx, room_b, all_rooms.index(room_b)),
                "target_id": "", "target_label": decoy.label,
                "target_label_plural": _pluralize(decoy.label),
                "gt_answer": "no",
                "choices": ["yes", "no"],
            }
    rng.shuffle(objs_b)
    target = objs_b[0]
    return {
        "room_a_id": room_a, "room_b_id": room_b,
        "portal_0_id": portal["id"],
        "portal_0_proxy_id": portal["id"],
        "room_b_name": _infer_room_name(scene_ctx, room_b, rooms.index(room_b)),
        "target_id": target.id, "target_label": target.label,
        "target_label_plural": _pluralize(target.label),
        "gt_answer": "yes",
        "choices": ["yes", "no"],
    }


# ---------------------------------------------------------------------------
# T08 — Configuration Judgment
# ---------------------------------------------------------------------------

@register_instantiator("T08")
def _instantiate_T08(spec, scene_ctx, rng):
    """Pick a small group (3-5) from the SAME room so view_breaking_projection
    generates valid positions around a within-room centroid."""
    rooms = scene_ctx.room_ids()
    rng.shuffle(rooms)
    min_size = int(getattr(spec, "trigger", {}).get("group_size_min", 3))
    max_size = int(getattr(spec, "trigger", {}).get("group_size_max", 5))
    for room in rooms:
        objs_in_room = [
            o for o in _candidate_objects(scene_ctx)
            if _is_meaningful_object(o)
            and scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), room)
        ]
        if len(objs_in_room) < min_size:
            continue
        rng.shuffle(objs_in_room)
        group = None
        for seed in objs_in_room:
            seen_labels: set = set()
            local_group = []
            for obj in sorted(objs_in_room, key=lambda o: _xy_distance(seed, o)):
                if obj.label in seen_labels:
                    continue
                if local_group and _xy_distance(seed, obj) > 2.8:
                    continue
                local_group.append(obj)
                seen_labels.add(obj.label)
                if len(local_group) >= min(max_size, len(objs_in_room)):
                    break
            if len(local_group) < min_size:
                continue
            local_group = local_group[:rng.randint(min_size, min(max_size, len(local_group)))]
            local_centres = np.stack([_center(o) for o in local_group])[:, :2]
            radius = max(float(np.linalg.norm(c - local_centres.mean(axis=0))) for c in local_centres)
            if radius > 2.2:
                continue
            group = local_group
            break
        if group is None:
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
    by_label: Dict[str, List[Any]] = {}
    for o in _candidate_objects(scene_ctx):
        by_label.setdefault(o.label.lower(), []).append(o)

    connected = set()
    try:
        for p in scene_ctx._portals():
            a, b = p.get("room_a"), p.get("room_b")
            if a and b:
                connected.add((a, b))
                connected.add((b, a))
    except Exception:
        pass

    room_pairs = [(a, b) for a in rooms for b in rooms if a != b]
    rng.shuffle(room_pairs)
    room_pairs.sort(key=lambda ab: 0 if ab in connected else 1)

    labels = list(by_label.items())
    rng.shuffle(labels)
    chosen = None
    for zone_a, zone_b in room_pairs:
        for lab, items in labels:
            in_a = [
                o for o in items
                if scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), zone_a)
            ]
            in_b = [
                o for o in items
                if scene_ctx.is_position_in_room_id(0.5 * (o.bmin + o.bmax), zone_b)
            ]
            # Both evidence slots use zone_count_check; an empty-zone slot is
            # unsatisfiable, so require at least one target instance per zone.
            if in_a and in_b and len(in_a) <= 5 and len(in_b) <= 5:
                chosen = (zone_a, zone_b, lab, [o.id for o in in_a], [o.id for o in in_b])
                break
        if chosen:
            break
    if not chosen:
        return None
    zone_a, zone_b, target_label, ids_in_a, ids_in_b = chosen
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

# Only solid furniture that would realistically be moved through a doorway
_T06_MOVEABLE_FURNITURE = {
    "sofa", "couch", "settee", "table", "coffee table", "dining table", "desk",
    "chair", "armchair", "stool", "bench", "wardrobe", "cabinet", "sideboard",
    "bed", "bunk bed", "bookshelf", "bookcase", "dresser", "chest of drawers",
    "nightstand", "bedside table", "refrigerator", "washing machine",
    "dishwasher", "dryer", "luggage", "suitcase", "trolley",
}

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
    cands = [o for o in cands if 0.30 <= _xy_max_dim(o) <= 3.0
             and o.label.lower() in _T06_MOVEABLE_FURNITURE]
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
    gt = "yes" if p["width"] > _xy_max_dim(tgt) else "no"
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
        items = list(items)
        rng.shuffle(items)
        best_cluster = None
        best_radius = float("inf")
        for seed in items:
            neighbours = sorted(items, key=lambda o: _xy_distance(seed, o))[:6]
            if len(neighbours) < 3:
                continue
            centre = np.mean([_center(o) for o in neighbours], axis=0)
            radius = max(float(np.linalg.norm((_center(o) - centre)[:2])) for o in neighbours)
            if radius <= 2.0 and radius < best_radius:
                best_cluster = neighbours
                best_radius = radius
        if not best_cluster:
            continue
        cluster = best_cluster
        centre = np.mean([_center(o) for o in cluster], axis=0)
        occluder = None
        best_score = float("inf")
        for cand in _candidate_objects(scene_ctx):
            if cand.label.lower() == label:
                continue
            cdiag = float(np.linalg.norm(cand.bmax - cand.bmin))
            cc = _center(cand)
            d = float(np.linalg.norm(cc - centre))
            # Want a sizeable occluder (≥0.4m diag) within 2m of cluster centre
            if cdiag >= 0.60 and d < 1.8:
                score = d - 0.15 * cdiag
                if score < best_score:
                    occluder = cand
                    best_score = score
        if occluder is None:
            continue
        oc = _center(occluder)
        hidden_candidates = sorted(
            cluster,
            key=lambda o: (
                float(np.linalg.norm((_center(o) - oc)[:2])),
                -float(np.linalg.norm(_dims(o))),
            ),
        )[: min(3, len(cluster))]
        if not hidden_candidates:
            continue
        plural = label + ("s" if not label.endswith("s") else "")
        total = len(cluster)
        ti = {
            "target_label": label, "target_label_plural": plural,
            "occluder_id": occluder.id, "occluder_label": occluder.label,
            "num_evidence_slots": len(hidden_candidates),
            "cluster_instance_ids": [o.id for o in cluster],
            "gt_answer": str(total),
            "choices": [str(total), str(max(1, total-1)), str(total+1), str(max(1, total-2))],
        }
        for i, obj in enumerate(hidden_candidates):
            ti[f"hidden_instance_{i}_id"] = obj.id
        return ti
    raise SceneRequirementUnmet("T26: cluster found but no suitable occluder within 2 m")


# ---------------------------------------------------------------------------
# T29 — Contact Relationship   (AABB-based)
# ---------------------------------------------------------------------------

def _aabb_overlap_xy(a, b):
    lo = np.maximum(a.bmin[:2], b.bmin[:2])
    hi = np.minimum(a.bmax[:2], b.bmax[:2])
    return float(np.prod(np.clip(hi - lo, 0, None)))


def _aabb_area_xy(o):
    d = np.clip(o.bmax[:2] - o.bmin[:2], 1e-6, None)
    return float(d[0] * d[1])


def _aabb_gap_xy(a, b):
    gap_x = max(float(a.bmin[0] - b.bmax[0]), float(b.bmin[0] - a.bmax[0]), 0.0)
    gap_y = max(float(a.bmin[1] - b.bmax[1]), float(b.bmin[1] - a.bmax[1]), 0.0)
    return float(math.hypot(gap_x, gap_y))


_T29_SUPPORT_SURFACES = {
    "table", "desk", "coffee table", "dining table", "side table", "nightstand",
    "bedside table", "cabinet", "sideboard", "dresser", "chest of drawers",
    "shelf", "bookshelf", "bookcase", "counter", "sink",
}
_T29_SUPPORTED_OBJECTS = {
    "lamp", "table lamp", "vase", "plant", "flowerpot", "book", "bottle",
    "cup", "mug", "bowl", "plate", "clock", "candle", "box", "remote",
    "phone", "ornament",
}
_T29_BESIDE_OBJECTS = {
    "chair", "armchair", "stool", "bench", "ottoman", "sofa", "couch", "bed",
    "table", "desk", "coffee table", "dining table", "side table", "nightstand",
    "cabinet", "wardrobe", "dresser", "bookshelf", "bookcase", "plant",
    "floor lamp",
}


def _t29_can_rest_on(top, bottom) -> bool:
    return (
        top.label.lower() in _T29_SUPPORTED_OBJECTS
        and bottom.label.lower() in _T29_SUPPORT_SURFACES
        and _aabb_area_xy(top) <= 0.75 * _aabb_area_xy(bottom)
    )


def _t29_can_be_beside(a, b) -> bool:
    return (
        a.label.lower() in _T29_BESIDE_OBJECTS
        and b.label.lower() in _T29_BESIDE_OBJECTS
        and _aabb_overlap_xy(a, b) <= 0.05 * min(_aabb_area_xy(a), _aabb_area_xy(b))
    )


@register_instantiator("T29")
def _instantiate_T29(spec, scene_ctx, rng):
    objs = _candidate_objects(scene_ctx)
    if len(objs) < 2:
        raise SceneRequirementUnmet("T29 needs ≥2 candidate objects")
    pairs = []
    for i, a in enumerate(objs):
        for b in objs[i+1:]:
            # vertical relationship
            ov = _aabb_overlap_xy(a, b)
            if ov > 0.01 and abs(a.bmax[2] - b.bmin[2]) < 0.08 and _t29_can_rest_on(b, a):
                pairs.append((b, a, "resting_on"))         # b's bottom ≈ a's top → b resting on a
            elif ov > 0.01 and abs(b.bmax[2] - a.bmin[2]) < 0.08 and _t29_can_rest_on(a, b):
                pairs.append((a, b, "resting_on"))         # a's bottom ≈ b's top → a resting on b
            elif _aabb_gap_xy(a, b) <= 0.20 and abs(a.bmin[2] - b.bmin[2]) < 0.10 and _t29_can_be_beside(a, b):
                pairs.append((a, b, "beside"))             # share floor level
    if not pairs:
        raise SceneRequirementUnmet(
            "T29 needs a physically plausible support/contact pair: either a "
            "small supported object on a known support surface, or two plausible "
            "floor objects beside each other."
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
                all_rooms = list(g.keys())
                def _rname(rid):
                    idx = all_rooms.index(rid) if rid in all_rooms else 0
                    return _infer_room_name(scene_ctx, rid, idx)
                return {
                    "start_room": _rname(start), "end_room": _rname(end),
                    "excluded_room": _rname(excl),
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
    all_room_ids = scene_ctx.room_ids()
    room_b_raw = p.get("room_b", "")
    room_b_display = (
        _infer_room_name(scene_ctx, room_b_raw, all_room_ids.index(room_b_raw))
        if room_b_raw in all_room_ids else str(room_b_raw)
    )
    return {
        "passage_id": p["id"], "passage_proxy_id": p["id"],
        "passage_plane": "vertical",
        "left_wall_proxy": p["id"], "right_wall_proxy": p["id"],
        "side_a_label":  p.get("room_a", "side A"),
        "side_b_label":  p.get("room_b", "side B"),
        "agent_type":    agent,
        "room_b_name":   room_b_display,
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
    room_b_raw = p["room_b"]
    room_b_display = (
        _infer_room_name(scene_ctx, room_b_raw, rooms.index(room_b_raw))
        if room_b_raw in rooms else str(room_b_raw)
    )
    return {
        "portal_id": p["id"],
        "portal_proxy_id": p["id"],
        "portal_plane": p["id"],
        "room_a_id": p["room_a"],
        "room_b_id": p["room_b"],
        "room_b_name": room_b_display,
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
        digest = hashlib.blake2b(template_id.encode("utf-8"), digest_size=4).digest()
        template_seed = int.from_bytes(digest, "big") & 0x7FFF_FFFF
        self._rng.seed(self._base_seed ^ template_seed)
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
            diag = getattr(self, "_last_init_diag", {}) or {}
            if diag.get("n_accepted", 0) == 0:
                return None, "init_view_none"
            return None, "init_below_threshold"

        # T24 post-init GT fixup: the portal bearing depends on the actual
        # init_view direction, so we recompute gt_answer here once init_view
        # is known.  This overwrites the placeholder "ahead" set by the
        # instantiator.
        if spec.template_id == "T24":
            portal_id = task_instance.get("portal_proxy_id") or task_instance.get("portal_id")
            portal_centre = self.scene_ctx.get_object_centre(portal_id)
            if portal_centre is not None:
                cp = np.asarray(init_view.position, dtype=float)
                ct = np.asarray(init_view.target, dtype=float)
                fw = ct - cp
                fw /= np.linalg.norm(fw) + 1e-9
                d = portal_centre - cp
                d /= np.linalg.norm(d) + 1e-9
                fx, fy, dx, dy = fw[0], fw[1], d[0], d[1]
                cross = fx * dy - fy * dx
                dot = float(np.clip(fx * dx + fy * dy, -1.0, 1.0))
                yaw_deg = math.degrees(math.acos(dot))
                if cross < 0:
                    yaw_deg = -yaw_deg
                if abs(yaw_deg) <= 30:
                    task_instance["gt_answer"] = "ahead"
                elif abs(yaw_deg) >= 150:
                    task_instance["gt_answer"] = "behind"
                elif yaw_deg > 0:
                    task_instance["gt_answer"] = "right"
                else:
                    task_instance["gt_answer"] = "left"

        # 3. Run expert beam search
        expert = find_robust_expert_trajectory(
            spec, init_view, task_instance, self.scene_ctx,
            max_steps=spec.max_steps,
        )
        if not expert.found or not expert.actions:
            return None, "expert_not_found"

        # 4. Compute coverage and score for the expert trajectory.
        # Use submit_only=False so multi-slot sequential tasks (T27, T23, T32…)
        # earn full coverage when the trajectory visits each evidence region,
        # even if it can't be in all regions simultaneously at the final frame.
        traj = [(np.asarray(v.position), np.asarray(v.target))
                for v in expert.view_sequence]
        submit_view_coverage = compute_coverage(
            spec, traj[-1:] if traj else [], task_instance, self.scene_ctx,
            submit_only=True,
        )
        trajectory_evidence_coverage = compute_coverage(
            spec, traj, task_instance, self.scene_ctx, submit_only=False,
        )
        submit_only = getattr(spec, "coverage_mode", "submit") != "trajectory"
        coverage = (
            submit_view_coverage if submit_only else trajectory_evidence_coverage
        )

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

        answer = str(task_instance.get("gt_answer", ""))
        score = episode_score(
            spec, traj, predicted_answer=answer, gt_answer=answer,
            task_instance=task_instance, scene_ctx=self.scene_ctx,
        )

        # 6. Question text (pick template + light render)
        question = self._render_question(spec, task_instance)

        # 6. Wrap as APLActiveTaskItem
        choices = list(task_instance.get("choices") or [])
        if choices:
            self._rng.shuffle(choices)
            task_instance["choices"] = choices
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
                "coverage_semantics": (
                    "submit_final" if submit_only else "trajectory_memory"
                ),
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
                "coverage_mode": spec.coverage_mode,
                "gamma": spec.gamma,
            },
            expert_trajectory=list(expert.view_sequence),
            coverage=float(coverage),
            submit_view_coverage=float(submit_view_coverage),
            trajectory_evidence_coverage=float(trajectory_evidence_coverage),
            trajectory_reliability=dict(getattr(expert, "diagnostics", {}) or {}),
            score=float(score),
            min_steps=int(len(expert.actions)),
        )
        return item, "ok"

    # ------------------------------------------------------------------
    # init_view sampler: must FAIL slot predicates
    # ------------------------------------------------------------------
    # NOTE: a local ``_expand_pattern_slots`` used to live here and was a
    # duplicate of ``evaluation.template_spec.expand_evidence_slots``.  It
    # had no callers and has been removed; use the canonical helper.

    def _resolve_init_value(self, value: Any, task_instance: Dict[str, Any]) -> Any:
        if isinstance(value, str):
            s = value.strip()
            if s.startswith("{{") and s.endswith("}}"):
                return task_instance.get(s[2:-2].strip())
            return task_instance.get(s, value)
        if isinstance(value, list):
            return [self._resolve_init_value(v, task_instance) for v in value]
        return value

    def _infer_init_anchor_specs(self, spec: TemplateSpec,
                                 task_instance: Dict[str, Any]) -> List[Any]:
        cfg = getattr(spec, "init_view", {}) or {}
        explicit = cfg.get("visible_anchors") or cfg.get("anchors")
        if explicit:
            vals = self._resolve_init_value(explicit, task_instance)
            return vals if isinstance(vals, list) else [vals]

        tid = spec.template_id
        ti = task_instance
        if tid == "T04":
            return [[ti.get("obj_a_id"), ti.get("obj_b_id")]]
        if tid == "T05":
            return [ti.get("subject_id")]
        if tid == "T08":
            return [ti.get("group_ids") or ti.get("group_centroid_proxy")]
        if tid == "T11":
            return [ti.get("group_centroid_proxy") or ti.get("member_id")]
        if tid == "T13":
            return [[ti.get("front_id"), ti.get("back_id")]]
        if tid in ("T17", "T18", "T19"):
            return [ti.get("target_id") or ti.get("hidden_region") or ti.get("occluder_id")]
        if tid == "T21":
            return [[ti.get("obj_a_id"), ti.get("obj_b_id"), ti.get("reference_id")]]
        if tid == "T26":
            return [[ti.get("occluder_id"), ti.get("hidden_instance_0_id")]]
        if tid == "T27":
            ids = ti.get("target_ids") or ti.get("zone_targets") or []
            return [ids[0]] if ids else []
        if tid == "T29":
            return [[ti.get("obj_a_id"), ti.get("obj_b_id")]]
        for key in (
            "target_id", "subject_id", "obj_a_id", "occluder_id",
            "portal_id", "portal_0_id", "passage_id", "bottleneck_0_id",
        ):
            val = ti.get(key)
            if val:
                return [val]
        return []

    def _point_for_anchor(self, anchor: Any) -> Optional[np.ndarray]:
        if anchor is None or anchor == "":
            return None
        if isinstance(anchor, (list, tuple)):
            if anchor and all(isinstance(x, (int, float)) for x in anchor):
                arr = np.asarray(anchor, dtype=float)
                return arr if arr.size >= 3 else None
            pts = [self._point_for_anchor(a) for a in anchor]
            pts = [p for p in pts if p is not None]
            return np.mean(np.stack(pts), axis=0) if pts else None
        if isinstance(anchor, np.ndarray):
            arr = np.asarray(anchor, dtype=float)
            return arr if arr.size >= 3 else None
        try:
            c = self.scene_ctx.get_object_centre(anchor)
            if c is not None:
                return np.asarray(c, dtype=float)
        except Exception:
            pass
        try:
            pd = self.scene_ctx._portal_dict(anchor)
            if pd is not None and pd.get("position") is not None:
                return np.asarray(pd.get("position"), dtype=float)
        except Exception:
            pass
        return None

    def _anchor_visible_relaxed(self, anchor: Any, cp: np.ndarray, ct: np.ndarray,
                                hfov: float) -> bool:
        if anchor is None or anchor == "":
            return False
        if isinstance(anchor, (list, tuple)) and not (
            anchor and all(isinstance(x, (int, float)) for x in anchor)
        ):
            return any(self._anchor_visible_relaxed(a, cp, ct, hfov) for a in anchor)
        if isinstance(anchor, (list, tuple, np.ndarray)):
            p = self._point_for_anchor(anchor)
            if p is None:
                return False
            f = np.asarray(ct, dtype=float) - np.asarray(cp, dtype=float)
            d = p - np.asarray(cp, dtype=float)
            if np.linalg.norm(f[:2]) < 1e-6 or np.linalg.norm(d[:2]) < 1e-6:
                return False
            f = f[:2] / (np.linalg.norm(f[:2]) + 1e-9)
            d = d[:2] / (np.linalg.norm(d[:2]) + 1e-9)
            ang = math.degrees(math.acos(float(np.clip(np.dot(f, d), -1.0, 1.0))))
            return ang <= 0.5 * float(hfov)
        try:
            from ...evaluation.predicates import Visible
            if Visible(cp, ct, hfov, self.scene_ctx,
                       obj=anchor, min_corners=1, max_occ=0.95):
                return True
        except Exception:
            pass
        try:
            _, in_frame = self.scene_ctx.project_aabb_corners(anchor, cp, ct)
            return bool(in_frame.sum() > 0)
        except Exception:
            p = self._point_for_anchor(anchor)
            return self._anchor_visible_relaxed(p, cp, ct, hfov) if p is not None else False

    def _append_anchor_biased_positions(self, candidates: List[np.ndarray],
                                        anchors: List[Any]) -> List[np.ndarray]:
        pts = [self._point_for_anchor(a) for a in anchors]
        pts = [p for p in pts if p is not None]
        if not pts:
            return candidates
        scene_ctx = self.scene_ctx
        out = list(candidates)
        for p in pts[:2]:
            base_z = 0.8
            for r in (0.8, 1.2, 1.8, 2.6, 3.5):
                for k in range(16):
                    ang = 2.0 * math.pi * (k / 16.0)
                    cp = np.array([p[0] + r * math.cos(ang),
                                   p[1] + r * math.sin(ang),
                                   base_z], dtype=float)
                    try:
                        if scene_ctx.is_position_valid(cp):
                            out.append(cp)
                    except Exception:
                        continue
        return out

    def _sample_failing_init(self, spec: TemplateSpec,
                             task_instance: Dict[str, Any]) -> Optional[ViewState]:
        """Pick an init view that (a) is NOT a winning view at submit time, and
        (b) maximizes "illusion score" so the question is actually challenging.

        Algorithm
        ---------
        1. Sample ``n_pos`` candidate positions uniformly across all room
           polygons.  At each position try ``yaws_per_pos`` random yaws.
        2. For every (cp, ct) candidate compute three signals:

           * ``n_strict``   — number of slots strictly satisfied
             (``slot_satisfied_at``).
           * ``n_relaxed``  — number of slots almost-satisfied
             (``slot_almost_satisfied_at`` with relax_factor 0.5).  This is
             the "objects in awareness" signal: half the predicates of a
             slot pass, so the agent sees something relevant but not
             enough to submit.
           * ``validator_score`` — sum of per-trigger-field contributions
             from the YAML ``trigger:`` block (see ``init_validators.py``).
             Empty / unregistered today; populated per-template later.

        3. Compose a total score:

               score = base + 0.5 * (n_relaxed / n_total) + validator_score

           where ``base = 1.0`` if the candidate is a valid "fails at
           submit" view (``n_strict < n_total``) and the validator did NOT
           hard-reject, otherwise the candidate is discarded.

        4. Among accepted candidates keep the best.  If the best total
           score is below ``INIT_MIN_TOTAL_SCORE`` (default 1.0, i.e. only
           the base "fails strict" guarantee), return None — the scene
           cannot host this template's intended init view.

        Diagnostics
        -----------
        After the call ``self._last_init_diag`` carries a dict with the
        accept/reject counts, the best score, and the validator
        breakdown of the winning candidate.  ``trace_T05.py`` (and other
        tracers) read this to explain why an init view was chosen or why
        none was returned.
        """
        from ...evaluation.template_spec import expand_evidence_slots
        from ...evaluation.region_generators import _sample_positions_in_poly
        from ...evaluation.coverage import fraction_passed

        rs = np.random.RandomState(self._rng.randint(0, 2**31 - 1))
        # Sampling budget.  Matches the legacy 64x4 = 256 candidates so that
        # adding the per-candidate scoring pass does not blow up the time per
        # task.  Per-template validators can raise this when wired in.
        n_pos = 64
        scene_ctx = self.scene_ctx
        scene_ctx._ensure_room_index()
        polys = getattr(scene_ctx, "_room_polygons_by_id", {})
        if polys:
            n_rooms = len(polys)
            n_per_room = max(4, (n_pos + n_rooms - 1) // n_rooms)
            candidates = []
            for poly in polys.values():
                candidates.extend(
                    _sample_positions_in_poly(scene_ctx, poly, n_per_room, rs)
                )
        else:
            candidates = scene_ctx.sample_positions_in_room(
                num_points=n_pos, rng=rs,
            )

        hfov = scene_ctx.default_hfov_deg()
        init_anchors = self._infer_init_anchor_specs(spec, task_instance)
        if not init_anchors:
            return None
        candidates = self._append_anchor_biased_positions(candidates, init_anchors)
        primary_anchor_point = None
        for a in init_anchors:
            primary_anchor_point = self._point_for_anchor(a)
            if primary_anchor_point is not None:
                break
        if primary_anchor_point is None:
            return None
        eff_slots_raw = expand_evidence_slots(spec, task_instance)
        # Resolve slot variable bindings once — saves O(n_pos * yaws * n_slots)
        # repeated dict substitutions inside the inner loop.
        eff_slots = [resolve_slot(s, task_instance) for s in eff_slots_raw]
        n_total = len(eff_slots)
        if n_total == 0:
            return None  # nothing to fail — shouldn't happen for live templates

        # Minimum total score required.  A candidate that ONLY satisfies the
        # "fails strict" base condition scores exactly 1.0; that's the legacy
        # floor.  Per-template validators will push this up as they are
        # registered, which automatically tightens acceptance over time
        # without further code changes here.
        INIT_MIN_TOTAL_SCORE = 1.0

        best_view: Optional[ViewState] = None
        best_score: float = -math.inf
        best_breakdown: Dict[str, float] = {}
        n_examined = 0
        n_winning_view = 0          # candidates that would have ALL slots satisfied
        n_hard_rejected = 0         # rejected by a validator
        n_anchor_rejected = 0       # rejected because no relevant anchor is visible
        n_accepted = 0              # n_strict < n_total AND not hard-rejected

        for cp in candidates:
            d = np.asarray(primary_anchor_point, dtype=float) - np.asarray(cp, dtype=float)
            if float(np.linalg.norm(d[:2])) < 1e-6:
                continue
            ct = np.asarray(primary_anchor_point, dtype=float).copy()
            n_examined += 1

            if not any(self._anchor_visible_relaxed(a, cp, ct, hfov) for a in init_anchors):
                n_anchor_rejected += 1
                continue

            fracs = [
                fraction_passed(s, cp, ct, hfov, scene_ctx)
                for s in eff_slots
            ]
            n_strict = 0
            for s, fr in zip(eff_slots, fracs):
                if s.tier2_override:
                    ok = slot_satisfied(
                        s,
                        trajectory=[(cp, ct)],
                        task_instance=task_instance,
                        scene_ctx=scene_ctx,
                        hfov_deg=hfov,
                        submit_only=True,
                    )
                else:
                    ok = fr >= s.threshold
                if ok:
                    n_strict += 1
            if n_strict >= n_total:
                n_winning_view += 1
                continue

            passes, v_score, breakdown = score_init_view(
                spec, task_instance, cp, ct, hfov, scene_ctx,
            )
            if not passes:
                n_hard_rejected += 1
                continue

            n_relaxed = sum(
                1 for s, fr in zip(eff_slots, fracs)
                if fr >= s.threshold * 0.5
            )

            total = 1.0 + 0.5 * (n_relaxed / n_total) + v_score
            n_accepted += 1

            if total > best_score:
                best_score = total
                best_view = self._make_view(cp, ct)
                best_breakdown = {
                    "n_strict": float(n_strict),
                    "n_relaxed": float(n_relaxed),
                    "n_total": float(n_total),
                    "validator_score": float(v_score),
                    **{f"trigger.{k}": float(v) for k, v in breakdown.items()},
                }

                    # All slots strictly satisfied — this would be a winning

        # Diagnostics: always stored, even on failure.
        self._last_init_diag = {
            "n_examined": n_examined,
            "n_winning_view": n_winning_view,
            "n_hard_rejected": n_hard_rejected,
            "n_anchor_rejected": n_anchor_rejected,
            "n_accepted": n_accepted,
            "best_score": best_score if best_score > -math.inf else None,
            "min_required": INIT_MIN_TOTAL_SCORE,
            "best_breakdown": best_breakdown,
        }

        if best_view is None or best_score < INIT_MIN_TOTAL_SCORE:
            return None
        return best_view

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
        return self._postprocess_question(q.strip())

    def _postprocess_question(self, q: str) -> str:
        """Final text cleanup for clarity and grammatical consistency."""
        if not q:
            return q

        # Avoid deictic "that" references that are ambiguous in static text.
        q = re.sub(r"\bWhat type of object is that\?", "What type of object is shown here?", q, flags=re.IGNORECASE)
        q = re.sub(r"\bIs that one\b", "Is the object", q, flags=re.IGNORECASE)
        q = re.sub(r"\bIs that\b", "Is the object", q, flags=re.IGNORECASE)
        q = re.sub(r"\bthat narrow\b", "the narrow", q, flags=re.IGNORECASE)

        # Better wording for plural existential questions.
        q = re.sub(
            r"\bIs there\s+(?:a|an)\s+([A-Za-z][A-Za-z\-]*)\b",
            lambda m: f"Are there any {m.group(1)}" if _looks_plural(m.group(1)) else m.group(0),
            q,
            flags=re.IGNORECASE,
        )

        # Fix indefinite article agreement for singular nouns.
        def _fix_article(match):
            noun = match.group(2)
            if _looks_plural(noun):
                return noun
            article = "an" if noun[:1].lower() in "aeiou" else "a"
            return f"{article} {noun}"

        q = re.sub(r"\b(a|an)\s+([A-Za-z][A-Za-z\-]*)\b", _fix_article, q, flags=re.IGNORECASE)
        return q
