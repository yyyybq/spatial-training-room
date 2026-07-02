"""
task_generation/apl_tasks/choice_generators.py

Registry of answer-choice generators referenced from YAML
(`answer_choices_generator: <name>`).

Each generator has the signature::

    fn(task_instance: dict, scene_ctx, rng: random.Random) -> List[str]

The first element of the returned list MUST be the correct answer; the
caller will shuffle as needed. If a generator cannot produce a valid set
(e.g. missing distractors), it should return ``[]`` so the caller can skip.

Templates whose YAML names a registered generator can opt-in to use the
authoritative list; otherwise the per-template instantiator may build its
own choices and overwrite ``task_instance['choices']`` directly.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

CHOICE_REGISTRY: Dict[str, Callable] = {}


def register_choice(name: str):
    def deco(fn: Callable) -> Callable:
        CHOICE_REGISTRY[name] = fn
        return fn
    return deco


def build_choices(name: str, task_instance: dict, scene_ctx, rng) -> List[str]:
    """
    Look up `name` in CHOICE_REGISTRY and call it. Returns [] if the name is
    not registered (so callers can fall back to instantiator-built choices).
    """
    fn = CHOICE_REGISTRY.get(name)
    if fn is None:
        return []
    try:
        return list(fn(task_instance, scene_ctx, rng))
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Concrete generators
# ---------------------------------------------------------------------------

@register_choice("binary_yes_no")
def _binary_yes_no(ti, ctx, rng) -> List[str]:
    correct = (ti.get("gt_answer") or ti.get("answer") or "yes").lower()
    return [correct, "no" if correct == "yes" else "yes"]


@register_choice("binary_same_different")
def _binary_same_different(ti, ctx, rng) -> List[str]:
    correct = ti.get("gt_answer", "same")
    return [correct, "different" if correct == "same" else "same"]


@register_choice("binary_complete_incomplete")
def _binary_complete_incomplete(ti, ctx, rng) -> List[str]:
    correct = ti.get("gt_answer", "complete")
    return [correct, "incomplete" if correct == "complete" else "complete"]


@register_choice("binary_continuous_separate")
def _binary_continuous_separate(ti, ctx, rng) -> List[str]:
    correct = ti.get("gt_answer", "continuous")
    return [correct, "separate" if correct == "continuous" else "continuous"]


@register_choice("pair_labels")
def _pair_labels(ti, ctx, rng) -> List[str]:
    # Use label fields (not ID fields) so choices are human-readable
    a_label = ti.get("obj_a_label")
    b_label = ti.get("obj_b_label")
    correct = ti.get("gt_answer")
    if not (correct and a_label and b_label):
        return []
    other = b_label if correct == a_label else a_label
    return [correct, other]


@register_choice("pair_from_triple")
def _pair_from_triple(ti, ctx, rng) -> List[str]:
    # T05 uses one subject plus two reference objects; choices are the refs.
    b_label = ti.get("ref_b_label")
    c_label = ti.get("ref_c_label")
    correct = ti.get("gt_answer")
    if not (correct and b_label and c_label):
        return []
    other = c_label if correct == b_label else b_label
    return [correct, other]


@register_choice("similar_aabb_labels")
def _similar_aabb_labels(ti, ctx, rng) -> List[str]:
    """Correct label + 2-3 other labels from the scene."""
    correct = ti.get("gt_answer") or ti.get("target_label")
    if not correct:
        return []
    existing = list(ti.get("choices") or [])
    if existing:
        out = [correct]
        for choice in existing:
            if str(choice).lower() != str(correct).lower() and choice not in out:
                out.append(choice)
            if len(out) >= 4:
                break
        if len(out) >= 2:
            return out
    seen = []
    for o in getattr(ctx, "objects", []):
        if o.label.lower() == correct.lower() or o.label in seen:
            continue
        seen.append(o.label)
        if len(seen) >= 3:
            break
    return [correct] + seen


@register_choice("directional_choices")
def _directional_choices(ti, ctx, rng) -> List[str]:
    correct = ti.get("gt_answer", "left")
    pool = ["left", "right", "ahead", "behind"]
    if correct not in pool:
        pool.append(correct)
    return [correct] + [p for p in pool if p != correct]


@register_choice("count_range")
def _count_range(ti, ctx, rng) -> List[str]:
    correct = str(ti.get("gt_answer", "1"))
    try:
        c = int(correct)
    except ValueError:
        return [correct]
    pool = sorted({c, max(0, c - 1), c + 1, c + 2})
    return [str(c)] + [str(x) for x in pool if x != c]


@register_choice("count_range_or_zone_comparison")
def _count_range_or_zone(ti, ctx, rng) -> List[str]:
    return _count_range(ti, ctx, rng)


@register_choice("count_and_arrangement")
def _count_and_arrangement(ti, ctx, rng) -> List[str]:
    correct = ti.get("gt_answer", "single")
    pool = ["single", "multiple"]
    if correct not in pool:
        pool.append(correct)
    return [correct] + [p for p in pool if p != correct]


@register_choice("contact_relations")
def _contact_relations(ti, ctx, rng) -> List[str]:
    correct = ti.get("gt_answer", "beside")
    pool = ["touching", "resting_on", "beside", "above"]
    if correct not in pool:
        pool.append(correct)
    return [correct] + [p for p in pool if p != correct]


@register_choice("config_shapes")
def _config_shapes(ti, ctx, rng) -> List[str]:
    correct = ti.get("gt_answer", "row")
    pool = ["row", "cluster", "ring", "L-shape"]
    if correct not in pool:
        pool.append(correct)
    return [correct] + [p for p in pool if p != correct]


@register_choice("property_choices")
def _property_choices(ti, ctx, rng) -> List[str]:
    correct = ti.get("gt_answer")
    distractors = ti.get("distractor_choices") or []
    if not correct:
        return []
    return [correct] + [d for d in distractors if d != correct]


@register_choice("label_face_choices")
def _label_face_choices(ti, ctx, rng) -> List[str]:
    return _property_choices(ti, ctx, rng)


@register_choice("back_face_property_choices")
def _back_face_property_choices(ti, ctx, rng) -> List[str]:
    return _property_choices(ti, ctx, rng)
