"""
evaluation/coverage.py

Slot-coverage computation for APL active templates.

Coverage definition
-------------------
For a template with N evidence slots and a trajectory τ = [(p_t, T_t), ...],
each slot s has:

    slot_satisfied(τ, s) = max over τ of fraction_passed(s, p, T)
                                   ≥ s.threshold

`fraction_passed` = (#predicates that hold) / |s.predicates|

Tier-2 overrides
----------------
If a slot specifies `tier2_override`, the named callable in QUALITY_REGISTRY
replaces the standard predicate evaluation. A Tier-2 callable takes:

    fn(slot, trajectory, task_instance, scene_ctx) -> bool
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .predicates import evaluate_predicate
from .template_spec import EvidenceSlot, TemplateSpec, expand_evidence_slots
from ..core import scene_context_ext  # noqa: F401
from ..core.scene_context import SceneContext


# Registry filled by `evaluation/quality_overrides.py`
QUALITY_REGISTRY: dict[str, callable] = {}


def register_quality(name: str):
    def deco(fn):
        QUALITY_REGISTRY[name] = fn
        return fn
    return deco


# ---------------------------------------------------------------------------
# Argument substitution helpers
# ---------------------------------------------------------------------------

def _resolve(value, task_instance: dict):
    """
    Replace any '{{var}}' string in `value` (recursively) with task_instance[var].
    """
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("{{") and s.endswith("}}"):
            key = s[2:-2].strip()
            return task_instance.get(key, value)
        return value
    if isinstance(value, dict):
        return {k: _resolve(v, task_instance) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, task_instance) for v in value]
    return value


def resolve_slot(slot: EvidenceSlot, task_instance: dict) -> EvidenceSlot:
    """Returns a NEW EvidenceSlot with all template variables substituted."""
    from .template_spec import PredicateSpec
    new_preds = [
        PredicateSpec(name=p.name, args=_resolve(p.args, task_instance))
        for p in slot.predicates
    ]
    return EvidenceSlot(
        slot_id=slot.slot_id,
        region_generator=slot.region_generator,
        region_args=_resolve(slot.region_args, task_instance),
        predicates=new_preds,
        threshold=slot.threshold,
        tier2_override=slot.tier2_override,
    )


# ---------------------------------------------------------------------------
# Slot satisfaction
# ---------------------------------------------------------------------------

def fraction_passed(
    slot: EvidenceSlot,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    hfov_deg: float,
    scene_ctx: SceneContext,
) -> float:
    if not slot.predicates:
        return 1.0
    n_pass = 0
    for p in slot.predicates:
        try:
            if evaluate_predicate(p.name, cam_pos, cam_target, hfov_deg,
                                  scene_ctx, **(p.args or {})):
                n_pass += 1
        except Exception:
            pass
    return n_pass / len(slot.predicates)


def slot_satisfied_at(
    slot: EvidenceSlot,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    hfov_deg: float,
    scene_ctx: SceneContext,
) -> bool:
    return fraction_passed(slot, cam_pos, cam_target, hfov_deg, scene_ctx) >= slot.threshold


def slot_almost_satisfied_at(
    slot: EvidenceSlot,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    hfov_deg: float,
    scene_ctx: SceneContext,
    relax_factor: float = 0.5,
) -> bool:
    """Relaxed version of ``slot_satisfied_at`` used by init-view samplers.

    The strict slot is satisfied when ``fraction_passed >= threshold``; the
    relaxed version accepts ``fraction_passed >= threshold * relax_factor``.
    For a 2-predicate slot with threshold 1.0 this means "at least one
    predicate passes" — i.e., the agent's view has *some* relationship to
    the slot's objects even though the slot itself does NOT count.

    This signal lets init-view samplers prefer candidates where the question
    objects are at least partially in awareness over candidates where the
    agent is staring at an empty wall.
    """
    if not slot.predicates:
        return False
    bar = slot.threshold * float(relax_factor)
    return fraction_passed(slot, cam_pos, cam_target, hfov_deg, scene_ctx) >= bar


def slot_satisfied(
    slot: EvidenceSlot,
    trajectory: Sequence[Tuple[np.ndarray, np.ndarray]],
    task_instance: dict,
    scene_ctx: SceneContext,
    hfov_deg: float | None = None,
    submit_only: bool = False,
) -> bool:
    """
    Returns True iff `slot` is satisfied.

    submit_only=False (legacy / informational):
        Slot counts as satisfied if ANY frame in trajectory passes it. Useful
        for diagnostics and for Φ in potential-shaped rewards (where partial
        progress is the whole point).

    submit_only=True (anti-cheat scoring path):
        Only the LAST view in trajectory is evaluated. This is what
        `episode_score` uses, enforcing the v3 rule "评分用提交时质量,
        避免驻留刷分".

    Tier-2 overrides receive the trajectory unchanged in either mode; their
    own implementation is responsible for honoring the submit-time semantics
    (the canonical pattern is `slot_satisfied_at(slot, *traj[-1], ...)`).
    """
    if hfov_deg is None:
        hfov_deg = scene_ctx.default_hfov_deg()
    resolved = resolve_slot(slot, task_instance)

    if resolved.tier2_override:
        fn = QUALITY_REGISTRY.get(resolved.tier2_override)
        if fn is None:
            raise KeyError(
                f"Tier-2 override '{resolved.tier2_override}' not registered. "
                f"Available: {list(QUALITY_REGISTRY)}"
            )
        # Tier-2 callables ALWAYS receive the full trajectory. Each callable
        # is responsible for honoring submit-time semantics internally
        # (canonical: read trajectory[-1] for the submit view, scan all
        # frames for trajectory-level properties such as "did agent cover
        # 270° of yaw inside room_b").
        return bool(fn(resolved, trajectory, task_instance, scene_ctx))

    if submit_only:
        if not trajectory:
            return False
        cp, ct = trajectory[-1]
        return slot_satisfied_at(resolved, np.asarray(cp), np.asarray(ct),
                                 hfov_deg, scene_ctx)

    for (cp, ct) in trajectory:
        if slot_satisfied_at(resolved, np.asarray(cp), np.asarray(ct),
                             hfov_deg, scene_ctx):
            return True
    return False


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def compute_coverage(
    template: TemplateSpec,
    trajectory: Sequence[Tuple[np.ndarray, np.ndarray]],
    task_instance: dict,
    scene_ctx: SceneContext,
    submit_only: bool = False,
) -> float:
    """
    Aggregate coverage in [0, 1].

    submit_only:
      • False (default) — informational coverage (any-frame). Used by Φ.
      • True  — submit-time coverage (last frame only). Used by `episode_score`.

    coverage_aggregator:
      • 'all'  → product of slot indicators (all-or-nothing per slot)
      • 'mean' → mean of slot indicators
    """
    slots = expand_evidence_slots(template, task_instance)
    if not slots:
        return 1.0
    hfov = scene_ctx.default_hfov_deg()
    flags = [
        1.0 if slot_satisfied(s, trajectory, task_instance, scene_ctx, hfov,
                              submit_only=submit_only) else 0.0
        for s in slots
    ]
    if template.coverage_aggregator == "mean":
        return float(np.mean(flags))
    return float(np.prod(flags))   # default: all
