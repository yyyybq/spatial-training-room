"""
evaluation/potential.py

Potential function Φ(s) for shaping.

Definition
----------
Φ(s) = average across slots of best per-frame fraction-passed at state s.

For multi-frame trajectories Φ is computed at the latest state (the result is
"per-state", same as the look-ahead used by step_rewards).

This file additionally exposes:

    breadth_first_potential_search(template, init_view, task_instance,
                                   scene_ctx, max_depth, action_config)
        → List[(action_seq, terminal_view, potential)]

which BFS-expands ActionPrimitive transitions starting from init_view to
build candidate trajectories. Used by `expert.py` to find a high-potential
trajectory.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .coverage import (
    QUALITY_REGISTRY,
    fraction_passed,
    resolve_slot,
)
from .template_spec import TemplateSpec, ActionConfig as TplActionConfig, expand_evidence_slots
from ..action_space.action_primitives import ActionConfig, ActionPrimitive
from ..action_space.action_executor import ViewStateExecutor
from ..core import scene_context_ext  # noqa: F401
from ..core.data_types import ViewState
from ..core.scene_context import SceneContext


# ---------------------------------------------------------------------------
# Per-state potential
# ---------------------------------------------------------------------------

def potential_at(
    template: TemplateSpec,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    task_instance: dict,
    scene_ctx: SceneContext,
) -> float:
    """
    Φ at a single (cam_pos, cam_target). Average per-slot fraction passed.
    Tier-2 overrides are NOT used at the per-state level (they're holistic);
    we approximate with predicates only here.
    """
    slots = expand_evidence_slots(template, task_instance)
    if not slots:
        return 1.0
    hfov = scene_ctx.default_hfov_deg()
    fractions: List[float] = []
    for slot in slots:
        resolved = resolve_slot(slot, task_instance)
        fractions.append(
            fraction_passed(resolved, cam_pos, cam_target, hfov, scene_ctx)
        )
    return float(np.mean(fractions))


def compute_potential(
    template: TemplateSpec,
    trajectory: Sequence[Tuple[np.ndarray, np.ndarray]],
    task_instance: dict,
    scene_ctx: SceneContext,
) -> float:
    """Φ at the *latest* state in trajectory."""
    if not trajectory:
        return 0.0
    cp, ct = trajectory[-1]
    return potential_at(template, np.asarray(cp), np.asarray(ct),
                        task_instance, scene_ctx)


# ---------------------------------------------------------------------------
# BFS in action space (used by expert.py)
# ---------------------------------------------------------------------------

def _round_state_key(view: ViewState, pos_step: float, ang_step_deg: float):
    """Discretise state for visited-set."""
    p = np.asarray(view.position, dtype=float)
    f = np.asarray(view.target, dtype=float) - p
    if np.linalg.norm(f) < 1e-9:
        yaw = 0
    else:
        yaw = np.arctan2(f[1], f[0])
    yaw_deg = np.degrees(yaw)
    return (
        round(float(p[0]) / pos_step),
        round(float(p[1]) / pos_step),
        round(float(p[2]) / max(pos_step, 0.1)),
        round(float(yaw_deg) / ang_step_deg),
    )


_DEFAULT_ACTIONS = (
    ActionPrimitive.MOVE_FORWARD,
    ActionPrimitive.MOVE_BACKWARD,
    ActionPrimitive.MOVE_LEFT,
    ActionPrimitive.MOVE_RIGHT,
    ActionPrimitive.TURN_LEFT,
    ActionPrimitive.TURN_RIGHT,
)


def breadth_first_potential_search(
    template: TemplateSpec,
    init_view: ViewState,
    task_instance: dict,
    scene_ctx: SceneContext,
    max_depth: int = 8,
    action_config: ActionConfig | None = None,
    beam_width: int = 8,
):
    """
    Beam search in action space: at each depth keep top-`beam_width` partial
    trajectories ranked by a trajectory-aware score.

    Score = cov_so_far * 2 + phi_instant
    where:
      - cov_so_far = fraction of slots satisfied at ANY prior frame in this beam
      - phi_instant = instantaneous potential at the current frame (gradient info)

    This allows multi-slot sequential tasks (e.g. T27 zone-counting) to escape
    from the first zone toward the second: once zone_a is visited cov_so_far=0.5
    even during transit, so beams heading for zone_b score higher once they
    arrive (cov_so_far→1.0) and are never pruned below beams camping in zone_a.

    Returns list of:
        (action_seq, view_seq, final_potential)
    sorted by final_potential descending.
    """
    if action_config is None:
        ac = template.action_config
        action_config = ActionConfig(
            move_distance=ac.move_m,
            turn_angle=ac.turn_deg,
            look_angle=ac.look_deg,
        )
    executor = ViewStateExecutor(action_config)

    slots = expand_evidence_slots(template, task_instance)
    hfov = scene_ctx.default_hfov_deg()

    # Precompute region centroids for proximity-based gradient.
    # Allows BFS to navigate toward unsatisfied zone centres even when
    # InRoom / Visible predicates are 0 everywhere in the corridor.
    from .region_generators import sample_region as _sr
    import random as _rand_
    _rng_pre = _rand_.Random(7)
    _slot_centroids: List[np.ndarray | None] = []
    for _sl in slots:
        try:
            _rs = resolve_slot(_sl, task_instance)
            _samps = _sr(_rs.region_generator, scene_ctx, _rng_pre, **_rs.region_args)
            if _samps:
                _slot_centroids.append(
                    np.mean([np.asarray(_cp) for _cp, _ in _samps[:8]], axis=0)
                )
            else:
                _slot_centroids.append(None)
        except Exception:
            _slot_centroids.append(None)

    def _check_slot_sat(view: ViewState, slot_idx: int) -> bool:
        from .coverage import resolve_slot, slot_satisfied_at
        resolved = resolve_slot(slots[slot_idx], task_instance)
        return slot_satisfied_at(resolved,
                                 np.asarray(view.position),
                                 np.asarray(view.target),
                                 hfov, scene_ctx)

    def _prox_bonus(cam_pos_arr: np.ndarray, sat_mask_: frozenset) -> float:
        """Proximity bonus toward unsatisfied slot region centroids.
        Normalised to [0, 1] per unsatisfied slot via 1/(1+d).
        """
        total = 0.0
        for _i, _c in enumerate(_slot_centroids):
            if _i in sat_mask_ or _c is None:
                continue
            _d = float(np.linalg.norm(cam_pos_arr[:2] - _c[:2]))
            total += 1.0 / (1.0 + _d)
        return total / max(n_slots, 1)

    def _phi_remaining(view: ViewState, sat_mask_: frozenset) -> float:
        """Mean potential over UNSATISFIED slots only.

        This creates a gradient toward the *next* unvisited zone so that
        beams heading from zone_a toward zone_b score HIGHER than beams
        camping in zone_a.  For single-slot templates the behaviour is
        identical to the old per-state potential.
        """
        remaining = [i for i in range(n_slots) if i not in sat_mask_]
        if not remaining:
            return 1.0  # all slots already satisfied
        fracs: List[float] = []
        for i in remaining:
            resolved = resolve_slot(slots[i], task_instance)
            fracs.append(
                fraction_passed(resolved,
                                np.asarray(view.position),
                                np.asarray(view.target),
                                hfov, scene_ctx)
            )
        return float(np.mean(fracs))

    def _beam_score(cov_so_far: float, phi_rem: float, prox: float) -> float:
        # cov_so_far (cumulative) dominates: beams that have already visited
        # more zones are always preferred.  phi_rem + prox guide navigation
        # toward remaining unsatisfied zones via predicate gradient + proximity.
        return cov_so_far * 4.0 + phi_rem + prox

    # Compute which slots are already satisfied at init
    n_slots = len(slots)
    if n_slots > 0:
        init_sat = frozenset(
            i for i in range(n_slots) if _check_slot_sat(init_view, i)
        )
    else:
        init_sat = frozenset()
    init_cov = len(init_sat) / max(n_slots, 1)
    init_phi = _phi_remaining(init_view, init_sat)
    init_prox = _prox_bonus(np.asarray(init_view.position), init_sat)

    # Beam item: (actions, views, score, visited, sat_mask)
    init_score = _beam_score(init_cov, init_phi, init_prox)
    beams = [([], [init_view], init_score, {_round_state_key(init_view, 0.25, 22.5)}, init_sat)]

    best_seen = [(list(beams[0][0]), list(beams[0][1]), init_cov)]

    for _depth in range(max_depth):
        candidates = []
        for actions, views, _score_v, visited, sat_mask in beams:
            cur = views[-1]
            for a in _DEFAULT_ACTIONS:
                new_view = executor.execute(a, cur)
                key = _round_state_key(new_view, 0.25, 22.5)
                if key in visited:
                    continue
                # Validity check
                pos = np.asarray(new_view.position)
                if not scene_ctx.is_position_valid(pos):
                    continue
                # Update cumulative slot satisfaction BEFORE scoring
                # so phi_rem reflects progress toward REMAINING zones.
                if n_slots > 0:
                    new_sat = sat_mask | frozenset(
                        i for i in range(n_slots)
                        if i not in sat_mask and _check_slot_sat(new_view, i)
                    )
                else:
                    new_sat = sat_mask
                phi_new = _phi_remaining(new_view, new_sat)
                prox_new = _prox_bonus(np.asarray(new_view.position), new_sat)
                cov_new = len(new_sat) / max(n_slots, 1)
                new_score = _beam_score(cov_new, phi_new, prox_new)
                new_visited = visited | {key}
                candidates.append((
                    actions + [a],
                    views + [new_view],
                    new_score,
                    new_visited,
                    new_sat,
                ))
        if not candidates:
            break
        # Keep top beam_width by trajectory-aware score
        candidates.sort(key=lambda x: x[2], reverse=True)
        beams = candidates[:beam_width]
        # Track top beams (report instantaneous phi for compatibility).
        # Expose more than just the top-1 so the expert can pick a SHORTER
        # trajectory that already meets full coverage even if a deeper beam
        # has marginally higher Φ.
        for actions, views, score_v, _, sat in beams[:3]:
            cov = len(sat) / max(n_slots, 1)
            best_seen.append((list(actions), list(views), cov))

    best_seen.sort(key=lambda x: x[2], reverse=True)
    return best_seen

