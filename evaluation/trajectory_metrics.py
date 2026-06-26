"""Trajectory-level action quality metrics.

This module implements four evaluation components:
1) Step Efficiency (steps_to_success)
2) Path Optimality (SPL)
3) Information Efficiency (belief reduction / information gain per step)
4) Counterfactual action regret (chosen vs random vs oracle)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .coverage import fraction_passed, resolve_slot
from .template_spec import TemplateSpec, expand_evidence_slots
from ..action_space.action_executor import ViewStateExecutor
from ..action_space.action_primitives import ActionConfig, ActionPrimitive
from ..core.data_types import ViewState
from ..core.scene_context import SceneContext


ViewTuple = Tuple[np.ndarray, np.ndarray]


DEFAULT_ACTION_SET: Tuple[ActionPrimitive, ...] = (
    ActionPrimitive.MOVE_FORWARD,
    ActionPrimitive.MOVE_BACKWARD,
    ActionPrimitive.MOVE_LEFT,
    ActionPrimitive.MOVE_RIGHT,
    ActionPrimitive.TURN_LEFT,
    ActionPrimitive.TURN_RIGHT,
    ActionPrimitive.LOOK_UP,
    ActionPrimitive.LOOK_DOWN,
    ActionPrimitive.LOOK_FORWARD,
    ActionPrimitive.STOP,
)


@dataclass
class CounterfactualTrace:
    step: int
    chosen_action: str
    chosen_value: float
    random_value: float
    oracle_action: str
    oracle_value: float
    regret: float
    necessity: float
    redundant: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "chosen_action": self.chosen_action,
            "chosen_value": self.chosen_value,
            "random_value": self.random_value,
            "oracle_action": self.oracle_action,
            "oracle_value": self.oracle_value,
            "regret": self.regret,
            "necessity": self.necessity,
            "redundant": self.redundant,
        }


def _as_np_view(view_like: Any) -> Optional[ViewTuple]:
    if isinstance(view_like, dict):
        pos = view_like.get("position") or view_like.get("pos")
        tgt = view_like.get("target") or view_like.get("look_at")
    elif isinstance(view_like, ViewState):
        pos = view_like.position
        tgt = view_like.target
    elif isinstance(view_like, (list, tuple)) and len(view_like) == 2:
        pos, tgt = view_like
    else:
        return None

    if pos is None or tgt is None or len(pos) != 3 or len(tgt) != 3:
        return None
    try:
        p = np.asarray(pos, dtype=float)
        t = np.asarray(tgt, dtype=float)
    except Exception:
        return None
    return p, t


def normalize_trajectory(trajectory_like: Sequence[Any]) -> List[ViewTuple]:
    out: List[ViewTuple] = []
    for item in trajectory_like or []:
        parsed = _as_np_view(item)
        if parsed is not None:
            out.append(parsed)
    return out


def trajectory_path_length(trajectory: Sequence[ViewTuple]) -> float:
    if len(trajectory) < 2:
        return 0.0
    length = 0.0
    for i in range(len(trajectory) - 1):
        p0 = np.asarray(trajectory[i][0], dtype=float)
        p1 = np.asarray(trajectory[i + 1][0], dtype=float)
        length += float(np.linalg.norm(p1 - p0))
    return float(length)


def steps_to_success(trajectory: Sequence[ViewTuple], success: bool) -> Optional[int]:
    if not success:
        return None
    return max(0, len(trajectory) - 1)


def spl(success: bool, path_length: float, shortest_path_length: float) -> float:
    if not success:
        return 0.0
    if path_length <= 1e-9 and shortest_path_length <= 1e-9:
        return 1.0
    denom = max(path_length, shortest_path_length, 1e-9)
    return float(shortest_path_length / denom)


def belief_score(
    template: TemplateSpec,
    cam_pos: np.ndarray,
    cam_target: np.ndarray,
    task_instance: dict,
    scene_ctx: SceneContext,
) -> float:
    """Belief reduction score B(s) in [0, 1].

    Defined as the minimum per-slot fraction_passed at this state, so the score
    is bottlenecked by the least-satisfied slot.  Slots with a tier2_override
    are evaluated by invoking the Tier-2 callable on a single-frame trajectory
    (the current view), matching the submit-time semantics of that callable.
    """
    from .coverage import QUALITY_REGISTRY
    slots = expand_evidence_slots(template, task_instance)
    if not slots:
        return 1.0
    hfov = scene_ctx.default_hfov_deg()
    cp = np.asarray(cam_pos, dtype=float)
    ct = np.asarray(cam_target, dtype=float)
    single_frame = [(cp, ct)]
    fractions: List[float] = []
    for slot in slots:
        resolved = resolve_slot(slot, task_instance)
        if resolved.tier2_override:
            fn = QUALITY_REGISTRY.get(resolved.tier2_override)
            if fn is not None:
                frac = 1.0 if fn(resolved, single_frame, task_instance, scene_ctx) else 0.0
            else:
                frac = fraction_passed(resolved, cp, ct, hfov, scene_ctx)
        else:
            frac = fraction_passed(resolved, cp, ct, hfov, scene_ctx)
        fractions.append(float(frac))
    return float(min(fractions)) if fractions else 1.0


def information_gain_per_step(
    template: TemplateSpec,
    trajectory: Sequence[ViewTuple],
    task_instance: dict,
    scene_ctx: SceneContext,
) -> List[float]:
    if len(trajectory) < 2:
        return []
    bs = [
        belief_score(template, np.asarray(cp), np.asarray(ct), task_instance, scene_ctx)
        for (cp, ct) in trajectory
    ]
    return [float(bs[i + 1] - bs[i]) for i in range(len(bs) - 1)]


def _to_action_config(spec: TemplateSpec) -> ActionConfig:
    ac = spec.action_config
    return ActionConfig(move_distance=ac.move_m, turn_angle=ac.turn_deg, look_angle=ac.look_deg)


def _parse_action_name(v: Any) -> Optional[ActionPrimitive]:
    if v is None:
        return None
    s = str(v)
    for a in ActionPrimitive:
        if s == a.value or s.lower() == a.value:
            return a
    return None


def _valid_action_successors(
    state: ViewTuple,
    action_set: Iterable[ActionPrimitive],
    executor: ViewStateExecutor,
    scene_ctx: SceneContext,
) -> List[Tuple[ActionPrimitive, ViewTuple]]:
    cp, ct = np.asarray(state[0], dtype=float), np.asarray(state[1], dtype=float)
    base = ViewState(position=cp.tolist(), target=ct.tolist())

    out: List[Tuple[ActionPrimitive, ViewTuple]] = []
    for a in action_set:
        nxt = executor.execute(a, base)
        npos = np.asarray(nxt.position, dtype=float)
        ntgt = np.asarray(nxt.target, dtype=float)
        if a != ActionPrimitive.STOP and not scene_ctx.is_position_valid(npos):
            continue
        out.append((a, (npos, ntgt)))
    if not out:
        out.append((ActionPrimitive.STOP, (cp, ct)))
    return out


def counterfactual_regret(
    template: TemplateSpec,
    trajectory: Sequence[ViewTuple],
    task_instance: dict,
    scene_ctx: SceneContext,
    action_sequence: Optional[Sequence[Any]] = None,
    action_set: Iterable[ActionPrimitive] = DEFAULT_ACTION_SET,
    necessity_eps: float = 1e-6,
) -> Dict[str, Any]:
    """Compare chosen action value against random and oracle actions.

    V(a | s_t) = B(step(s_t, a)) - B(s_t)
    where B is belief_score.
    """
    if len(trajectory) < 2:
        return {
            "num_steps": 0,
            "mean_regret": 0.0,
            "cumulative_regret": 0.0,
            "mean_necessity": 0.0,
            "redundant_step_ratio": 0.0,
            "trace": [],
        }

    executor = ViewStateExecutor(_to_action_config(template))
    parsed_actions = [
        _parse_action_name(a) for a in (action_sequence or [])
    ]

    trace: List[CounterfactualTrace] = []
    for t in range(len(trajectory) - 1):
        s_t = trajectory[t]
        s_next = trajectory[t + 1]

        b_t = belief_score(template, s_t[0], s_t[1], task_instance, scene_ctx)
        b_next = belief_score(template, s_next[0], s_next[1], task_instance, scene_ctx)
        chosen_value = float(b_next - b_t)

        candidates = _valid_action_successors(s_t, action_set, executor, scene_ctx)
        valued: List[Tuple[ActionPrimitive, float]] = []
        for a, nxt in candidates:
            v = float(
                belief_score(template, nxt[0], nxt[1], task_instance, scene_ctx) - b_t
            )
            valued.append((a, v))

        oracle_action, oracle_value = max(valued, key=lambda x: x[1])
        random_value = float(np.mean([x[1] for x in valued]))
        regret = float(oracle_value - chosen_value)
        necessity = chosen_value

        chosen_action = parsed_actions[t] if t < len(parsed_actions) else None
        chosen_action_name = chosen_action.value if chosen_action is not None else "unknown"
        trace.append(
            CounterfactualTrace(
                step=t,
                chosen_action=chosen_action_name,
                chosen_value=chosen_value,
                random_value=random_value,
                oracle_action=oracle_action.value,
                oracle_value=float(oracle_value),
                regret=regret,
                necessity=necessity,
                redundant=necessity <= necessity_eps,
            )
        )

    regrets = [x.regret for x in trace]
    necessities = [x.necessity for x in trace]
    redundant = [1.0 if x.redundant else 0.0 for x in trace]
    return {
        "num_steps": len(trace),
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "cumulative_regret": float(np.sum(regrets)) if regrets else 0.0,
        "mean_necessity": float(np.mean(necessities)) if necessities else 0.0,
        "redundant_step_ratio": float(np.mean(redundant)) if redundant else 0.0,
        "trace": [x.to_dict() for x in trace],
    }
