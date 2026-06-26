"""
evaluation/expert.py

Find an expert (gold) trajectory from `init_view` that — using only the
template's allowed action_config — drives Φ as high as possible while keeping
the trajectory short.

API
---
    find_expert_trajectory(template, init_view, task_instance, scene_ctx,
                           max_steps=None, beam_width=8)
        → ExpertTrajectoryResult
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import random
from typing import Any, Dict, List

import numpy as np

from .coverage import compute_coverage, resolve_slot, slot_satisfied
from .potential import breadth_first_potential_search
from .template_spec import TemplateSpec, expand_evidence_slots
from ..action_space.action_primitives import ActionConfig, ActionPrimitive
from ..core import scene_context_ext  # noqa: F401
from ..core.data_types import ViewState
from ..core.scene_context import SceneContext


@dataclass
class ExpertTrajectoryResult:
    actions: List[ActionPrimitive] = field(default_factory=list)
    view_sequence: List[ViewState] = field(default_factory=list)
    final_potential: float = 0.0
    final_coverage: float = 0.0
    found: bool = False
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _make_view(position: np.ndarray, target: np.ndarray) -> ViewState:
    fwd = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    norm = float(np.linalg.norm(fwd))
    forward = (fwd / norm).tolist() if norm > 1e-6 else [0.0, 1.0, 0.0]
    return ViewState(
        position=np.asarray(position, dtype=float).tolist(),
        target=np.asarray(target, dtype=float).tolist(),
        forward=forward,
    )


def _view_tuple(view: ViewState) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(view.position, dtype=float), np.asarray(view.target, dtype=float)


def _heading_delta_deg(a: ViewState, b: ViewState) -> float:
    pa, ta = _view_tuple(a)
    pb, tb = _view_tuple(b)
    fa = ta - pa
    fb = tb - pb
    if np.linalg.norm(fa[:2]) < 1e-8 or np.linalg.norm(fb[:2]) < 1e-8:
        return 0.0
    ya = math.degrees(math.atan2(fa[1], fa[0]))
    yb = math.degrees(math.atan2(fb[1], fb[0]))
    return (yb - ya + 180.0) % 360.0 - 180.0


def _turn_actions(delta_deg: float, turn_deg: float) -> List[ActionPrimitive]:
    if abs(delta_deg) < max(1.0, turn_deg * 0.5):
        return []
    n = max(1, int(round(abs(delta_deg) / max(turn_deg, 1e-6))))
    act = ActionPrimitive.TURN_LEFT if delta_deg > 0 else ActionPrimitive.TURN_RIGHT
    return [act] * n


def _segment_actions(a: ViewState, b: ViewState, move_m: float, turn_deg: float) -> List[ActionPrimitive]:
    pa, _ = _view_tuple(a)
    pb, _ = _view_tuple(b)
    dist = float(np.linalg.norm((pb - pa)[:2]))
    n_move = max(1, int(math.ceil(dist / max(move_m, 1e-6)))) if dist > 1e-4 else 0
    return [ActionPrimitive.MOVE_FORWARD] * n_move


def _interpolate_segment(
    start: ViewState,
    end: ViewState,
    scene_ctx: SceneContext,
    move_m: float,
) -> List[ViewState] | None:
    """Return intermediate views from start(exclusive) to end(inclusive)."""
    p0, _ = _view_tuple(start)
    p1, t1 = _view_tuple(end)
    dist = float(np.linalg.norm((p1 - p0)[:2]))
    n = max(1, int(math.ceil(dist / max(move_m, 1e-6))))
    views: List[ViewState] = []
    for i in range(1, n + 1):
        alpha = i / n
        pos = (1.0 - alpha) * p0 + alpha * p1
        target = t1 if i == n else pos + (t1 - p1)
        if not scene_ctx.is_position_valid(pos):
            return None
        views.append(_make_view(pos, target))
    return views


def _trajectory_arrays(views: List[ViewState]) -> List[tuple[np.ndarray, np.ndarray]]:
    return [_view_tuple(v) for v in views]


def _finalize_stitched_views(
    template: TemplateSpec,
    task_instance: dict,
    scene_ctx: SceneContext,
    init_view: ViewState,
    waypoints: List[ViewState],
    max_steps: int,
    diagnostics: Dict[str, Any],
) -> ExpertTrajectoryResult:
    ac = template.action_config
    views: List[ViewState] = [init_view]
    actions: List[ActionPrimitive] = []
    current = init_view
    for wp in waypoints:
        segment = _interpolate_segment(current, wp, scene_ctx, ac.move_m)
        if segment is None:
            return ExpertTrajectoryResult(
                view_sequence=[init_view],
                found=False,
                diagnostics={
                    **diagnostics,
                    "failed": "invalid_interpolated_position",
                    "num_waypoints": len(waypoints),
                },
            )
        actions.extend(_segment_actions(current, wp, ac.move_m, ac.turn_deg))
        views.extend(segment)
        current = wp

    max_allowed_steps = max(max_steps * 3, len(waypoints))
    if len(views) - 1 > max_allowed_steps:
        return ExpertTrajectoryResult(
            actions=actions,
            view_sequence=views,
            found=False,
            diagnostics={
                **diagnostics,
                "failed": "too_many_steps",
                "steps": len(views) - 1,
                "max_allowed_steps": max_allowed_steps,
                "num_waypoints": len(waypoints),
            },
        )

    submit_only = getattr(template, "coverage_mode", "submit") != "trajectory"
    traj = _trajectory_arrays(views)
    cov = compute_coverage(template, traj, task_instance, scene_ctx,
                           submit_only=submit_only)
    found = cov >= template.min_coverage_for_credit
    if found:
        actions.append(ActionPrimitive.STOP)
    steps = max(0, len(views) - 1)
    return ExpertTrajectoryResult(
        actions=actions,
        view_sequence=views,
        final_potential=float(cov),
        final_coverage=float(cov),
        found=found,
        diagnostics={
            **diagnostics,
            "found": bool(found),
            "coverage": float(cov),
            "steps": steps,
            "num_waypoints": len(waypoints),
            "stable_full_coverage": bool(found),
            "coverage_range": [float(cov), float(cov)],
            "step_range_full_coverage": [
                steps if found else None,
                steps if found else None,
            ],
        },
    )


def _t27_target_scan_waypoints(
    template: TemplateSpec,
    init_view: ViewState,
    task_instance: dict,
    scene_ctx: SceneContext,
    max_steps: int,
    per_target_samples: int = 48,
) -> ExpertTrajectoryResult:
    from .predicates import Visible
    from .region_generators import sample_region

    rng = random.Random(2707)
    zone_targets = [
        (task_instance.get("zone_a_id"), list(task_instance.get("zone_a_targets") or [])),
        (task_instance.get("zone_b_id"), list(task_instance.get("zone_b_targets") or [])),
    ]
    waypoints: List[ViewState] = []
    missing: List[str] = []
    hfov = scene_ctx.default_hfov_deg()
    current = init_view
    for room_id, target_ids in zone_targets:
        for tid in target_ids:
            samples = sample_region(
                "around_object", scene_ctx, rng,
                target=tid, dist_min=0.8, dist_max=3.0, n=per_target_samples,
            )
            candidates: List[ViewState] = []
            for cp, ct in samples:
                cp_a = np.asarray(cp, dtype=float)
                ct_a = np.asarray(ct, dtype=float)
                if room_id and not scene_ctx.is_position_in_room_id(cp_a, room_id):
                    continue
                if Visible(cp_a, ct_a, hfov, scene_ctx,
                           obj=tid, min_corners=3, max_occ=0.40):
                    candidates.append(_make_view(cp_a, ct_a))
            if not candidates:
                missing.append(str(tid))
                continue
            cur_pos, _ = _view_tuple(current)
            chosen = min(
                candidates,
                key=lambda v: float(np.linalg.norm((_view_tuple(v)[0] - cur_pos)[:2])),
            )
            waypoints.append(chosen)
            current = chosen
    diagnostics = {
        "method": "waypoint_stitch",
        "variant": "t27_target_scan",
        "selected": {
            "per_target_samples": per_target_samples,
            "max_steps": max_steps,
        },
        "missing_target_waypoints": missing,
    }
    if missing or not waypoints:
        return ExpertTrajectoryResult(
            view_sequence=[init_view],
            found=False,
            diagnostics={**diagnostics, "failed": "missing_target_waypoint"},
        )
    return _finalize_stitched_views(
        template, task_instance, scene_ctx, init_view, waypoints, max_steps,
        diagnostics,
    )


def _candidate_views_for_slot(
    slot,
    template: TemplateSpec,
    task_instance: dict,
    scene_ctx: SceneContext,
    rng: random.Random,
    per_slot_samples: int,
) -> List[ViewState]:
    from .region_generators import sample_region

    resolved = resolve_slot(slot, task_instance)
    args = dict(resolved.region_args)
    args.setdefault("n", per_slot_samples)
    try:
        samples = sample_region(
            resolved.region_generator, scene_ctx, rng, **args
        )
    except Exception:
        return []
    hfov = scene_ctx.default_hfov_deg()
    out: List[ViewState] = []
    for cp, ct in samples:
        traj = [(np.asarray(cp, dtype=float), np.asarray(ct, dtype=float))]
        try:
            ok = slot_satisfied(
                resolved, traj, task_instance, scene_ctx, hfov,
                submit_only=getattr(template, "coverage_mode", "submit") != "trajectory",
            )
        except Exception:
            ok = False
        if ok:
            out.append(_make_view(np.asarray(cp), np.asarray(ct)))
    return out


def _slot_satisfaction_mask(
    view: ViewState,
    slots,
    task_instance: dict,
    scene_ctx: SceneContext,
) -> frozenset[int]:
    hfov = scene_ctx.default_hfov_deg()
    traj = [_view_tuple(view)]
    sat = []
    for idx, slot in enumerate(slots):
        try:
            if slot_satisfied(slot, traj, task_instance, scene_ctx, hfov, submit_only=False):
                sat.append(idx)
        except Exception:
            pass
    return frozenset(sat)


def _route_waypoints(
    init_view: ViewState,
    slots,
    candidates_by_slot: List[List[ViewState]],
    task_instance: dict,
    scene_ctx: SceneContext,
) -> List[ViewState] | None:
    n_slots = len(slots)
    if n_slots == 0:
        return None
    max_orders = 120
    slot_orders = itertools.permutations(range(n_slots))
    best_route = None
    best_len = math.inf
    for order_idx, order in enumerate(slot_orders):
        if order_idx >= max_orders:
            break
        current = init_view
        covered = set()
        route: List[ViewState] = []
        total_dist = 0.0
        for slot_idx in order:
            if slot_idx in covered:
                continue
            pool = candidates_by_slot[slot_idx]
            if not pool:
                break
            cur_pos, _ = _view_tuple(current)
            cand = min(
                pool,
                key=lambda v: float(np.linalg.norm((_view_tuple(v)[0] - cur_pos)[:2])),
            )
            cand_pos, _ = _view_tuple(cand)
            total_dist += float(np.linalg.norm((cand_pos - cur_pos)[:2]))
            route.append(cand)
            current = cand
            covered.update(_slot_satisfaction_mask(cand, slots, task_instance, scene_ctx))
        if len(covered) == n_slots and total_dist < best_len:
            best_route = route
            best_len = total_dist
    return best_route


def find_waypoint_stitched_trajectory(
    template: TemplateSpec,
    init_view: ViewState,
    task_instance: dict,
    scene_ctx: SceneContext,
    max_steps: int | None = None,
    per_slot_samples: int = 160,
) -> ExpertTrajectoryResult:
    """Build a trajectory by sampling evidence-slot waypoints and stitching them.

    This is intentionally conservative: the stitched trajectory is accepted
    only if every interpolated position is valid and the final coverage check
    meets the template's own threshold. Beam search remains the fallback.
    """
    if max_steps is None:
        max_steps = template.max_steps

    slots = [resolve_slot(s, task_instance) for s in expand_evidence_slots(template, task_instance)]
    if not slots:
        return ExpertTrajectoryResult(view_sequence=[init_view], found=False)

    if template.template_id == "T27":
        return _t27_target_scan_waypoints(
            template, init_view, task_instance, scene_ctx, max_steps=max_steps
        )

    seed_src = f"{template.template_id}:{task_instance.get('gt_answer', '')}:{len(slots)}"
    seed = int.from_bytes(seed_src.encode("utf-8")[:8], "little", signed=False)
    rng = random.Random(seed)
    candidates_by_slot = [
        _candidate_views_for_slot(
            slot, template, task_instance, scene_ctx, rng, per_slot_samples
        )
        for slot in slots
    ]
    if any(not cands for cands in candidates_by_slot):
        return ExpertTrajectoryResult(
            view_sequence=[init_view],
            found=False,
            diagnostics={
                "method": "waypoint_stitch",
                "failed": "no_candidate_for_slot",
                "candidate_counts": [len(c) for c in candidates_by_slot],
            },
        )

    waypoints = _route_waypoints(
        init_view, slots, candidates_by_slot, task_instance, scene_ctx
    )
    if not waypoints:
        return ExpertTrajectoryResult(
            view_sequence=[init_view],
            found=False,
            diagnostics={
                "method": "waypoint_stitch",
                "failed": "route_did_not_cover_slots",
                "candidate_counts": [len(c) for c in candidates_by_slot],
            },
        )

    return _finalize_stitched_views(
        template, task_instance, scene_ctx, init_view, waypoints, max_steps,
        {
            "method": "waypoint_stitch",
            "selected": {
                "per_slot_samples": per_slot_samples,
                "max_steps": max_steps,
            },
            "candidate_counts": [len(c) for c in candidates_by_slot],
        },
    )


def find_expert_trajectory(
    template: TemplateSpec,
    init_view: ViewState,
    task_instance: dict,
    scene_ctx: SceneContext,
    max_steps: int | None = None,
    beam_width: int = 8,
) -> ExpertTrajectoryResult:
    if max_steps is None:
        max_steps = template.max_steps

    ac = template.action_config
    action_config = ActionConfig(
        move_distance=ac.move_m,
        turn_angle=ac.turn_deg,
        look_angle=ac.look_deg,
    )

    candidates = breadth_first_potential_search(
        template, init_view, task_instance, scene_ctx,
        max_depth=max_steps, action_config=action_config,
        beam_width=beam_width,
    )

    # Pick the candidate with the BEST coverage; tie-break by shorter actions.
    # Use submit_only=False so multi-slot sequential tasks (e.g. T27 zone
    # counting) can earn full coverage from a trajectory that visits each zone
    # at different steps, rather than requiring all slots satisfied at the
    # final frame simultaneously (which is geometrically impossible).
    # Score ordering: (real_cov, -length, bfs_phi).  Putting length BEFORE
    # phi means that, at equal real coverage, the SHORTER trajectory wins.
    # This directly improves episode_score = cov_factor * γ^T which is
    # dominated by trajectory length once coverage is full.
    best = None
    best_cov = -1.0
    best_len_neg = -10**9
    best_phi = -1.0
    submit_only = getattr(template, "coverage_mode", "submit") != "trajectory"
    for actions, views, phi in candidates[:64]:
        traj = [(np.asarray(v.position), np.asarray(v.target)) for v in views]
        cov = compute_coverage(template, traj, task_instance, scene_ctx,
                               submit_only=submit_only)
        score = (cov, -len(actions), phi)
        cmp = (best_cov, best_len_neg, best_phi)
        if score > cmp:
            best = ExpertTrajectoryResult(
                actions=list(actions),
                view_sequence=list(views),
                final_potential=phi,
                final_coverage=cov,
                found=cov >= template.min_coverage_for_credit,
            )
            best_cov = cov
            best_len_neg = -len(actions)
            best_phi = phi
    if best is None:
        # Fallback: just return init view
        return ExpertTrajectoryResult(
            actions=[], view_sequence=[init_view], final_potential=0.0,
            final_coverage=0.0, found=False,
        )
    # Append STOP for clarity
    if best.actions and best.actions[-1] != ActionPrimitive.STOP:
        best.actions.append(ActionPrimitive.STOP)
    return best


def find_robust_expert_trajectory(
    template: TemplateSpec,
    init_view: ViewState,
    task_instance: dict,
    scene_ctx: SceneContext,
    max_steps: int | None = None,
) -> ExpertTrajectoryResult:
    """Run a small beam/depth sensitivity sweep and keep the best trajectory.

    This does not claim global path optimality. It guards generation against a
    single narrow beam pruning the good route by comparing several deterministic
    search budgets and recording whether they agree.
    """
    if max_steps is None:
        max_steps = template.max_steps

    stitched = find_waypoint_stitched_trajectory(
        template, init_view, task_instance, scene_ctx, max_steps=max_steps
    )
    if stitched.found:
        return stitched

    if getattr(template, "coverage_mode", "submit") == "trajectory":
        if template.template_id in {"T05", "T06", "T27"}:
            return stitched
        cfg = {"beam_width": 4, "max_steps": min(max_steps, 8)}
        res = find_expert_trajectory(
            template, init_view, task_instance, scene_ctx,
            max_steps=cfg["max_steps"], beam_width=cfg["beam_width"],
        )
        res.diagnostics = {
            "method": "waypoint_stitch_limited_fallback",
            "waypoint_stitch": dict(getattr(stitched, "diagnostics", {}) or {}),
            "limited_beam": {
                "beam_width": cfg["beam_width"],
                "max_steps": cfg["max_steps"],
                "found": res.found,
                "coverage": float(res.final_coverage),
                "steps": max(0, len(res.view_sequence) - 1),
                "potential": float(res.final_potential),
            },
            "stable_full_coverage": bool(res.found),
            "coverage_range": [float(res.final_coverage), float(res.final_coverage)],
            "step_range_full_coverage": [
                max(0, len(res.view_sequence) - 1) if res.found else None,
                max(0, len(res.view_sequence) - 1) if res.found else None,
            ],
        }
        return res

    configs = [
        {"beam_width": 8, "max_steps": max_steps},
        {"beam_width": 16, "max_steps": max_steps + 2},
    ]

    results: List[tuple[dict, ExpertTrajectoryResult]] = []
    for cfg in configs:
        res = find_expert_trajectory(
            template, init_view, task_instance, scene_ctx,
            max_steps=cfg["max_steps"], beam_width=cfg["beam_width"],
        )
        results.append((cfg, res))

    def _move_count(res: ExpertTrajectoryResult) -> int:
        return max(0, len(res.view_sequence) - 1)

    best_cfg, best = max(
        results,
        key=lambda item: (
            item[1].final_coverage,
            -_move_count(item[1]),
            item[1].final_potential,
        ),
    )
    successful = [res for _, res in results if res.found]
    lengths = [_move_count(res) for res in successful]
    coverages = [res.final_coverage for _, res in results]
    best.diagnostics = {
        "method": "beam_sensitivity",
        "waypoint_stitch": dict(getattr(stitched, "diagnostics", {}) or {}),
        "selected": best_cfg,
        "configs": [
            {
                "beam_width": cfg["beam_width"],
                "max_steps": cfg["max_steps"],
                "found": res.found,
                "coverage": float(res.final_coverage),
                "steps": _move_count(res),
                "potential": float(res.final_potential),
            }
            for cfg, res in results
        ],
        "stable_full_coverage": bool(
            successful and len(successful) == len(results)
            and max(lengths) == min(lengths)
        ),
        "coverage_range": [
            float(min(coverages)) if coverages else 0.0,
            float(max(coverages)) if coverages else 0.0,
        ],
        "step_range_full_coverage": [
            int(min(lengths)) if lengths else None,
            int(max(lengths)) if lengths else None,
        ],
    }
    return best
