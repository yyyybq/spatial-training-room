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
from typing import List

import numpy as np

from .coverage import compute_coverage
from .potential import breadth_first_potential_search
from .template_spec import TemplateSpec
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
    for actions, views, phi in candidates[:64]:
        traj = [(np.asarray(v.position), np.asarray(v.target)) for v in views]
        cov = compute_coverage(template, traj, task_instance, scene_ctx,
                               submit_only=False)
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
