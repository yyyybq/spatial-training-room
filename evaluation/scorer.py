"""
evaluation/scorer.py

Episode scoring + step-level potential-shaped rewards.

API
---
    episode_score(template, trajectory, predicted_answer, gt_answer,
                  task_instance, scene_ctx) -> float in [0, 1]

    step_rewards(template, trajectory, task_instance, scene_ctx,
                 episode_score_value) -> List[float]

Math note (v3 step-cost guarantee)
----------------------------------
Per-step discount γ implies an implicit per-step "cost" of (1−γ) on the
attainable terminal score. With γ = 0.95 this is 5 % of the maximum reward
per extra step. The shaping rewards r_t = Φ(s_{t+1}) − Φ(s_t) are bounded
in [−1, +1] because Φ ∈ [0, 1]. Hence:

    max single-step shaping gain   = +1   (numerically)
    max attainable terminal credit = γ^T  (≤ 1)

A *meaningful* gain at step t therefore corresponds to ΔΦ_t ≥ 1 − γ
(otherwise the implicit cost from one extra step erases it). Templates that
want a stricter budget should LOWER γ (e.g. 0.90 for a stiffer penalty);
nothing else needs to change.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .coverage import compute_coverage
from .potential import compute_potential
from .template_spec import TemplateSpec
from ..core import scene_context_ext  # noqa: F401
from ..core.scene_context import SceneContext


def _answers_equal(predicted, gt) -> bool:
    if predicted is None or gt is None:
        return False
    if isinstance(predicted, str) and isinstance(gt, str):
        return predicted.strip().lower() == gt.strip().lower()
    return predicted == gt


def episode_score(
    template: TemplateSpec,
    trajectory: Sequence[Tuple[np.ndarray, np.ndarray]],
    predicted_answer,
    gt_answer,
    task_instance: dict,
    scene_ctx: SceneContext,
) -> float:
    """
    Score = 1[â=a*] · CovFactor · γ^T

    * CovFactor = min(1, Coverage / min_coverage_for_credit)
    * T = effective trajectory length (number of submitted ViewStates − 1).
    """
    if not _answers_equal(predicted_answer, gt_answer):
        return 0.0
    coverage_mode = getattr(template, "coverage_mode", "submit")
    submit_only = coverage_mode != "trajectory"
    coverage = compute_coverage(template, trajectory, task_instance, scene_ctx,
                                submit_only=submit_only)
    min_cov = max(template.min_coverage_for_credit, 1e-9)
    cov_factor = min(1.0, coverage / min_cov)
    if cov_factor <= 0.0:
        return 0.0
    T = max(0, len(trajectory) - 1)
    return float(cov_factor * (template.gamma ** T))


def step_rewards(
    template: TemplateSpec,
    trajectory: Sequence[Tuple[np.ndarray, np.ndarray]],
    task_instance: dict,
    scene_ctx: SceneContext,
    episode_score_value: float,
) -> List[float]:
    """
    Potential-shaped rewards:

        r_t = Φ(s_{t+1}) - Φ(s_t)         for t = 0 … T-2
        r_{T-1} += episode_score_value     (sparse terminal)

    Φ is *not* persisted to JSONL, only the resulting r_t list.
    """
    if len(trajectory) < 2:
        return [episode_score_value] if trajectory else []

    phis = [compute_potential(template, [trajectory[i]], task_instance, scene_ctx)
            for i in range(len(trajectory))]
    rewards: List[float] = []
    for i in range(len(trajectory) - 1):
        rewards.append(float(phis[i + 1] - phis[i]))
    if rewards:
        rewards[-1] += float(episode_score_value)
    return rewards
