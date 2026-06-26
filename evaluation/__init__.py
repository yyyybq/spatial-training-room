"""
evaluation/__init__.py
Public API for the APL evaluation module.
"""

# Touch quality_overrides so its @register_quality decorators run on import
from . import quality_overrides   # noqa: F401

from .template_spec import (   # noqa: F401
    TemplateSpec, EvidenceSlot, PredicateSpec, ActionConfig,
    load_template, load_all_templates,
)
from .predicates import PREDICATE_REGISTRY, evaluate_predicate   # noqa: F401
from .region_generators import REGION_REGISTRY, sample_region    # noqa: F401
from .coverage import (   # noqa: F401
    QUALITY_REGISTRY, register_quality,
    fraction_passed, slot_satisfied_at, slot_almost_satisfied_at,
    slot_satisfied, compute_coverage, resolve_slot,
)
from .scorer import episode_score, step_rewards   # noqa: F401
from .potential import (   # noqa: F401
    potential_at, compute_potential, breadth_first_potential_search,
)
from .expert import (   # noqa: F401
    find_expert_trajectory,
    find_robust_expert_trajectory,
    ExpertTrajectoryResult,
)
from .trajectory_metrics import (   # noqa: F401
    DEFAULT_ACTION_SET,
    normalize_trajectory,
    trajectory_path_length,
    steps_to_success,
    spl,
    belief_score,
    information_gain_per_step,
    counterfactual_regret,
)

__all__ = [
    "TemplateSpec", "EvidenceSlot", "PredicateSpec", "ActionConfig",
    "load_template", "load_all_templates",
    "PREDICATE_REGISTRY", "evaluate_predicate",
    "REGION_REGISTRY", "sample_region",
    "QUALITY_REGISTRY", "register_quality",
    "fraction_passed", "slot_satisfied_at", "slot_almost_satisfied_at",
    "slot_satisfied", "compute_coverage", "resolve_slot",
    "episode_score", "step_rewards",
    "potential_at", "compute_potential", "breadth_first_potential_search",
    "find_expert_trajectory", "find_robust_expert_trajectory",
    "ExpertTrajectoryResult",
    "DEFAULT_ACTION_SET",
    "normalize_trajectory",
    "trajectory_path_length",
    "steps_to_success",
    "spl",
    "belief_score",
    "information_gain_per_step",
    "counterfactual_regret",
]
