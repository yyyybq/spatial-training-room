"""
BaseAPLGenerator — common base for all APL task generators.

Extends core.BaseTaskGenerator with APL-specific helpers:
- scene context management
- action executor setup
- sequence validation
- JSONL I/O
"""

from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.task_base import BaseTaskGenerator
from ..core.scene_context import SceneContext
from ..core.data_types import ViewState
from ..action_space.action_primitives import ActionConfig, ActionPrimitive
from ..action_space.action_executor import ViewStateExecutor
from ..action_space.action_sequences import GoalDirectedPlanner, SequenceValidator


class BaseAPLGenerator(BaseTaskGenerator):
    """
    Shared base for APLPassiveGenerator and APLActiveGenerator.

    Adds:
    - self.scene_ctx : SceneContext
    - self.executor  : ViewStateExecutor
    - self.planner   : GoalDirectedPlanner
    - self.validator : SequenceValidator
    - helpers: _difficulty_from_steps, _assign_difficulty
    """

    def __init__(self, scene_path: str, config: Dict[str, Any]):
        super().__init__(scene_path, config)

        # Full scene context (loads geometry, objects, etc.)
        self.scene_ctx = SceneContext.load(scene_path)

        # Action space setup
        action_cfg = ActionConfig(
            move_distance=config.get("move_distance", 0.5),
            turn_angle=config.get("turn_angle", 45.0),
            look_angle=config.get("look_angle", 15.0),
            max_sequence_length=config.get("max_steps", 7),
        )
        self.action_config = action_cfg
        self.executor = ViewStateExecutor(action_cfg)
        self.planner = GoalDirectedPlanner(
            action_cfg,
            distance_tol=config.get("distance_tol", 0.20),
            angle_tol=config.get("angle_tol", 10.0),
        )
        self.validator = SequenceValidator(self.scene_ctx, action_cfg)

        self.camera_height: float = config.get("camera_height", 0.8)
        self.min_visible_corners: int = config.get("min_visible_corners", 2)
        self.max_occ_ratio: float = config.get("max_occ_ratio", 0.80)

    # -----------------------------------------------------------------------
    # Abstract interface (subclasses must implement)
    # -----------------------------------------------------------------------

    @abstractmethod
    def generate_batch(
        self,
        max_items: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_task(self, task: Any) -> tuple:
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Shared helpers
    # -----------------------------------------------------------------------

    def _difficulty_from_steps(self, num_steps: int) -> str:
        if num_steps <= 1:
            return "easy"
        if num_steps <= 3:
            return "medium"
        if num_steps <= 5:
            return "hard"
        return "expert"

    def _make_view(self, position: np.ndarray, target: np.ndarray) -> ViewState:
        fwd = target - position
        norm = float(np.linalg.norm(fwd))
        fwd_list = (fwd / norm).tolist() if norm > 1e-6 else [0.0, 1.0, 0.0]
        return ViewState(
            position=position.tolist(),
            target=target.tolist(),
            forward=fwd_list,
        )

    def _action_descriptions(self, actions: List[ActionPrimitive]) -> List[str]:
        """Human-readable description for each action."""
        desc_map = {
            ActionPrimitive.MOVE_FORWARD:  f"Move forward {self.action_config.move_distance:.1f}m",
            ActionPrimitive.MOVE_BACKWARD: f"Move backward {self.action_config.move_distance:.1f}m",
            ActionPrimitive.MOVE_LEFT:     f"Strafe left {self.action_config.move_distance:.1f}m",
            ActionPrimitive.MOVE_RIGHT:    f"Strafe right {self.action_config.move_distance:.1f}m",
            ActionPrimitive.TURN_LEFT:     f"Turn left {self.action_config.turn_angle:.0f}°",
            ActionPrimitive.TURN_RIGHT:    f"Turn right {self.action_config.turn_angle:.0f}°",
            ActionPrimitive.LOOK_UP:       f"Look up {self.action_config.look_angle:.0f}°",
            ActionPrimitive.LOOK_DOWN:     f"Look down {self.action_config.look_angle:.0f}°",
            ActionPrimitive.LOOK_FORWARD:  "Reset pitch to horizontal",
            ActionPrimitive.STOP:          "Stop",
        }
        return [desc_map.get(a, str(a)) for a in actions]
