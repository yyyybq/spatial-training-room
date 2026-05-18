"""
Goal-directed action sequence planning and validation.

Provides:
- GoalDirectedPlanner  — plan an action sequence from init_view to target_view
- SequenceValidator    — check that every step of a sequence stays in valid space
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from .action_primitives import ActionPrimitive, ActionConfig
from .action_executor import ViewStateExecutor
from ..core.data_types import ViewState
from ..core.scene_context import SceneContext


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_DISTANCE_TOL = 0.20    # metres – considered "at target distance"
_DEFAULT_ANGLE_TOL = 10.0       # degrees – considered "facing target"
_GREEDY_MAX_STEPS = 20          # safety cap for the greedy planner


# ---------------------------------------------------------------------------
# Greedy planner: init_view → target_view
# ---------------------------------------------------------------------------

class GoalDirectedPlanner:
    """
    Plans a short action sequence that brings `init_view` close to `target_view`.

    Strategy (greedy):
    1. If the heading differs by > angle_tol → turn left/right to reduce it.
    2. If the position differs by > distance_tol → move forward/backward.
    3. Repeat until within tolerance or max_steps reached.

    The planner operates in the discretised action space defined by `config`.
    """

    def __init__(
        self,
        config: ActionConfig,
        distance_tol: float = _DEFAULT_DISTANCE_TOL,
        angle_tol: float = _DEFAULT_ANGLE_TOL,
        max_steps: int = _GREEDY_MAX_STEPS,
    ):
        self.config = config
        self.executor = ViewStateExecutor(config)
        self.distance_tol = distance_tol
        self.angle_tol = angle_tol
        self.max_steps = max_steps

    def plan(
        self, init_view: ViewState, target_view: ViewState
    ) -> Tuple[List[ActionPrimitive], List[ViewState]]:
        """
        Return (action_sequence, trajectory) where trajectory[0] = init_view.

        If target cannot be reached within max_steps an empty sequence is returned
        but the partial trajectory is still provided.
        """
        sequence: List[ActionPrimitive] = []
        trajectory: List[ViewState] = [init_view]
        current = init_view

        target_pos = np.array(target_view.position, dtype=float)
        target_fwd = _forward_from_view(target_view)

        for _ in range(self.max_steps):
            cur_pos = np.array(current.position, dtype=float)
            cur_fwd = _forward_from_view(current)

            # --- Check termination ---
            pos_dist = float(np.linalg.norm(target_pos[:2] - cur_pos[:2]))
            angle_diff = _signed_angle_deg(cur_fwd, target_fwd)

            if pos_dist <= self.distance_tol and abs(angle_diff) <= self.angle_tol:
                break  # close enough

            # --- Decide next action ---
            # Priority: align heading first, then move
            if abs(angle_diff) > self.angle_tol:
                action = ActionPrimitive.TURN_LEFT if angle_diff > 0 else ActionPrimitive.TURN_RIGHT
            else:
                # Check if moving forward gets us closer
                test_fwd = self.executor.execute(ActionPrimitive.MOVE_FORWARD, current)
                test_back = self.executor.execute(ActionPrimitive.MOVE_BACKWARD, current)
                d_fwd = np.linalg.norm(
                    np.array(test_fwd.position[:2]) - target_pos[:2]
                )
                d_back = np.linalg.norm(
                    np.array(test_back.position[:2]) - target_pos[:2]
                )
                if d_fwd <= d_back:
                    action = ActionPrimitive.MOVE_FORWARD
                else:
                    action = ActionPrimitive.MOVE_BACKWARD

            sequence.append(action)
            current = self.executor.execute(action, current)
            trajectory.append(current)

        return sequence, trajectory

    def plan_turn_to_face(
        self, init_view: ViewState, target_object_pos: np.ndarray
    ) -> Tuple[List[ActionPrimitive], List[ViewState]]:
        """
        Plan turns so that the camera faces `target_object_pos`.
        No translation — only yaw turns.
        """
        sequence: List[ActionPrimitive] = []
        trajectory: List[ViewState] = [init_view]
        current = init_view

        for _ in range(self.max_steps):
            cur_pos = np.array(current.position, dtype=float)
            cur_tgt = np.array(current.target, dtype=float)
            desired_fwd = target_object_pos[:2] - cur_pos[:2]
            desired_fwd_norm = np.linalg.norm(desired_fwd)
            if desired_fwd_norm < 1e-6:
                break
            desired_fwd = desired_fwd / desired_fwd_norm

            cur_fwd = _forward_from_view(current)
            angle_diff = _signed_angle_deg(cur_fwd, desired_fwd)

            if abs(angle_diff) <= self.angle_tol:
                break

            action = ActionPrimitive.TURN_LEFT if angle_diff > 0 else ActionPrimitive.TURN_RIGHT
            sequence.append(action)
            current = self.executor.execute(action, current)
            trajectory.append(current)

        return sequence, trajectory


# ---------------------------------------------------------------------------
# Sequence validator
# ---------------------------------------------------------------------------

class SequenceValidator:
    """
    Validates that every step in an action sequence remains in valid scene space.
    """

    def __init__(self, scene_context: SceneContext, config: ActionConfig):
        self.ctx = scene_context
        self.executor = ViewStateExecutor(config)

    def validate(
        self,
        actions: List[ActionPrimitive],
        init_view: ViewState,
        min_wall_dist: float = 0.10,
        min_object_dist: float = 0.15,
    ) -> Tuple[bool, str]:
        """
        Simulate the sequence and check that each intermediate position is valid.

        Returns:
            (is_valid, reason_if_invalid)
        """
        current = init_view
        for i, action in enumerate(actions):
            if action == ActionPrimitive.STOP:
                break
            current = self.executor.execute(action, current)
            pos = np.array(current.position, dtype=float)

            if not self.ctx.is_position_valid(pos, min_wall_dist, min_object_dist):
                return False, f"Step {i}: position {pos.tolist()} is invalid (out of room or too close to obstacle)"

        return True, ""

    def filter_valid(
        self,
        sequences: List[List[ActionPrimitive]],
        init_view: ViewState,
    ) -> List[List[ActionPrimitive]]:
        """Filter out sequences that fail validation."""
        return [seq for seq in sequences if self.validate(seq, init_view)[0]]


# ---------------------------------------------------------------------------
# Turn-angle sequence builder (utility)
# ---------------------------------------------------------------------------

def build_turn_sequence(
    angle_deg: float,
    step_deg: float = 45.0,
) -> List[ActionPrimitive]:
    """
    Build a minimal sequence of discrete turns to cover `angle_deg`.
    Positive angle → turn left; negative → turn right.
    """
    sequence: List[ActionPrimitive] = []
    remaining = angle_deg
    action = ActionPrimitive.TURN_LEFT if remaining > 0 else ActionPrimitive.TURN_RIGHT
    while abs(remaining) > step_deg * 0.5:
        sequence.append(action)
        remaining -= math.copysign(step_deg, remaining)
    return sequence


def build_move_sequence(
    distance: float,
    step_dist: float = 0.5,
    forward: bool = True,
) -> List[ActionPrimitive]:
    """
    Build a sequence of move actions to cover `distance` metres.
    """
    action = ActionPrimitive.MOVE_FORWARD if forward else ActionPrimitive.MOVE_BACKWARD
    n_steps = max(1, round(distance / step_dist))
    return [action] * n_steps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _forward_from_view(view: ViewState) -> np.ndarray:
    """Return normalised 2-D forward direction of `view`."""
    pos = np.array(view.position[:2], dtype=float)
    tgt = np.array(view.target[:2], dtype=float)
    fwd = tgt - pos
    norm = np.linalg.norm(fwd)
    if norm < 1e-6:
        return np.array([0.0, 1.0])
    return fwd / norm


def _signed_angle_deg(fwd_current: np.ndarray, fwd_desired: np.ndarray) -> float:
    """
    Signed angle (degrees) from `fwd_current` to `fwd_desired`.
    Positive = rotate counter-clockwise (left turn).
    """
    cos_a = float(np.clip(np.dot(fwd_current, fwd_desired), -1.0, 1.0))
    angle = math.degrees(math.acos(cos_a))
    cross_z = fwd_current[0] * fwd_desired[1] - fwd_current[1] * fwd_desired[0]
    return angle if cross_z >= 0 else -angle
