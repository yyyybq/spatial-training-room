"""
ActionExecutor bridging the (position, target) ViewState API with ViewManipulator.

Two interfaces are provided:
- ViewStateExecutor  — pure numpy, no ViewManipulator dependency (fast, no scipy needed)
- ViewManipulatorBridge — wraps ViewManipulator for c2w-matrix-based execution
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .action_primitives import ActionPrimitive, ActionConfig
from ..core.data_types import ViewState
from ..utils.occlusion import camtoworld_from_pos_target


# ---------------------------------------------------------------------------
# Pure numpy executor  (position/target based)
# ---------------------------------------------------------------------------

class ViewStateExecutor:
    """
    Executes ActionPrimitive operations on (position, target) pairs.

    This is the primary executor used by all generators.  It does NOT depend
    on ViewManipulator so there is no scipy dependency.
    """

    def __init__(self, config: ActionConfig):
        self.config = config

    # ------------------------------------------------------------------
    def execute(
        self,
        action: ActionPrimitive,
        view: ViewState,
        distance_scale: float = 1.0,
        angle_scale: float = 1.0,
    ) -> ViewState:
        """Apply one action to `view` and return the new ViewState."""
        pos = np.array(view.position, dtype=float)
        tgt = np.array(view.target, dtype=float)

        new_pos, new_tgt = self._apply(action, pos, tgt, distance_scale, angle_scale)

        # Recompute forward vector
        fwd = new_tgt - new_pos
        norm = np.linalg.norm(fwd)
        fwd = (fwd / norm).tolist() if norm > 1e-6 else [0.0, 1.0, 0.0]

        return ViewState(
            position=new_pos.tolist(),
            target=new_tgt.tolist(),
            forward=fwd,
        )

    def execute_sequence(
        self,
        actions: List[ActionPrimitive],
        init_view: ViewState,
        distance_scale: float = 1.0,
        angle_scale: float = 1.0,
    ) -> List[ViewState]:
        """
        Execute a sequence of actions.  Returns list of ViewStates
        including the initial state (index 0).
        """
        trajectory: List[ViewState] = [init_view]
        current = init_view
        for action in actions:
            if action == ActionPrimitive.STOP:
                break
            current = self.execute(action, current, distance_scale, angle_scale)
            trajectory.append(current)
        return trajectory

    # ------------------------------------------------------------------
    # Internal math
    # ------------------------------------------------------------------

    def _apply(
        self,
        action: ActionPrimitive,
        pos: np.ndarray,
        tgt: np.ndarray,
        distance_scale: float,
        angle_scale: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Core geometry — all rotations done in-place on pos/tgt copies."""
        pos = pos.copy()
        tgt = tgt.copy()

        if action == ActionPrimitive.STOP:
            return pos, tgt

        # Derived axes
        fwd = tgt - pos
        fwd_len = np.linalg.norm(fwd)
        if fwd_len > 1e-6:
            fwd = fwd / fwd_len

        # Right = cross(forward_xy, Z)  — keep movement horizontal
        right = np.array([-fwd[1], fwd[0], 0.0], dtype=float)
        r_len = np.linalg.norm(right)
        if r_len > 1e-6:
            right = right / r_len

        move_dist = self.config.move_distance * distance_scale
        turn_rad = math.radians(self.config.turn_angle * angle_scale)
        look_rad = math.radians(self.config.look_angle * angle_scale)

        # --- Movement ---
        if action == ActionPrimitive.MOVE_FORWARD:
            delta = fwd * move_dist
            pos += delta; tgt += delta

        elif action == ActionPrimitive.MOVE_BACKWARD:
            delta = fwd * move_dist
            pos -= delta; tgt -= delta

        elif action == ActionPrimitive.MOVE_LEFT:
            delta = right * move_dist
            pos -= delta; tgt -= delta

        elif action == ActionPrimitive.MOVE_RIGHT:
            delta = right * move_dist
            pos += delta; tgt += delta

        # --- Yaw (turn) ---
        elif action in (ActionPrimitive.TURN_LEFT, ActionPrimitive.TURN_RIGHT):
            a = turn_rad if action == ActionPrimitive.TURN_LEFT else -turn_rad
            rel = tgt - pos
            cos_a, sin_a = math.cos(a), math.sin(a)
            new_x = rel[0] * cos_a - rel[1] * sin_a
            new_y = rel[0] * sin_a + rel[1] * cos_a
            tgt[0] = pos[0] + new_x
            tgt[1] = pos[1] + new_y

        # --- Pitch (look) ---
        elif action in (ActionPrimitive.LOOK_UP, ActionPrimitive.LOOK_DOWN):
            a = look_rad if action == ActionPrimitive.LOOK_UP else -look_rad
            rel = tgt - pos
            # Rotate around right axis
            fwd_xy_len = math.sqrt(rel[0] ** 2 + rel[1] ** 2)
            current_pitch = math.atan2(rel[2], fwd_xy_len)
            new_pitch = current_pitch + a
            # Clamp to ±75°
            new_pitch = max(-math.radians(75), min(math.radians(75), new_pitch))
            fwd_dir = fwd[:2] if np.linalg.norm(fwd[:2]) > 1e-6 else np.array([0.0, 1.0])
            fwd_dir = fwd_dir / np.linalg.norm(fwd_dir)
            rel_len = np.linalg.norm(rel)
            tgt[0] = pos[0] + fwd_dir[0] * rel_len * math.cos(new_pitch)
            tgt[1] = pos[1] + fwd_dir[1] * rel_len * math.cos(new_pitch)
            tgt[2] = pos[2] + rel_len * math.sin(new_pitch)

        elif action == ActionPrimitive.LOOK_FORWARD:
            tgt[2] = pos[2]  # flatten pitch

        return pos, tgt


# ---------------------------------------------------------------------------
# ViewManipulator bridge (optional, for c2w-matrix consumers)
# ---------------------------------------------------------------------------

class ViewManipulatorBridge:
    """
    Wraps ViewManipulator to expose the same (position, target) interface
    as ViewStateExecutor.  Import is lazy so scipy is not required at module level.
    """

    def __init__(self, config: ActionConfig):
        self.config = config
        self._vm = None  # lazy init

    def _get_vm(self):
        if self._vm is None:
            from ..motion.view_manipulator import ViewManipulator
            self._vm = ViewManipulator(
                step_translation=self.config.move_distance,
                step_rotation_deg=self.config.turn_angle,
                world_up_axis="Z",
                is_discrete=False,
                image_y_down=True,
            )
        return self._vm

    # Map ActionPrimitive → ViewManipulator step key
    _ACTION_MAP = {
        ActionPrimitive.MOVE_FORWARD: "w",
        ActionPrimitive.MOVE_BACKWARD: "s",
        ActionPrimitive.MOVE_LEFT: "a",
        ActionPrimitive.MOVE_RIGHT: "d",
        ActionPrimitive.TURN_LEFT: "q",
        ActionPrimitive.TURN_RIGHT: "e",
        ActionPrimitive.LOOK_UP: "r",
        ActionPrimitive.LOOK_DOWN: "f",
    }

    def execute(self, action: ActionPrimitive, view: ViewState) -> ViewState:
        if action == ActionPrimitive.STOP:
            return view

        vm = self._get_vm()
        c2w = camtoworld_from_pos_target(
            np.array(view.position), np.array(view.target)
        )
        vm.reset(c2w)

        key = self._ACTION_MAP.get(action)
        if key is None:
            return view
        new_c2w = vm.step(key)

        new_pos = new_c2w[:3, 3]
        fwd_cam = np.array([0.0, 0.0, 1.0])
        fwd_world = new_c2w[:3, :3] @ fwd_cam
        new_tgt = new_pos + fwd_world

        return ViewState(
            position=new_pos.tolist(),
            target=new_tgt.tolist(),
            forward=fwd_world.tolist(),
        )

    def execute_sequence(
        self, actions: List[ActionPrimitive], init_view: ViewState
    ) -> List[ViewState]:
        trajectory = [init_view]
        current = init_view
        for action in actions:
            if action == ActionPrimitive.STOP:
                break
            current = self.execute(action, current)
            trajectory.append(current)
        return trajectory
