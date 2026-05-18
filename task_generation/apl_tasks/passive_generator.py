"""
APLPassiveGenerator — generates Instruction-Following (passive) APL tasks.

Supported P0 task types:
  - distance_absolute : "Move to N meters away from the <object>"
  - direction_face    : "Turn to look at the <object>"

P1 task types (partially supported):
  - relative_position : "Stand to the left/right/front/behind of the <object>"

Usage::

    gen = APLPassiveGenerator(
        scene_path="/data/.../0013_840910",
        config={"max_items_per_object": 3, "target_distances": [1.0, 2.0]}
    )
    tasks = gen.generate_batch(max_items=100)
    gen.save_batch_to_jsonl(tasks, "out/passive_tasks.jsonl")
"""

from __future__ import annotations

import math
import random
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base_generator import BaseAPLGenerator
from .apl_types import (
    PassiveTaskType,
    Difficulty,
    steps_to_difficulty,
    DISTANCE_TEMPLATES,
    FACE_TEMPLATES,
    RELATIVE_POSITION_TEMPLATES,
)
from ...core.data_types import APLPassiveTaskItem, ViewState, make_task_id
from ...core.scene_context import SceneContext, DEFAULT_CAMERA_HEIGHT
from ...action_space.action_primitives import ActionPrimitive
from ...action_space.action_sequences import GoalDirectedPlanner, build_turn_sequence, build_move_sequence


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class APLPassiveGenerator(BaseAPLGenerator):
    """
    Generates APL passive (instruction-following) tasks for a single scene.

    Config keys:
        target_distances        List[float]  distances to generate for DISTANCE tasks
                                             default: [0.5, 1.0, 1.5, 2.0]
        max_items_per_object    int          cap per object  (default 4)
        task_types              List[str]    which PassiveTaskType values to generate
                                             default: all P0+P1 types
        num_init_views          int          candidate init views per object (default 6)
        camera_height           float        (default 0.8)
        move_distance           float        action step in metres (default 0.5)
        turn_angle              float        action turn in degrees (default 45)
        max_steps               int          max sequence length (default 7)
        distance_tol            float        (default 0.20)
        angle_tol               float        (default 10.0)
        seed                    int          random seed
    """

    def __init__(self, scene_path: str, config: Dict[str, Any]):
        super().__init__(scene_path, config)
        self._rng = np.random.RandomState(config.get("seed", 42))
        self._py_rng = random.Random(config.get("seed", 42))

        self.target_distances: List[float] = config.get("target_distances", [0.5, 1.0, 1.5, 2.0])
        self.max_items_per_object: int = config.get("max_items_per_object", 4)
        self.num_init_views: int = config.get("num_init_views", 6)

        raw_types = config.get("task_types", [
            PassiveTaskType.DISTANCE_ABSOLUTE.value,
            PassiveTaskType.DIRECTION_FACE.value,
            PassiveTaskType.RELATIVE_POSITION.value,
        ])
        self.task_types: List[PassiveTaskType] = [PassiveTaskType(t) for t in raw_types]

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def generate_batch(
        self,
        max_items: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[APLPassiveTaskItem]:
        tasks: List[APLPassiveTaskItem] = []
        objects = self.scene_ctx.valid_objects

        if not objects:
            return tasks

        for obj in objects:
            if len(tasks) >= max_items:
                break
            per_obj = min(self.max_items_per_object, max_items - len(tasks))

            for task_type in self.task_types:
                if len(tasks) >= max_items:
                    break
                new_tasks = self._generate_for_object(obj, task_type, per_obj)
                tasks.extend(new_tasks)

        if filters:
            tasks = [t for t in tasks if self._apply_filters(t, filters)]

        return tasks[:max_items]

    def validate_task(self, task: APLPassiveTaskItem) -> Tuple[bool, Optional[str]]:
        if not task.action_sequence:
            return False, "Empty action sequence"
        if len(task.action_sequence) > self.action_config.max_sequence_length:
            return False, "Sequence too long"

        init_view = task.init_view
        is_valid, reason = self.validator.validate(
            [ActionPrimitive(a) for a in task.action_sequence], init_view
        )
        return is_valid, reason

    # -----------------------------------------------------------------------
    # Per-object generation
    # -----------------------------------------------------------------------

    def _generate_for_object(
        self,
        obj,
        task_type: PassiveTaskType,
        max_items: int,
    ) -> List[APLPassiveTaskItem]:
        if task_type == PassiveTaskType.DISTANCE_ABSOLUTE:
            return self._gen_distance_tasks(obj, max_items)
        elif task_type == PassiveTaskType.DIRECTION_FACE:
            return self._gen_face_tasks(obj, max_items)
        elif task_type == PassiveTaskType.RELATIVE_POSITION:
            return self._gen_relative_position_tasks(obj, max_items)
        return []

    # -----------------------------------------------------------------------
    # P0: Distance tasks
    # -----------------------------------------------------------------------

    def _gen_distance_tasks(self, obj, max_items: int) -> List[APLPassiveTaskItem]:
        tasks: List[APLPassiveTaskItem] = []
        obj_center = 0.5 * (obj.bmin + obj.bmax)

        for target_dist in self.target_distances:
            if len(tasks) >= max_items:
                break

            # Sample init positions at various random distances/angles
            init_positions = self.scene_ctx.sample_positions_in_room(
                num_points=self.num_init_views,
                camera_height=self.camera_height,
                rng=self._rng,
            )

            for init_pos in init_positions:
                if len(tasks) >= max_items:
                    break

                # Target positions: ring around object at target_dist
                target_positions = self.scene_ctx.sample_positions_around_object(
                    obj, target_dist,
                    num_points=4,
                    camera_height=self.camera_height,
                    distance_tolerance=0.15,
                    rng=self._rng,
                )
                if not target_positions:
                    continue

                target_pos = target_positions[0]
                # Camera faces the object in target view
                target_tgt = obj_center.copy()
                target_tgt[2] = self.camera_height
                target_view = self._make_view(target_pos, target_tgt)

                # Init view: camera faces general forward (toward object is fine too)
                init_tgt = obj_center.copy()
                init_tgt[2] = self.camera_height
                init_view = self._make_view(init_pos, init_tgt)

                # Plan sequence
                seq, _traj = self.planner.plan(init_view, target_view)

                if not seq or len(seq) > self.action_config.max_sequence_length:
                    continue

                # Validate
                is_ok, reason = self.validator.validate(
                    [ActionPrimitive(a) for a in seq], init_view
                )
                if not is_ok:
                    continue

                # Verify final distance is approximately correct
                final_view = _traj[-1]
                final_pos = np.array(final_view.position)
                actual_dist = float(np.linalg.norm(final_pos[:2] - obj_center[:2]))
                if abs(actual_dist - target_dist) > self.config.get("distance_tol", 0.30):
                    continue

                # Pick NL template
                templates = DISTANCE_TEMPLATES.get(
                    target_dist,
                    [f"Move to {target_dist:.1f} meters away from the {{object}}."]
                )
                instruction = self._py_rng.choice(templates).format(object=obj.label)
                difficulty = steps_to_difficulty(len(seq))

                task = APLPassiveTaskItem(
                    task_id=make_task_id("apl_passive"),
                    scene_name=self.scene_ctx.scene_name,
                    instruction=instruction,
                    instruction_type=PassiveTaskType.DISTANCE_ABSOLUTE.value,
                    init_view=init_view,
                    target_view=target_view,
                    action_sequence=[a.value for a in seq],
                    action_descriptions=self._action_descriptions(seq),
                    target_object=obj.label,
                    target_object_id=obj.id,
                    target_distance=target_dist,
                    difficulty=difficulty.value,
                    metadata={
                        "actual_final_distance": round(actual_dist, 3),
                        "scene_name": self.scene_ctx.scene_name,
                    },
                )
                tasks.append(task)

        return tasks

    # -----------------------------------------------------------------------
    # P0: Face / direction tasks
    # -----------------------------------------------------------------------

    def _gen_face_tasks(self, obj, max_items: int) -> List[APLPassiveTaskItem]:
        tasks: List[APLPassiveTaskItem] = []
        obj_center = 0.5 * (obj.bmin + obj.bmax)
        obj_center_2d = obj_center[:2]

        init_positions = self.scene_ctx.sample_positions_in_room(
            num_points=self.num_init_views,
            camera_height=self.camera_height,
            rng=self._rng,
        )

        for init_pos in init_positions:
            if len(tasks) >= max_items:
                break

            # Random initial heading (not facing the object)
            random_angle = self._rng.uniform(0, 2 * math.pi)
            init_fwd = np.array([math.cos(random_angle), math.sin(random_angle), 0.0])
            init_tgt = init_pos + init_fwd
            init_view = self._make_view(init_pos, init_tgt)

            # Target view: same position, facing the object
            target_tgt = np.array([obj_center[0], obj_center[1], self.camera_height])
            target_view = self._make_view(init_pos, target_tgt)

            # Check the initial heading is NOT already facing the object
            bearing = self.scene_ctx.bearing_to_object(init_pos, init_tgt, obj)
            if abs(bearing) < self.action_config.turn_angle * 0.5:
                continue  # already facing it — not interesting

            # Plan turns only
            seq, _traj = self.planner.plan_turn_to_face(init_view, obj_center)
            if not seq or len(seq) > self.action_config.max_sequence_length:
                continue

            instruction = self._py_rng.choice(FACE_TEMPLATES).format(object=obj.label)
            difficulty = steps_to_difficulty(len(seq))

            task = APLPassiveTaskItem(
                task_id=make_task_id("apl_passive"),
                scene_name=self.scene_ctx.scene_name,
                instruction=instruction,
                instruction_type=PassiveTaskType.DIRECTION_FACE.value,
                init_view=init_view,
                target_view=target_view,
                action_sequence=[a.value for a in seq],
                action_descriptions=self._action_descriptions(seq),
                target_object=obj.label,
                target_object_id=obj.id,
                target_direction="face",
                difficulty=difficulty.value,
                metadata={"initial_bearing_deg": round(float(bearing), 2)},
            )
            tasks.append(task)

        return tasks

    # -----------------------------------------------------------------------
    # P1: Relative position tasks
    # -----------------------------------------------------------------------

    def _gen_relative_position_tasks(self, obj, max_items: int) -> List[APLPassiveTaskItem]:
        """
        Generate tasks where the model must stand on a specific side of the object.
        Sides: left, right, front, behind (relative to object's long axis or
        global +Y axis as a proxy).
        """
        tasks: List[APLPassiveTaskItem] = []
        obj_center = 0.5 * (obj.bmin + obj.bmax)

        # Object "front" direction: use the longer horizontal axis as proxy
        extent = obj.bmax - obj.bmin
        if extent[0] >= extent[1]:
            obj_front = np.array([1.0, 0.0])   # front is along X
        else:
            obj_front = np.array([0.0, 1.0])   # front is along Y
        obj_right = np.array([-obj_front[1], obj_front[0]])

        side_offsets = {
            "front":  obj_front,
            "behind": -obj_front,
            "right":  obj_right,
            "left":   -obj_right,
        }

        stand_dist = 1.0  # stand ~1m from object side

        for side, direction_vec in side_offsets.items():
            if len(tasks) >= max_items:
                break

            # Target position = object centre + direction * stand_dist
            tgt_xy = obj_center[:2] + direction_vec * stand_dist
            tgt_pos = np.array([tgt_xy[0], tgt_xy[1], self.camera_height])

            if not self.scene_ctx.is_position_valid(tgt_pos):
                continue

            # Target view faces the object
            tgt_look = np.array([obj_center[0], obj_center[1], self.camera_height])
            target_view = self._make_view(tgt_pos, tgt_look)

            # Sample random init position
            init_positions = self.scene_ctx.sample_positions_in_room(
                num_points=self.num_init_views,
                camera_height=self.camera_height,
                rng=self._rng,
            )
            if not init_positions:
                continue

            init_pos = init_positions[0]
            init_tgt = np.array([obj_center[0], obj_center[1], self.camera_height])
            init_view = self._make_view(init_pos, init_tgt)

            # Plan sequence
            seq, _traj = self.planner.plan(init_view, target_view)
            if not seq or len(seq) > self.action_config.max_sequence_length:
                continue

            is_ok, _ = self.validator.validate(
                [ActionPrimitive(a) for a in seq], init_view
            )
            if not is_ok:
                continue

            templates = RELATIVE_POSITION_TEMPLATES.get(side, [f"Stand to the {side} of the {{object}}."])
            instruction = self._py_rng.choice(templates).format(object=obj.label)
            difficulty = steps_to_difficulty(len(seq))

            task = APLPassiveTaskItem(
                task_id=make_task_id("apl_passive"),
                scene_name=self.scene_ctx.scene_name,
                instruction=instruction,
                instruction_type=PassiveTaskType.RELATIVE_POSITION.value,
                init_view=init_view,
                target_view=target_view,
                action_sequence=[a.value for a in seq],
                action_descriptions=self._action_descriptions(seq),
                target_object=obj.label,
                target_object_id=obj.id,
                target_direction=side,
                difficulty=difficulty.value,
                metadata={"side": side},
            )
            tasks.append(task)

        return tasks

    # -----------------------------------------------------------------------
    # Filter helper
    # -----------------------------------------------------------------------

    def _apply_filters(self, task: APLPassiveTaskItem, filters: Dict[str, Any]) -> bool:
        if "difficulty" in filters:
            allowed = filters["difficulty"]
            if isinstance(allowed, str):
                allowed = [allowed]
            if task.difficulty not in allowed:
                return False
        if "max_steps" in filters:
            if task.num_steps > filters["max_steps"]:
                return False
        if "instruction_type" in filters:
            if task.instruction_type != filters["instruction_type"]:
                return False
        return True
