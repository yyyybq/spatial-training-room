"""
APLActiveGenerator — generates Question-Driven Navigation (active) APL tasks.

Legacy note:
  This module predates the template-driven Active APL system in
  template_active_generator.py. It is kept for backward compatibility with
  run_factory.py --mode apl. New release training data should use
  run_factory.py --mode template --template-id T?? or the template sweep tools.

Supported P0 task types:
  - visibility_single : "What is to your left/right/behind?"  (need to turn)
  - visibility_hidden : "What is [relationship] of [anchor]?" (need to move to see)

P1 task types (scaffolded):
  - spatial_distance  : "Is A closer to B or C?"
  - next_action       : "What should you do next to see X?"

Usage::

    gen = APLActiveGenerator(
        scene_path="/data/.../0013_840910",
        config={"num_candidates": 8}
    )
    tasks = gen.generate_batch(max_items=50)
    gen.save_batch_to_jsonl(tasks, "out/active_tasks.jsonl")
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base_generator import BaseAPLGenerator
from .apl_types import (
    ActiveTaskType,
    Difficulty,
    steps_to_difficulty,
    VISIBILITY_LEFT_TEMPLATES,
    VISIBILITY_RIGHT_TEMPLATES,
    VISIBILITY_BEHIND_TEMPLATES,
    RELATIVE_WHAT_TEMPLATES,
)
from ...core.data_types import APLActiveTaskItem, ViewState, make_task_id
from ...core.scene_context import SceneContext, DEFAULT_CAMERA_HEIGHT
from ...action_space.action_primitives import ActionPrimitive
from ...action_space.action_sequences import (
    GoalDirectedPlanner,
    build_turn_sequence,
    _forward_from_view,
    _signed_angle_deg,
)
from ...utils.occlusion import AABB


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class APLActiveGenerator(BaseAPLGenerator):
    """
    Generates APL active (question-driven navigation) tasks for a single scene.

    Config keys:
        task_types          List[str]   ActiveTaskType values to generate
                                        default: visibility_single, visibility_hidden
        num_candidates      int         init views to try per object (default 8)
        num_distractors     int         wrong-answer choices for MC (default 3)
        max_bearing_hidden  float       degrees — object must be at least this far
                                        off-axis to count as "not visible" (default 60°)
        camera_height       float       (default 0.8)
        move_distance       float       (default 0.5)
        turn_angle          float       (default 45.0)
        max_steps           int         (default 5)
        seed                int
    """

    def __init__(self, scene_path: str, config: Dict[str, Any]):
        super().__init__(scene_path, config)
        self._rng = np.random.RandomState(config.get("seed", 42))
        self._py_rng = random.Random(config.get("seed", 42))

        raw_types = config.get("task_types", [
            ActiveTaskType.VISIBILITY_SINGLE.value,
            ActiveTaskType.VISIBILITY_HIDDEN.value,
        ])
        self.task_types: List[ActiveTaskType] = [ActiveTaskType(t) for t in raw_types]
        self.num_candidates: int = config.get("num_candidates", 8)
        self.num_distractors: int = config.get("num_distractors", 3)
        # Minimum bearing offset (degrees) from forward for the object to count
        # as "not currently visible" (needs a turn to see)
        self.min_bearing_hidden: float = config.get("min_bearing_hidden", 60.0)

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def generate_batch(
        self,
        max_items: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[APLActiveTaskItem]:
        tasks: List[APLActiveTaskItem] = []
        objects = self.scene_ctx.valid_objects

        if not objects:
            return tasks

        for task_type in self.task_types:
            if len(tasks) >= max_items:
                break
            new_tasks = self._generate_for_type(task_type, max_items - len(tasks))
            tasks.extend(new_tasks)

        if filters:
            tasks = [t for t in tasks if self._apply_filters(t, filters)]

        return tasks[:max_items]

    def validate_task(self, task: APLActiveTaskItem) -> Tuple[bool, Optional[str]]:
        if not task.action_sequence:
            return False, "Empty action sequence"
        is_ok, reason = self.validator.validate(
            [ActionPrimitive(a) for a in task.action_sequence], task.init_view
        )
        return is_ok, reason

    # -----------------------------------------------------------------------
    # Dispatch by type
    # -----------------------------------------------------------------------

    def _generate_for_type(
        self, task_type: ActiveTaskType, max_items: int
    ) -> List[APLActiveTaskItem]:
        if task_type == ActiveTaskType.VISIBILITY_SINGLE:
            return self._gen_visibility_single(max_items)
        elif task_type == ActiveTaskType.VISIBILITY_HIDDEN:
            return self._gen_visibility_hidden(max_items)
        elif task_type == ActiveTaskType.SPATIAL_DISTANCE:
            return self._gen_spatial_distance(max_items)
        elif task_type == ActiveTaskType.NEXT_ACTION:
            return self._gen_next_action(max_items)
        return []

    # -----------------------------------------------------------------------
    # P0a: Visibility-single  ("What is to your left?")
    # -----------------------------------------------------------------------

    def _gen_visibility_single(self, max_items: int) -> List[APLActiveTaskItem]:
        """
        For each init view, pick a turn direction (left/right/behind) and find
        which object becomes most visible after that turn.  The question is
        "What is to your [direction]?" and the answer is that object's label.
        """
        tasks: List[APLActiveTaskItem] = []
        objects = self.scene_ctx.valid_objects

        init_positions = self.scene_ctx.sample_positions_in_room(
            num_points=self.num_candidates,
            camera_height=self.camera_height,
            rng=self._rng,
        )

        for init_pos in init_positions:
            if len(tasks) >= max_items:
                break

            # Random facing direction
            angle = self._rng.uniform(0, 2 * math.pi)
            init_fwd = np.array([math.cos(angle), math.sin(angle), 0.0])
            init_tgt = init_pos + init_fwd
            init_view = self._make_view(init_pos, init_tgt)

            # Try left, right, behind
            for direction, turn_deg in [("left", 90.0), ("right", -90.0), ("behind", 180.0)]:
                if len(tasks) >= max_items:
                    break

                # Build turn actions
                turn_actions = build_turn_sequence(
                    turn_deg, step_deg=self.action_config.turn_angle
                )
                if len(turn_actions) > self.action_config.max_sequence_length:
                    continue

                # Simulate turn to get target view
                target_view = self._simulate_actions(init_view, turn_actions)

                # Find best-visible object in target view that was NOT visible in init view
                target_obj = self._best_newly_visible_object(init_view, target_view, objects)
                if target_obj is None:
                    continue

                # Make MC choices: correct + distractors
                correct_label = target_obj.label
                distractor_labels = self._sample_distractors(correct_label, objects)
                choices, answer_choice = self._make_mc_choices(
                    correct_label, distractor_labels
                )

                templates = {
                    "left": VISIBILITY_LEFT_TEMPLATES,
                    "right": VISIBILITY_RIGHT_TEMPLATES,
                    "behind": VISIBILITY_BEHIND_TEMPLATES,
                }[direction]
                question = self._py_rng.choice(templates)
                difficulty = steps_to_difficulty(len(turn_actions))

                task = APLActiveTaskItem(
                    task_id=make_task_id("apl_active"),
                    scene_name=self.scene_ctx.scene_name,
                    question=question,
                    question_type=ActiveTaskType.VISIBILITY_SINGLE.value,
                    answer=correct_label,
                    init_view=init_view,
                    target_view=target_view,
                    action_sequence=[a.value for a in turn_actions],
                    action_descriptions=self._action_descriptions(turn_actions),
                    choices=choices,
                    answer_choice=answer_choice,
                    target_object=target_obj.label,
                    target_object_id=target_obj.id,
                    num_steps=len(turn_actions),
                    reasoning_required=False,
                    difficulty=difficulty.value,
                    metadata={"turn_direction": direction, "turn_deg": turn_deg},
                )
                tasks.append(task)

        return tasks

    # -----------------------------------------------------------------------
    # P0b: Visibility-hidden  ("What is [direction] of the [anchor]?")
    # -----------------------------------------------------------------------

    def _gen_visibility_hidden(self, max_items: int) -> List[APLActiveTaskItem]:
        """
        Pick a target object that is hidden (outside FOV) and an anchor object
        that IS visible.  The question asks about the spatial relationship of
        the hidden object to the anchor.  The model must navigate to see the
        hidden object.
        """
        tasks: List[APLActiveTaskItem] = []
        objects = self.scene_ctx.valid_objects

        if len(objects) < 2:
            return tasks

        init_positions = self.scene_ctx.sample_positions_in_room(
            num_points=self.num_candidates,
            camera_height=self.camera_height,
            rng=self._rng,
        )

        for init_pos in init_positions:
            if len(tasks) >= max_items:
                break

            # Random facing
            angle = self._rng.uniform(0, 2 * math.pi)
            init_fwd = np.array([math.cos(angle), math.sin(angle), 0.0])
            init_tgt = init_pos + init_fwd
            init_view = self._make_view(init_pos, init_tgt)

            visible_objs = self.scene_ctx.get_visible_objects(
                init_pos, init_tgt,
                min_visible_corners=self.min_visible_corners,
                max_occ_ratio=self.max_occ_ratio,
            )
            if not visible_objs:
                continue

            # Find an object that is clearly NOT visible (high bearing offset)
            hidden_obj = self._pick_hidden_object(init_view, objects, visible_objs)
            if hidden_obj is None:
                continue

            anchor_obj = self._py_rng.choice(visible_objs)
            if anchor_obj.id == hidden_obj.id:
                continue

            # Spatial relationship of hidden_obj relative to anchor_obj (world coords)
            direction_label = self._spatial_direction(anchor_obj, hidden_obj)

            # Plan path: turn + possibly move to make hidden_obj visible
            target_pos, target_view = self._find_view_for_object(hidden_obj, init_pos)
            if target_view is None:
                continue

            seq, _traj = self.planner.plan(init_view, target_view)
            if not seq or len(seq) > self.action_config.max_sequence_length:
                continue

            is_ok, _ = self.validator.validate(
                [ActionPrimitive(a) for a in seq], init_view
            )
            if not is_ok:
                continue

            # Verify hidden_obj is now visible in target view
            tgt_pos = np.array(target_view.position)
            tgt_tgt = np.array(target_view.target)
            if not self.scene_ctx.is_object_visible(hidden_obj, tgt_pos, tgt_tgt):
                continue

            question = self._py_rng.choice(RELATIVE_WHAT_TEMPLATES).format(
                direction=direction_label, anchor=anchor_obj.label
            )
            correct_label = hidden_obj.label
            distractor_labels = self._sample_distractors(correct_label, objects)
            choices, answer_choice = self._make_mc_choices(correct_label, distractor_labels)
            difficulty = steps_to_difficulty(len(seq))

            task = APLActiveTaskItem(
                task_id=make_task_id("apl_active"),
                scene_name=self.scene_ctx.scene_name,
                question=question,
                question_type=ActiveTaskType.VISIBILITY_HIDDEN.value,
                answer=correct_label,
                init_view=init_view,
                target_view=target_view,
                action_sequence=[a.value for a in seq],
                action_descriptions=self._action_descriptions(seq),
                choices=choices,
                answer_choice=answer_choice,
                target_object=hidden_obj.label,
                target_object_id=hidden_obj.id,
                anchor_object=anchor_obj.label,
                anchor_object_id=anchor_obj.id,
                num_steps=len(seq),
                reasoning_required=True,
                difficulty=difficulty.value,
                metadata={
                    "spatial_relation": direction_label,
                    "init_visible_count": len(visible_objs),
                },
            )
            tasks.append(task)

        return tasks

    # -----------------------------------------------------------------------
    # P1: Spatial distance  ("Is A closer to B or C?")
    # -----------------------------------------------------------------------

    def _gen_spatial_distance(self, max_items: int) -> List[APLActiveTaskItem]:
        """
        Scaffolded P1 task — generates basic relative-distance questions.
        """
        tasks: List[APLActiveTaskItem] = []
        objects = self.scene_ctx.valid_objects

        if len(objects) < 3:
            return tasks

        init_positions = self.scene_ctx.sample_positions_in_room(
            num_points=self.num_candidates,
            camera_height=self.camera_height,
            rng=self._rng,
        )

        for init_pos in init_positions:
            if len(tasks) >= max_items:
                break

            angle = self._rng.uniform(0, 2 * math.pi)
            init_fwd = np.array([math.cos(angle), math.sin(angle), 0.0])
            init_view = self._make_view(init_pos, init_pos + init_fwd)

            # Pick 3 random objects: subject A, reference B, reference C
            sample = self._py_rng.sample(objects, min(3, len(objects)))
            obj_a, obj_b, obj_c = sample[0], sample[1], sample[2]

            dist_ab = self.scene_ctx.distance_to_object(
                0.5 * (obj_a.bmin + obj_a.bmax), obj_b
            )
            dist_ac = self.scene_ctx.distance_to_object(
                0.5 * (obj_a.bmin + obj_a.bmax), obj_c
            )

            if abs(dist_ab - dist_ac) < 0.3:
                continue  # too ambiguous

            closer_obj = obj_b if dist_ab < dist_ac else obj_c
            question = (
                f"Is the {obj_a.label} closer to the {obj_b.label} "
                f"or the {obj_c.label}?"
            )
            choices = [
                f"The {obj_b.label}",
                f"The {obj_c.label}",
                "They are about the same distance",
                "Cannot determine from here",
            ]
            answer = f"The {closer_obj.label}"
            answer_choice = "A" if closer_obj.id == obj_b.id else "B"

            # Target view: position from which all 3 objects are visible
            tgt_pos, tgt_view = self._find_view_for_objects([obj_a, obj_b, obj_c], init_pos)
            if tgt_view is None:
                continue

            seq, _traj = self.planner.plan(init_view, tgt_view)
            if not seq or len(seq) > self.action_config.max_sequence_length:
                continue

            tasks.append(APLActiveTaskItem(
                task_id=make_task_id("apl_active"),
                scene_name=self.scene_ctx.scene_name,
                question=question,
                question_type=ActiveTaskType.SPATIAL_DISTANCE.value,
                answer=answer,
                init_view=init_view,
                target_view=tgt_view,
                action_sequence=[a.value for a in seq],
                action_descriptions=self._action_descriptions(seq),
                choices=choices,
                answer_choice=answer_choice,
                target_object=obj_a.label,
                target_object_id=obj_a.id,
                anchor_object=obj_b.label,
                anchor_object_id=obj_b.id,
                num_steps=len(seq),
                reasoning_required=True,
                difficulty=steps_to_difficulty(len(seq)).value,
                metadata={
                    "dist_ab": round(dist_ab, 3),
                    "dist_ac": round(dist_ac, 3),
                    "obj_c": obj_c.label,
                },
            ))

        return tasks

    # -----------------------------------------------------------------------
    # P1: Next action prediction  ("What should you do next to see X?")
    # -----------------------------------------------------------------------

    def _gen_next_action(self, max_items: int) -> List[APLActiveTaskItem]:
        """
        Given init_view and a target object (not currently visible), predict
        the single best next action to get closer to seeing it.
        """
        tasks: List[APLActiveTaskItem] = []
        objects = self.scene_ctx.valid_objects

        init_positions = self.scene_ctx.sample_positions_in_room(
            num_points=self.num_candidates,
            camera_height=self.camera_height,
            rng=self._rng,
        )

        # Action choice labels
        action_labels = {
            ActionPrimitive.MOVE_FORWARD: "Move forward",
            ActionPrimitive.MOVE_BACKWARD: "Move backward",
            ActionPrimitive.TURN_LEFT:    "Turn left",
            ActionPrimitive.TURN_RIGHT:   "Turn right",
        }
        candidate_actions = list(action_labels.keys())

        for init_pos in init_positions:
            if len(tasks) >= max_items:
                break

            angle = self._rng.uniform(0, 2 * math.pi)
            init_fwd = np.array([math.cos(angle), math.sin(angle), 0.0])
            init_view = self._make_view(init_pos, init_pos + init_fwd)

            visible_objs = self.scene_ctx.get_visible_objects(
                init_pos, init_pos + init_fwd,
                min_visible_corners=self.min_visible_corners,
                max_occ_ratio=self.max_occ_ratio,
            )
            hidden_obj = self._pick_hidden_object(init_view, objects, visible_objs)
            if hidden_obj is None:
                continue

            # Find best single action that maximises visibility of hidden_obj
            best_action, best_view = self._best_next_action(
                init_view, hidden_obj, candidate_actions
            )
            if best_action is None:
                continue

            question = (
                f"You want to see the {hidden_obj.label}. "
                f"What is the best next action?"
            )
            choices_text = [action_labels[a] for a in candidate_actions]
            answer_text = action_labels[best_action]
            answer_choice_letter = "ABCD"[candidate_actions.index(best_action)]

            task = APLActiveTaskItem(
                task_id=make_task_id("apl_active"),
                scene_name=self.scene_ctx.scene_name,
                question=question,
                question_type=ActiveTaskType.NEXT_ACTION.value,
                answer=answer_text,
                init_view=init_view,
                target_view=best_view,
                action_sequence=[best_action.value],
                action_descriptions=self._action_descriptions([best_action]),
                choices=choices_text,
                answer_choice=answer_choice_letter,
                target_object=hidden_obj.label,
                target_object_id=hidden_obj.id,
                num_steps=1,
                reasoning_required=False,
                difficulty=Difficulty.EASY.value,
                metadata={"candidate_actions": [a.value for a in candidate_actions]},
            )
            tasks.append(task)

        return tasks

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _simulate_actions(
        self, view: ViewState, actions: List[ActionPrimitive]
    ) -> ViewState:
        current = view
        for action in actions:
            current = self.executor.execute(action, current)
        return current

    def _best_newly_visible_object(
        self,
        init_view: ViewState,
        target_view: ViewState,
        objects: List[AABB],
    ) -> Optional[AABB]:
        """Return the most prominently visible object in target_view that was hidden in init_view."""
        init_pos = np.array(init_view.position)
        init_tgt = np.array(init_view.target)
        tgt_pos  = np.array(target_view.position)
        tgt_tgt  = np.array(target_view.target)

        best_obj = None
        best_occ = 1.0

        for obj in objects:
            was_visible = self.scene_ctx.is_object_visible(
                obj, init_pos, init_tgt,
                self.min_visible_corners, self.max_occ_ratio
            )
            if was_visible:
                continue

            is_visible = self.scene_ctx.is_object_visible(
                obj, tgt_pos, tgt_tgt,
                self.min_visible_corners, self.max_occ_ratio
            )
            if is_visible:
                from ...utils.occlusion import occluded_area_on_image
                from ...bench_generation.batch_utils import create_intrinsics
                from ...utils.occlusion import camtoworld_from_pos_target
                c2w = camtoworld_from_pos_target(tgt_pos, tgt_tgt)
                occ, _ = occluded_area_on_image(
                    obj, c2w, create_intrinsics(),
                    [b for b in self.scene_ctx.all_blockers if b.id != obj.id],
                )
                if occ < best_occ:
                    best_occ = occ
                    best_obj = obj

        return best_obj

    def _pick_hidden_object(
        self,
        init_view: ViewState,
        all_objects: List[AABB],
        visible_objects: List[AABB],
    ) -> Optional[AABB]:
        """Pick an object that is not visible and has large bearing offset."""
        visible_ids = {o.id for o in visible_objects}
        init_pos = np.array(init_view.position)
        init_tgt = np.array(init_view.target)

        candidates = []
        for obj in all_objects:
            if obj.id in visible_ids:
                continue
            bearing = self.scene_ctx.bearing_to_object(init_pos, init_tgt, obj)
            if abs(bearing) >= self.min_bearing_hidden:
                candidates.append((abs(bearing), obj))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        # prefer the most offset object
        return candidates[0][1]

    def _find_view_for_object(
        self, obj: AABB, init_pos: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[ViewState]]:
        """Find a valid position from which `obj` is visible."""
        positions = self.scene_ctx.sample_positions_around_object(
            obj, target_distance=1.5, num_points=6,
            camera_height=self.camera_height, rng=self._rng
        )
        obj_center = 0.5 * (obj.bmin + obj.bmax)
        for pos in positions:
            tgt = np.array([obj_center[0], obj_center[1], self.camera_height])
            if self.scene_ctx.is_object_visible(obj, pos, tgt):
                return pos, self._make_view(pos, tgt)
        return None, None

    def _find_view_for_objects(
        self, objs: List[AABB], init_pos: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[ViewState]]:
        """Find a valid position from which all objects in `objs` are visible."""
        # Try positions in the room
        positions = self.scene_ctx.sample_positions_in_room(
            num_points=20, camera_height=self.camera_height, rng=self._rng
        )
        centroid = np.mean([0.5 * (o.bmin + o.bmax) for o in objs], axis=0)

        for pos in positions:
            tgt = np.array([centroid[0], centroid[1], self.camera_height])
            if all(self.scene_ctx.is_object_visible(obj, pos, tgt) for obj in objs):
                return pos, self._make_view(pos, tgt)
        return None, None

    def _best_next_action(
        self,
        init_view: ViewState,
        target_obj: AABB,
        candidates: List[ActionPrimitive],
    ) -> Tuple[Optional[ActionPrimitive], Optional[ViewState]]:
        """Choose the single action that gives best visibility of `target_obj`."""
        from ...utils.occlusion import occluded_area_on_image, camtoworld_from_pos_target
        from ...bench_generation.batch_utils import create_intrinsics

        best_action = None
        best_view = None
        best_score = float("inf")  # lower occ ratio = better

        intrinsics = create_intrinsics()

        for action in candidates:
            next_view = self.executor.execute(action, init_view)
            next_pos = np.array(next_view.position)
            if not self.scene_ctx.is_position_valid(next_pos):
                continue
            next_tgt = np.array(next_view.target)
            c2w = camtoworld_from_pos_target(next_pos, next_tgt)
            blockers = [b for b in self.scene_ctx.all_blockers if b.id != target_obj.id]
            occ, _ = occluded_area_on_image(target_obj, c2w, intrinsics, blockers)
            if occ < best_score:
                best_score = occ
                best_action = action
                best_view = next_view

        return best_action, best_view

    def _spatial_direction(self, anchor: AABB, target: AABB) -> str:
        """Return a simple cardinal direction label of target relative to anchor."""
        anchor_c = 0.5 * (anchor.bmin + anchor.bmax)
        target_c = 0.5 * (target.bmin + target.bmax)
        delta = target_c - anchor_c
        if abs(delta[0]) >= abs(delta[1]):
            return "to the right of" if delta[0] > 0 else "to the left of"
        else:
            return "in front of" if delta[1] > 0 else "behind"

    def _sample_distractors(
        self, correct_label: str, all_objects: List[AABB]
    ) -> List[str]:
        others = [o.label for o in all_objects if o.label != correct_label]
        others = list(set(others))
        self._py_rng.shuffle(others)
        return others[: self.num_distractors]

    def _make_mc_choices(
        self, correct: str, distractors: List[str]
    ) -> Tuple[List[str], str]:
        """Build 4-option MC choices and return (choices, correct_letter)."""
        all_choices = [correct] + distractors[: 3]
        # Pad if not enough distractors
        while len(all_choices) < 4:
            all_choices.append("None of the above")
        self._py_rng.shuffle(all_choices)
        letter = "ABCD"[all_choices.index(correct)]
        return all_choices, letter

    def _apply_filters(self, task: APLActiveTaskItem, filters: Dict[str, Any]) -> bool:
        if "difficulty" in filters:
            allowed = filters["difficulty"]
            if isinstance(allowed, str):
                allowed = [allowed]
            if task.difficulty not in allowed:
                return False
        if "max_steps" in filters:
            if task.num_steps > filters["max_steps"]:
                return False
        if "question_type" in filters:
            if task.question_type != filters["question_type"]:
                return False
        return True
