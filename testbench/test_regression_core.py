"""Core regression checks for template engine, registries, and scoring semantics.

Run:
  C:\\Users\\user\\miniconda3\\python.exe testbench\\test_regression_core.py
"""
from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PKG_ROOT = Path(__file__).resolve().parent.parent
PARENT = PKG_ROOT.parent
_PKG_ALIAS = "spatial_training_room"
if _PKG_ALIAS not in sys.modules:
    if str(PARENT) not in sys.path:
        sys.path.insert(0, str(PARENT))
    spec = importlib.util.spec_from_file_location(
        _PKG_ALIAS,
        PKG_ROOT / "__init__.py",
        submodule_search_locations=[str(PKG_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PKG_ALIAS] = module
    spec.loader.exec_module(module)


from spatial_training_room.evaluation.coverage import compute_coverage
from spatial_training_room.evaluation.predicates import PREDICATE_REGISTRY
from spatial_training_room.evaluation.region_generators import REGION_REGISTRY
from spatial_training_room.evaluation.scorer import episode_score, step_rewards
from spatial_training_room.evaluation.template_spec import TemplateSpec
from spatial_training_room.evaluation.trajectory_metrics import (
    counterfactual_regret,
    information_gain_per_step,
    spl,
    steps_to_success,
    trajectory_path_length,
)
from spatial_training_room.task_generation.apl_tasks.choice_generators import CHOICE_REGISTRY
from spatial_training_room.task_generation.apl_tasks.init_validators import INIT_VALIDATORS
from spatial_training_room.task_generation.apl_tasks.template_handler import all_template_handlers


class _DummyScene:
    def default_hfov_deg(self) -> float:
        return 90.0

    def is_position_valid(self, pos) -> bool:
        return True


class RegressionCoreTests(unittest.TestCase):
    def test_registry_sizes_and_handler_inventory(self):
        handlers = all_template_handlers()
        self.assertEqual(len(handlers), 22)
        self.assertEqual(len(CHOICE_REGISTRY), 15)
        self.assertEqual(len(PREDICATE_REGISTRY), 24)
        self.assertEqual(len(REGION_REGISTRY), 16)
        self.assertGreaterEqual(len(INIT_VALIDATORS), 19)

        for tid in ("T05", "T20", "T33"):
            self.assertIn(tid, handlers)
            self.assertEqual(handlers[tid].spec.template_id, tid)
            self.assertTrue(callable(handlers[tid].instantiator))

        for key in (
            "target_invisible_at_init",
            "max_both_pairs_visible_at_init",
            "label_side_visible_at_init",
            "passage_not_orthogonally_visible_at_init",
        ):
            self.assertIn(key, INIT_VALIDATORS)

    def test_episode_score_answer_gate_short_circuit(self):
        spec = TemplateSpec.from_dict(
            {
                "template_id": "TXA",
                "name": "x",
                "subclass": "C",
                "evidence_slots": [
                    {
                        "slot_id": "s",
                        "region_generator": "inside_room",
                        "region_args": {},
                        "predicates": [{"name": "Visible", "args": {"obj": "{{obj_id}}"}}],
                    }
                ],
                "gamma": 0.9,
                "min_coverage_for_credit": 1.0,
            }
        )
        traj = [
            (np.array([0.0, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
            (np.array([0.5, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
        ]

        with patch(
            "spatial_training_room.evaluation.scorer.compute_coverage",
            side_effect=AssertionError("compute_coverage must not be called when answer is wrong"),
        ):
            score = episode_score(spec, traj, "A", "B", {"obj_id": 1}, _DummyScene())
        self.assertEqual(score, 0.0)

    def test_episode_score_cov_and_gamma_factor(self):
        spec = TemplateSpec.from_dict(
            {
                "template_id": "TXB",
                "name": "x",
                "subclass": "C",
                "evidence_slots": [
                    {
                        "slot_id": "s",
                        "region_generator": "inside_room",
                        "region_args": {},
                        "predicates": [{"name": "Visible", "args": {"obj": "{{obj_id}}"}}],
                    }
                ],
                "gamma": 0.9,
                "min_coverage_for_credit": 1.0,
            }
        )
        traj = [
            (np.array([0.0, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
            (np.array([0.5, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
            (np.array([1.0, 0.0, 0.8]), np.array([1.5, 0.0, 0.8])),
        ]

        with patch("spatial_training_room.evaluation.scorer.compute_coverage", return_value=0.5):
            score = episode_score(spec, traj, "A", "A", {"obj_id": 1}, _DummyScene())

        expected = 0.5 * (0.9 ** 2)
        self.assertTrue(math.isclose(score, expected, rel_tol=1e-9))

    def test_step_rewards_potential_shaping_plus_terminal(self):
        spec = TemplateSpec.from_dict(
            {
                "template_id": "TXC",
                "name": "x",
                "subclass": "C",
                "evidence_slots": [
                    {
                        "slot_id": "s",
                        "region_generator": "inside_room",
                        "region_args": {},
                        "predicates": [{"name": "Visible", "args": {"obj": "{{obj_id}}"}}],
                    }
                ],
            }
        )
        traj = [
            (np.array([0.0, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
            (np.array([0.5, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
            (np.array([1.0, 0.0, 0.8]), np.array([1.5, 0.0, 0.8])),
        ]
        with patch(
            "spatial_training_room.evaluation.scorer.compute_potential",
            side_effect=[0.10, 0.30, 0.25],
        ):
            rewards = step_rewards(spec, traj, {"obj_id": 1}, _DummyScene(), episode_score_value=0.80)

        self.assertEqual(len(rewards), 2)
        self.assertTrue(math.isclose(rewards[0], 0.20, rel_tol=1e-9))
        self.assertTrue(math.isclose(rewards[1], 0.75, rel_tol=1e-9))

    def test_compute_coverage_submit_only_semantics(self):
        spec = TemplateSpec.from_dict(
            {
                "template_id": "TXD",
                "name": "x",
                "subclass": "C",
                "coverage_aggregator": "mean",
                "evidence_slots": [
                    {
                        "slot_id": "s",
                        "region_generator": "inside_room",
                        "region_args": {},
                        "predicates": [{"name": "Visible", "args": {"obj": "{{obj_id}}"}}],
                        "threshold": 1.0,
                    }
                ],
            }
        )
        trajectory = [
            (np.array([-1.0, 0.0, 0.8]), np.array([0.0, 0.0, 0.8])),
            (np.array([+1.0, 0.0, 0.8]), np.array([0.0, 0.0, 0.8])),
        ]

        def _fake_pred(name, cam_pos, cam_target, hfov, scene_ctx, **kwargs):
            return float(cam_pos[0]) < 0.0

        with patch("spatial_training_room.evaluation.coverage.evaluate_predicate", side_effect=_fake_pred):
            cov_any = compute_coverage(spec, trajectory, {"obj_id": 7}, _DummyScene(), submit_only=False)
            cov_submit = compute_coverage(spec, trajectory, {"obj_id": 7}, _DummyScene(), submit_only=True)

        self.assertEqual(cov_any, 1.0)
        self.assertEqual(cov_submit, 0.0)

    def test_step_efficiency_and_spl(self):
        traj = [
            (np.array([0.0, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
            (np.array([0.5, 0.0, 0.8]), np.array([1.5, 0.0, 0.8])),
            (np.array([1.0, 0.0, 0.8]), np.array([2.0, 0.0, 0.8])),
        ]
        optimal = [
            (np.array([0.0, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
            (np.array([1.0, 0.0, 0.8]), np.array([2.0, 0.0, 0.8])),
        ]
        sts = steps_to_success(traj, success=True)
        self.assertEqual(sts, 2)

        l_agent = trajectory_path_length(traj)
        l_opt = trajectory_path_length(optimal)
        self.assertTrue(math.isclose(l_agent, 1.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(l_opt, 1.0, rel_tol=1e-9))
        self.assertTrue(math.isclose(spl(True, l_agent, l_opt), 1.0, rel_tol=1e-9))
        self.assertEqual(spl(False, l_agent, l_opt), 0.0)

    def test_information_gain_and_counterfactual_regret(self):
        spec = TemplateSpec.from_dict(
            {
                "template_id": "TXE",
                "name": "x",
                "subclass": "C",
                "evidence_slots": [
                    {
                        "slot_id": "s",
                        "region_generator": "inside_room",
                        "region_args": {},
                        "predicates": [{"name": "Visible", "args": {"obj": "{{obj_id}}"}}],
                        "threshold": 1.0,
                    }
                ],
            }
        )
        traj = [
            (np.array([0.0, 0.0, 0.8]), np.array([1.0, 0.0, 0.8])),
            (np.array([0.5, 0.0, 0.8]), np.array([1.5, 0.0, 0.8])),
        ]

        def _fake_pred(name, cam_pos, cam_target, hfov, scene_ctx, **kwargs):
            return float(cam_pos[0]) >= 0.25

        with patch("spatial_training_room.evaluation.coverage.evaluate_predicate", side_effect=_fake_pred):
            ig = information_gain_per_step(spec, traj, {"obj_id": 1}, _DummyScene())
            cf = counterfactual_regret(
                spec,
                traj,
                {"obj_id": 1},
                _DummyScene(),
                action_sequence=["move_forward"],
            )

        self.assertEqual(len(ig), 1)
        self.assertGreaterEqual(ig[0], 0.0)
        self.assertEqual(cf["num_steps"], 1)
        self.assertIn("trace", cf)
        self.assertEqual(len(cf["trace"]), 1)
        self.assertIn("regret", cf["trace"][0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
