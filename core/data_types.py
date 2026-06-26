"""
Unified data type definitions for the APL data factory.
All task items share a common serialization interface (to_jsonl_dict / from_dict).
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import uuid


# ---------------------------------------------------------------------------
# Primitive view state
# ---------------------------------------------------------------------------

@dataclass
class ViewState:
    """Represents a single camera pose in the scene."""
    position: List[float]   # [x, y, z]  world coords
    target: List[float]     # [x, y, z]  look-at point
    forward: Optional[List[float]] = None   # derived, optional

    def to_dict(self) -> Dict[str, Any]:
        d = {"position": self.position, "target": self.target}
        if self.forward is not None:
            d["forward"] = self.forward
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ViewState":
        return cls(
            position=list(d["position"]),
            target=list(d["target"]),
            forward=list(d["forward"]) if d.get("forward") else None,
        )


# ---------------------------------------------------------------------------
# QA task item
# ---------------------------------------------------------------------------

@dataclass
class QATaskItem:
    """Static single-view question-answering task."""
    task_id: str
    scene_name: str
    question: str
    choices: List[str]
    answer: str                         # Option letter ('A','B','C','D') or text
    question_type: str                  # e.g. 'object_count_mca'
    view: ViewState
    visible_objects: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "qa"
    difficulty: str = "medium"

    def to_jsonl_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "scene_name": self.scene_name,
            "question": self.question,
            "choices": self.choices,
            "answer": self.answer,
            "question_type": self.question_type,
            "view": self.view.to_dict(),
            "visible_objects": self.visible_objects,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QATaskItem":
        return cls(
            task_id=d["task_id"],
            scene_name=d["scene_name"],
            question=d["question"],
            choices=d["choices"],
            answer=d["answer"],
            question_type=d.get("question_type", ""),
            view=ViewState.from_dict(d["view"]),
            visible_objects=d.get("visible_objects", []),
            metadata=d.get("metadata", {}),
            difficulty=d.get("difficulty", "medium"),
        )


# ---------------------------------------------------------------------------
# APL passive task item  (Instruction Following)
# ---------------------------------------------------------------------------

@dataclass
class APLPassiveTaskItem:
    """
    Passive APL task: model follows a natural-language instruction by
    executing a sequence of actions to reach a target view.
    """
    task_id: str
    scene_name: str
    instruction: str                    # NL instruction
    instruction_type: str               # 'distance' | 'direction' | 'relative_position' | 'multi_step'
    init_view: ViewState
    target_view: ViewState
    action_sequence: List[str]          # List of ActionPrimitive values
    action_descriptions: List[str] = field(default_factory=list)  # Human-readable
    target_object: Optional[str] = None
    target_object_id: Optional[str] = None
    target_distance: Optional[float] = None    # metres (for distance tasks)
    target_direction: Optional[str] = None     # 'left'|'right'|'front'|'behind' (for direction tasks)
    num_steps: int = 0
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "apl_passive"

    def __post_init__(self):
        self.num_steps = len(self.action_sequence)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "scene_name": self.scene_name,
            "instruction": self.instruction,
            "instruction_type": self.instruction_type,
            "init_view": self.init_view.to_dict(),
            "target_view": self.target_view.to_dict(),
            "action_sequence": self.action_sequence,
            "action_descriptions": self.action_descriptions,
            "target_object": self.target_object,
            "target_object_id": self.target_object_id,
            "target_distance": self.target_distance,
            "target_direction": self.target_direction,
            "num_steps": self.num_steps,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "APLPassiveTaskItem":
        item = cls(
            task_id=d["task_id"],
            scene_name=d["scene_name"],
            instruction=d["instruction"],
            instruction_type=d["instruction_type"],
            init_view=ViewState.from_dict(d["init_view"]),
            target_view=ViewState.from_dict(d["target_view"]),
            action_sequence=d["action_sequence"],
            action_descriptions=d.get("action_descriptions", []),
            target_object=d.get("target_object"),
            target_object_id=d.get("target_object_id"),
            target_distance=d.get("target_distance"),
            target_direction=d.get("target_direction"),
            difficulty=d.get("difficulty", "medium"),
            metadata=d.get("metadata", {}),
        )
        return item


# ---------------------------------------------------------------------------
# APL active task item  (Question-Driven Navigation)
# ---------------------------------------------------------------------------

@dataclass
class APLActiveTaskItem:
    """
    Active APL task: model must navigate to a position where it can answer
    a question that is unanswerable from the initial viewpoint.
    """
    task_id: str
    scene_name: str
    question: str
    question_type: str                  # 'visibility' | 'spatial_reasoning' | 'next_action' | 'memory'
    answer: str
    init_view: ViewState
    target_view: ViewState
    action_sequence: List[str]
    action_descriptions: List[str] = field(default_factory=list)
    choices: Optional[List[str]] = None          # For MC format
    answer_choice: Optional[str] = None          # 'A'|'B'|'C'|'D'
    target_object: Optional[str] = None
    target_object_id: Optional[str] = None
    anchor_object: Optional[str] = None          # reference object in question
    anchor_object_id: Optional[str] = None
    num_steps: int = 0
    reasoning_required: bool = False
    difficulty: str = "medium"
    metadata: Dict[str, Any] = field(default_factory=dict)
    task_type: str = "apl_active"

    # Template-driven extensions ------------------------------------------
    template_id: Optional[str] = None       # e.g. 'T01'
    subclass: Optional[str] = None          # e.g. 'C1.1'
    quality_spec: Dict[str, Any] = field(default_factory=dict)
    expert_trajectory: List[ViewState] = field(default_factory=list)
    coverage: float = 0.0
    submit_view_coverage: float = 0.0
    trajectory_evidence_coverage: float = 0.0
    trajectory_reliability: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    min_steps: int = 0

    def __post_init__(self):
        self.num_steps = len(self.action_sequence)

    def to_jsonl_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "scene_name": self.scene_name,
            "question": self.question,
            "question_type": self.question_type,
            "answer": self.answer,
            "gt_answer": self.answer,   # alias used by review tooling
            "choices": self.choices,
            "answer_choice": self.answer_choice,
            "init_view": self.init_view.to_dict(),
            "target_view": self.target_view.to_dict(),
            "action_sequence": self.action_sequence,
            "action_descriptions": self.action_descriptions,
            "target_object": self.target_object,
            "target_object_id": self.target_object_id,
            "anchor_object": self.anchor_object,
            "anchor_object_id": self.anchor_object_id,
            "num_steps": self.num_steps,
            "reasoning_required": self.reasoning_required,
            "difficulty": self.difficulty,
            "metadata": self.metadata,
            "template_id": self.template_id,
            "subclass": self.subclass,
            "quality_spec": self.quality_spec,
            "expert_trajectory": [v.to_dict() for v in self.expert_trajectory],
            "coverage": self.coverage,
            "submit_view_coverage": self.submit_view_coverage,
            "trajectory_evidence_coverage": self.trajectory_evidence_coverage,
            "trajectory_reliability": self.trajectory_reliability,
            "score": self.score,
            "min_steps": self.min_steps,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "APLActiveTaskItem":
        item = cls(
            task_id=d["task_id"],
            scene_name=d["scene_name"],
            question=d["question"],
            question_type=d["question_type"],
            answer=d["answer"],
            init_view=ViewState.from_dict(d["init_view"]),
            target_view=ViewState.from_dict(d["target_view"]),
            action_sequence=d["action_sequence"],
            action_descriptions=d.get("action_descriptions", []),
            choices=d.get("choices"),
            answer_choice=d.get("answer_choice"),
            target_object=d.get("target_object"),
            target_object_id=d.get("target_object_id"),
            anchor_object=d.get("anchor_object"),
            anchor_object_id=d.get("anchor_object_id"),
            reasoning_required=d.get("reasoning_required", False),
            difficulty=d.get("difficulty", "medium"),
            metadata=d.get("metadata", {}),
            template_id=d.get("template_id"),
            subclass=d.get("subclass"),
            quality_spec=d.get("quality_spec", {}) or {},
            expert_trajectory=[
                ViewState.from_dict(v) for v in d.get("expert_trajectory", [])
            ],
            coverage=float(d.get("coverage", 0.0)),
            submit_view_coverage=float(d.get("submit_view_coverage", d.get("coverage", 0.0))),
            trajectory_evidence_coverage=float(d.get("trajectory_evidence_coverage", d.get("coverage", 0.0))),
            trajectory_reliability=d.get("trajectory_reliability", {}) or {},
            score=float(d.get("score", 0.0)),
            min_steps=int(d.get("min_steps", 0)),
        )
        return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_task_id(prefix: str = "task") -> str:
    """Generate a unique task ID."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"
