"""
Core data types and structures for the unified data factory.
Supports both QA tasks and Action-Perception Loop tasks.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class TaskType(Enum):
    """Task type enumeration"""
    QA = "qa"                          # Static question-answering
    APL_PASSIVE = "apl_passive"        # Passive: instruction following
    APL_ACTIVE = "apl_active"          # Active: question requires view change


@dataclass
class ViewState:
    """Represents a view/camera state in 3D space"""
    position: List[float]              # [x, y, z]
    target: List[float]                # [target_x, target_y, target_z]
    forward: List[float] = field(default_factory=lambda: [0.0, 0.0, 1.0])
    
    def to_dict(self) -> Dict:
        return {
            'position': self.position,
            'target': self.target,
            'forward': self.forward,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ViewState':
        return cls(
            position=data.get('position'),
            target=data.get('target'),
            forward=data.get('forward', [0.0, 0.0, 1.0])
        )


@dataclass
class QATaskItem:
    """A single QA (question-answering) task"""
    task_id: str
    scene_name: str
    question: str
    choices: List[str]
    answer: str                        # 'A', 'B', 'C', or 'D'
    question_type: str                 # 'distance_mca', 'object_count_mca', etc.
    view_state: ViewState
    visible_objects: List[Dict] = field(default_factory=list)  # [{'id': ..., 'label': ...}, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_jsonl_dict(self) -> Dict:
        """Convert to JSONL-serializable format"""
        return {
            'task_id': self.task_id,
            'task_type': 'qa',
            'scene_name': self.scene_name,
            'question': self.question,
            'choices': self.choices,
            'answer': self.answer,
            'question_type': self.question_type,
            'view': self.view_state.to_dict(),
            'visible_objects': self.visible_objects,
            'metadata': self.metadata,
        }


@dataclass
class APLPassiveTaskItem:
    """
    Passive APL task: Instruction Following
    Model follows a natural language instruction to execute actions and reach a target view.
    """
    task_id: str
    scene_name: str
    instruction: str                   # "Move to 1 meter away from the cup"
    instruction_type: str              # 'distance', 'direction', 'relative', 'multi_step'
    
    init_view: ViewState               # Initial camera view
    target_view: ViewState             # Expected view after following instruction
    action_sequence: List[str]         # Ground truth actions: ['move_forward', 'turn_left', ...]
    action_descriptions: List[str] = field(default_factory=list)  # Human-readable descriptions
    
    # Validation info
    init_visible_objects: List[Dict] = field(default_factory=list)
    target_visible_objects: List[Dict] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_jsonl_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'task_type': 'apl_passive',
            'scene_name': self.scene_name,
            'instruction': self.instruction,
            'instruction_type': self.instruction_type,
            'init_view': self.init_view.to_dict(),
            'target_view': self.target_view.to_dict(),
            'action_sequence': self.action_sequence,
            'action_descriptions': self.action_descriptions,
            'init_visible_objects': self.init_visible_objects,
            'target_visible_objects': self.target_visible_objects,
            'metadata': self.metadata,
        }


@dataclass
class APLActiveTaskItem:
    """
    Active APL task: Question requires view transformation to answer
    Model is given a question that cannot be answered from the current view,
    must execute actions to transform the view, then answer the question.
    """
    task_id: str
    scene_name: str
    question: str                      # "What is on the right side of the table?"
    question_type: str                 # 'visibility', 'spatial_inference', 'next_action', etc.
    answer: str                        # Answer at target view
    
    init_view: ViewState               # Current view (question not answerable here)
    target_view: ViewState             # View after transformation (question answerable)
    
    # Required transformation
    action_sequence: List[str]         # ['turn_right', 'move_forward', ...]
    action_descriptions: List[str] = field(default_factory=list)
    
    # Difficulty/complexity
    num_steps: int = 1
    reasoning_required: bool = False   # Whether answer requires spatial reasoning
    
    # For multi-choice format
    choices: List[str] = field(default_factory=list)  # Optional
    answer_choice: Optional[str] = None  # 'A', 'B', 'C', etc. if multiple choice
    
    # Visibility and context
    init_visible_objects: List[Dict] = field(default_factory=list)
    target_visible_objects: List[Dict] = field(default_factory=list)
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_jsonl_dict(self) -> Dict:
        result = {
            'task_id': self.task_id,
            'task_type': 'apl_active',
            'scene_name': self.scene_name,
            'question': self.question,
            'question_type': self.question_type,
            'answer': self.answer,
            'init_view': self.init_view.to_dict(),
            'target_view': self.target_view.to_dict(),
            'action_sequence': self.action_sequence,
            'action_descriptions': self.action_descriptions,
            'num_steps': self.num_steps,
            'reasoning_required': self.reasoning_required,
            'init_visible_objects': self.init_visible_objects,
            'target_visible_objects': self.target_visible_objects,
            'metadata': self.metadata,
        }
        
        if self.choices:
            result['choices'] = self.choices
            result['answer_choice'] = self.answer_choice
        
        return result


# Type union for all task items
TaskItem = QATaskItem | APLPassiveTaskItem | APLActiveTaskItem
