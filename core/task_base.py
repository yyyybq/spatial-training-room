"""
Base class and utilities for all task generators.
Provides common interface and scene context management.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from dataclasses import dataclass
import numpy as np


@dataclass
class SceneContext:
    """Context information about a 3D scene"""
    scene_path: Path
    scene_name: str
    
    # Loaded from scene files
    labels: List[Dict] = None          # Object labels and bounding boxes
    room_polygons: List[np.ndarray] = None  # Room boundaries (2D)
    occupancy_bounds: Optional[tuple] = None  # (min_height, max_height)
    
    # Computed
    aabbs: List[Any] = None            # AABBs from labels
    wall_aabbs: List[Any] = None       # Wall AABBs from structure.json
    valid_spawn_regions: List[Any] = None  # Legal camera placement regions
    
    def __post_init__(self):
        """Load scene data from disk"""
        if isinstance(self.scene_path, str):
            self.scene_path = Path(self.scene_path)
        
        if not self.scene_name:
            self.scene_name = self.scene_path.name
        
        self._load_scene_data()
    
    def _load_scene_data(self):
        """Load scene labels, room polygons, etc."""
        # Implementation will use existing utilities from utils/occlusion.py
        pass
    
    @property
    def num_objects(self) -> int:
        """Number of objects in scene"""
        return len(self.labels) if self.labels else 0


class BaseTaskGenerator(ABC):
    """
    Base class for all task generators (QA, APL-Passive, APL-Active).
    Provides common utilities and enforces interface.
    """
    
    def __init__(self, scene_path: str, config: Dict[str, Any]):
        """
        Args:
            scene_path: Path to 3D scene directory
            config: Configuration dict for this generator
        """
        self.scene_path = Path(scene_path)
        self.config = config
        self.scene_context = SceneContext(
            scene_path=self.scene_path,
            scene_name=self.scene_path.name
        )
        self.rng = np.random.RandomState(config.get('seed', 42))
    
    @abstractmethod
    def generate_batch(
        self,
        max_items: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """
        Generate a batch of task items.
        
        Args:
            max_items: Maximum number of items to generate
            filters: Optional filtering criteria (e.g., question_type, num_steps)
        
        Returns:
            List of task items (QATaskItem, APLPassiveTaskItem, or APLActiveTaskItem)
        """
        pass
    
    @abstractmethod
    def validate_task(self, task: Any) -> tuple[bool, Optional[str]]:
        """
        Validate a generated task item.
        
        Args:
            task: Task item to validate
        
        Returns:
            (is_valid, failure_reason)
        """
        pass
    
    def filter_tasks(
        self,
        tasks: List[Any],
        filters: Dict[str, Any]
    ) -> List[Any]:
        """
        Filter tasks based on criteria.
        
        Args:
            tasks: List of task items
            filters: Filtering criteria
        
        Returns:
            Filtered list
        """
        result = []
        for task in tasks:
            valid = True
            
            # Example filters (subclasses can override)
            if 'min_visible_objects' in filters:
                if len(task.init_visible_objects) < filters['min_visible_objects']:
                    valid = False
            
            if valid:
                result.append(task)
        
        return result
    
    def save_batch_to_jsonl(
        self,
        tasks: List[Any],
        output_path: str
    ) -> int:
        """
        Save tasks to JSONL format.
        
        Args:
            tasks: List of task items
            output_path: Path to output JSONL file
        
        Returns:
            Number of items written
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for task in tasks:
                # Assume each task has to_jsonl_dict() method
                jsonl_dict = task.to_jsonl_dict() if hasattr(task, 'to_jsonl_dict') else task.__dict__
                f.write(json.dumps(jsonl_dict, ensure_ascii=False) + '\n')
        
        return len(tasks)
    
    def load_batch_from_jsonl(self, input_path: str) -> List[Dict]:
        """Load tasks from JSONL file"""
        input_path = Path(input_path)
        tasks = []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        
        return tasks


class TaskGeneratorFactory:
    """Factory for creating appropriate task generators"""
    
    _generators: Dict[str, type] = {}
    
    @classmethod
    def register(cls, task_type: str, generator_class: type):
        """Register a generator class"""
        cls._generators[task_type] = generator_class
    
    @classmethod
    def create(
        cls,
        task_type: str,
        scene_path: str,
        config: Dict[str, Any]
    ) -> BaseTaskGenerator:
        """
        Create a task generator instance.
        
        Args:
            task_type: Type of generator ('qa', 'apl_passive', 'apl_active')
            scene_path: Path to scene
            config: Configuration dict
        
        Returns:
            Generator instance
        """
        if task_type not in cls._generators:
            raise ValueError(f"Unknown task type: {task_type}. Registered: {list(cls._generators.keys())}")
        
        generator_class = cls._generators[task_type]
        return generator_class(scene_path, config)
