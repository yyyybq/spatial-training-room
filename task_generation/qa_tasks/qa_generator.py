"""
QAGenerator — thin wrapper that exposes the existing batch QA generation
logic through the new task_generation framework.

Re-uses bench_generation.qa_batch_generator under the hood so no logic
is duplicated.  Converts its JSONL dicts to typed QATaskItem objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..base_generator import BaseAPLGenerator
from ...core.data_types import QATaskItem, ViewState, make_task_id
from ...core.scene_context import SceneContext


class QAGenerator(BaseAPLGenerator):
    """
    Wraps the existing bench_generation QA pipeline.

    Config keys:
        max_items_per_view  int     (default 2)
        render              bool    whether to render preview images (default False)
        question_types      List[str]  filter to specific question types (default all)
        seed                int
    """

    def __init__(self, scene_path: str, config: Dict[str, Any]):
        super().__init__(scene_path, config)
        self.max_items_per_view: int = config.get("max_items_per_view", 2)
        self.render: bool = config.get("render", False)
        self.question_types: Optional[List[str]] = config.get("question_types", None)

    # -----------------------------------------------------------------------

    def generate_batch(
        self,
        max_items: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[QATaskItem]:
        """
        Delegate to bench_generation and convert the dicts to QATaskItem.
        """
        # Import lazily to avoid heavy deps at module load time
        from ...bench_generation.qa_batch_generator import BatchQAGenerator  # type: ignore

        raw_items: List[Dict] = []
        try:
            gen = BatchQAGenerator(
                scene_path=str(self.scene_path),
                max_items=max_items,
                max_items_per_view=self.max_items_per_view,
                render=self.render,
            )
            raw_items = gen.generate()
        except Exception as exc:
            # Gracefully degrade if the legacy generator is not importable
            import warnings
            warnings.warn(
                f"QAGenerator: legacy BatchQAGenerator failed ({exc}). "
                "Returning empty list.",
                stacklevel=2,
            )
            return []

        tasks: List[QATaskItem] = []
        for d in raw_items:
            if len(tasks) >= max_items:
                break
            try:
                task = self._dict_to_task(d)
                if self.question_types and task.question_type not in self.question_types:
                    continue
                if filters and not self._apply_filters(task, filters):
                    continue
                tasks.append(task)
            except Exception:
                continue

        return tasks

    def validate_task(self, task: QATaskItem) -> Tuple[bool, Optional[str]]:
        if not task.question:
            return False, "Empty question"
        if not task.choices:
            return False, "No choices"
        if not task.answer:
            return False, "No answer"
        return True, None

    # -----------------------------------------------------------------------

    def _dict_to_task(self, d: Dict[str, Any]) -> QATaskItem:
        view_data = d.get("view") or d.get("camera") or {}
        view = ViewState(
            position=view_data.get("position", [0.0, 0.0, 0.8]),
            target=view_data.get("target", [0.0, 1.0, 0.8]),
        )
        return QATaskItem(
            task_id=d.get("task_id") or make_task_id("qa"),
            scene_name=d.get("scene_name") or self.scene_ctx.scene_name,
            question=d.get("question", ""),
            choices=d.get("choices", []),
            answer=d.get("answer", ""),
            question_type=d.get("question_type", "unknown"),
            view=view,
            visible_objects=d.get("visible_objects", []),
            metadata=d.get("metadata", {}),
        )

    def _apply_filters(self, task: QATaskItem, filters: Dict[str, Any]) -> bool:
        if "question_type" in filters:
            if task.question_type != filters["question_type"]:
                return False
        return True
