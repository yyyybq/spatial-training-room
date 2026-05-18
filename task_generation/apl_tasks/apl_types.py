"""
APL task type definitions: enumerations, difficulty levels, and
NL template registries used by the generators.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List


# ---------------------------------------------------------------------------
# Task type enumerations
# ---------------------------------------------------------------------------

class PassiveTaskType(str, Enum):
    """Instruction-following task types (P0 → P2 priority)."""
    # P0
    DISTANCE_ABSOLUTE = "distance_absolute"   # "Move to 1m away from the cup"
    DIRECTION_FACE    = "direction_face"      # "Turn to look at the chair"
    # P1
    RELATIVE_POSITION = "relative_position"  # "Stand to the left of the table"
    MULTI_STEP        = "multi_step"          # "Turn left, then move forward"
    # P2
    DIRECTION_ROOM    = "direction_room"      # "Move to the right side of the room"
    EQUIDISTANT       = "equidistant"         # "Stand equidistant from A and B"


class ActiveTaskType(str, Enum):
    """Question-driven navigation task types (P0 → P2 priority)."""
    # P0
    VISIBILITY_SINGLE  = "visibility_single"   # "What is on your left?"
    VISIBILITY_HIDDEN  = "visibility_hidden"   # "What colour is the back of the chair?"
    # P1
    SPATIAL_DISTANCE   = "spatial_distance"    # "Is the cup closer to A or B?"
    NEXT_ACTION        = "next_action"         # "What should you do next to see X?"
    # P2
    MEMORY_REASONING   = "memory_reasoning"    # "Where was X before you turned?"
    CONSTRAINT_SATISFY = "constraint_satisfy"  # "Find a spot where you can see both A and B"


class Difficulty(str, Enum):
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"
    EXPERT = "expert"


# ---------------------------------------------------------------------------
# NL template registries
# ---------------------------------------------------------------------------

# Passive: distance instructions
DISTANCE_TEMPLATES: Dict[float, List[str]] = {
    0.5: [
        "Get very close to the {object} — within 0.5 meters.",
        "Move to half a meter away from the {object}.",
    ],
    1.0: [
        "Move to 1 meter away from the {object}.",
        "Position yourself 1 meter from the {object}.",
        "Stand about 1 meter in front of the {object}.",
    ],
    1.5: [
        "Move to 1.5 meters away from the {object}.",
        "Stand about one and a half meters from the {object}.",
    ],
    2.0: [
        "Move to 2 meters away from the {object}.",
        "Step back until you are 2 meters from the {object}.",
    ],
    3.0: [
        "Move to 3 meters away from the {object}.",
        "Position yourself 3 meters from the {object}.",
    ],
}

# Passive: direction/face instructions
FACE_TEMPLATES: List[str] = [
    "Turn to look at the {object}.",
    "Face the {object}.",
    "Rotate so the {object} is directly in front of you.",
    "Adjust your view so you are looking at the {object}.",
]

# Passive: relative position
RELATIVE_POSITION_TEMPLATES: Dict[str, List[str]] = {
    "left":   ["Stand to the left of the {object}.",
               "Position yourself on the left side of the {object}."],
    "right":  ["Stand to the right of the {object}.",
               "Position yourself on the right side of the {object}."],
    "front":  ["Stand in front of the {object}.",
               "Position yourself facing the {object} from the front."],
    "behind": ["Stand behind the {object}.",
               "Position yourself on the far side of the {object}."],
}

# Active: visibility questions
VISIBILITY_LEFT_TEMPLATES: List[str] = [
    "What object is to your left?",
    "What is on your left side?",
    "What can you see if you look to the left?",
]
VISIBILITY_RIGHT_TEMPLATES: List[str] = [
    "What object is to your right?",
    "What is on your right side?",
    "What can you see if you look to the right?",
]
VISIBILITY_BEHIND_TEMPLATES: List[str] = [
    "What is behind you?",
    "What object is directly behind your current position?",
]

# Active: what-is-X-relative-to-Y templates
RELATIVE_WHAT_TEMPLATES: List[str] = [
    "What is {direction} of the {anchor}?",
    "What object is located {direction} of the {anchor}?",
    "Looking at the {anchor}, what is {direction} of it?",
]

# Difficulty thresholds (number of steps)
DIFFICULTY_STEP_THRESHOLDS = {
    Difficulty.EASY:   (1, 1),   # 1 step
    Difficulty.MEDIUM: (2, 3),   # 2-3 steps
    Difficulty.HARD:   (4, 5),   # 4-5 steps
    Difficulty.EXPERT: (6, 999), # 6+
}


def steps_to_difficulty(num_steps: int) -> Difficulty:
    for diff, (lo, hi) in DIFFICULTY_STEP_THRESHOLDS.items():
        if lo <= num_steps <= hi:
            return diff
    return Difficulty.EXPERT
