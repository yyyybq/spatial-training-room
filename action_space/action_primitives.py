"""
Action space definition and utilities for Action-Perception Loop tasks.
Defines primitive actions, action sequences, and execution logic.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np


class ActionPrimitive(str, Enum):
    """Basic action primitives"""
    # Movement actions
    MOVE_FORWARD = "move_forward"      # Move forward along camera direction
    MOVE_BACKWARD = "move_backward"    # Move backward
    MOVE_LEFT = "move_left"            # Strafe left
    MOVE_RIGHT = "move_right"          # Strafe right
    
    # Rotation actions (camera yaw)
    TURN_LEFT = "turn_left"            # Rotate counter-clockwise (yaw)
    TURN_RIGHT = "turn_right"          # Rotate clockwise (yaw)
    
    # Pitch actions (look up/down)
    LOOK_UP = "look_up"                # Pitch up (look up)
    LOOK_DOWN = "look_down"            # Pitch down (look down)
    LOOK_FORWARD = "look_forward"      # Reset pitch to forward
    
    # Termination
    STOP = "stop"                      # End action sequence


@dataclass
class ActionConfig:
    """Configuration for action execution"""
    
    # Movement parameters
    move_distance: float = 0.5         # Distance per move action (meters)
    move_variants: List[float] = None  # Optional: [0.3, 0.5, 1.0] for varied distances
    
    # Rotation parameters
    turn_angle: float = 45.0           # Angle per turn action (degrees)
    turn_variants: List[float] = None  # Optional: [22.5, 45.0, 90.0]
    
    # Pitch parameters
    look_angle: float = 15.0           # Angle per look action (degrees)
    look_variants: List[float] = None
    
    # Sequence constraints
    max_sequence_length: int = 10      # Maximum actions in a sequence
    allow_consecutive_moves: bool = True
    allow_consecutive_turns: bool = True
    
    def __post_init__(self):
        if self.move_variants is None:
            self.move_variants = [self.move_distance]
        if self.turn_variants is None:
            self.turn_variants = [self.turn_angle]
        if self.look_variants is None:
            self.look_variants = [self.look_angle]


class ActionExecutor:
    """
    Executes actions and updates camera state.
    Interface to ViewManipulator or custom view transformation.
    """
    
    def __init__(self, config: ActionConfig):
        self.config = config
        self.current_state = None  # Will be set by execute()
    
    def execute_action(
        self,
        action: ActionPrimitive,
        current_position: np.ndarray,
        current_target: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute a single action and return new position and target.
        
        Args:
            action: Action to execute
            current_position: Current camera position [x, y, z]
            current_target: Current look-at target [x, y, z]
            **kwargs: Additional parameters (e.g., distance_scale, angle_scale)
        
        Returns:
            (new_position, new_target)
        """
        
        if action == ActionPrimitive.STOP:
            return current_position, current_target
        
        # Get movement/rotation parameters
        distance_scale = kwargs.get('distance_scale', 1.0)
        angle_scale = kwargs.get('angle_scale', 1.0)
        
        # Forward direction (camera looking direction)
        forward = current_target - current_position
        forward_dist = np.linalg.norm(forward)
        if forward_dist > 1e-6:
            forward = forward / forward_dist
        
        # Right direction (perpendicular to forward in XY plane)
        right = np.array([-forward[1], forward[0], 0.0])
        right_dist = np.linalg.norm(right)
        if right_dist > 1e-6:
            right = right / right_dist
        
        # Up direction
        up = np.array([0.0, 0.0, 1.0])
        
        new_position = current_position.copy()
        new_target = current_target.copy()
        
        # Execute movement actions
        if action == ActionPrimitive.MOVE_FORWARD:
            move_dist = self.config.move_distance * distance_scale
            new_position = new_position + forward * move_dist
            new_target = new_target + forward * move_dist
        
        elif action == ActionPrimitive.MOVE_BACKWARD:
            move_dist = self.config.move_distance * distance_scale
            new_position = new_position - forward * move_dist
            new_target = new_target - forward * move_dist
        
        elif action == ActionPrimitive.MOVE_LEFT:
            move_dist = self.config.move_distance * distance_scale
            new_position = new_position - right * move_dist
            new_target = new_target - right * move_dist
        
        elif action == ActionPrimitive.MOVE_RIGHT:
            move_dist = self.config.move_distance * distance_scale
            new_position = new_position + right * move_dist
            new_target = new_target + right * move_dist
        
        # Execute rotation actions (yaw - turn left/right)
        elif action in [ActionPrimitive.TURN_LEFT, ActionPrimitive.TURN_RIGHT]:
            angle_rad = np.radians(self.config.turn_angle * angle_scale)
            if action == ActionPrimitive.TURN_RIGHT:
                angle_rad = -angle_rad
            
            # Rotate target around position (yaw rotation in XY plane)
            rel_target = new_target - new_position
            rel_x = rel_target[0]
            rel_y = rel_target[1]
            
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            new_rel_x = rel_x * cos_a - rel_y * sin_a
            new_rel_y = rel_x * sin_a + rel_y * cos_a
            
            new_target[0] = new_position[0] + new_rel_x
            new_target[1] = new_position[1] + new_rel_y
        
        # Execute pitch actions (look up/down)
        elif action == ActionPrimitive.LOOK_UP:
            angle_rad = np.radians(self.config.look_angle * angle_scale)
            rel_target = new_target - new_position
            
            # Rotate around right axis (pitch)
            rel_y = rel_target[1]
            rel_z = rel_target[2]
            
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            new_rel_y = rel_y * cos_a - rel_z * sin_a
            new_rel_z = rel_y * sin_a + rel_z * cos_a
            
            new_target[1] = new_position[1] + new_rel_y
            new_target[2] = new_position[2] + new_rel_z
        
        elif action == ActionPrimitive.LOOK_DOWN:
            angle_rad = -np.radians(self.config.look_angle * angle_scale)
            rel_target = new_target - new_position
            
            rel_y = rel_target[1]
            rel_z = rel_target[2]
            
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            new_rel_y = rel_y * cos_a - rel_z * sin_a
            new_rel_z = rel_y * sin_a + rel_z * cos_a
            
            new_target[1] = new_position[1] + new_rel_y
            new_target[2] = new_position[2] + new_rel_z
        
        elif action == ActionPrimitive.LOOK_FORWARD:
            # Reset pitch to forward (horizontal)
            rel_target = new_target - new_position
            rel_target[2] = 0.0  # Remove pitch
            new_target = new_position + rel_target
        
        return new_position, new_target
    
    def execute_sequence(
        self,
        actions: List[ActionPrimitive],
        initial_position: np.ndarray,
        initial_target: np.ndarray,
        **kwargs
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Execute a sequence of actions.
        
        Returns:
            List of (position, target) tuples for each step (including initial state)
        """
        trajectory = [(initial_position.copy(), initial_target.copy())]
        
        pos = initial_position.copy()
        tgt = initial_target.copy()
        
        for action in actions:
            if action == ActionPrimitive.STOP:
                break
            
            pos, tgt = self.execute_action(action, pos, tgt, **kwargs)
            trajectory.append((pos.copy(), tgt.copy()))
        
        return trajectory


class ActionSequenceValidator:
    """Validates action sequences for legality and feasibility"""
    
    def __init__(self, scene_context=None, config: ActionConfig = None):
        self.scene_context = scene_context
        self.config = config or ActionConfig()
    
    def validate_sequence(
        self,
        actions: List[ActionPrimitive],
        initial_position: np.ndarray,
        scene_path: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate an action sequence.
        
        Args:
            actions: Sequence of actions
            initial_position: Starting camera position
            scene_path: Path to scene (for collision checking)
        
        Returns:
            (is_valid, failure_reason)
        """
        
        # Check sequence length
        if len(actions) > self.config.max_sequence_length:
            return False, f"Sequence too long ({len(actions)} > {self.config.max_sequence_length})"
        
        # Check for valid action types
        for action in actions:
            if not isinstance(action, (str, ActionPrimitive)):
                return False, f"Invalid action type: {type(action)}"
        
        # Could add more checks:
        # - Collision detection (all positions valid)
        # - Room boundary enforcement
        # - Maximum distance from start
        
        return True, None


class ActionSequenceSampler:
    """Samples action sequences for generating tasks"""
    
    def __init__(self, config: ActionConfig):
        self.config = config
        self.executor = ActionExecutor(config)
    
    def sample_random_sequence(
        self,
        num_steps: int,
        initial_position: np.ndarray,
        initial_target: np.ndarray,
        action_pool: Optional[List[ActionPrimitive]] = None,
        seed: Optional[int] = None
    ) -> List[ActionPrimitive]:
        """
        Sample a random action sequence.
        
        Args:
            num_steps: Number of actions
            initial_position: Starting position
            initial_target: Starting target
            action_pool: Actions to sample from (default: all primitives)
            seed: Random seed
        
        Returns:
            List of actions
        """
        if seed is not None:
            np.random.seed(seed)
        
        if action_pool is None:
            action_pool = [
                ActionPrimitive.MOVE_FORWARD,
                ActionPrimitive.TURN_LEFT,
                ActionPrimitive.TURN_RIGHT,
                ActionPrimitive.LOOK_UP,
                ActionPrimitive.LOOK_DOWN,
            ]
        
        sequence = []
        for _ in range(num_steps):
            action = ActionPrimitive(np.random.choice([a.value for a in action_pool]))
            sequence.append(action)
        
        return sequence
    
    def sample_directed_sequence(
        self,
        initial_position: np.ndarray,
        initial_target: np.ndarray,
        target_object_position: np.ndarray,
        max_steps: int = 5,
        seed: Optional[int] = None
    ) -> Tuple[List[ActionPrimitive], List[Tuple[np.ndarray, np.ndarray]]]:
        """
        Sample an action sequence that moves toward a target object.
        (Simple greedy approach)
        
        Returns:
            (action_sequence, trajectory)
        """
        if seed is not None:
            np.random.seed(seed)
        
        sequence = []
        trajectory = [(initial_position.copy(), initial_target.copy())]
        
        pos = initial_position.copy()
        tgt = initial_target.copy()
        
        for _ in range(max_steps):
            # Simple strategy: turn toward target, then move forward
            forward = tgt - pos
            forward = forward / (np.linalg.norm(forward) + 1e-6)
            
            to_target = target_object_position - pos
            to_target_dist = np.linalg.norm(to_target)
            
            if to_target_dist < 0.5:  # Close enough
                break
            
            # Decide action: turn or move
            angle_to_target = np.arccos(np.clip(np.dot(forward[:2], to_target[:2]) / (np.linalg.norm(forward[:2]) * np.linalg.norm(to_target[:2]) + 1e-6), -1, 1))
            
            if angle_to_target > np.radians(15):
                # Need to turn
                action = ActionPrimitive.TURN_RIGHT if to_target[0] < 0 else ActionPrimitive.TURN_LEFT
            else:
                # Move forward
                action = ActionPrimitive.MOVE_FORWARD
            
            sequence.append(action)
            pos, tgt = self.executor.execute_action(action, pos, tgt)
            trajectory.append((pos.copy(), tgt.copy()))
        
        return sequence, trajectory
