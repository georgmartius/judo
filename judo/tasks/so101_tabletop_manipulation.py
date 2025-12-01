# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
from mujoco import MjSpec  # type: ignore

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import quadratic_norm

XML_PATH = str(MODEL_PATH / "xml/so101_tabletop_manipulation.xml")
OBJECTS_PATH = Path("objects")


@dataclass
class ObjectPlacement:
    """Represents an object placement on the table."""
    
    name: str  # Object name (e.g., "can", "bread", "bottle")
    position: np.ndarray  # 3D position (x, y, z)
    orientation: np.ndarray  # Quaternion (w, x, y, z)
    xml_path: str  # Path to the object XML file


@dataclass
class TaskGoal:
    """Represents a task goal for an object."""
    
    object_name: str  # Name of the object to manipulate
    goal_position: np.ndarray  # Target 3D position (x, y, z)


@slider("w_object_position", 0.0, 10.0, 0.1)
@slider("w_ee_velocity", 0.0, 1.0, 0.01)
@dataclass
class SO101TabletopManipulationConfig(TaskConfig):
    """Configuration for the SO101 tabletop manipulation task."""
    
    w_object_position: float = 1.0  # Weight for object-to-goal distance
    w_ee_velocity: float = 0.01  # Weight for end-effector velocity penalty
    num_objects: int = 3  # Number of objects to place on the table
    min_object_distance: float = 0.08  # Minimum distance between objects


class SO101TabletopManipulation(Task[SO101TabletopManipulationConfig]):
    """Defines the SO101 tabletop manipulation task.
    
    The robot must manipulate objects on a table to reach goal positions.
    Objects are procedurally placed on the table, and the task is to move
    one of the objects to its goal location.
    """

    name: str = "so101_tabletop_manipulation"
    config_t: type[SO101TabletopManipulationConfig] = SO101TabletopManipulationConfig

    # Available objects for manipulation
    AVAILABLE_OBJECTS = [
        "bottle",
        "bread",
        "can",
        "cereal",
        "lemon",
        "milk",
    ]

    def __init__(
        self,
        model_path: str = XML_PATH,
        sim_model_path: str | None = None,
        config: Optional[SO101TabletopManipulationConfig] = None,
    ) -> None:
        """Initializes the SO101 tabletop manipulation task.
        
        Args:
            model_path: Path to the base model XML file (not used, scene is generated).
            sim_model_path: Optional path to simulation model.
            config: Optional task configuration. If None, uses default config.
        """
        # Initialize config first (before building scene)
        if config is None:
            self.config = self.config_t()
        else:
            self.config = config
        
        self.table_bounds_x: tuple = (-0.3, 0.3)  # Table X bounds
        self.table_bounds_y: tuple = (-0.3, 0.3)  # Table Y bounds
        self.table_height: float = 0.8  # Height of the table top
 

        # First, create the scene with objects
        self.object_placements: list[ObjectPlacement] = []
        self.task_goal: TaskGoal | None = None
        
        # Build the scene XML with objects
        scene_xml_path = self._build_scene_with_objects(model_path)
        
        # Initialize the task with the generated scene
        super().__init__(model_path=scene_xml_path, sim_model_path=sim_model_path)
        # super().__init__(model_path=MODEL_PATH / "xml" / "so101_tabletop_manipulation_generated.xml", sim_model_path=sim_model_path)
        
        # Get sensor index for end-effector position
        self.ee_pos_adr = self.get_sensor_start_index("ee_pos")
        
        # Get joint indices for the arm (excluding gripper)
        self.shoulder_pan_adr = self.get_joint_position_start_index("shoulder_pan")
        self.arm_pos_slice = slice(self.shoulder_pan_adr, self.shoulder_pan_adr + 5)  # 5 arm joints
        
        # Get velocity indices
        shoulder_pan_vel_adr = self.get_joint_velocity_start_index("shoulder_pan")
        self.arm_vel_slice = slice(shoulder_pan_vel_adr, shoulder_pan_vel_adr + 5)
        
        # Home position for the arm (neutral pose)
        self.qpos_home = np.array([
            0.0,  # shoulder_pan
            0.0,  # shoulder_lift
            0.0,  # elbow_flex
            0.0,  # wrist_flex
            0.0,  # wrist_roll
            0.0,  # gripper
        ])
  
        self.reset()

    def _build_scene_with_objects(self, model_path: str) -> str:
        """Builds the scene XML with procedurally placed objects.
        
        Returns:
            Path to the generated scene XML file.
        """
        # Generate object placements
        self.object_placements = self.generate_object_arrangement(
            num_objects=self.config.num_objects
        )
        
        # Select a random object as the task goal
        if self.object_placements:
            goal_idx = np.random.randint(0, len(self.object_placements))
            goal_object = self.object_placements[goal_idx]
            self.task_goal = self._generate_goal_for_object(goal_object)
        
        # Read the base XML content
        base_xml_content = Path(model_path).read_text()
        
        # Prepare the new objects XML
        new_objects_xml = []
        
        # Add objects
        for i, obj_placement in enumerate(self.object_placements):
            x, y, z = obj_placement.position
            qw, qx, qy, qz = obj_placement.orientation
            
            new_objects_xml.append(f'    <body name="{obj_placement.name}_{i}" pos="{x} {y} {z}" quat="{qw} {qx} {qy} {qz}">')
            new_objects_xml.append(f'      <freejoint/>')
            new_objects_xml.append(f'      <include file="{obj_placement.xml_path}"/>')
            new_objects_xml.append('    </body>')
        
        # Add goal visualization if we have a task goal
        if self.task_goal:
            gx, gy, gz = self.task_goal.goal_position
            new_objects_xml.append(f'    <body name="goal_marker" pos="{gx} {gy} {gz}" mocap="true">')
            new_objects_xml.append('      <geom name="goal_sphere" type="sphere" size="0.02" rgba="0 1 0 0.3" contype="0" conaffinity="0"/>')
            new_objects_xml.append('    </body>')
        
        # Inject the new objects before the closing </worldbody> tag
        # We look for the last occurrence of </worldbody>
        split_token = '</worldbody>'
        if split_token not in base_xml_content:
            raise ValueError(f"Could not find {split_token} in {XML_PATH}")
            
        parts = base_xml_content.rsplit(split_token, 1)
        
        final_xml_content = parts[0] + '\n' + '\n'.join(new_objects_xml) + '\n  ' + split_token + parts[1]
        
        # Write to file
        output_path = MODEL_PATH / "xml" / "so101_tabletop_manipulation_generated.xml"
        output_path.write_text(final_xml_content)
        
        return str(output_path)

    def generate_object_arrangement(
        self,
        num_objects: int | None = None,
        object_subset: list[str] | None = None,
        seed: int | None = None,
    ) -> list[ObjectPlacement]:
        """Generates a procedural arrangement of objects on the table.
        
        Args:
            num_objects: Number of objects to place. If None, uses config value.
            object_subset: Subset of object names to use. If None, uses all available objects.
            seed: Random seed for reproducibility. If None, uses current random state.
            
        Returns:
            List of ObjectPlacement instances describing the object arrangement.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if num_objects is None:
            num_objects = self.config.num_objects
        
        if object_subset is None:
            object_subset = self.AVAILABLE_OBJECTS
        
        # Select random objects from the subset
        num_to_select = min(num_objects, len(object_subset))  # type: ignore
        selected_objects = np.random.choice(
            object_subset,
            size=num_to_select,
            replace=False
        )
        
        placements: list[ObjectPlacement] = []
        positions: list[np.ndarray] = []
        
        for obj_name in selected_objects:
            # Try to find a valid position (with collision avoidance)
            max_attempts = 50
            for _ in range(max_attempts):
                # Random position on the table
                x = np.random.uniform(*self.table_bounds_x)
                y = np.random.uniform(*self.table_bounds_y)
                z = self.table_height + 0.05  # Slightly above table surface
                
                position = np.array([x, y, z])
                
                # Check if position is far enough from other objects
                if self._is_valid_position(position, positions):
                    positions.append(position)
                    
                    # Random orientation (rotation around Z-axis)
                    angle = np.random.uniform(0, 2 * np.pi)
                    quat = np.array([
                        np.cos(angle / 2),  # w
                        0,  # x
                        0,  # y
                        np.sin(angle / 2),  # z
                    ])
                    
                    # Create placement
                    xml_path = str(OBJECTS_PATH / f"{obj_name}.xml")
                    placement = ObjectPlacement(
                        name=obj_name,
                        position=position,
                        orientation=quat,
                        xml_path=xml_path,
                    )
                    placements.append(placement)
                    break
        
        return placements

    def _is_valid_position(
        self,
        position: np.ndarray,
        existing_positions: list[np.ndarray]
    ) -> bool:
        """Checks if a position is valid (far enough from existing objects).
        
        Args:
            position: Position to check.
            existing_positions: List of existing object positions.
            
        Returns:
            True if the position is valid, False otherwise.
        """
        for existing_pos in existing_positions:
            # Check XY distance only (ignore Z)
            distance = np.linalg.norm(position[:2] - existing_pos[:2])
            if distance < self.config.min_object_distance:
                return False
        return True

    def _generate_goal_for_object(self, obj_placement: ObjectPlacement) -> TaskGoal:
        """Generates a goal position for an object.
        
        Args:
            obj_placement: The object placement to generate a goal for.
            
        Returns:
            TaskGoal instance with the goal position.
        """
        # Generate a random goal position on the table
        # Make sure it's different from the current position
        while True:
            goal_x = np.random.uniform(*self.table_bounds_x)
            goal_y = np.random.uniform(*self.table_bounds_y)
            goal_z = self.table_height + 0.05
            
            goal_position = np.array([goal_x, goal_y, goal_z])
            
            # Check if goal is far enough from current position
            distance = np.linalg.norm(goal_position[:2] - obj_placement.position[:2])
            if distance > 0.1:  # At least 10cm away
                break
        
        return TaskGoal(
            object_name=obj_placement.name,
            goal_position=goal_position,
        )

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the SO101 tabletop manipulation reward.

        The reward has two terms:
            * `position_reward`, penalizing the distance between the object and the goal.
            * `velocity_reward`, penalizing the velocity of the arm joints.

        Since we return rewards, each penalty term is returned as negative. The max reward is zero.
        
        Args:
            states: The rolled out states. Shape=(num_rollouts, T, nq + nv).
            sensors: The rolled out sensor readings. Shape=(num_rollouts, T, total_num_sensor_dims).
            controls: The rolled out controls. Shape=(num_rollouts, T, nu).
            system_metadata: Additional metadata from the system.
            
        Returns:
            rewards: The reward for each rollout. Shape=(num_rollouts,).
        """
        batch_size = states.shape[0]

        # Get arm velocities from states
        arm_vel = states[..., self.arm_vel_slice]  # (num_rollouts, T, 5)
        
        # For now, use a simple reward based on end-effector proximity to the goal object
        # In a more sophisticated version, you would track the object position
        if self.task_goal is not None:
            # Get end-effector position from sensors
            ee_pos = sensors[..., self.ee_pos_adr : self.ee_pos_adr + 3]  # (num_rollouts, T, 3)
            
            # Distance to goal (simplified - should track object position)
            goal_pos = self.task_goal.goal_position
            ee_to_goal = ee_pos - goal_pos
            position_cost = quadratic_norm(ee_to_goal)  # (num_rollouts, T)
            position_reward = -self.config.w_object_position * position_cost.sum(-1)  # (num_rollouts,)
        else:
            position_reward = np.zeros(batch_size)

        # Velocity penalty
        velocity_cost = quadratic_norm(arm_vel)  # (num_rollouts, T)
        velocity_reward = -self.config.w_ee_velocity * velocity_cost.sum(-1)  # (num_rollouts,)
        
        assert position_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
    
        return position_reward + velocity_reward

    def reset(self) -> None:
        """Resets the model to a default state."""
        # Reset to home position with small random noise
        self.data.qpos[:len(self.qpos_home)] = self.qpos_home + np.random.uniform(
            -0.1, 0.1, size=self.qpos_home.shape
        )
        self.data.qvel[:] = 0.0
        
        mujoco.mj_forward(self.model, self.data)  # type: ignore
    
    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns the simulation metadata including object placements and goal."""
        return {
            "object_placements": self.object_placements,
            "task_goal": self.task_goal,
        }
