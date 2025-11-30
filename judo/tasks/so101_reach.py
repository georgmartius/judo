# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from judo import MODEL_PATH
from judo.gui import slider
from judo.tasks.base import Task, TaskConfig
from judo.tasks.cost_functions import quadratic_norm
from judo.utils.fields import np_1d_field

XML_PATH = str(MODEL_PATH / "xml/so101_reach.xml")


@slider("w_ee_position", 0.0, 10.0, 0.1)
@slider("w_ee_velocity", 0.0, 1.0, 0.01)
@dataclass
class SO101ReachConfig(TaskConfig):
    """Reward configuration for the SO101 reaching task."""

    w_ee_position: float = 1.0
    w_ee_velocity: float = 0.01
    target_pos: np.ndarray = np_1d_field(
        np.array([0.2, 0.0, 0.2]),
        names=["x", "y", "z"],
        mins=[-0.3, -0.3, 0.05],
        maxs=[0.3, 0.3, 0.4],
        steps=[0.01, 0.01, 0.01],
        vis_name="target_position",
        xyz_vis_indices=[0, 1, 2],
        xyz_vis_defaults=[0.0, 0.0, 0.0],
    )


class SO101Reach(Task[SO101ReachConfig]):
    """Defines the SO101 reaching task.
    
    The robot must reach to a randomized target location in 3D space.
    The end-effector position is visualized with a transparent green sphere.
    """

    name: str = "so101_reach"
    config_t: type[SO101ReachConfig] = SO101ReachConfig

    def __init__(self, model_path: str = XML_PATH, sim_model_path: str | None = None) -> None:
        """Initializes the SO101 reaching task."""
        super().__init__(model_path=model_path, sim_model_path=sim_model_path)
        
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

    def reward(
        self,
        states: np.ndarray,
        sensors: np.ndarray,
        controls: np.ndarray,
        system_metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Implements the SO101 reaching reward.

        The reward has two terms:
            * `position_reward`, penalizing the distance between the end-effector and the target.
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

        # Get end-effector position from sensors
        ee_pos = sensors[..., self.ee_pos_adr : self.ee_pos_adr + 3]  # (num_rollouts, T, 3)
        
        # Get arm velocities from states
        arm_vel = states[..., self.arm_vel_slice]  # (num_rollouts, T, 5)
        
        # Target position
        target_pos = self.config.target_pos  # (3,)
        
        # Position error
        ee_to_target = ee_pos - target_pos  # (num_rollouts, T, 3)
        position_cost = quadratic_norm(ee_to_target)  # (num_rollouts, T)
        position_reward = -self.config.w_ee_position * position_cost.sum(-1)  # (num_rollouts,)
        
        # Velocity penalty
        velocity_cost = quadratic_norm(arm_vel)  # (num_rollouts, T)
        velocity_reward = -self.config.w_ee_velocity * velocity_cost.sum(-1)  # (num_rollouts,)
        
        assert position_reward.shape == (batch_size,)
        assert velocity_reward.shape == (batch_size,)
    
        return position_reward + velocity_reward

    def reset(self) -> None:
        """Resets the model to a default state with randomized target."""
        # Reset to home position with small random noise
        self.data.qpos[:] = self.qpos_home + np.random.uniform(-0.1, 0.1, size=self.qpos_home.shape)
        self.data.qvel[:] = 0.0
        
        # Randomize target position
        self._randomize_target()
        
        mujoco.mj_forward(self.model, self.data)
    
    def _randomize_target(self) -> None:
        """Randomizes the target position within the workspace."""
        # Define workspace bounds (reachable space for SO101)
        x_range = (-0.25, 0.25)
        y_range = (-0.25, 0.25)
        z_range = (0.05, 0.35)
        
        # Generate random target position
        target_x = np.random.uniform(*x_range)
        target_y = np.random.uniform(*y_range)
        target_z = np.random.uniform(*z_range)
        
        target_pos = np.array([target_x, target_y, target_z])
        
        # Update the mocap body position for visualization
        self.data.mocap_pos[0] = target_pos
        
        # Update the config (this will be used in the reward function)
        self.config.target_pos[:] = target_pos
    
    def post_sim_step(self) -> None:
        """Checks if the target is reached and randomizes a new target if so."""
        # Get current end-effector position
        ee_pos = self.data.sensordata[self.ee_pos_adr : self.ee_pos_adr + 3]
        
        # Check if target is reached (within 3cm)
        distance = np.linalg.norm(ee_pos - self.config.target_pos)
        if distance < 0.03:
            self._randomize_target()
            mujoco.mj_forward(self.model, self.data)
    
    def get_sim_metadata(self) -> dict[str, Any]:
        """Returns the simulation's target position."""
        return {"target_pos": self.config.target_pos.copy()}
