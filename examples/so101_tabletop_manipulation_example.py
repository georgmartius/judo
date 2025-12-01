#!/usr/bin/env python3
"""Example script demonstrating the SO101 tabletop manipulation task."""

import numpy as np
from judo.tasks import SO101TabletopManipulation, SO101TabletopManipulationConfig


def main():
    """Main function to demonstrate the tabletop manipulation task."""
    
    print("=" * 60)
    print("SO101 Tabletop Manipulation Task - Example")
    print("=" * 60)
    
    # Create a custom configuration
    config = SO101TabletopManipulationConfig(
        w_object_position=1.5,
        w_ee_velocity=0.02,
        num_objects=4,
        table_bounds_x=(-0.25, 0.25),
        table_bounds_y=(-0.25, 0.25),
        table_height=0.8,
        min_object_distance=0.1,
    )
    
    print("\n1. Creating task with custom configuration...")
    print(f"   - Number of objects: {config.num_objects}")
    print(f"   - Table bounds X: {config.table_bounds_x}")
    print(f"   - Table bounds Y: {config.table_bounds_y}")
    print(f"   - Min object distance: {config.min_object_distance}m")
    
    # Create the task with the custom config
    task = SO101TabletopManipulation(config=config)
    
    print("\n2. Object placements:")
    for i, placement in enumerate(task.object_placements):
        print(f"   Object {i + 1}: {placement.name}")
        print(f"      Position: [{placement.position[0]:.3f}, {placement.position[1]:.3f}, {placement.position[2]:.3f}]")
        print(f"      Orientation (quat): [{placement.orientation[0]:.3f}, {placement.orientation[1]:.3f}, "
              f"{placement.orientation[2]:.3f}, {placement.orientation[3]:.3f}]")
    
    print("\n3. Task goal:")
    if task.task_goal:
        print(f"   Target object: {task.task_goal.object_name}")
        print(f"   Goal position: [{task.task_goal.goal_position[0]:.3f}, "
              f"{task.task_goal.goal_position[1]:.3f}, {task.task_goal.goal_position[2]:.3f}]")
    
    print("\n4. Generating a new custom arrangement...")
    new_placements = task.generate_object_arrangement(
        num_objects=3,
        object_subset=["can", "bottle", "bread"],
        seed=42  # For reproducibility
    )
    
    print(f"   Generated {len(new_placements)} objects:")
    for i, placement in enumerate(new_placements):
        print(f"      - {placement.name} at position "
              f"[{placement.position[0]:.3f}, {placement.position[1]:.3f}, {placement.position[2]:.3f}]")
    
    print("\n5. Task information:")
    print(f"   - Number of actuators: {task.nu}")
    print(f"   - Timestep: {task.dt}s")
    print(f"   - Actuator control ranges:")
    for i, (low, high) in enumerate(task.actuator_ctrlrange):
        print(f"      Actuator {i}: [{low:.3f}, {high:.3f}]")
    
    print("\n6. Simulating a random rollout...")
    # Create random control inputs
    num_rollouts = 5
    horizon = 10
    num_controls = task.nu
    
    # Random states and sensors (for demonstration)
    states = np.random.randn(num_rollouts, horizon, task.model.nq + task.model.nv)
    sensors = np.random.randn(num_rollouts, horizon, task.model.nsensordata)
    controls = np.random.randn(num_rollouts, horizon, num_controls)
    
    # Compute rewards
    rewards = task.reward(states, sensors, controls)
    
    print(f"   Computed rewards for {num_rollouts} rollouts:")
    for i, reward in enumerate(rewards):
        print(f"      Rollout {i + 1}: {reward:.3f}")
    
    print("\n7. Metadata:")
    metadata = task.get_sim_metadata()
    print(f"   - Number of object placements: {len(metadata['object_placements'])}")
    print(f"   - Has task goal: {metadata['task_goal'] is not None}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
