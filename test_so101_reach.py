#!/usr/bin/env python3
"""Test script to verify SO101Reach task loads correctly."""

import numpy as np
from judo.tasks.so101_reach import SO101Reach

def main():
    print("Loading SO101Reach task...")
    task = SO101Reach()
    
    print(f"Task name: {task.name}")
    print(f"Number of actuators: {task.nu}")
    print(f"Model timestep: {task.dt}")
    print(f"Actuator control ranges:\n{task.actuator_ctrlrange}")
    
    # Test reset
    print("\nTesting reset...")
    task.reset()
    print(f"Initial qpos: {task.data.qpos}")
    print(f"Target position: {task.config.target_pos}")
    
    # Test reward computation
    print("\nTesting reward computation...")
    # Create dummy rollout data
    batch_size = 2
    T = 10
    nq = task.model.nq
    nv = task.model.nv
    nu = task.nu
    nsensordata = task.model.nsensordata
    
    states = np.random.randn(batch_size, T, nq + nv)
    sensors = np.random.randn(batch_size, T, nsensordata)
    controls = np.random.randn(batch_size, T, nu)
    
    rewards = task.reward(states, sensors, controls)
    print(f"Rewards shape: {rewards.shape}")
    print(f"Rewards: {rewards}")
    
    print("\n✓ SO101Reach task loaded and tested successfully!")

if __name__ == "__main__":
    main()
