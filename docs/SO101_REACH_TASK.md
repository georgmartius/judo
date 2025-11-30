# SO101 Reaching Task

## Overview

The SO101 Reaching Task is a simple reaching task for the LeRobot SO101 robotic arm. The robot must reach to randomized target locations in 3D space.

## Features

- **Randomized Target Positions**: Target positions are randomized within the robot's workspace on reset and when the target is reached
- **End-Effector Tracking**: The task tracks the end-effector position using MuJoCo sensors
- **Visual Feedback**: 
  - A small transparent green sphere (1cm radius) visualizes the target position in the MuJoCo viewer
  - The GUI provides interactive sliders to manually adjust the target position
  - End-effector trajectories are visualized as thin lines showing the optimizer's planned paths
- **Automatic Target Switching**: When the end-effector reaches within 3cm of the target, a new random target is generated

## Task Configuration

The task configuration (`SO101ReachConfig`) includes the following parameters:

- `w_ee_position` (default: 1.0): Weight for the end-effector position error
- `w_ee_velocity` (default: 0.01): Weight for the arm velocity penalty
- `target_pos`: 3D target position [x, y, z] with workspace bounds:
  - x: [-0.3, 0.3]
  - y: [-0.3, 0.3]
  - z: [0.05, 0.4]

## Reward Function

The reward function has two components:

1. **Position Reward**: Penalizes the squared distance between the end-effector and the target position
2. **Velocity Reward**: Penalizes the squared velocity of the arm joints to encourage smooth motion

The total reward is:
```
reward = -w_ee_position * ||ee_pos - target_pos||² - w_ee_velocity * ||arm_vel||²
```

## Files

- **Task Definition**: `judo/tasks/so101_reach.py`
- **XML Model**: `judo/models/xml/so101_reach.xml`
- **Robot Meshes**: `judo/models/meshes/LeRobot-SO101/`

## Usage

```python
from judo.tasks.so101_reach import SO101Reach

# Create the task
task = SO101Reach()

# Reset to initialize with a random target
task.reset()

# Access task properties
print(f"Number of actuators: {task.nu}")
print(f"Target position: {task.config.target_pos}")
print(f"Actuator ranges: {task.actuator_ctrlrange}")
```

## Implementation Details

### Robot Model

The SO101 is a 6-DOF robotic arm with the following joints:
1. `shoulder_pan`: Base rotation
2. `shoulder_lift`: Shoulder pitch
3. `elbow_flex`: Elbow flexion
4. `wrist_flex`: Wrist pitch
5. `wrist_roll`: Wrist roll
6. `gripper`: Gripper open/close

All joints use STS3215 servo actuators with position control.

### Workspace

The target positions are randomized within a reachable workspace:
- X range: -0.25 to 0.25 meters
- Y range: -0.25 to 0.25 meters  
- Z range: 0.05 to 0.35 meters (above the floor)

### Sensors

The task uses two `framepos` sensors:
- `ee_pos`: Tracks the end-effector position at the `gripperframe` site for reward computation
- `trace_ee`: Provides trajectory visualization of the end-effector during planning (thin lines showing optimizer paths)

## Testing

A test script is provided to verify the task loads correctly:

```bash
.venv/bin/python test_so101_reach.py
```

This will:
1. Load the SO101Reach task
2. Display task properties
3. Test the reset function
4. Test the reward computation
