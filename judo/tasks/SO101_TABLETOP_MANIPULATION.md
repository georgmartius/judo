# SO101 Tabletop Manipulation Task

This task implements a procedural tabletop manipulation environment for the SO101 robot.

## Overview

The SO101 robot is placed on a table with procedurally generated objects. The task is to manipulate one of the objects to reach a goal position.

## Features

- **Procedural Object Generation**: Objects are randomly placed on the table with collision avoidance
- **Customizable Object Selection**: Choose which objects to include in the scene
- **Goal-based Rewards**: The robot is rewarded for moving the target object closer to its goal position
- **Modifiable Arrangements**: The `generate_object_arrangement()` function can be called to create custom object layouts

## Available Objects

The following objects are available for manipulation:
- `bottle`
- `bread`
- `can`
- `cereal`
- `lemon`
- `milk`

## Usage

### Basic Usage

```python
from judo.tasks import SO101TabletopManipulation

# Create the task with default settings (3 random objects)
task = SO101TabletopManipulation()
```

### Custom Object Arrangement

You can generate custom object arrangements using the `generate_object_arrangement()` method:

```python
# Generate a specific arrangement with 5 objects
placements = task.generate_object_arrangement(
    num_objects=5,
    object_subset=["can", "bottle", "bread"],  # Only use these objects
    seed=42  # For reproducibility
)
```

### Configuration

The task can be configured by passing a config object to the constructor:

```python
from judo.tasks import SO101TabletopManipulation, SO101TabletopManipulationConfig

# Create custom configuration
config = SO101TabletopManipulationConfig(
    w_object_position=2.0,  # Weight for object-to-goal distance
    w_ee_velocity=0.05,  # Weight for end-effector velocity penalty
    num_objects=4,  # Number of objects to place
    table_bounds_x=(-0.25, 0.25),  # Table X bounds
    table_bounds_y=(-0.25, 0.25),  # Table Y bounds
    table_height=0.8,  # Height of the table top
    min_object_distance=0.1,  # Minimum distance between objects
)

# Pass config to constructor (important: config is used during scene generation)
task = SO101TabletopManipulation(config=config)
```

## Object Placement

The `ObjectPlacement` dataclass describes each object's placement:

```python
@dataclass
class ObjectPlacement:
    name: str  # Object name (e.g., "can", "bread", "bottle")
    position: np.ndarray  # 3D position (x, y, z)
    orientation: np.ndarray  # Quaternion (w, x, y, z)
    xml_path: str  # Path to the object XML file
```

## Task Goal

The `TaskGoal` dataclass describes the manipulation goal:

```python
@dataclass
class TaskGoal:
    object_name: str  # Name of the object to manipulate
    goal_position: np.ndarray  # Target 3D position (x, y, z)
```

## Reward Function

The reward function has two components:

1. **Position Reward**: Penalizes the distance between the end-effector and the goal position
   - Weight: `w_object_position` (default: 1.0)
   
2. **Velocity Reward**: Penalizes high joint velocities to encourage smooth motion
   - Weight: `w_ee_velocity` (default: 0.01)

The total reward is the sum of these two components (both negative, so max reward is 0).

## Example: Modifying Object Arrangement

Here's how you can create a custom scene and modify it:

```python
from judo.tasks import SO101TabletopManipulation

# Create task
task = SO101TabletopManipulation()

# Generate a new arrangement with specific objects
new_placements = task.generate_object_arrangement(
    num_objects=3,
    object_subset=["can", "bottle", "lemon"],
    seed=123
)

# Access the current object placements
for placement in task.object_placements:
    print(f"Object: {placement.name}")
    print(f"Position: {placement.position}")
    print(f"Orientation: {placement.orientation}")

# Access the task goal
if task.task_goal:
    print(f"Goal object: {task.task_goal.object_name}")
    print(f"Goal position: {task.task_goal.goal_position}")
```

## Files

- **XML Model**: `/judo/models/xml/so101_tabletop_manipulation.xml` - Base scene with SO101 robot on table
- **Task Implementation**: `/judo/tasks/so101_tabletop_manipulation.py` - Task logic and procedural generation
- **Objects**: `/judo/models/xml/objects/` - Individual object XML files

## Future Improvements

- Add object position tracking to the reward function (currently uses end-effector position as a proxy)
- Implement grasp detection and object manipulation constraints
- Add more complex task goals (e.g., stacking, sorting)
- Support for multi-object manipulation tasks
