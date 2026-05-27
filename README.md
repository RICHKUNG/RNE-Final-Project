# RNE-Final-Project

Autonomous 3-task mission for the RNE course robot. The robot navigates a Unity-simulated environment, locates objects using YOLO, approaches them with visual servoing, grabs them with a 5-DOF arm, and crosses a bridge between tasks.

## Package layout

```
rne_final_project/
└── rne_final_pkg/
    ├── rne_final_pkg/
    │   ├── final_mission.py   — main state machine
    │   ├── car_driver.py      — wheel publisher
    │   ├── arm_driver.py      — arm publisher
    │   ├── nav_client.py      — AMCL + path follower
    │   ├── yolo_client.py     — YOLO topic subscriber
    │   ├── yolo_align.py      — standalone visual-servo test node
    │   └── topic_check.py     — pre-flight diagnostics
    └── config/
        ├── mission.yaml       — tunable thresholds and timeouts
        └── goals.yaml         — map coordinates (fill after SLAM)
```

## Build and run

```bash
# inside the pros_car Docker container
cd ~/richkung/RNE/pros_car && ./car_control.sh
r                                          # colcon build + source
ros2 run rne_final_pkg final_mission       # run the mission
```

## ROS2 topics

| Direction | Topic | Type | Purpose |
|-----------|-------|------|---------|
| Subscribed | `/amcl_pose` | `PoseWithCovarianceStamped` | localization |
| Subscribed | `/plan` | `Path` | path from Nav2 |
| Subscribed | `/yolo/target_info` | `Float32MultiArray` | `[found, distance_m, delta_x_px]` |
| Subscribed | `/yolo/target_marker` | `Marker` | 3-D object pose for TF-based arm targeting |
| Subscribed | `/yolo/bridge_info` | `Float32MultiArray` | `[found, delta_x_px, area_ratio]` |
| Published | `/goal_pose` | `PoseStamped` | Nav2 goal |
| Published | `/car_C_rear_wheel` | `Float32MultiArray` | `[left, right]` wheel speeds |
| Published | `/car_C_front_wheel` | `Float32MultiArray` | `[left, right]` wheel speeds |
| Published | `/robot_arm` | `JointTrajectoryPoint` | 5-joint arm command |

## State machine

```
IDLE
 │
 └─▶ T1_NAV ─▶ T1_SPIN ─▶ T1_OBS ─▶ T1_APPR ─▶ T1_GRAB
                                                      │
                          ┌───────────────────────────┘
                          ▼
               T2_NAV_APPROACH ─▶ T2_BRIDGE_ALIGN ─▶ T2_CROSS
                                                          │
                                                          ▼
                                          T2_SPIN ─▶ T2_OBS ─▶ T2_APPR ─▶ T2_GRAB ─▶ T2_EXIT
                                                                                            │
                          ┌─────────────────────────────────────────────────────────────────┘
                          ▼
               T3_NAV ─▶ T3_SPIN ─▶ T3_OBS ─▶ T3_APPR ─▶ T3_UNLOCK ─▶ T3_CLEAR ─▶ DONE
```

### State descriptions

| State | Behavior |
|-------|----------|
| `T*_NAV` | Navigate a sequence of waypoints from `goals.yaml` using path following |
| `T*_SPIN` | Rotate slowly until YOLO reports the target visible (timeout → continue) |
| `T*_OBS` | Pause for `observe_wait_seconds` to stabilize the detection |
| `T*_APPR` | Visual servo: align on `delta_x`, drive forward until within `grab_distance_threshold` |
| `T*_GRAB` | Stop, run arm grab sequence, reset arm |
| `T2_BRIDGE_ALIGN` | Center on bridge using `bridge_delta_x` and wait for `bridge_min_area_ratio` |
| `T2_CROSS` | Drive forward for `bridge_cross_seconds`, correcting with `bridge_delta_x` |
| `T2_EXIT` | Navigate to `task2_bridge_exit` coordinate |
| `T3_CLEAR` | Timed back-up, rotate, and forward to clear area after Task 3 |

## Configuration

### `config/goals.yaml` — map coordinates

Fill these in after running SLAM and inspecting poses in Foxglove (`/amcl_pose`):

```yaml
task1_search_waypoints:
  - [x, y]   # waypoint 1
  - [x, y]   # waypoint 2

task2_bridge_approach: [x, y]
task2_bridge_exit:     [x, y]

task3_search_waypoints:
  - [x, y]
  - [x, y]
```

### `config/mission.yaml` — tunable parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grab_distance_threshold` | 0.5 m | Stop approaching when target is closer than this |
| `observe_wait_seconds` | 5.5 s | Pause duration after spin-search finds the target |
| `nav_arrive_threshold` | 0.5 m | Distance to waypoint considered "arrived" |
| `search_rotation_timeout` | 30.0 s | Give up spin-search and continue after this |
| `plan_follow_min_dist` | 0.5 m | Skip plan waypoints closer than this |
| `plan_forward_angle_deg` | 20.0 ° | Angle within which to drive straight (not steer) |
| `visual_servo_rotate_threshold_px` | 180 px | `delta_x` threshold to trigger rotation |
| `bridge_deadband_px` | 100 px | Bridge centering deadband |
| `bridge_min_area_ratio` | 0.03 | Minimum bridge bounding-box area to consider aligned |
| `bridge_cross_seconds` | 6.0 s | Fixed time to cross the bridge |
| `clear_backward_seconds` | 2.0 s | Task 3 clear: reverse duration |
| `clear_rotate_seconds` | 1.0 s | Task 3 clear: rotate duration |
| `clear_forward_seconds` | 1.0 s | Task 3 clear: forward duration |

## Arm driver

5-DOF arm joints (index 0–4):

| Index | Joint | Range |
|-------|-------|-------|
| 0 | base rotation | 0–180° |
| 1 | shoulder | 0–120° |
| 2 | elbow | 0–150° |
| 3 | wrist | 50–180° |
| 4 | gripper | 10° closed / 70° open |

`grab_sequence()` uses hardcoded `PRE_GRAB_POS` angles — tune these values in [arm_driver.py](rne_final_pkg/rne_final_pkg/arm_driver.py) to match the physical grab position. If the `map → arm_ik_base` TF is available, the mission will compute a TF-based target; otherwise it falls back to the default `(x=0.15, z=0.05)`.

## Utility nodes

```bash
# Pre-flight check — lists missing topics and TF frames
ros2 run rne_final_pkg topic_check

# Standalone YOLO visual servo (for manual testing)
ros2 run rne_final_pkg yolo_align
```

## Dependencies

- ROS2 (Humble or later)
- `rclpy`, `tf2_ros`, `tf2_geometry_msgs`
- `geometry_msgs`, `nav_msgs`, `std_msgs`, `visualization_msgs`, `trajectory_msgs`
- `pyyaml`
- `ament_index_python`

These are satisfied by the `pros_car` Docker image.
