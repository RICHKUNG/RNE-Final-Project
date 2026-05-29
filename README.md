# RNE-Final-Project

Autonomous 3-task mission for the RNE course robot. The robot navigates a Unity-simulated environment, locates objects using YOLO, approaches them with visual servoing, grabs them with a 5-DOF arm, and crosses a bridge between tasks.

## Package layout

```
rne_final_project/
└── rne_final_pkg/
    ├── rne_final_pkg/
    │   ├── final_mission.py   — 3-task linear mission state machine
    │   ├── get_bear_node.py   — multi-bear retrieval state machine
    │   ├── car_driver.py      — wheel publisher
    │   ├── arm_driver.py      — arm publisher (used by final_mission)
    │   ├── nav_client.py      — AMCL + path follower
    │   ├── yolo_client.py     — YOLO topic subscriber (used by final_mission)
    │   ├── yolo_align.py      — standalone visual-servo test node
    │   └── topic_check.py     — pre-flight diagnostics
    └── config/
        ├── mission.yaml       — thresholds for final_mission
        ├── get_bear.yaml      — thresholds and policy for get_bear_node
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

---

## get_bear_node — multi-bear retrieval

`get_bear_node.py` is a standalone, looping state machine that autonomously collects multiple bears and delivers them to the home position one by one.

### Run

```bash
# inside the pros_car Docker container
r                                          # colcon build + source
ros2 run rne_final_pkg get_bear_node
```

Requires `pros_car` (arm + wheels), `ros2_yolo_integration` (YOLO + depth), and Nav2 to be running in parallel.

### ROS2 topics

| Direction | Topic | Type | Purpose |
|-----------|-------|------|---------|
| Subscribed | `/yolo/bear_info` | `Float32MultiArray` | `[found, dist_m, delta_x_px, pixel_x, pixel_y]` |
| Subscribed | `/camera/depth/camera_info` | `CameraInfo` | camera intrinsics for backprojection |
| Subscribed | `/amcl_pose` | `PoseWithCovarianceStamped` | robot localization |
| Subscribed | `/plan` | `Path` | Nav2 global path |
| Published | `/goal_pose` | `PoseStamped` | Nav2 navigation goal |
| Published | `/clicked_point` | `PointStamped` | bear map position → triggers `arm_controller_2D` grab |
| Published | `/robot_arm` | `JointTrajectoryPoint` | direct arm command (stow on start, open gripper on drop) |
| Published | `/initialpose` | `PoseWithCovarianceStamped` | bootstrap AMCL if not yet localized |

### State machine

```
SEARCH_SPIN ──bear found──▶ LOCALIZE ──▶ NAV_TO_BEAR ──▶ VISUAL_SERVO
    ▲                                                          │
    │  timeout+count<total                                     ▼
    │                                                         GRAB
    │                                                          │
EXPLORE ◀──timeout+count<total                           GRAB_WAIT
    │                                                          │
    │                                                    VERIFY_GRASP
    │                                                    ╱           ╲
    │                                           confirmed             failed (retry/skip)
    │                                                ▼
    │                                         RETURN_HOME
    │                                                │
    │                                              DROP
    │                                                │
    └──────────────────────────────────── BACK_AWAY ◀┘

Any state with linear movement ──stuck──▶ RECOVERY ──done──▶ (original state)
count == total_bear_count ──▶ DONE
```

### State descriptions

| State | Behavior |
|-------|----------|
| `SEARCH_SPIN` | Rotate slowly in `search_spin_phase_seconds` bursts, nudging forward `search_nudge_seconds` between bursts to escape walls. Ignores bears within `home_ignore_radius` of home. |
| `LOCALIZE` | Backproject YOLO pixel + depth → map frame via TF. Send Nav2 goal at `bear_stop_distance_m` in front of bear. |
| `NAV_TO_BEAR` | Follow Nav2 global path. Falls back to `VISUAL_SERVO` if plan times out and bear is still visible. |
| `VISUAL_SERVO` | Align on `delta_x` (rotate) then drive forward until `dist < grab_distance_threshold`. |
| `GRAB` | Publish bear map position to `/clicked_point` to set target and trigger `arm_controller_2D`. |
| `GRAB_WAIT` | Wait `grab_wait_seconds` for the arm background thread to complete. |
| `VERIFY_GRASP` | Back up `verify_backup_seconds`, then count frames where bear depth < `verify_arm_depth_threshold`. If `close_frames >= verify_close_min_frames` → grasp confirmed, else retry up to `grab_retry_max` times. |
| `RETURN_HOME` | Nav2 to `(home_x, home_y)`. Bear detection suppressed while `_has_bear = True`. |
| `DROP` | Publish `[π, 0, π/2]` to `/robot_arm` to open gripper. Start `ignore_home_seconds` cooldown. |
| `BACK_AWAY` | Reverse `back_away_backward_seconds` then rotate `back_away_rotate_seconds` to clear the drop zone. |
| `EXPLORE` | Forward `explore_forward_seconds` + rotate `explore_rotate_seconds` to move to a new search position. Interrupts immediately if a bear appears. |
| `RECOVERY` | Tiered escape: level 1 back-up only → level 2 back+left → level 3 back+right → level 4 aggressive back+left. Returns to the state that triggered it. |
| `DONE` | All `total_bear_count` bears delivered, or search timed out after final delivery. |

### Bear detection filters (`_bear_visible`)

`_bear_visible()` suppresses detection in three cases:
1. **Carrying** — `_has_bear = True` (between `VERIFY_GRASP` success and `DROP`)
2. **Post-drop cooldown** — `time < ignore_home_until` (lasts `ignore_home_seconds` after each drop)
3. **Near home** — robot within `home_ignore_radius` metres of `(home_x, home_y)`

`_raw_bear_visible()` bypasses all filters and is used only inside `VERIFY_GRASP`.

### Stuck detection

`_drive(action)` / `_stop()` wrappers track whether the last command was a linear move:
- Linear commands (`FORWARD*`, `BACKWARD*`) set `_last_cmd_moving = True`
- Rotation commands and `_stop()` set it to `False`

At the top of each `_tick()`, if `_last_cmd_moving` is True and the current state is not in `{RECOVERY, DONE, GRAB, GRAB_WAIT, DROP, LOCALIZE}`, `_check_stuck(state)` runs:
- No position change over `stuck_timeout` seconds → `_recovery_level += 1`, enter `RECOVERY`
- Successful movement → `_recovery_level` decrements back toward 0

### Configuration (`config/get_bear.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `total_bear_count` | 5 | Mission ends when this many bears are delivered |
| `home_x` / `home_y` | 0.0 / 0.0 | Delivery position in map frame |
| `home_ignore_radius` | 0.8 m | Bears inside this radius are ignored |
| `ignore_home_seconds` | 5.0 s | Post-drop bear detection cooldown |
| `bear_stop_distance_m` | 1.0 m | Nav2 goal placed this far in front of bear |
| `grab_distance_threshold` | 0.4 m | Visual servo stop distance |
| `visual_servo_rotate_threshold_px` | 50 px | `delta_x` rotation threshold |
| `search_rotation_timeout` | 30.0 s | Per-round search time before `EXPLORE` |
| `search_spin_phase_seconds` | 8.0 s | Spin duration before each forward nudge |
| `search_nudge_seconds` | 1.5 s | Forward nudge duration |
| `explore_forward_seconds` | 3.0 s | Forward drive during `EXPLORE` |
| `explore_rotate_seconds` | 2.0 s | Rotation during `EXPLORE` |
| `grab_wait_seconds` | 10.0 s | Wait for arm background thread |
| `verify_backup_seconds` | 0.5 s | Reverse before observing in `VERIFY_GRASP` |
| `verify_duration_seconds` | 2.0 s | Observation window in `VERIFY_GRASP` |
| `verify_arm_depth_threshold` | 0.35 m | Depth below which bear is considered on arm |
| `verify_close_min_frames` | 3 | Minimum close-depth frames to confirm grasp |
| `grab_retry_max` | 2 | Max retries before skipping a bear |
| `drop_wait_seconds` | 2.0 s | Wait after opening gripper |
| `back_away_backward_seconds` | 1.5 s | Reverse duration after drop |
| `back_away_rotate_seconds` | 1.0 s | Rotation duration after drop |
| `stuck_check_interval` | 0.5 s | Position sampling interval for stuck detection |
| `stuck_move_threshold` | 0.03 m | Minimum movement per interval to not be stuck |
| `stuck_timeout` | 2.0 s | Time without movement before triggering recovery |
