# `get_bear` — multi-bear retrieval (Nav2)

> Reference doc for `get_bear_node.py`. See the [top-level README](../README.md) for the project overview.

`get_bear_node.py` is a standalone, looping state machine that autonomously collects multiple bears and delivers them to the home position one by one.

## Run

```bash
# inside the pros_car Docker container
r                                          # colcon build + source
ros2 run rne_final_pkg get_bear
```

Requires `pros_car` (arm + wheels), `ros2_yolo_integration` (YOLO + depth), and Nav2 to be running in parallel.

## ROS2 topics

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

## State machine

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

## State descriptions

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

## Bear detection filters (`_bear_visible`)

`_bear_visible()` suppresses detection in three cases:
1. **Carrying** — `_has_bear = True` (between `VERIFY_GRASP` success and `DROP`)
2. **Post-drop cooldown** — `time < ignore_home_until` (lasts `ignore_home_seconds` after each drop)
3. **Near home** — robot within `home_ignore_radius` metres of `(home_x, home_y)`

`_raw_bear_visible()` bypasses all filters and is used only inside `VERIFY_GRASP`.

## Stuck detection

`_drive(action)` / `_stop()` wrappers track whether the last command was a linear move:
- Linear commands (`FORWARD*`, `BACKWARD*`) set `_last_cmd_moving = True`
- Rotation commands and `_stop()` set it to `False`

At the top of each `_tick()`, if `_last_cmd_moving` is True and the current state is not in `{RECOVERY, DONE, GRAB, GRAB_WAIT, DROP, LOCALIZE}`, `_check_stuck(state)` runs:
- No position change over `stuck_timeout` seconds → `_recovery_level += 1`, enter `RECOVERY`
- Successful movement → `_recovery_level` decrements back toward 0

## Configuration (`config/get_bear.yaml`)

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
