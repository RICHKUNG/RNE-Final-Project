"""
get_bear_node.py — multi-bear retrieval state machine.

States
------
SEARCH_SPIN   rotate until a bear is detected outside the home ignore zone
LOCALIZE      backproject bear pixel → map coords, send Nav2 goal
NAV_TO_BEAR   follow Nav2 plan to stop_dist in front of bear
VISUAL_SERVO  close-range centering + approach
GRAB          publish /clicked_point to trigger arm grab
GRAB_WAIT     timed wait for arm_controller_2D background thread to finish
VERIFY_GRASP  back up, count YOLO frames to confirm pick-up succeeded
RETURN_HOME   Nav2 back to home pose (bear detection suppressed while carrying)
DROP          open gripper to release bear at home
BACK_AWAY     reverse + rotate to leave the drop zone
EXPLORE       relocate to new position after a failed search round
RECOVERY      tiered stuck recovery; returns to saved state when done
DONE          stop (all bears collected, or search timed out)
"""
import math
import os
import time

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint
from ament_index_python.packages import get_package_share_directory
import yaml

from rne_final_pkg.car_driver import CarDriver
from rne_final_pkg.nav_client import NavClient

# Only linear commands can reveal a stuck-against-wall condition via position tracking.
# Rotation commands don't change position, so they're excluded from stuck detection.
_LINEAR_ACTIONS = frozenset({"FORWARD", "FORWARD_SLOW", "BACKWARD", "BACKWARD_SLOW"})

# States where stuck detection must not run (already recovering, or car is stopped by design).
_NO_STUCK_CHECK = frozenset({"RECOVERY", "BRIDGE_AVOID", "DONE", "GRAB", "GRAB_WAIT", "DROP", "LOCALIZE"})


class GetBearNode(Node):
    def __init__(self):
        super().__init__("get_bear")

        self._load_params()

        self.car = CarDriver(self)
        self.nav = NavClient(self)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Bear detection: [found, dist, delta_x, pixel_x, pixel_y]
        self._bear_info = None
        self.create_subscription(
            Float32MultiArray, "/yolo/bear_info", self._bear_cb, 10
        )

        # Bridge safety: [found, delta_x, area_ratio] — published periodically by yolo_node
        self._bridge_info = None
        self._bridge_avoid_dx = 0.0
        self.create_subscription(
            Float32MultiArray, "/yolo/bridge_info", self._bridge_cb, 10
        )

        # Camera intrinsics — publisher uses TRANSIENT_LOCAL so we must match
        self._camera_info = None
        camera_info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            CameraInfo, "/camera/depth/camera_info", self._camera_info_cb, camera_info_qos
        )

        # /initialpose publisher — fires once at startup if AMCL is not yet localized
        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

        # Arm publishers
        self._arm_pub = self.create_publisher(JointTrajectoryPoint, "/robot_arm", 10)
        self._clicked_point_pub = self.create_publisher(PointStamped, "/clicked_point", 10)

        # Retry stow until arm_writer subscribes (polls every 0.5 s, gives up after 15 s)
        self._stow_attempts = 0
        self._stow_timer = self.create_timer(0.5, self._stow_arm_once)

        # Core state
        self._state = "SEARCH_SPIN"
        self._state_start = None
        self._nav_plan_wait_start = None
        self._bear_map_pos = None

        # SEARCH_SPIN phase tracking (spin → nudge → spin → …)
        self._search_phase_start = None
        self._search_nudging = False

        # Multi-bear mission tracking
        self._has_bear = False
        self._delivered_count = 0
        self._grab_retry_count = 0
        self._ignore_home_until = 0.0

        # VERIFY_GRASP frame counters (reset on each entry)
        self._verify_close_frames = 0
        self._verify_frames = 0

        # Stuck detection
        self._last_cmd_moving = False   # True only after a linear drive command
        self._last_pos_check_time = None
        self._last_pos_x = None
        self._last_pos_y = None
        self._stuck_since = None
        self._recovery_level = 0
        self._recovery_return_state = "SEARCH_SPIN"

        self.create_timer(0.1, self._tick)

        timeout = self.params.get("initial_pose_timeout", 3.0)
        self.create_timer(timeout, self._maybe_publish_initialpose)

        self.get_logger().info("GetBearNode ready (multi-bear mode).")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_params(self):
        share = get_package_share_directory("rne_final_pkg")
        path = os.path.join(share, "config", "get_bear.yaml")
        with open(path) as f:
            self.params = yaml.safe_load(f)

        # Use goals.yaml:origin as home if not already set in get_bear.yaml
        goals_path = os.path.join(share, "config", "goals.yaml")
        if os.path.exists(goals_path):
            with open(goals_path) as f:
                goals = yaml.safe_load(f)
            origin = goals.get("origin", None)
            if origin and len(origin) >= 2:
                self.params.setdefault("home_x", float(origin[0]))
                self.params.setdefault("home_y", float(origin[1]))

    # ------------------------------------------------------------------
    # /initialpose — publish once if AMCL has no pose yet
    # ------------------------------------------------------------------

    def _maybe_publish_initialpose(self):
        if self.nav.position is not None:
            self.get_logger().info("[INIT_POSE] AMCL already localized — skipping")
            return

        x   = self.params.get("initial_pose_x", 0.0)
        y   = self.params.get("initial_pose_y", 0.0)
        yaw = self.params.get("initial_pose_yaw", 0.0)

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(x)
        msg.pose.pose.position.y = float(y)
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.covariance[0]  = 0.25
        msg.pose.covariance[7]  = 0.25
        msg.pose.covariance[35] = 0.068

        self._initialpose_pub.publish(msg)
        self.get_logger().info(
            f"[INIT_POSE] Published /initialpose  x={x}  y={y}  yaw={yaw}"
        )

    # ------------------------------------------------------------------
    # Arm stow — retries until arm_writer has subscribed
    # ------------------------------------------------------------------

    def _stow_arm_once(self):
        self._stow_attempts += 1
        if self._arm_pub.get_subscription_count() == 0:
            if self._stow_attempts >= 30:
                self.get_logger().warn("[STOW] arm_writer never subscribed — skipping stow")
                self.destroy_timer(self._stow_timer)
            return
        msg = JointTrajectoryPoint()
        msg.positions = [math.pi, 0.0, math.pi / 2]
        msg.velocities = [0.0, 0.0, 0.0]
        self._arm_pub.publish(msg)
        self.get_logger().info(
            f"[STOW] Published stow pose (attempt {self._stow_attempts})"
        )
        self.destroy_timer(self._stow_timer)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _bear_cb(self, msg):
        self._bear_info = list(msg.data)

    def _bridge_cb(self, msg):
        self._bridge_info = list(msg.data)

    def _bridge_danger(self):
        """Return True when a bridge is close enough to warrant avoidance."""
        if not self._bridge_info or len(self._bridge_info) < 3:
            return False
        return (
            self._bridge_info[0] == 1.0
            and self._bridge_info[2] > self.params.get("bridge_safety_area_threshold", 0.05)
        )

    def _camera_info_cb(self, msg):
        self._camera_info = msg

    # ------------------------------------------------------------------
    # Car drive wrappers — track linear movement for global stuck detection
    # ------------------------------------------------------------------

    def _drive(self, action: str):
        """Publish a car action. Sets _last_cmd_moving True only for linear commands."""
        self._last_cmd_moving = action in _LINEAR_ACTIONS
        self.car.publish(action)

    def _stop(self):
        """Stop the car and clear the moving flag."""
        self._last_cmd_moving = False
        self.car.stop()

    # ------------------------------------------------------------------
    # Bear-info helpers
    # ------------------------------------------------------------------

    def _raw_bear_visible(self):
        """Unfiltered YOLO check — only use inside VERIFY_GRASP."""
        return (
            self._bear_info is not None
            and len(self._bear_info) >= 1
            and self._bear_info[0] == 1.0
        )

    def _bear_visible(self):
        """
        Filtered check. Returns False when:
          - we are already carrying a bear (_has_bear)
          - within the post-drop ignore window (_ignore_home_until)
          - robot is inside home_ignore_radius of home
        """
        if self._has_bear:
            return False
        now = time.time()
        if now < self._ignore_home_until:
            return False
        if self.nav.position is not None:
            rx, ry = self.nav.position
            home_x = self.params.get("home_x", 0.0)
            home_y = self.params.get("home_y", 0.0)
            ignore_r = self.params.get("home_ignore_radius", 0.8)
            if math.hypot(rx - home_x, ry - home_y) < ignore_r:
                return False
        return self._raw_bear_visible()

    def _bear_distance(self):
        if self._bear_info and len(self._bear_info) >= 2:
            return self._bear_info[1]
        return float("inf")

    def _bear_delta_x(self):
        if self._bear_info and len(self._bear_info) >= 3:
            return self._bear_info[2]
        return 0.0

    def _bear_pixel(self):
        if self._bear_info and len(self._bear_info) >= 5:
            return self._bear_info[3], self._bear_info[4]
        return None

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _goto(self, state: str):
        self.get_logger().info(f"State: {self._state} → {state}")
        self._state = state
        self._state_start = None
        self._nav_plan_wait_start = None
        self._last_cmd_moving = False
        # Reset per-state stuck context; recovery_level persists across states
        self._last_pos_check_time = None
        self._stuck_since = None

    def _tick(self):
        # Global stuck detection — runs whenever the last command was a linear move
        # and the current state is not one where the car is intentionally stopped.
        if self._last_cmd_moving and self._state not in _NO_STUCK_CHECK:
            if self._check_stuck(self._state):
                return

        s = self._state

        if s == "SEARCH_SPIN":
            self._state_search_spin()
        elif s == "LOCALIZE":
            self._state_localize()
        elif s == "NAV_TO_BEAR":
            self._state_nav_to_bear()
        elif s == "VISUAL_SERVO":
            self._state_visual_servo()
        elif s == "GRAB":
            self._state_grab()
        elif s == "GRAB_WAIT":
            self._state_grab_wait()
        elif s == "VERIFY_GRASP":
            self._state_verify_grasp()
        elif s == "RETURN_HOME":
            self._state_return_home()
        elif s == "DROP":
            self._state_drop()
        elif s == "BACK_AWAY":
            self._state_back_away()
        elif s == "EXPLORE":
            self._state_explore()
        elif s == "BRIDGE_AVOID":
            self._state_bridge_avoid()
        elif s == "RECOVERY":
            self._state_recovery()
        elif s == "DONE":
            self._stop()

    # ------------------------------------------------------------------
    # SEARCH_SPIN — rotate until bear visible (filtered)
    # ------------------------------------------------------------------

    def _state_search_spin(self):
        if self._state_start is None:
            self._state_start = time.time()
            self._search_phase_start = time.time()
            self._search_nudging = False

        if self._bear_visible():
            self.get_logger().info(
                f"[SEARCH] Bear found  dist={self._bear_distance():.2f}m  "
                f"dx={self._bear_delta_x():.0f}px"
            )
            self._stop()
            self._goto("LOCALIZE")
            return

        now = time.time()
        elapsed = now - self._state_start
        timeout = self.params["search_rotation_timeout"]
        if elapsed > timeout:
            if self._delivered_count < self.params.get("total_bear_count", 3):
                self.get_logger().info(
                    f"[SEARCH] Timeout — no bear found  "
                    f"({self._delivered_count}/{self.params.get('total_bear_count', 3)} delivered) → EXPLORE"
                )
                self._goto("EXPLORE")
            else:
                self.get_logger().info(
                    f"[SEARCH] All {self._delivered_count} bears collected. Done."
                )
                self._stop()
                self._goto("DONE")
            return

        phase_elapsed = now - self._search_phase_start
        spin_dur  = self.params.get("search_spin_phase_seconds", 8.0)
        nudge_dur = self.params.get("search_nudge_seconds", 1.5)

        if self._search_nudging:
            if phase_elapsed >= nudge_dur:
                self.get_logger().info("[SEARCH] Nudge done — resuming spin")
                self._search_nudging = False
                self._search_phase_start = now
            else:
                self.get_logger().info(
                    f"[SEARCH] Nudging forward  {phase_elapsed:.1f}/{nudge_dur:.1f}s"
                )
                self._drive("FORWARD_SLOW")
        else:
            if phase_elapsed >= spin_dur:
                self.get_logger().info("[SEARCH] Spin phase done — nudging forward to escape wall")
                self._search_nudging = True
                self._search_phase_start = now
            else:
                self.get_logger().info(
                    f"[SEARCH] No bear  elapsed={elapsed:.1f}/{timeout:.0f}s → CLOCKWISE_ROTATION_SLOW"
                )
                self._drive("CLOCKWISE_ROTATION_SLOW")

    # ------------------------------------------------------------------
    # LOCALIZE — backproject pixel → map frame, send Nav2 goal
    # ------------------------------------------------------------------

    def _state_localize(self):
        if not self._bear_visible():
            self.get_logger().warn("[LOCALIZE] Bear lost — returning to SEARCH_SPIN")
            self._goto("SEARCH_SPIN")
            return

        if self._camera_info is None:
            self.get_logger().warn("[LOCALIZE] Waiting for /camera/depth/camera_info …")
            return

        if self.nav.position is None:
            self.get_logger().warn("[LOCALIZE] Waiting for /amcl_pose …")
            return

        pixel = self._bear_pixel()
        if pixel is None:
            self.get_logger().warn("[LOCALIZE] No pixel coords in bear_info")
            return

        depth = self._bear_distance()
        if depth <= 0:
            self.get_logger().warn(f"[LOCALIZE] Invalid depth {depth:.3f} — retrying")
            return

        px, py = pixel
        K = self._camera_info.k
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]

        X_cam = (px - cx) * depth / fx
        Y_cam = (py - cy) * depth / fy
        Z_cam = depth

        pt = PointStamped()
        pt.header.frame_id = self.params["camera_frame"]
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = X_cam
        pt.point.y = Y_cam
        pt.point.z = Z_cam

        try:
            tf = self.tf_buffer.lookup_transform(
                "map",
                self.params["camera_frame"],
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            pt_map = tf2_geometry_msgs.do_transform_point(pt, tf)
            bear_x = pt_map.point.x
            bear_y = pt_map.point.y
        except Exception as e:
            self.get_logger().warn(f"[LOCALIZE] TF failed: {e}")
            return

        self._bear_map_pos = (bear_x, bear_y)
        self.get_logger().info(
            f"[LOCALIZE] Bear map pos ({bear_x:.2f}, {bear_y:.2f})  depth={depth:.2f}m"
        )

        rx, ry = self.nav.position
        dx = bear_x - rx
        dy = bear_y - ry
        dist_to_bear = math.hypot(dx, dy)
        stop_dist = self.params["bear_stop_distance_m"]
        arrive_th = self.params["nav_arrive_threshold"]
        travel_dist = dist_to_bear - stop_dist

        if travel_dist <= arrive_th:
            self.get_logger().info(
                f"[LOCALIZE] travel_dist={travel_dist:.2f}m ≤ arrive_th={arrive_th:.2f}m — skipping Nav2"
            )
            self._goto("VISUAL_SERVO")
            return

        goal_x = bear_x - (dx / dist_to_bear) * stop_dist
        goal_y = bear_y - (dy / dist_to_bear) * stop_dist
        self.get_logger().info(
            f"[LOCALIZE] Nav2 goal ({goal_x:.2f}, {goal_y:.2f})  travel={travel_dist:.2f}m"
        )
        self.nav.send_goal(goal_x, goal_y)
        self._goto("NAV_TO_BEAR")

    # ------------------------------------------------------------------
    # NAV_TO_BEAR — follow Nav2 plan toward bear
    # ------------------------------------------------------------------

    def _state_nav_to_bear(self):
        if not self.nav.has_plan or self.nav.position is None:
            # Reset stuck context while waiting — we're intentionally stopped
            self._last_pos_check_time = None
            self._stuck_since = None
            if self._nav_plan_wait_start is None:
                self._nav_plan_wait_start = time.time()
            waited = time.time() - self._nav_plan_wait_start
            timeout = self.params.get("nav_plan_timeout_s", 5.0)
            self.get_logger().info(f"[NAV] Waiting for plan… ({waited:.1f}/{timeout:.0f}s)")
            self._stop()
            if waited > timeout:
                self._nav_plan_wait_start = None
                if self._bear_visible():
                    self.get_logger().warn("[NAV] Plan timeout — bear visible, going to VISUAL_SERVO")
                    self._goto("VISUAL_SERVO")
                else:
                    self.get_logger().warn("[NAV] Plan timeout — bear lost, returning to SEARCH_SPIN")
                    self._goto("SEARCH_SPIN")
            return

        self._nav_plan_wait_start = None

        if self.nav.arrived(self.params["nav_arrive_threshold"]):
            self.get_logger().info(
                f"[NAV] Arrived  dist={self.nav.distance_to_goal():.2f}m → VISUAL_SERVO"
            )
            self._stop()
            self._goto("VISUAL_SERVO")
            return

        self._drive_toward_plan()

    # ------------------------------------------------------------------
    # VISUAL_SERVO — close-range centering + approach
    # ------------------------------------------------------------------

    def _state_visual_servo(self):
        # Bridge safety: if bridge detected in front, divert before continuing approach
        if self._bridge_danger():
            area = self._bridge_info[2]
            dx   = self._bridge_info[1]
            self.get_logger().warn(
                f"[VISUAL_SERVO] Bridge ahead  area={area:.3f}  dx={dx:.0f}px → BRIDGE_AVOID"
            )
            self._bridge_avoid_dx = dx
            self._stop()
            self._goto("BRIDGE_AVOID")
            return

        if not self._bear_visible():
            self.get_logger().info(
                "[VISUAL_SERVO] Bear not visible → CLOCKWISE_ROTATION_SLOW"
            )
            self._drive("CLOCKWISE_ROTATION_SLOW")
            return

        dx   = self._bear_delta_x()
        dist = self._bear_distance()
        rotate_th = self.params["visual_servo_rotate_threshold_px"]
        grab_th   = self.params["grab_distance_threshold"]

        if dx > rotate_th:
            action = "CLOCKWISE_ROTATION_SLOW"
        elif dx < -rotate_th:
            action = "COUNTERCLOCKWISE_ROTATION_SLOW"
        elif 0 < dist < grab_th:
            action = "→ GRAB"
        else:
            action = "FORWARD_SLOW"

        self.get_logger().info(
            f"[VISUAL_SERVO] dist={dist:.2f}m  dx={dx:.0f}px  "
            f"(rotate_th=±{rotate_th:.0f}  grab_th={grab_th:.2f}m) → {action}"
        )

        if action == "→ GRAB":
            self._stop()
            self._goto("GRAB")
        else:
            self._drive(action)

    # ------------------------------------------------------------------
    # GRAB — set arm target + trigger grab, then hand off to GRAB_WAIT
    # ------------------------------------------------------------------

    def _state_grab(self):
        self._stop()

        if self._bear_map_pos is None:
            self.get_logger().warn("[GRAB] No bear map position — returning to SEARCH_SPIN")
            self._goto("SEARCH_SPIN")
            return

        bear_x, bear_y = self._bear_map_pos
        pt = PointStamped()
        pt.header.frame_id = "map"
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = float(bear_x)
        pt.point.y = float(bear_y)
        pt.point.z = 0.0
        self._clicked_point_pub.publish(pt)
        self.get_logger().info(
            f"[GRAB] Published /clicked_point  map=({bear_x:.2f}, {bear_y:.2f})"
        )
        self._goto("GRAB_WAIT")

    # ------------------------------------------------------------------
    # GRAB_WAIT — timed wait for arm_controller_2D background thread
    # ------------------------------------------------------------------

    def _state_grab_wait(self):
        if self._state_start is None:
            self._state_start = time.time()
            wait = self.params.get("grab_wait_seconds", 12.0)
            self.get_logger().info(f"[GRAB_WAIT] Waiting {wait:.0f}s for arm sequence…")
            return

        elapsed = time.time() - self._state_start
        wait = self.params.get("grab_wait_seconds", 12.0)
        if elapsed >= wait:
            self.get_logger().info("[GRAB_WAIT] Done → VERIFY_GRASP")
            self._goto("VERIFY_GRASP")

    # ------------------------------------------------------------------
    # VERIFY_GRASP — back up, count close-depth frames to confirm pick-up
    # ------------------------------------------------------------------

    def _state_verify_grasp(self):
        """
        Confirm grasp by depth: a bear held on the arm appears at close range
        (< verify_arm_depth_threshold) in the center of the FoV.
        A bear still on the ground after backing up appears at normal distance.
        """
        now = time.time()
        if self._state_start is None:
            self._state_start = now
            self._verify_close_frames = 0
            self._verify_frames = 0

        elapsed    = now - self._state_start
        backup_dur = self.params.get("verify_backup_seconds", 0.5)
        verify_dur = self.params.get("verify_duration_seconds", 2.0)

        if elapsed < backup_dur:
            self._drive("BACKWARD_SLOW")
            return

        self._stop()
        observe_elapsed = elapsed - backup_dur

        if observe_elapsed < verify_dur:
            self._verify_frames += 1
            if self._raw_bear_visible():
                dist = self._bear_distance()
                arm_depth_th = self.params.get("verify_arm_depth_threshold", 0.35)
                if dist > 0 and dist < arm_depth_th:
                    self._verify_close_frames += 1
            self.get_logger().info(
                f"[VERIFY] Observing {observe_elapsed:.1f}/{verify_dur:.1f}s  "
                f"close_frames={self._verify_close_frames}/{self._verify_frames}  "
                f"dist={self._bear_distance():.2f}m"
            )
            return

        # --- Decision: bear close = on arm = grasp succeeded ---
        min_close = self.params.get("verify_close_min_frames", 3)
        if self._verify_close_frames >= min_close:
            self.get_logger().info(
                f"[VERIFY] Grasp confirmed  close_frames={self._verify_close_frames}/{self._verify_frames}"
            )
            self._has_bear = True
            self._grab_retry_count = 0
            self._goto("RETURN_HOME")
        else:
            self._grab_retry_count += 1
            max_retry = self.params.get("grab_retry_max", 2)
            self.get_logger().warn(
                f"[VERIFY] Bear not on arm  close_frames={self._verify_close_frames}/{self._verify_frames}  "
                f"retry {self._grab_retry_count}/{max_retry}"
            )
            if self._grab_retry_count >= max_retry:
                self.get_logger().warn("[VERIFY] Max retries — skipping this bear")
                self._grab_retry_count = 0
                self._goto("SEARCH_SPIN")
            else:
                self._goto("VISUAL_SERVO")

    # ------------------------------------------------------------------
    # RETURN_HOME — Nav2 back to home pose
    # ------------------------------------------------------------------

    def _state_return_home(self):
        if self._state_start is None:
            home_x = self.params.get("home_x", 0.0)
            home_y = self.params.get("home_y", 0.0)
            self.get_logger().info(f"[HOME] Navigating to home ({home_x:.2f}, {home_y:.2f})")
            self.nav.send_goal(home_x, home_y)
            self._state_start = time.time()

        if not self.nav.has_plan or self.nav.position is None:
            # Reset stuck context while waiting — we're intentionally stopped
            self._last_pos_check_time = None
            self._stuck_since = None
            if self._nav_plan_wait_start is None:
                self._nav_plan_wait_start = time.time()
            waited = time.time() - self._nav_plan_wait_start
            timeout = self.params.get("nav_plan_timeout_s", 5.0)
            self.get_logger().info(f"[HOME] Waiting for plan… ({waited:.1f}/{timeout:.0f}s)")
            self._stop()
            if waited > timeout:
                self._nav_plan_wait_start = None
                home_x = self.params.get("home_x", 0.0)
                home_y = self.params.get("home_y", 0.0)
                self.get_logger().warn("[HOME] Plan timeout — resending goal")
                self.nav.send_goal(home_x, home_y)
            return

        self._nav_plan_wait_start = None

        if self.nav.arrived(self.params["nav_arrive_threshold"]):
            self.get_logger().info(
                f"[HOME] Arrived  dist={self.nav.distance_to_goal():.2f}m → DROP"
            )
            self._stop()
            self._goto("DROP")
            return

        self._drive_toward_plan()

    # ------------------------------------------------------------------
    # DROP — open gripper to release bear
    # ------------------------------------------------------------------

    def _state_drop(self):
        if self._state_start is None:
            self._state_start = time.time()
            self._stop()
            # Stow pose with joint-2 open (90° = π/2): releases the FixedJoint in Unity
            msg = JointTrajectoryPoint()
            msg.positions = [math.pi, 0.0, math.pi / 2]
            msg.velocities = [0.0, 0.0, 0.0]
            self._arm_pub.publish(msg)
            self.get_logger().info("[DROP] Gripper open command sent")

        elapsed = time.time() - self._state_start
        if elapsed >= self.params.get("drop_wait_seconds", 2.0):
            self._has_bear = False
            self._delivered_count += 1
            self._ignore_home_until = time.time() + self.params.get("ignore_home_seconds", 5.0)
            self.get_logger().info(
                f"[DROP] Bear released. Total delivered: {self._delivered_count}"
            )
            self._goto("BACK_AWAY")

    # ------------------------------------------------------------------
    # BACK_AWAY — reverse + rotate to leave the drop zone
    # ------------------------------------------------------------------

    def _state_back_away(self):
        if self._state_start is None:
            self._state_start = time.time()

        elapsed      = time.time() - self._state_start
        backward_dur = self.params.get("back_away_backward_seconds", 1.5)
        rotate_dur   = self.params.get("back_away_rotate_seconds", 1.0)

        if elapsed < backward_dur:
            self._drive("BACKWARD_SLOW")
        elif elapsed < backward_dur + rotate_dur:
            self._drive("CLOCKWISE_ROTATION_SLOW")
        else:
            self._stop()
            self._goto("SEARCH_SPIN")

    # ------------------------------------------------------------------
    # EXPLORE — relocate to a new position, then retry SEARCH_SPIN
    # ------------------------------------------------------------------

    def _state_explore(self):
        """Move to a new spot so SEARCH_SPIN covers different ground next round."""
        if self._bear_visible():
            self.get_logger().info("[EXPLORE] Bear spotted — going to LOCALIZE")
            self._stop()
            self._goto("LOCALIZE")
            return

        if self._state_start is None:
            self._state_start = time.time()
            self.get_logger().info("[EXPLORE] Moving to new search position…")

        elapsed  = time.time() - self._state_start
        fwd_dur  = self.params.get("explore_forward_seconds", 3.0)
        rot_dur  = self.params.get("explore_rotate_seconds", 2.0)

        if elapsed < fwd_dur:
            self._drive("FORWARD_SLOW")
        elif elapsed < fwd_dur + rot_dur:
            self._drive("CLOCKWISE_ROTATION_SLOW")
        else:
            self.get_logger().info("[EXPLORE] Done — retrying SEARCH_SPIN")
            self._goto("SEARCH_SPIN")

    # ------------------------------------------------------------------
    # BRIDGE_AVOID — back up then turn away from detected bridge
    # ------------------------------------------------------------------

    def _state_bridge_avoid(self):
        if self._state_start is None:
            self._state_start = time.time()
            self.get_logger().warn(
                f"[BRIDGE_AVOID] Backing up then turning away from bridge  dx={self._bridge_avoid_dx:.0f}px"
            )

        elapsed = time.time() - self._state_start
        back_t  = self.params.get("bridge_avoid_backward_seconds", 0.5)
        turn_t  = self.params.get("bridge_avoid_turn_seconds", 0.8)

        if elapsed < back_t:
            self._drive("BACKWARD_SLOW")
        elif elapsed < back_t + turn_t:
            # bridge right of center → turn left; bridge left → turn right
            if self._bridge_avoid_dx >= 0:
                self._drive("COUNTERCLOCKWISE_ROTATION_SLOW")
            else:
                self._drive("CLOCKWISE_ROTATION_SLOW")
        else:
            self.get_logger().info("[BRIDGE_AVOID] Done → VISUAL_SERVO")
            self._goto("VISUAL_SERVO")

    # ------------------------------------------------------------------
    # RECOVERY — tiered stuck recovery, returns to saved state when done
    # ------------------------------------------------------------------

    def _state_recovery(self):
        if self._state_start is None:
            self._state_start = time.time()
            self._stop()
            self.get_logger().warn(
                f"[RECOVERY] Level {self._recovery_level} — will return to {self._recovery_return_state}"
            )

        elapsed = time.time() - self._state_start
        level = min(self._recovery_level, 4)

        if level <= 1:
            # Back up 0.5 s
            if elapsed < 0.5:
                self._drive("BACKWARD_SLOW")
            else:
                self._goto(self._recovery_return_state)

        elif level == 2:
            # Back 0.5 s + rotate left 0.8 s
            if elapsed < 0.5:
                self._drive("BACKWARD_SLOW")
            elif elapsed < 1.3:
                self._drive("COUNTERCLOCKWISE_ROTATION_SLOW")
            else:
                self._goto(self._recovery_return_state)

        elif level == 3:
            # Back 0.5 s + rotate right 0.8 s
            if elapsed < 0.5:
                self._drive("BACKWARD_SLOW")
            elif elapsed < 1.3:
                self._drive("CLOCKWISE_ROTATION_SLOW")
            else:
                self._goto(self._recovery_return_state)

        else:
            # Back 0.7 s + harder rotate left 1.2 s
            if elapsed < 0.7:
                self._drive("BACKWARD_SLOW")
            elif elapsed < 1.9:
                self._drive("COUNTERCLOCKWISE_ROTATION")
            else:
                self._goto(self._recovery_return_state)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _drive_toward_plan(self):
        target = self.nav.get_next_waypoint(self.params["plan_follow_min_dist"])
        if target is None:
            self._stop()
            return

        rx, ry = self.nav.position
        tx, ty = target
        target_yaw = math.atan2(ty - ry, tx - rx)
        diff = math.degrees(target_yaw - self.nav.yaw)
        diff = (diff + 180.0) % 360.0 - 180.0

        fwd_th = self.params["plan_forward_angle_deg"]
        if abs(diff) < fwd_th:
            action = "FORWARD"
        elif diff > 0:
            action = "COUNTERCLOCKWISE_ROTATION"
        else:
            action = "CLOCKWISE_ROTATION"

        self.get_logger().info(
            f"[NAV] pos=({rx:.2f},{ry:.2f})  target=({tx:.2f},{ty:.2f})  "
            f"err={diff:.1f}°  dist={self.nav.distance_to_goal():.2f}m → {action}"
        )
        self._drive(action)

    def _check_stuck(self, return_state: str) -> bool:
        """
        Position-based stuck detector for linear motion.
        Returns True and triggers RECOVERY when no movement is detected
        for stuck_timeout seconds while driving linearly.
        """
        if self.nav.position is None:
            return False

        now = time.time()
        pos_x, pos_y = self.nav.position

        if self._last_pos_check_time is None:
            self._last_pos_check_time = now
            self._last_pos_x = pos_x
            self._last_pos_y = pos_y
            return False

        check_interval = self.params.get("stuck_check_interval", 0.5)
        if now - self._last_pos_check_time < check_interval:
            return False

        moved = math.hypot(pos_x - self._last_pos_x, pos_y - self._last_pos_y)
        self._last_pos_check_time = now
        self._last_pos_x = pos_x
        self._last_pos_y = pos_y

        move_threshold = self.params.get("stuck_move_threshold", 0.03)

        if moved < move_threshold:
            if self._stuck_since is None:
                self._stuck_since = now
            elif now - self._stuck_since >= self.params.get("stuck_timeout", 2.0):
                self._stuck_since = None
                self._recovery_level += 1
                self._recovery_return_state = return_state
                self.get_logger().warn(
                    f"[STUCK] No movement for {self.params.get('stuck_timeout', 2.0):.1f}s "
                    f"in {return_state} — RECOVERY level {self._recovery_level}"
                )
                self._goto("RECOVERY")
                return True
        else:
            self._stuck_since = None
            if self._recovery_level > 0:
                self._recovery_level = max(0, self._recovery_level - 1)

        return False


def main(args=None):
    rclpy.init(args=args)
    node = GetBearNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.car.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
