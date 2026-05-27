"""
get_bear_node.py — autonomous bear-retrieval state machine.

States
------
SEARCH_SPIN   rotate in place until a bear is detected
LOCALIZE      backproject bear pixel → map coords, send Nav2 goal
NAV_TO_BEAR   follow Nav2 plan to (stop_dist) meters in front of bear
VISUAL_SERVO  close-range wheel-based centering + approach
GRAB          publish /clicked_point → arm auto-controller handles the grab
DONE          stop and hold
"""
import math
import os
import time

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import Float32MultiArray
from trajectory_msgs.msg import JointTrajectoryPoint
from action_interface.action import ArmGoal
from ament_index_python.packages import get_package_share_directory
import yaml

from rne_final_pkg.car_driver import CarDriver
from rne_final_pkg.nav_client import NavClient


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

        # /robot_arm publisher — stow arm on startup
        self._arm_pub = self.create_publisher(
            JointTrajectoryPoint, "/robot_arm", 10
        )
        # Retry publishing stow until arm_writer subscribes (polls every 0.5 s, gives up after 15 s)
        self._stow_attempts = 0
        self._stow_timer = self.create_timer(0.5, self._stow_arm_once)

        # Arm action client — sends catch2 goal to arm_action_server
        self._arm_action_client = ActionClient(self, ArmGoal, "arm_action_server")

        self._state = "SEARCH_SPIN"
        self._state_start = None
        self._bear_map_pos = None   # (x, y) in map frame
        self._grab_goal_sent = False

        self.create_timer(0.1, self._tick)

        # Publish /initialpose after a short delay if AMCL hasn't localized yet
        timeout = self.params.get("initial_pose_timeout", 3.0)
        self.create_timer(timeout, self._maybe_publish_initialpose)

        self.get_logger().info("GetBearNode ready.")

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_params(self):
        share = get_package_share_directory("rne_final_pkg")
        path = os.path.join(share, "config", "get_bear.yaml")
        with open(path) as f:
            self.params = yaml.safe_load(f)

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
        # Diagonal covariance: 0.25 m² position, 0.068 rad² yaw
        msg.pose.covariance[0]  = 0.25
        msg.pose.covariance[7]  = 0.25
        msg.pose.covariance[35] = 0.068

        self._initialpose_pub.publish(msg)
        self.get_logger().info(
            f"[INIT_POSE] Published /initialpose  x={x}  y={y}  yaw={yaw}"
        )

    # ------------------------------------------------------------------
    # Arm stow — fires once 1 s after startup
    # ------------------------------------------------------------------

    def _stow_arm_once(self):
        self._stow_attempts += 1
        # Wait until arm_writer has subscribed before sending; give up after 30 tries (15 s)
        if self._arm_pub.get_subscription_count() == 0:
            if self._stow_attempts >= 30:
                self.get_logger().warn("[STOW] arm_writer never subscribed — skipping stow")
                self.destroy_timer(self._stow_timer)
            return
        # joints_reset from arm_config.yaml — matches Manual Arm Control 'b' key
        angles_deg = [90.0, 30.0, 160.0, 180.0, 70.0]
        msg = JointTrajectoryPoint()
        msg.positions = [math.radians(a) for a in angles_deg]
        msg.velocities = []       # must be empty — matches arm_commute_node format
        msg.accelerations = []
        msg.effort = []
        self._arm_pub.publish(msg)
        self.get_logger().info(
            f"[STOW] Published stow pose to /robot_arm (attempt {self._stow_attempts})"
        )
        self.destroy_timer(self._stow_timer)

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _bear_cb(self, msg):
        self._bear_info = list(msg.data)

    def _camera_info_cb(self, msg):
        self._camera_info = msg

    # ------------------------------------------------------------------
    # Bear-info helpers
    # ------------------------------------------------------------------

    def _bear_visible(self):
        return (
            self._bear_info is not None
            and len(self._bear_info) >= 1
            and self._bear_info[0] == 1.0
        )

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

    def _tick(self):
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
        elif s == "DONE":
            self.car.stop()

    # ------------------------------------------------------------------
    # SEARCH_SPIN — rotate until bear visible
    # ------------------------------------------------------------------

    def _state_search_spin(self):
        if self._state_start is None:
            self._state_start = time.time()

        if self._bear_visible():
            self.get_logger().info(
                f"[SEARCH] Bear found  dist={self._bear_distance():.2f}m  "
                f"dx={self._bear_delta_x():.0f}px"
            )
            self.car.stop()
            self._goto("LOCALIZE")
            return

        elapsed = time.time() - self._state_start
        timeout = self.params["search_rotation_timeout"]
        if elapsed > timeout:
            self.get_logger().warn("[SEARCH] Timeout — no bear found. Stopping.")
            self.car.stop()
            self._goto("DONE")
            return

        self.get_logger().info(
            f"[SEARCH] No bear  elapsed={elapsed:.1f}/{timeout:.0f}s → CLOCKWISE_ROTATION_SLOW"
        )
        self.car.publish("CLOCKWISE_ROTATION_SLOW")

    # ------------------------------------------------------------------
    # LOCALIZE — backproject pixel to map frame, send Nav2 goal
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

        # Backproject to camera_optical_frame (X right, Y down, Z forward)
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

        if dist_to_bear <= stop_dist:
            self.get_logger().info(
                f"[LOCALIZE] Already {dist_to_bear:.2f}m from bear — skipping Nav2"
            )
            self._goto("VISUAL_SERVO")
            return

        goal_x = bear_x - (dx / dist_to_bear) * stop_dist
        goal_y = bear_y - (dy / dist_to_bear) * stop_dist
        self.get_logger().info(
            f"[LOCALIZE] Nav2 goal ({goal_x:.2f}, {goal_y:.2f})  stop_dist={stop_dist:.1f}m"
        )
        self.nav.send_goal(goal_x, goal_y)
        self._goto("NAV_TO_BEAR")

    # ------------------------------------------------------------------
    # NAV_TO_BEAR — follow Nav2 plan
    # ------------------------------------------------------------------

    def _state_nav_to_bear(self):
        if not self.nav.has_plan or self.nav.position is None:
            self.get_logger().info("[NAV] Waiting for plan…")
            self.car.stop()
            return

        if self.nav.arrived(self.params["nav_arrive_threshold"]):
            self.get_logger().info(
                f"[NAV] Arrived  dist={self.nav.distance_to_goal():.2f}m → VISUAL_SERVO"
            )
            self.car.stop()
            self._goto("VISUAL_SERVO")
            return

        target = self.nav.get_next_waypoint(self.params["plan_follow_min_dist"])
        if target is None:
            self.car.stop()
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
        self.car.publish(action)

    # ------------------------------------------------------------------
    # VISUAL_SERVO — close-range centering + approach
    # ------------------------------------------------------------------

    def _state_visual_servo(self):
        if not self._bear_visible():
            self.get_logger().info(
                "[VISUAL_SERVO] Bear not visible → CLOCKWISE_ROTATION_SLOW"
            )
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
            return

        dx = self._bear_delta_x()
        dist = self._bear_distance()
        rotate_th = self.params["visual_servo_rotate_threshold_px"]
        grab_th = self.params["grab_distance_threshold"]

        if dx > rotate_th:
            action = "CLOCKWISE_ROTATION_SLOW"
        elif dx < -rotate_th:
            action = "COUNTERCLOCKWISE_ROTATION_SLOW"
        elif dist < 0 or (0 < dist < grab_th):
            action = "→ GRAB"
        else:
            action = "FORWARD_SLOW"

        self.get_logger().info(
            f"[VISUAL_SERVO] dist={dist:.2f}m  dx={dx:.0f}px  "
            f"(rotate_th=±{rotate_th:.0f}  grab_th={grab_th:.2f}m) → {action}"
        )

        if action == "CLOCKWISE_ROTATION_SLOW":
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
        elif action == "COUNTERCLOCKWISE_ROTATION_SLOW":
            self.car.publish("COUNTERCLOCKWISE_ROTATION_SLOW")
        elif action == "→ GRAB":
            self.car.stop()
            self._goto("GRAB")
        else:
            self.car.publish("FORWARD_SLOW")

    # ------------------------------------------------------------------
    # GRAB — send catch2 action goal to arm_action_server
    # ------------------------------------------------------------------

    def _state_grab(self):
        if self._grab_goal_sent:
            return  # waiting for result callback

        self.car.stop()
        self._grab_goal_sent = True

        if not self._arm_action_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn(
                "[GRAB] arm_action_server not available — is arm_control_pkg running? Skipping."
            )
            self._goto("DONE")
            return

        goal = ArmGoal.Goal()
        goal.mode = "catch2"
        self.get_logger().info("[GRAB] Sending catch2 action goal to arm_action_server")
        send_future = self._arm_action_client.send_goal_async(goal)
        send_future.add_done_callback(self._grab_goal_response_cb)

    def _grab_goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn("[GRAB] catch2 goal rejected by arm_action_server")
            self._goto("DONE")
            return
        self.get_logger().info("[GRAB] catch2 accepted — arm executing grab sequence")
        goal_handle.get_result_async().add_done_callback(self._grab_result_cb)

    def _grab_result_cb(self, future):
        result = future.result().result
        self.get_logger().info(
            f"[GRAB] catch2 complete  success={result.success}  msg={result.message}"
        )
        self._goto("DONE")


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
