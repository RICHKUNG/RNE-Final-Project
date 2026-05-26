import math
import time
from enum import Enum, auto

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from ament_index_python.packages import get_package_share_directory
import os
import yaml

from rne_final_pkg.car_driver import CarDriver
from rne_final_pkg.arm_driver import ArmDriver
from rne_final_pkg.nav_client import NavClient
from rne_final_pkg.yolo_client import YoloClient


class S(Enum):
    IDLE = auto()

    T1_NAV    = auto()
    T1_SPIN   = auto()
    T1_OBS    = auto()
    T1_APPR   = auto()
    T1_GRAB   = auto()

    T2_NAV_APPROACH  = auto()
    T2_BRIDGE_ALIGN  = auto()
    T2_CROSS         = auto()
    T2_SPIN          = auto()
    T2_OBS           = auto()
    T2_APPR          = auto()
    T2_GRAB          = auto()
    T2_EXIT          = auto()

    T3_NAV    = auto()
    T3_SPIN   = auto()
    T3_OBS    = auto()
    T3_APPR   = auto()
    T3_UNLOCK = auto()
    T3_CLEAR  = auto()

    DONE = auto()


class FinalMission(Node):
    def __init__(self):
        super().__init__("final_mission")

        self._load_params()

        self.car  = CarDriver(self)
        self.arm  = ArmDriver(self)
        self.nav  = NavClient(self)
        self.yolo = YoloClient(self)

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._state       = S.IDLE
        self._state_start = None
        self._grab_busy   = False
        self._wp_idx      = 0

        # 10 Hz control loop
        self.create_timer(0.1, self._tick)
        self.get_logger().info("FinalMission node ready. Starting in 2 s…")
        self.create_timer(2.0, self._start, clock=self.get_clock())

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _load_params(self):
        share = get_package_share_directory("rne_final_pkg")
        mission_path = os.path.join(share, "config", "mission.yaml")
        goals_path   = os.path.join(share, "config", "goals.yaml")

        with open(mission_path) as f:
            self.params = yaml.safe_load(f)
        with open(goals_path) as f:
            self.goals = yaml.safe_load(f)

    def _start(self):
        self._goto(S.T1_NAV)

    # ------------------------------------------------------------------
    # State machine core
    # ------------------------------------------------------------------

    def _goto(self, state: S):
        self.get_logger().info(f"State: {self._state.name} -> {state.name}")
        self._state       = state
        self._state_start = None
        self._wp_idx      = 0

    def _tick(self):
        s = self._state

        if s == S.IDLE:
            pass

        # ---- Task 1 ----
        elif s == S.T1_NAV:
            self._state_nav_waypoints("task1_search_waypoints", S.T1_SPIN)
        elif s == S.T1_SPIN:
            self._state_spin_search(S.T1_OBS)
        elif s == S.T1_OBS:
            self._state_observe(S.T1_APPR)
        elif s == S.T1_APPR:
            self._state_approach(S.T1_GRAB)
        elif s == S.T1_GRAB:
            self._state_grab(S.T2_NAV_APPROACH)

        # ---- Task 2 ----
        elif s == S.T2_NAV_APPROACH:
            goal = self.goals["task2_bridge_approach"]
            self._state_nav_single(goal, S.T2_BRIDGE_ALIGN)
        elif s == S.T2_BRIDGE_ALIGN:
            self._state_bridge_align(S.T2_CROSS)
        elif s == S.T2_CROSS:
            self._state_cross_bridge(S.T2_SPIN)
        elif s == S.T2_SPIN:
            self._state_spin_search(S.T2_OBS)
        elif s == S.T2_OBS:
            self._state_observe(S.T2_APPR)
        elif s == S.T2_APPR:
            self._state_approach(S.T2_GRAB)
        elif s == S.T2_GRAB:
            self._state_grab(S.T2_EXIT)
        elif s == S.T2_EXIT:
            goal = self.goals["task2_bridge_exit"]
            self._state_nav_single(goal, S.T3_NAV)

        # ---- Task 3 ----
        elif s == S.T3_NAV:
            self._state_nav_waypoints("task3_search_waypoints", S.T3_SPIN)
        elif s == S.T3_SPIN:
            self._state_spin_search(S.T3_OBS)
        elif s == S.T3_OBS:
            self._state_observe(S.T3_APPR)
        elif s == S.T3_APPR:
            self._state_approach(S.T3_UNLOCK)
        elif s == S.T3_UNLOCK:
            self._state_grab(S.T3_CLEAR)
        elif s == S.T3_CLEAR:
            self._state_clear(S.DONE)

        elif s == S.DONE:
            self.car.stop()
            self.get_logger().info("Mission complete.")

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _state_nav_waypoints(self, goals_key, next_state):
        waypoints = self.goals[goals_key]
        if self._wp_idx >= len(waypoints):
            self._goto(next_state)
            return

        wp = waypoints[self._wp_idx]
        if self._state_start is None:
            self.nav.send_goal(wp[0], wp[1])
            self._state_start = time.time()

        self._follow_plan()

        if self.nav.arrived(self.params["nav_arrive_threshold"]):
            self._wp_idx      += 1
            self._state_start  = None
            if self._wp_idx >= len(waypoints):
                self._goto(next_state)
            else:
                nxt = waypoints[self._wp_idx]
                self.nav.send_goal(nxt[0], nxt[1])

    def _state_nav_single(self, goal, next_state):
        if self._state_start is None:
            self.nav.send_goal(goal[0], goal[1])
            self._state_start = time.time()

        self._follow_plan()

        if self.nav.arrived(self.params["nav_arrive_threshold"]):
            self.car.stop()
            self._goto(next_state)

    def _follow_plan(self):
        if not self.nav.has_plan or self.nav.position is None:
            self.car.stop()
            return

        target = self.nav.get_next_waypoint(self.params["plan_follow_min_dist"])
        if target is None:
            self.car.stop()
            return

        car_x, car_y = self.nav.position
        tx, ty = target
        target_yaw = math.atan2(ty - car_y, tx - car_x)

        diff = math.degrees(target_yaw - self.nav.yaw)
        diff = (diff + 180.0) % 360.0 - 180.0

        if abs(diff) < self.params["plan_forward_angle_deg"]:
            self.car.publish("FORWARD")
        elif diff > 0:
            self.car.publish("COUNTERCLOCKWISE_ROTATION")
        else:
            self.car.publish("CLOCKWISE_ROTATION")

    # ------------------------------------------------------------------
    # Search / observe
    # ------------------------------------------------------------------

    def _state_spin_search(self, next_state):
        if self._state_start is None:
            self._state_start = time.time()

        if self.yolo.is_visible():
            self.car.stop()
            self._goto(next_state)
            return

        elapsed = time.time() - self._state_start
        if elapsed > self.params["search_rotation_timeout"]:
            self.get_logger().warn("Spin search timed out; advancing anyway.")
            self.car.stop()
            self._goto(next_state)
            return

        self.car.publish("CLOCKWISE_ROTATION_SLOW")

    def _state_observe(self, next_state):
        if self._state_start is None:
            self._state_start = time.time()
            self.car.stop()

        if time.time() - self._state_start >= self.params["observe_wait_seconds"]:
            self._goto(next_state)

    # ------------------------------------------------------------------
    # Detection visual servo
    # ------------------------------------------------------------------

    def _state_approach(self, next_state):
        if not self.yolo.is_visible():
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
            return

        dx   = self.yolo.delta_x()
        dist = self.yolo.distance()
        rotate_th = self.params["visual_servo_rotate_threshold_px"]
        grab_th   = self.params["grab_distance_threshold"]

        if dx > rotate_th:
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
        elif dx < -rotate_th:
            self.car.publish("COUNTERCLOCKWISE_ROTATION_SLOW")
        elif 0 < dist < grab_th:
            self.car.stop()
            self._goto(next_state)
        elif dist < 0:
            # depth invalid — assume close enough
            self.car.stop()
            self._goto(next_state)
        else:
            self.car.publish("FORWARD_SLOW")

    # ------------------------------------------------------------------
    # Arm grab
    # ------------------------------------------------------------------

    def _state_grab(self, next_state):
        if self._grab_busy:
            return

        self._grab_busy = True
        self.car.stop()

        x_target, z_target = self._get_arm_target_or_default()
        self.arm.grab_sequence(x_target=x_target, z_target=z_target)

        time.sleep(1.0)
        self.arm.reset()

        self._grab_busy = False
        self._goto(next_state)

    def _get_arm_target_or_default(self):
        default = (0.15, 0.05)
        marker = self.yolo.marker()
        if marker is None:
            self.get_logger().warn("No marker; using default grab target.")
            return default

        pt_map = PointStamped()
        pt_map.header.frame_id = "map"
        pt_map.header.stamp = self.get_clock().now().to_msg()
        pt_map.point = marker.pose.position

        try:
            tf = self.tf_buffer.lookup_transform(
                "arm_ik_base", "map",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            pt_arm = tf2_geometry_msgs.do_transform_point(pt_map, tf)
            x = pt_arm.point.x
            z = pt_arm.point.z
            self.get_logger().info(f"Arm target from TF: x={x:.3f}, z={z:.3f}")
            return (x, z)
        except Exception as e:
            self.get_logger().warn(f"TF grab failed: {e}; using default.")
            return default

    # ------------------------------------------------------------------
    # Bridge crossing (Task 2)
    # ------------------------------------------------------------------

    def _state_bridge_align(self, next_state):
        if not self.yolo.bridge_visible():
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
            return

        dx   = self.yolo.bridge_delta_x()
        area = self.yolo.bridge_area_ratio()

        deadband = self.params["bridge_deadband_px"]
        min_area = self.params["bridge_min_area_ratio"]

        if area < min_area:
            self.car.publish("FORWARD_SLOW")
            return

        if dx > deadband:
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
        elif dx < -deadband:
            self.car.publish("COUNTERCLOCKWISE_ROTATION_SLOW")
        else:
            self.car.stop()
            self._goto(next_state)

    def _state_cross_bridge(self, next_state):
        if self._state_start is None:
            self._state_start = time.time()

        if self.yolo.bridge_visible():
            dx       = self.yolo.bridge_delta_x()
            deadband = self.params["bridge_deadband_px"]
            if dx > deadband:
                self.car.publish("CLOCKWISE_ROTATION_SLOW")
            elif dx < -deadband:
                self.car.publish("COUNTERCLOCKWISE_ROTATION_SLOW")
            else:
                self.car.publish("FORWARD_SLOW")
        else:
            self.car.publish("FORWARD_SLOW")

        if time.time() - self._state_start > self.params["bridge_cross_seconds"]:
            self.car.stop()
            self._state_start = None
            self._goto(next_state)

    # ------------------------------------------------------------------
    # Task 3 clear (fallback timed sequence)
    # ------------------------------------------------------------------

    def _state_clear(self, next_state):
        self.car.publish("BACKWARD_SLOW")
        time.sleep(self.params["clear_backward_seconds"])

        self.car.publish("CLOCKWISE_ROTATION_SLOW")
        time.sleep(self.params["clear_rotate_seconds"])

        self.car.publish("FORWARD_SLOW")
        time.sleep(self.params["clear_forward_seconds"])

        self.car.stop()
        self._goto(next_state)


def main(args=None):
    rclpy.init(args=args)
    node = FinalMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.car.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
