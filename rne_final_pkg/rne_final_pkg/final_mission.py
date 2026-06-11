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
    T1_VERIFY = auto()

    T2_NAV_APPROACH  = auto()
    T2_BRIDGE_ALIGN  = auto()
    T2_CROSS         = auto()
    T2_SPIN          = auto()
    T2_OBS           = auto()
    T2_APPR          = auto()
    T2_GRAB          = auto()
    T2_VERIFY        = auto()
    T2_EXIT          = auto()

    T3_NAV    = auto()
    T3_SPIN   = auto()
    T3_OBS    = auto()
    T3_APPR   = auto()
    T3_UNLOCK = auto()
    T3_VERIFY = auto()
    T3_CLEAR  = auto()

    BRIDGE_AVOID = auto()

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

        self._no_plan_since  = None
        self._recovery_until = 0.0
        self._avoid_return   = S.IDLE   # state to resume after BRIDGE_AVOID
        self._avoid_dx       = 0.0      # bridge delta_x captured at avoidance start
        self._verify_retry   = S.IDLE   # APPR state to retry on verify failure

        # 10 Hz control loop
        self.create_timer(0.1, self._tick)
        self.get_logger().info("FinalMission node ready. Starting in 2 s…")
        self._start_timer = self.create_timer(2.0, self._start, clock=self.get_clock())

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
        self._start_timer.cancel()
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
            self._state_grab(S.T1_VERIFY)
        elif s == S.T1_VERIFY:
            self._state_verify_grab(S.T2_NAV_APPROACH, S.T1_APPR)

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
            self._state_grab(S.T2_VERIFY)
        elif s == S.T2_VERIFY:
            self._state_verify_grab(S.T2_EXIT, S.T2_APPR)
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
            self._state_grab(S.T3_VERIFY)
        elif s == S.T3_VERIFY:
            self._state_verify_grab(S.T3_CLEAR, S.T3_APPR)
        elif s == S.T3_CLEAR:
            self._state_clear(S.DONE)

        elif s == S.BRIDGE_AVOID:
            self._state_bridge_avoid()

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
            self.get_logger().info(
                f"[NAV_WP] sending goal {self._wp_idx + 1}/{len(waypoints)}: "
                f"({wp[0]:.2f}, {wp[1]:.2f})"
            )
            self.nav.send_goal(wp[0], wp[1])
            self._state_start = time.time()

        self._follow_plan()

        if self.nav.arrived(self.params["nav_arrive_threshold"]):
            self.get_logger().info(
                f"[NAV_WP] arrived at wp {self._wp_idx + 1}/{len(waypoints)}  "
                f"dist={self.nav.distance_to_goal():.2f}m"
            )
            self._wp_idx      += 1
            self._state_start  = None
            if self._wp_idx >= len(waypoints):
                self.get_logger().info("[NAV_WP] all waypoints done")
                self._goto(next_state)
            else:
                nxt = waypoints[self._wp_idx]
                self.get_logger().info(
                    f"[NAV_WP] sending goal {self._wp_idx + 1}/{len(waypoints)}: "
                    f"({nxt[0]:.2f}, {nxt[1]:.2f})"
                )
                self.nav.send_goal(nxt[0], nxt[1])

    def _state_nav_single(self, goal, next_state):
        if self._state_start is None:
            self.get_logger().info(f"[NAV_SINGLE] sending goal ({goal[0]:.2f}, {goal[1]:.2f})")
            self.nav.send_goal(goal[0], goal[1])
            self._state_start = time.time()

        self._follow_plan()

        if self.nav.arrived(self.params["nav_arrive_threshold"]):
            self.get_logger().info(
                f"[NAV_SINGLE] arrived  dist={self.nav.distance_to_goal():.2f}m"
            )
            self.car.stop()
            self._goto(next_state)

    def _follow_plan(self):
        now = time.time()

        # stuck recovery: back up if triggered
        if now < self._recovery_until:
            remaining = self._recovery_until - now
            self.get_logger().info(f"[STUCK] recovering — BACKWARD_SLOW ({remaining:.1f}s left)")
            self.car.publish("BACKWARD_SLOW")
            return

        if not self.nav.has_plan or self.nav.position is None:
            if self._no_plan_since is None:
                self._no_plan_since = now
                self.get_logger().warn(
                    f"[NAV] no global plan / position  "
                    f"(has_plan={self.nav.has_plan}  pos={'ok' if self.nav.position else 'None'})"
                )
            else:
                waiting = now - self._no_plan_since
                self.get_logger().info(f"[NAV] waiting for plan… {waiting:.1f}s")
                if waiting > 3.0:
                    self.get_logger().warn("[STUCK] no global plan for 3s — backing up")
                    self._recovery_until = now + 1.5
                    self._no_plan_since  = None
            self.car.stop()
            return

        # plan is healthy — reset the no-plan timer
        if self._no_plan_since is not None:
            self.get_logger().info("[NAV] plan received — resuming")
            self._no_plan_since = None

        target = self.nav.get_next_waypoint(self.params["plan_follow_min_dist"])
        if target is None:
            self.car.stop()
            return

        car_x, car_y = self.nav.position
        tx, ty = target
        target_yaw = math.atan2(ty - car_y, tx - car_x)

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
            f"[NAV] pos=({car_x:.2f},{car_y:.2f})  target=({tx:.2f},{ty:.2f})  "
            f"heading_err={diff:.1f}°  dist_to_goal={self.nav.distance_to_goal():.2f}m → {action}"
        )
        self.car.publish(action)

    # ------------------------------------------------------------------
    # Search / observe
    # ------------------------------------------------------------------

    def _state_spin_search(self, next_state):
        if self._state_start is None:
            self._state_start = time.time()

        if self.yolo.is_visible():
            self.get_logger().info(
                f"[SPIN_SEARCH] target found  dist={self.yolo.distance():.2f}m  "
                f"dx={self.yolo.delta_x():.0f}px → stopping, go to {next_state.name}"
            )
            self.car.stop()
            self._goto(next_state)
            return

        elapsed = time.time() - self._state_start
        if elapsed > self.params["search_rotation_timeout"]:
            self.get_logger().warn("Spin search timed out; advancing anyway.")
            self.car.stop()
            self._goto(next_state)
            return

        self.get_logger().info(
            f"[SPIN_SEARCH] no target  elapsed={elapsed:.1f}s → CLOCKWISE_ROTATION_SLOW"
        )
        self.car.publish("CLOCKWISE_ROTATION_SLOW")

    def _state_observe(self, next_state):
        if self._state_start is None:
            self._state_start = time.time()
            self.car.stop()
            self.get_logger().info(
                f"[OBSERVE] pausing {self.params['observe_wait_seconds']:.1f}s  "
                f"yolo_visible={self.yolo.is_visible()}  "
                f"dist={self.yolo.distance():.2f}m  dx={self.yolo.delta_x():.0f}px"
            )

        elapsed = time.time() - self._state_start
        remaining = self.params["observe_wait_seconds"] - elapsed
        if remaining > 0:
            self.get_logger().info(f"[OBSERVE] {remaining:.1f}s remaining")
        else:
            self.get_logger().info("[OBSERVE] done → proceeding")
            self._goto(next_state)

    # ------------------------------------------------------------------
    # Detection visual servo
    # ------------------------------------------------------------------

    def _state_approach(self, next_state):
        # Bridge safety: if seg detects a bridge ahead, divert before continuing
        bridge_area = self.params.get("bridge_safety_area_threshold", 0.05)
        if self.yolo.bridge_visible() and self.yolo.bridge_area_ratio() > bridge_area:
            self.get_logger().warn(
                f"[APPROACH] Bridge ahead  area={self.yolo.bridge_area_ratio():.3f}  "
                f"dx={self.yolo.bridge_delta_x():.0f}px → BRIDGE_AVOID"
            )
            self._avoid_return = self._state
            self._avoid_dx = self.yolo.bridge_delta_x()
            self.car.stop()
            self._goto(S.BRIDGE_AVOID)
            return

        if not self.yolo.is_visible():
            self.get_logger().info("[APPROACH] no target → CLOCKWISE_ROTATION_SLOW")
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
            return

        dx   = self.yolo.delta_x()
        dist = self.yolo.distance()
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
            f"[APPROACH] dist={dist:.2f}m  dx={dx:.0f}px  "
            f"(rotate_th=±{rotate_th:.0f}  grab_th={grab_th:.2f}m) → {action}"
        )

        if action == "CLOCKWISE_ROTATION_SLOW":
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
        elif action == "COUNTERCLOCKWISE_ROTATION_SLOW":
            self.car.publish("COUNTERCLOCKWISE_ROTATION_SLOW")
        elif action.startswith("→ GRAB"):
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
        self.get_logger().info("[GRAB] starting grab sequence")

        x_target, z_target = self._get_arm_target_or_default()
        self.get_logger().info(f"[GRAB] arm target x={x_target:.3f}  z={z_target:.3f}")
        self.arm.grab_sequence(x_target=x_target, z_target=z_target)

        self.get_logger().info("[GRAB] grab done — resetting arm")
        time.sleep(1.0)
        self.arm.reset()

        self.get_logger().info("[GRAB] arm reset — advancing")
        self._grab_busy = False
        self._goto(next_state)

    def _get_arm_target_or_default(self):
        default = (0.15, 0.05)
        marker = self.yolo.marker()
        if marker is None:
            self.get_logger().warn("No marker; using default grab target.")
            return default

        # Use the frame the marker was actually published in (camera frame)
        pt = PointStamped()
        pt.header.frame_id = marker.header.frame_id
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point = marker.pose.position

        try:
            tf = self.tf_buffer.lookup_transform(
                "arm_ik_base", marker.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5),
            )
            pt_arm = tf2_geometry_msgs.do_transform_point(pt, tf)
            x = pt_arm.point.x
            z = pt_arm.point.z
            self.get_logger().info(f"Arm target from TF: x={x:.3f}, z={z:.3f}")
            return (x, z)
        except Exception as e:
            self.get_logger().warn(f"TF grab failed: {e}; using default.")
            return default

    # ------------------------------------------------------------------
    # Grab verification — back up briefly and confirm bear is gone
    # ------------------------------------------------------------------

    def _state_verify_grab(self, success_state: S, retry_state: S):
        """
        After a grab attempt: back up, then check whether the bear is still
        visible at grab range.  If it is → grab failed, retry approach.
        If not (bear gone or now on the arm at very close range) → success.
        """
        if self._state_start is None:
            self._state_start = time.time()

        elapsed  = time.time() - self._state_start
        back_t   = self.params.get("verify_backup_seconds", 0.4)
        obs_t    = back_t + self.params.get("verify_observe_seconds", 0.8)

        if elapsed < back_t:
            self.car.publish("BACKWARD_SLOW")
            return

        self.car.stop()

        if elapsed < obs_t:
            return   # still observing

        dist     = self.yolo.distance()
        grab_th  = self.params["grab_distance_threshold"]
        # Bear still at grab distance with valid depth → grab failed
        still_there = self.yolo.is_visible() and 0 < dist < grab_th

        if still_there:
            self.get_logger().warn(
                f"[VERIFY] Grab failed — bear still at dist={dist:.2f}m → {retry_state.name}"
            )
            self._goto(retry_state)
        else:
            self.get_logger().info(
                f"[VERIFY] Grab OK (dist={dist:.2f}m) → {success_state.name}"
            )
            self._goto(success_state)

    # ------------------------------------------------------------------
    # Bridge avoidance — back up then turn away, then resume saved state
    # ------------------------------------------------------------------

    def _state_bridge_avoid(self):
        if self._state_start is None:
            self._state_start = time.time()
            self.get_logger().warn(
                f"[BRIDGE_AVOID] Backing up then turning away  dx={self._avoid_dx:.0f}px  "
                f"will return to {self._avoid_return.name}"
            )

        elapsed = time.time() - self._state_start
        back_t  = self.params.get("bridge_avoid_backward_seconds", 0.5)
        turn_t  = self.params.get("bridge_avoid_turn_seconds", 0.8)

        if elapsed < back_t:
            self.car.publish("BACKWARD_SLOW")
        elif elapsed < back_t + turn_t:
            # bridge right of center → turn left; bridge left → turn right
            if self._avoid_dx >= 0:
                self.car.publish("COUNTERCLOCKWISE_ROTATION_SLOW")
            else:
                self.car.publish("CLOCKWISE_ROTATION_SLOW")
        else:
            self.get_logger().info(f"[BRIDGE_AVOID] Done → {self._avoid_return.name}")
            self._goto(self._avoid_return)

    # ------------------------------------------------------------------
    # Bridge crossing (Task 2)
    # ------------------------------------------------------------------

    def _state_bridge_align(self, next_state):
        if not self.yolo.bridge_visible():
            self.get_logger().info("[BRIDGE_ALIGN] bridge not visible → CLOCKWISE_ROTATION_SLOW")
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
            return

        dx   = self.yolo.bridge_delta_x()
        area = self.yolo.bridge_area_ratio()

        deadband = self.params["bridge_deadband_px"]
        min_area = self.params["bridge_min_area_ratio"]

        if area < min_area:
            self.get_logger().info(
                f"[BRIDGE_ALIGN] area={area:.3f} < {min_area} (too far) → FORWARD_SLOW"
            )
            self.car.publish("FORWARD_SLOW")
            return

        if dx > deadband:
            action = "CLOCKWISE_ROTATION_SLOW"
        elif dx < -deadband:
            action = "COUNTERCLOCKWISE_ROTATION_SLOW"
        else:
            action = "→ ALIGNED"

        self.get_logger().info(
            f"[BRIDGE_ALIGN] dx={dx:.0f}px  area={area:.3f}  "
            f"(deadband=±{deadband:.0f}) → {action}"
        )

        if action == "CLOCKWISE_ROTATION_SLOW":
            self.car.publish("CLOCKWISE_ROTATION_SLOW")
        elif action == "COUNTERCLOCKWISE_ROTATION_SLOW":
            self.car.publish("COUNTERCLOCKWISE_ROTATION_SLOW")
        else:
            self.car.stop()
            self._goto(next_state)

    def _state_cross_bridge(self, next_state):
        if self._state_start is None:
            self._state_start = time.time()

        elapsed  = time.time() - self._state_start
        deadline = self.params["bridge_cross_seconds"]

        if self.yolo.bridge_visible():
            dx       = self.yolo.bridge_delta_x()
            deadband = self.params["bridge_deadband_px"]
            if dx > deadband:
                action = "CLOCKWISE_ROTATION_SLOW"
            elif dx < -deadband:
                action = "COUNTERCLOCKWISE_ROTATION_SLOW"
            else:
                action = "FORWARD_SLOW"
            self.get_logger().info(
                f"[CROSS_BRIDGE] bridge visible  dx={dx:.0f}px  "
                f"elapsed={elapsed:.1f}/{deadline:.1f}s → {action}"
            )
        else:
            action = "FORWARD_SLOW"
            self.get_logger().info(
                f"[CROSS_BRIDGE] bridge not visible  elapsed={elapsed:.1f}/{deadline:.1f}s → {action}"
            )

        self.car.publish(action)

        if elapsed > deadline:
            self.car.stop()
            self._state_start = None
            self._goto(next_state)

    # ------------------------------------------------------------------
    # Task 3 clear (fallback timed sequence)
    # ------------------------------------------------------------------

    def _state_clear(self, next_state):
        self.get_logger().info(
            f"[CLEAR] backing up {self.params['clear_backward_seconds']:.1f}s"
        )
        self.car.publish("BACKWARD_SLOW")
        time.sleep(self.params["clear_backward_seconds"])

        self.get_logger().info(
            f"[CLEAR] rotating {self.params['clear_rotate_seconds']:.1f}s"
        )
        self.car.publish("CLOCKWISE_ROTATION_SLOW")
        time.sleep(self.params["clear_rotate_seconds"])

        self.get_logger().info(
            f"[CLEAR] moving forward {self.params['clear_forward_seconds']:.1f}s"
        )
        self.car.publish("FORWARD_SLOW")
        time.sleep(self.params["clear_forward_seconds"])

        self.car.stop()
        self.get_logger().info("[CLEAR] done")
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
