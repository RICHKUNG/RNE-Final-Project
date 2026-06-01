"""
rectangle_mapping.py — open-loop rectangle SLAM mapping

FORWARD phases: odom dead-reckoning (reliable during straight motion).
TURN / SPIN phases: time-based (odom is unreliable while rotating because
  scan_matcher cannot match rapidly changing scans).

Usage (inside pros_car container):
    ros2 run rne_final_pkg rect_map

Tune constants below, then: colcon build → ros2 run rne_final_pkg rect_map
"""

import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

# ── Tunable ───────────────────────────────────────────────────────────────────
L             = 2.8    # long-side distance (m)
W             = 2.8
   # short-side distance (m)

FORWARD_SPEED = 300.0  # wheel units (≥200 to move in Unity)
ROT_SPEED     = 300.0  # wheel units for in-place rotation

T_TURN_90     = 3.05    # fallback seconds if IMU unavailable
PAUSE_SEC     = 0.8    # pause between phases
# ─────────────────────────────────────────────────────────────────────────────

#  (phase_name, param)
#  FORWARD   param = distance in metres
#  TURN_LEFT param = turn duration in seconds (None → use T_TURN_90)
#  SPIN_360  param = spin duration in seconds (None → use T_SPIN_360)
#  DONE      param = None
PLAN = [
    ("FORWARD",   L),
    ("TURN_LEFT", None),
    ("FORWARD",   W),
    ("TURN_LEFT", None),
    ("FORWARD",   L),
    ("TURN_LEFT", None),
    ("FORWARD",   W),
    ("DONE",      None),
]


def _yaw(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


class RectangleMapping(Node):

    def __init__(self):
        super().__init__("rectangle_mapping")

        self._front = self.create_publisher(Float32MultiArray, "/car_C_front_wheel", 1)
        self._rear  = self.create_publisher(Float32MultiArray, "/car_C_rear_wheel",  1)

        self._odom = None
        self.create_subscription(Odometry, "/odom", lambda m: setattr(self, "_odom", m), 10)

        # IMU: integrate |angular_velocity.z| for reliable turn tracking
        self._imu_stamp     = None   # time.time() of last IMU message
        self._accum_turn    = 0.0   # accumulated rad this TURN_LEFT phase
        self._use_imu_turn  = False  # decided once per turn in _begin_phase
        self.create_subscription(Imu, "/imu/data", self._imu_cb, 20)

        self._idx           = -1
        self._phase         = "INIT"
        self._param         = None
        self._phase_started = False

        # FORWARD tracking (odom)
        self._sx = 0.0
        self._sy = 0.0

        # Fallback time-based turn deadline (used when no IMU)
        self._phase_end = 0.0

        self._pause_end = 0.0

        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"RectangleMapping  L={L}m W={W}m  "
            f"fwd={FORWARD_SPEED} rot={ROT_SPEED}  "
            f"T_turn={T_TURN_90}s  "
            "— waiting for /odom"
        )

    # ── IMU callback ──────────────────────────────────────────────────────────

    def _imu_cb(self, msg):
        now = time.time()
        if self._imu_stamp is not None and self._phase_started and self._phase == "TURN_LEFT":
            dt = now - self._imu_stamp
            # Use absolute value: we always command CCW, so any rotation counts
            self._accum_turn += abs(msg.angular_velocity.z) * dt
        self._imu_stamp = now

    # ── helpers ────────────────────────────────────────────────────────────────

    def _pub(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self._front.publish(msg)
        self._rear.publish(msg)

    def _stop(self):
        self._pub(0.0, 0.0)

    def _pos(self):
        p = self._odom.pose.pose.position
        return p.x, p.y

    # ── phase management ───────────────────────────────────────────────────────

    def _advance(self):
        self._stop()
        self._idx += 1
        self._phase, self._param = PLAN[self._idx] if self._idx < len(PLAN) else ("DONE", None)
        self._phase_started = False
        self._pause_end     = time.time() + PAUSE_SEC
        self.get_logger().info(f"→ {self._phase}  param={self._param}")

    def _begin_phase(self):
        x, y = self._pos()
        self._sx = x
        self._sy = y
        self._phase_started = True

        if self._phase == "TURN_LEFT":
            self._accum_turn     = 0.0
            self._use_imu_turn   = (self._imu_stamp is not None)
            # Always set time fallback in case IMU is absent
            dur = self._param if self._param is not None else T_TURN_90
            self._phase_end = time.time() + dur

        self.get_logger().info(
            f"── {self._phase}  pos=({x:.2f},{y:.2f})"
            + (f"  mode={'IMU' if self._use_imu_turn else 'time-fallback'}"
               if self._phase == "TURN_LEFT" else "")
        )

    # ── main loop ──────────────────────────────────────────────────────────────

    def _tick(self):
        if self._odom is None:
            self.get_logger().warn("Waiting for /odom…", throttle_duration_sec=3.0)
            return

        if time.time() < self._pause_end:
            self._stop()
            return

        if self._phase == "INIT":
            self._advance()
            return

        if self._phase == "DONE":
            self._stop()
            self.get_logger().info(
                "Rectangle complete — run store_map.sh",
                throttle_duration_sec=10.0,
            )
            return

        if not self._phase_started:
            self._begin_phase()
            return

        # ── FORWARD: odom distance ────────────────────────────────────────────
        if self._phase == "FORWARD":
            x, y = self._pos()
            dist = math.sqrt((x - self._sx) ** 2 + (y - self._sy) ** 2)
            self.get_logger().info(
                f"[FWD] {dist:.2f}/{self._param:.2f}m",
                throttle_duration_sec=1.0,
            )
            if dist >= self._param:
                self._advance()
            else:
                self._pub(FORWARD_SPEED, FORWARD_SPEED)

        # ── TURN_LEFT: IMU-based (fallback to time if IMU absent at turn start)
        elif self._phase == "TURN_LEFT":
            if self._use_imu_turn:
                done = self._accum_turn >= math.pi / 2.0
                self.get_logger().info(
                    f"[TURN/IMU] {math.degrees(self._accum_turn):.1f}°/90°",
                    throttle_duration_sec=1.0,
                )
            else:
                done = self._phase_end - time.time() <= 0
                self.get_logger().info(
                    f"[TURN/time] {self._phase_end - time.time():.1f}s left",
                    throttle_duration_sec=1.0,
                )
            if done:
                self._advance()
            else:
                self._pub(-ROT_SPEED, ROT_SPEED)   # CCW: left=−, right=+


def main(args=None):
    rclpy.init(args=args)
    node = RectangleMapping()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
