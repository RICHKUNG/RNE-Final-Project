"""
build_map session node.

Usage (inside pros_car container, after colcon build):
    ros2 run rne_final_pkg build_map

Requires:
  - /scan (LaserScan) — from SLAM launch
  - /yolo/bridge_info (Float32MultiArray) — optional, from yolo container

On start it records the initial map pose and writes it to goals.yaml as 'origin'
so that final_mission.py knows where to return for the bear drop-off.
"""

import os
import time
from enum import Enum, auto

import rclpy
import rclpy.duration
import rclpy.time
import tf2_ros
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray

from rne_final_pkg.bridge_avoid import BridgeAvoider
from rne_final_pkg.wall_follow_mapping import WallFollower

# goals.yaml path in the source tree (writable, not the installed share dir)
_GOALS_YAML = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "goals.yaml"
)


class S(Enum):
    MAPPING = auto()
    STOPPED = auto()


class MappingManager(Node):
    MAPPING_DURATION  = 90.0   # seconds before auto-stop
    ORIGIN_SAMPLE_SEC = 5.0    # seconds after start to sample initial pose

    def __init__(self):
        super().__init__("mapping_manager")

        self._front_pub = self.create_publisher(
            Float32MultiArray, "/car_C_front_wheel", 1
        )
        self._rear_pub = self.create_publisher(
            Float32MultiArray, "/car_C_rear_wheel", 1
        )

        self._scan        = None
        self._bridge_info = None
        self._origin_saved = False

        self.create_subscription(LaserScan,          "/scan",             self._scan_cb,   1)
        self.create_subscription(Float32MultiArray,  "/yolo/bridge_info", self._bridge_cb, 10)

        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self._wall   = WallFollower()
        self._bridge = BridgeAvoider()

        self._state      = S.MAPPING
        self._start_time = time.time()

        # One-shot timer to record origin pose after SLAM stabilises
        self.create_timer(self.ORIGIN_SAMPLE_SEC, self._record_origin)

        self.create_timer(0.1, self._tick)
        self.get_logger().info(
            f"MappingManager ready — will auto-stop after {self.MAPPING_DURATION:.0f}s. "
            "Start SLAM then wait for /scan."
        )

    # ------------------------------------------------------------------ origin recording

    def _record_origin(self):
        """Try to get map→base_link TF and write origin to goals.yaml."""
        if self._origin_saved:
            return
        try:
            tf = self._tf_buffer.lookup_transform(
                "map", "base_link",
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            self._save_origin(x, y)
            self.get_logger().info(
                f"Origin recorded: ({x:.3f}, {y:.3f})  — saved to goals.yaml as 'origin'"
            )
            self._origin_saved = True
        except Exception as e:
            self.get_logger().warn(
                f"Could not get map→base_link TF yet ({e}). "
                "Will retry next timer tick."
            )

    def _save_origin(self, x, y):
        # Write to installed share (what running nodes read) and source tree (persistence)
        installed = os.path.join(
            get_package_share_directory("rne_final_pkg"), "config", "goals.yaml"
        )
        src = "/workspaces/src/rne_final_project/rne_final_pkg/config/goals.yaml"
        for path in (installed, src):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                data["origin"] = [round(x, 3), round(y, 3)]
                with open(path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False)
                self.get_logger().info(f"goals.yaml updated at {path}")
            except Exception as e:
                self.get_logger().warn(f"Could not write to {path}: {e}")

    # ------------------------------------------------------------------ callbacks

    def _scan_cb(self, msg):
        self._scan = msg

    def _bridge_cb(self, msg):
        self._bridge_info = list(msg.data)

    # ------------------------------------------------------------------ wheel control

    def _publish(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self._front_pub.publish(msg)
        self._rear_pub.publish(msg)

    def _stop(self):
        self._publish(0.0, 0.0)

    # ------------------------------------------------------------------ main loop

    def _tick(self):
        if self._state == S.STOPPED:
            return

        elapsed = time.time() - self._start_time

        if elapsed >= self.MAPPING_DURATION:
            self._stop()
            self._state = S.STOPPED
            self.get_logger().info(
                f"Mapping finished ({self.MAPPING_DURATION:.0f}s). "
                "Run store_map.sh to save the map, then switch to localization."
            )
            return

        if self._scan is None:
            self.get_logger().warn(
                "Waiting for /scan — is slam_unity.sh running?",
                throttle_duration_sec=5.0,
            )
            return

        remaining = self.MAPPING_DURATION - elapsed

        # Bridge avoidance has priority over wall following
        bridge_cmd = self._bridge.compute(self._bridge_info)
        if bridge_cmd is not None:
            v = bridge_cmd[0]
            self._publish(v[0], v[1])
            self.get_logger().info(
                f"[BRIDGE_AVOID] l={v[0]:.0f} r={v[1]:.0f}  "
                f"area={self._bridge_info[2]:.3f}  {remaining:.0f}s left",
                throttle_duration_sec=1.0,
            )
            return

        wall_cmd = self._wall.compute(self._scan)
        if wall_cmd is None:
            self._stop()
            return

        v, _, dbg = wall_cmd
        self._publish(v[0], v[1])
        self.get_logger().info(
            f"[WALL] {dbg}  →  l={v[0]:.0f} r={v[1]:.0f}  {remaining:.0f}s left",
            throttle_duration_sec=1.0,
        )


def main(args=None):
    rclpy.init(args=args)
    node = MappingManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node._stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
