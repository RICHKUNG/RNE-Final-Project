#!/usr/bin/env python3
"""
Standalone YOLO-align node for manual testing (Test 3 / Test 6).
Subscribes /yolo/target_info and drives wheels to center + approach the target.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class YoloAlign(Node):
    def __init__(self):
        super().__init__("yolo_align")

        self._front_pub = self.create_publisher(Float32MultiArray, "/car_C_front_wheel", 10)
        self._rear_pub  = self.create_publisher(Float32MultiArray, "/car_C_rear_wheel",  10)

        self.create_subscription(Float32MultiArray, "/yolo/target_info",
                                 self._target_cb, 10)

        self.align_threshold  = 50.0    # px — within this → stop rotating
        self.turn_speed       = 300.0   # wheel rad/s
        self.forward_speed    = 200.0
        self.stop_distance    = 0.60    # m — stop approaching when closer than this

        self.get_logger().info("yolo_align node started.")

    # ------------------------------------------------------------------

    def _publish(self, left, right):
        msg = Float32MultiArray()
        msg.data = [float(left), float(right)]
        self._front_pub.publish(msg)
        self._rear_pub.publish(msg)

    def stop(self):      self._publish(0.0, 0.0)
    def forward(self):   self._publish(self.forward_speed,  self.forward_speed)
    def turn_right(self): self._publish( self.turn_speed, -self.turn_speed)
    def turn_left(self):  self._publish(-self.turn_speed,  self.turn_speed)

    # ------------------------------------------------------------------

    def _target_cb(self, msg):
        if len(msg.data) < 3:
            self.stop()
            return

        found    = msg.data[0]
        distance = msg.data[1]
        delta_x  = msg.data[2]

        self.get_logger().info(
            f"found={found:.0f}  dist={distance:.2f}m  dx={delta_x:.0f}px"
        )

        if found < 0.5:
            self.stop()
            return

        if delta_x > self.align_threshold:
            self.turn_right()
        elif delta_x < -self.align_threshold:
            self.turn_left()
        else:
            # aligned — move forward unless already close enough
            if 0 < distance < self.stop_distance:
                self.stop()
            else:
                self.forward()


def main(args=None):
    rclpy.init(args=args)
    node = YoloAlign()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
