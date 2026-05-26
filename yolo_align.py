#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class YoloAlign(Node):
    def __init__(self):
        super().__init__("yolo_align")

        self.front_pub = self.create_publisher(
            Float32MultiArray,
            "/car_C_front_wheel",
            10
        )

        self.rear_pub = self.create_publisher(
            Float32MultiArray,
            "/car_C_rear_wheel",
            10
        )

        self.target_sub = self.create_subscription(
            Float32MultiArray,
            "/yolo/target_info",
            self.target_callback,
            10
        )

        self.threshold = 50.0
        self.turn_speed = 300.0
        self.forward_speed = 200.0
        self.stop_distance = 0.60

        self.get_logger().info("YOLO align node started.")

    def forward(self):
        self.publish_wheel(self.forward_speed, self.forward_speed)

    def publish_wheel(self, left_value, right_value):
        msg = Float32MultiArray()
        msg.data = [float(left_value), float(right_value)]

        self.front_pub.publish(msg)
        self.rear_pub.publish(msg)

    def stop(self):
        self.publish_wheel(0.0, 0.0)

    def turn_left(self):
        self.publish_wheel(-self.turn_speed, self.turn_speed)

    def turn_right(self):
        self.publish_wheel(self.turn_speed, -self.turn_speed)

    def target_callback(self, msg):
        if len(msg.data) < 3:
            self.stop()
            return

        found = msg.data[0]
        distance = msg.data[1]
        delta_x = msg.data[2]

        self.get_logger().info(
            f"found={found:.1f}, distance={distance:.2f}, delta_x={delta_x:.1f}"
        )

        if found < 0.5:
            self.stop()
            return

        if delta_x > self.threshold:
            # 目標在畫面右邊，車子右轉
            self.turn_right()

        elif delta_x < -self.threshold:
            # 目標在畫面左邊，車子左轉
            self.turn_left()

        else:
            # 已經對準，開始靠近；距離小於 stop_distance 就停
            # if distance > self.stop_distance:
            #     self.forward()
            # else:
            self.stop()


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
