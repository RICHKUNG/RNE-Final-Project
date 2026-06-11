from std_msgs.msg import Float32MultiArray

_RATIO = 50
_V = 6.0 * _RATIO
_V_SLOW = 3.0 * _RATIO
_R = 6.0 * _RATIO
_R_SLOW = 5.0 * _RATIO

# [rear_left, rear_right, front_left, front_right]
ACTION_MAPPINGS = {
    "FORWARD":                      [_V,      _V,      _V,      _V],
    "FORWARD_SLOW":                 [_V_SLOW, _V_SLOW, _V_SLOW, _V_SLOW],
    "BACKWARD":                     [-_V,     -_V,     -_V,     -_V],
    "BACKWARD_SLOW":                [-_V_SLOW,-_V_SLOW,-_V_SLOW,-_V_SLOW],
    "CLOCKWISE_ROTATION":           [_R,  -_R,  _R,  -_R],
    "CLOCKWISE_ROTATION_SLOW":      [_R_SLOW, -_R_SLOW, _R_SLOW, -_R_SLOW],
    "COUNTERCLOCKWISE_ROTATION":    [-_R,  _R,  -_R,  _R],
    "COUNTERCLOCKWISE_ROTATION_SLOW":[-_R_SLOW, _R_SLOW, -_R_SLOW, _R_SLOW],
    "STOP":                         [0.0, 0.0, 0.0, 0.0],
}


class CarDriver:
    def __init__(self, node):
        self._node = node
        self._pub_rear = node.create_publisher(Float32MultiArray, "car_C_rear_wheel", 1)
        self._pub_front = node.create_publisher(Float32MultiArray, "car_C_front_wheel", 1)

    def publish(self, action: str):
        vels = ACTION_MAPPINGS.get(action)
        if vels is None:
            self._node.get_logger().warn(f"Unknown car action: {action}")
            return
        rear_msg = Float32MultiArray()
        front_msg = Float32MultiArray()
        rear_msg.data = [float(vels[0]), float(vels[1])]
        front_msg.data = [float(vels[2]), float(vels[3])]
        self._pub_rear.publish(rear_msg)
        self._pub_front.publish(front_msg)

    def publish_velocities(self, left: float, right: float):
        """Differential drive with explicit speeds: left value drives both
        left wheels, right value both right wheels.  (left=-s, right=s) = CCW."""
        rear_msg = Float32MultiArray()
        front_msg = Float32MultiArray()
        rear_msg.data = [float(left), float(right)]
        front_msg.data = [float(left), float(right)]
        self._pub_rear.publish(rear_msg)
        self._pub_front.publish(front_msg)

    def stop(self):
        self.publish("STOP")
