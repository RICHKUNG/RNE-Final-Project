from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker


class YoloClient:
    def __init__(self, node):
        self._info = None
        self._marker = None
        self._bridge_info = None

        node.create_subscription(Float32MultiArray, "/yolo/target_info", self._info_cb, 10)
        node.create_subscription(Marker, "/yolo/target_marker", self._marker_cb, 10)
        node.create_subscription(Float32MultiArray, "/yolo/bridge_info", self._bridge_cb, 10)

    def _info_cb(self, msg):
        self._info = list(msg.data)

    def _marker_cb(self, msg):
        self._marker = msg

    def _bridge_cb(self, msg):
        self._bridge_info = list(msg.data)

    def is_visible(self):
        return self._info is not None and len(self._info) >= 1 and self._info[0] == 1.0

    def distance(self):
        return self._info[1] if self._info and len(self._info) >= 2 else float("inf")

    def delta_x(self):
        return self._info[2] if self._info and len(self._info) >= 3 else 0.0

    def marker(self):
        return self._marker

    def bridge_visible(self):
        return (
            self._bridge_info is not None
            and len(self._bridge_info) >= 1
            and self._bridge_info[0] == 1.0
        )

    def bridge_delta_x(self):
        return self._bridge_info[1] if self._bridge_info and len(self._bridge_info) >= 2 else 0.0

    def bridge_area_ratio(self):
        return self._bridge_info[2] if self._bridge_info and len(self._bridge_info) >= 3 else 0.0
