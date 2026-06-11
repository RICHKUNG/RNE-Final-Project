import math
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path


class NavClient:
    def __init__(self, node):
        self._node = node
        self._goal_pub = node.create_publisher(PoseStamped, "/goal_pose", 10)
        node.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._pose_cb, 10)
        node.create_subscription(Path, "/plan", self._plan_cb, 10)

        self.position = None
        self.yaw = 0.0
        self.plan_poses = []
        self.has_plan = False
        self._plan_idx = 0
        self._current_goal = None

    def _pose_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = [p.x, p.y]
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny, cosy)

    def _plan_cb(self, msg):
        self.plan_poses = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        self.has_plan = len(self.plan_poses) > 0
        self._plan_idx = 0

    def send_goal(self, x, y, yaw=0.0):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.orientation.w = math.cos(yaw / 2.0)
        self._goal_pub.publish(msg)
        self._current_goal = [float(x), float(y)]
        self.has_plan = False
        self._plan_idx = 0

    def get_next_waypoint(self, min_dist=0.5):
        if not self.plan_poses or self.position is None:
            return None
        while self._plan_idx < len(self.plan_poses) - 1:
            tx, ty = self.plan_poses[self._plan_idx]
            d = math.hypot(self.position[0] - tx, self.position[1] - ty)
            if d < min_dist:
                self._plan_idx += 1
            else:
                return (tx, ty)
        return None

    def distance_to_goal(self):
        if self.position is None or self._current_goal is None:
            return float("inf")
        return math.hypot(
            self.position[0] - self._current_goal[0],
            self.position[1] - self._current_goal[1],
        )

    def arrived(self, threshold=0.5):
        return self.distance_to_goal() < threshold
