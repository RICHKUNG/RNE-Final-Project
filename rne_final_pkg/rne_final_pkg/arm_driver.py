import math
import time
from trajectory_msgs.msg import JointTrajectoryPoint

# Joint index reference
# 0: base rotation  0–180°
# 1: shoulder       0–120°
# 2: elbow          0–150°
# 3: wrist          50–180°
# 4: gripper        10–70°  (10=closed, 70=open)

def _deg(*angles):
    return [math.radians(a) for a in angles]


RESET_POS  = _deg(90, 30, 160, 180, 10)

# Pre-grab: arm extended forward and slightly down — TUNE THESE VALUES
PRE_GRAB_POS = _deg(90, 70, 100, 120, 70)   # gripper open

GRIPPER_OPEN   = math.radians(70)
GRIPPER_CLOSED = math.radians(10)


class ArmDriver:
    def __init__(self, node):
        self._node = node
        self._pub = node.create_publisher(JointTrajectoryPoint, "robot_arm", 10)
        self._joint_pos = list(RESET_POS)

    def _publish(self):
        msg = JointTrajectoryPoint()
        msg.positions = self._joint_pos
        msg.velocities = [0.0] * len(self._joint_pos)
        self._pub.publish(msg)

    def reset(self):
        self._joint_pos = list(RESET_POS)
        self._publish()

    def open_gripper(self):
        self._joint_pos[-1] = GRIPPER_OPEN
        self._publish()

    def close_gripper(self):
        self._joint_pos[-1] = GRIPPER_CLOSED
        self._publish()

    def set_angles_deg(self, *angles_deg):
        self._joint_pos = _deg(*angles_deg)
        self._publish()

    def grab_sequence(self, x_target=0.15, z_target=0.05):
        """
        Fixed grab sequence (Level 2 — no IK).
        x_target / z_target are accepted for interface compatibility with Level 3
        but are ignored here; use hardcoded PRE_GRAB_POS instead.
        Tune PRE_GRAB_POS angles to match the physical grab position.
        """
        # 1. open gripper and extend arm
        self._joint_pos = list(PRE_GRAB_POS)
        self._joint_pos[-1] = GRIPPER_OPEN
        self._publish()
        time.sleep(1.5)

        # 2. close gripper
        self._joint_pos[-1] = GRIPPER_CLOSED
        self._publish()
        time.sleep(1.0)

        # 3. lift back toward reset
        self._joint_pos = list(RESET_POS)
        self._joint_pos[-1] = GRIPPER_CLOSED   # keep holding
        self._publish()
        time.sleep(1.5)
