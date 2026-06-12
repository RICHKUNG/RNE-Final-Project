import math
import time
from geometry_msgs.msg import PointStamped
from trajectory_msgs.msg import JointTrajectoryPoint
from std_msgs.msg import Bool

# Joint index reference
# 0: base rotation  0–180°
# 1: shoulder       0–120°
# 2: elbow          0–150°
# 3: wrist          50–180°
# 4: gripper        10–70°  (10=closed, 70=open)

def _deg(*angles):
    return [math.radians(a) for a in angles]


RESET_POS  = _deg(90, 30, 160, 180, 10)

# Stow: fold the claw out of the camera view — same pose get_bear_node uses.
# 3 joints only (base 180°, shoulder 0°, elbow 90°); arm_writer leaves the rest.
STOW_POS = [math.pi, 0.0, math.pi / 2]

# Safe pre-auto-arm pose (3 joints): raise the claw above the bear before
# publishing the target point, so auto_arm descends from above instead of
# sweeping forward from STOW_POS through the bear.
SAFE_PRE_ARM_POSE = _deg(0.0, 141.0, 90.0)

# Intermediate raise: arm pointing upward with gripper open.
# Reached BEFORE extending toward the bear so the arm descends from above
# rather than sweeping horizontally through the bear. TUNE if needed.
RAISE_POS = _deg(90, 10, 150, 180, 70)      # shoulder up, elbow still folded, gripper open

# Pre-grab: arm extended forward/down toward bear — TUNE THESE VALUES
PRE_GRAB_POS = _deg(90, 70, 100, 120, 70)   # gripper open

GRIPPER_OPEN   = math.radians(70)
GRIPPER_CLOSED = math.radians(10)


class ArmDriver:
    def __init__(self, node):
        self._node = node
        self._pub = node.create_publisher(JointTrajectoryPoint, "robot_arm", 10)
        # Trigger for the calibrated auto_arm_human IK grab when auto_arm already
        # has a valid /yolo/target_marker cached.
        self._auto_grab_pub = node.create_publisher(Bool, "/arm_auto_grab", 10)
        # /clicked_point sets a fresh target marker in RosCommunicator and
        # triggers the same auto_arm grab path through main2.on_clicked_point_grab.
        self._clicked_point_pub = node.create_publisher(PointStamped, "/clicked_point", 10)
        self._joint_pos = list(RESET_POS)

    def _publish_positions(self, positions):
        msg = JointTrajectoryPoint()
        msg.positions = list(positions)
        msg.velocities = [0.0] * len(msg.positions)
        self._pub.publish(msg)

    def _publish(self):
        self._publish_positions(self._joint_pos)

    def reset(self):
        self._joint_pos = list(RESET_POS)
        self._publish()

    def stow(self):
        self._publish_positions(STOW_POS)

    def safe_pre_arm_pose(self, angles_deg=None):
        positions = (
            _deg(*angles_deg)
            if angles_deg is not None
            else list(SAFE_PRE_ARM_POSE)
        )
        self._publish_positions(positions)

    def subscriber_count(self):
        return self._pub.get_subscription_count()

    def open_gripper(self):
        self._joint_pos[-1] = GRIPPER_OPEN
        self._publish()

    def close_gripper(self):
        self._joint_pos[-1] = GRIPPER_CLOSED
        self._publish()

    def set_angles_deg(self, *angles_deg):
        self._joint_pos = _deg(*angles_deg)
        self._publish()

    def auto_grab(self):
        """Trigger the calibrated auto_arm_human IK grab on the latest bear.

        Publishes Bool True to /arm_auto_grab; arm_controller_2D (the AutoArmMode /
        robot_control node) transforms /yolo/target_marker into the arm base frame,
        runs 2D IK and executes the full open→approach→close→retract sequence in a
        background thread (~several seconds — caller must wait, see bear.grab_wait_seconds).

        Use auto_grab_marker() when the caller must publish a cached target point
        immediately before triggering the grab.
        """
        self._auto_grab_pub.publish(Bool(data=True))

    def auto_grab_marker(self, marker, z_offset=0.0):
        """Publish a cached target marker as /clicked_point, which triggers auto_arm.

        z_offset (m) lowers the published target so the gripper descends onto the
        bear from above (top-down grab). Applied here in the marker frame (map),
        which equals lowering it in arm_ik_base as long as their z-axes are
        parallel (true for an upright arm on flat ground). arm_controller_2D's own
        offset is 0.0, so this is the single source of the grab z-offset.
        """
        pt = PointStamped()
        pt.header.frame_id = marker.header.frame_id
        pt.header.stamp = self._node.get_clock().now().to_msg()
        pt.point.x = marker.pose.position.x
        pt.point.y = marker.pose.position.y
        pt.point.z = marker.pose.position.z - z_offset
        self._clicked_point_pub.publish(pt)

    def grab_sequence(self, x_target=0.15, z_target=0.05):
        """
        Fixed grab sequence (Level 2 — no IK).
        x_target / z_target are accepted for interface compatibility but ignored.
        Tune RAISE_POS / PRE_GRAB_POS angles and sleep durations to match the
        physical setup.

        Approach order: RAISE (arm up, gripper open) → PRE_GRAB (descend onto
        bear) → close → RESET (lift away).  Going via RAISE avoids the arm
        sweeping horizontally through the bear during extension.
        """
        # 1. raise arm upward, open gripper — safe pre-position
        self._joint_pos = list(RAISE_POS)
        self._publish()
        time.sleep(1.0)

        # 2. descend / extend toward bear
        self._joint_pos = list(PRE_GRAB_POS)
        self._publish()
        time.sleep(1.0)

        # 3. close gripper
        self._joint_pos[-1] = GRIPPER_CLOSED
        self._publish()
        time.sleep(1.0)

        # 4. lift back toward reset while holding
        self._joint_pos = list(RESET_POS)
        self._joint_pos[-1] = GRIPPER_CLOSED
        self._publish()
        time.sleep(1.5)
