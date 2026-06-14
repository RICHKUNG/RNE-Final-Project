"""Experimental scripted mission — Task 3 (door) first, then ramp bear retrieval.

No Nav2.  Localization-assisted scripted routing only:
  * pose feedback: TF map -> base_footprint (preferred), /amcl_pose fallback
  * route progress and turning use map-frame x, y, yaw — never bare sleeps
  * motion/phase timing uses time.monotonic()

Perception:
  * /yolo/knob_info   — knob visual servo (Task 3)
  * /yolo/bear_info   — bear classification + grasp
  * /yolo/bridge_info — used as ramp info (legacy topic name; driven by
    ramp_yolo11n.pt segmentation, class 'ramp')

Run:  ros2 run rne_final_pkg scripted_final_mission
"""

import math
import os
import time
from collections import deque
from enum import Enum, auto

import rclpy
import rclpy.time
import rclpy.duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform)
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import Marker
from ament_index_python.packages import get_package_share_directory
import yaml

from rne_final_pkg.car_driver import CarDriver
from rne_final_pkg.arm_driver import ArmDriver
from rne_final_pkg.yolo_client import YoloClient

_BASE_FRAME = "base_footprint"


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _norm_ang(a):
    """Normalize angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class S(Enum):
    INIT = auto()

    TASK3_ROUTE_SEGMENT_1 = auto()       # straight to route.turn_point
    TASK3_TURN_LEFT = auto()             # to turn_point.yaw + 90°
    TASK3_ROUTE_SEGMENT_2 = auto()       # straight to route.door_lane_point
    TASK3_TURN_RIGHT_TO_DOOR = auto()    # to door_front.yaw
    TASK3_ROUTE_SEGMENT_3 = auto()       # short straight to route.door_front
    TASK3_KNOB_SERVO = auto()
    TASK3_DOOR_PRESS_COMMIT = auto()
    TASK3_TURN_FORWARD = auto()          # turn left back to yaw≈0 (face forward) after the push
    TASK3_ARM_SAFE_BEFORE_BACKUP = auto()
    BACK_UP_AFTER_TASK3 = auto()         # reverse a fixed distance before ramp search

    MOVE_TO_RAMP_OBSERVE_LONG_SIDE = auto()
    RAMP_SCAN_LONG_SIDE = auto()
    MOVE_TO_LONG_SHORT_CORNER = auto()   # perimeter corner: avoid diagonal over the bridge
    MOVE_TO_RAMP_OBSERVE_SHORT_SIDE = auto()
    RAMP_SCAN_SHORT_SIDE = auto()
    RAMP_ALIGN_BOTTOM = auto()           # servo near edge of ramp mask to the frame bottom
    RAMP_APPROACH = auto()
    RAMP_BEAR_CLASSIFY = auto()
    CLEAR_BLOCKING_BEAR = auto()
    GRASP_RAMP_BEAR = auto()
    RETURN_ORIGIN = auto()

    # Post-mission cleanup patrol (Mr.Kung 2026-06-14): after RETURN_ORIGIN
    # delivers the ramp bear, drive once more toward turn_point; a bear NOT on
    # the ramp gets grabbed and carried back to origin, then DONE. Gated by
    # cfg["patrol"]["enabled"]; see _state_patrol_*.
    PATROL_TO_TURN = auto()
    PATROL_GRAB = auto()
    PATROL_RETURN = auto()

    DONE = auto()
    FAILED = auto()


class ScriptedFinalMission(Node):
    # States allowed past safety.sea_x_limit: the door (and door_exit_point
    # ~x=3.56) sit behind the wall in the +x sea region, so the push and the
    # post-push reverse legitimately operate there. Everywhere else, x past the
    # limit is drift toward the sea and the guard escapes it.
    _SEA_EXEMPT = frozenset({
        S.TASK3_DOOR_PRESS_COMMIT,
        S.TASK3_TURN_FORWARD,
        S.TASK3_ARM_SAFE_BEFORE_BACKUP,
        S.BACK_UP_AFTER_TASK3,
    })

    def __init__(self, node_name="scripted_final_mission"):
        super().__init__(node_name)

        self._load_config()

        self.car = CarDriver(self)
        self.arm = ArmDriver(self)
        self.yolo = YoloClient(self)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_cb, 10
        )

        # Depth-camera intrinsics — used to back-project bear candidates into the
        # map frame so a grab retry can re-lock the SAME (ground) bear by world
        # position instead of the new overall-nearest one. Publisher latches with
        # TRANSIENT_LOCAL, so the QoS must match to receive it.
        self._camera_info = None
        camera_info_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            CameraInfo,
            self.cfg["bear"].get("camera_info_topic", "/camera/depth/camera_info"),
            self._camera_info_cb,
            camera_info_qos,
        )

        # /initialpose — hands-off start: repeat-publish the configured spawn
        # pose until AMCL localizes (same pattern as get_bear_node).
        self._initialpose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self._initialpose_timer = self.create_timer(
            float(self.cfg["init"]["initial_pose_wait_s"]),
            self._maybe_publish_initialpose,
        )

        # Stow the claw at startup — same pose + retry-until-subscribed as
        # get_bear_node (a one-shot publish races arm_writer's subscription).
        self._stow_attempts = 0
        self._stow_timer = self.create_timer(0.5, self._stow_arm_once)

        # pose = (x, y, yaw) in map frame; _pose_mono = monotonic stamp of update
        self.pose = None
        self._pose_src = "none"
        self._pose_mono = 0.0
        self._amcl = None
        self._amcl_mono = 0.0

        self._state = S.INIT
        self._state_t0 = time.monotonic()
        self._phase = 0
        self._phase_t0 = time.monotonic()
        self._phase_entered = False  # one-shot flag for phase entry actions
        self._anchor = None          # per-state scratch (target yaw / start xy …)
        self._clear_turn_target = None  # CLEAR_BLOCKING_BEAR 180° deposit-turn target yaw
        self._fail_reason = None
        self._done_logged = False

        self._ramp_window = deque(maxlen=int(self.cfg["ramp"]["found_window_frames"]))
        self._ramp_last_seq = None   # last counted seg message (see yolo_client.ramp_seq)
        self._ramp_last_counted = False
        self._ramp_last_hit = False
        self._ramp_reacquire_state = None
        self._observe_idx = 0        # index into the current side's observe chain
                                     # (survives MOVE→SCAN→MOVE; reset only on side switch)
        self._bridge_horizontal = False  # set if the ramp is seen on the outbound
                                     # leg (SEGMENT_1): bridge lies horizontal, so the
                                     # post-door route skips the long side and goes
                                     # straight to the short-side observe chain.
                                     # Persists across _goto (not reset there).
        self._approach_done = False  # ramp approach reached approach_done_area at least once
        self._charge_anchor = None   # RAMP_APPROACH charge-mode start xy (pose-measured distance)
        self._ramp_charge_point = None  # 上衝點: ramp-foot xy where the charge began;
                                        # recorded once, persists across _goto for RETURN alignment
        self._ramp_grasp_backup_done = False    # one-time GRASP_RAMP_BEAR entry backup fired
        self._ramp_grasp_backup_anchor = None   # entry-backup reverse start xy (pose-measured)
        self._ramp_grasp_backup_t0 = None       # entry-backup decision/reverse timer
        self._ramp_aligned = False   # RAMP_ALIGN_BOTTOM cleared its launch gate at least once;
                                     # gates triage classify (pre-align) vs grasp classify (post-align).
                                     # Persists across _goto (not reset there).
        self._classify_entries = 0
        self._classify_samples = []
        self._grab_retries = 0
        self._auto_grab_triggered = False
        self._auto_grab_t0 = None
        self._auto_grab_marker = None
        self._bear_grab_snapshot = None
        self._bear_commit_t0 = None
        self._direct_grab_phase = 0   # sub-step of the direct close-gripper grab
        self._direct_grab_t0 = 0.0
        # Bear-servo stop-and-look sub-FSM (deadtime-immune alignment under the
        # ~0.3 s camera latency). SETTLE = stop & wait for the image to catch up
        # then read; TURN/FORWARD/SEARCH = timed open-loop bursts with no vision
        # read. _servo_burst_s / _servo_burst_cmd parametrise the active burst.
        self._servo_sub = "SETTLE"
        self._servo_sub_t0 = time.monotonic()
        self._servo_burst_s = 0.0
        self._servo_burst_cmd = 0.0
        self._bear_last_settle_depth = None
        self._bear_depth_jump_rejected = False
        self._clear_lost_state = None
        self._grasp_verify_depth0 = None
        self._grasp_verify_seq0 = None
        self._grasp_verify_samples = []
        self._grasp_verify_last_seq = None
        self._grasp_verify_probe_m = 0.0
        self._grasp_verify_lock = None
        self._return_exit_mode = None
        self._return_reverse_wp = None
        self._return_reverse_start_dist = None
        self._return_reverse_wp_label = None
        self._return_exit_stuck_anchor = None  # (x, y, monotonic) watchdog for bridge-exit progress
        self._press_attempts = 0
        self._knob_invalid_t0 = None
        self._servo_settle_t0 = None     # knob servo: in-window settle timer
        self._press_commit_depth = None  # depth at servo commit → dynamic forward leg

        # stuck detection while driving forward (route legs)
        self._stuck_anchor = None    # (x, y, monotonic) of last confirmed movement
        self._recover_until = 0.0    # while monotonic < this, back up instead of driving

        # sea guard: True while actively escaping the +x sea edge (see _sea_guard)
        self._sea_escaping = False

        # post-mission patrol: lost-bear abandon timer (see _state_patrol_grab)
        self._patrol_grab_lost_t0 = None

        self.create_timer(0.1, self._tick)   # 10 Hz control loop
        self.get_logger().info(f"{self.get_name()} ready — waiting for pose + YOLO topics (INIT)")

    # ------------------------------------------------------------------
    # Config / pose
    # ------------------------------------------------------------------

    def _load_config(self):
        share = get_package_share_directory("rne_final_pkg")
        path = os.path.join(share, "config", "scripted_mission.yaml")
        with open(path) as f:
            self.cfg = yaml.safe_load(f)
        self.get_logger().info(f"[CONFIG] loaded {path}")

        # Resolve debug.start_state at startup so a typo fails immediately,
        # not minutes into a run.
        name = (self.cfg.get("debug") or {}).get("start_state") or ""
        if name:
            try:
                self._start_state = S[name]
            except KeyError:
                valid = ", ".join(s.name for s in S)
                raise ValueError(
                    f"debug.start_state '{name}' is not a valid state. Valid: {valid}"
                )
        else:
            self._start_state = S.TASK3_ROUTE_SEGMENT_1

    def _maybe_publish_initialpose(self):
        if self.pose is not None:
            self.get_logger().info("[INIT_POSE] localized — stopping auto initial pose")
            self._initialpose_timer.cancel()
            return

        x = float(self.cfg["init"]["initial_pose_x"])
        y = float(self.cfg["init"]["initial_pose_y"])
        yaw = float(self.cfg["init"]["initial_pose_yaw"])

        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = x
        msg.pose.pose.position.y = y
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.covariance[0] = 0.25
        msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.068

        self._initialpose_pub.publish(msg)
        self.get_logger().info(
            f"[INIT_POSE] published /initialpose x={x} y={y} yaw={yaw:.4f} "
            f"(repeats every {self.cfg['init']['initial_pose_wait_s']}s until localized)"
        )

    def _stow_arm_once(self):
        self._stow_attempts += 1
        if self.arm.subscriber_count() == 0:
            if self._stow_attempts >= 30:
                self.get_logger().warn("[STOW] arm_writer never subscribed — skipping stow")
                self._stow_timer.cancel()
            return
        self.arm.stow_closed()
        self.get_logger().info(
            f"[STOW] stow + closed claw published (attempt {self._stow_attempts})"
        )
        self._stow_timer.cancel()

    def _amcl_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._amcl = (p.x, p.y, _yaw_from_quat(q))
        self._amcl_mono = time.monotonic()

    def _camera_info_cb(self, msg):
        self._camera_info = msg

    def _update_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform("map", _BASE_FRAME, rclpy.time.Time())
            t = tf.transform.translation
            self.pose = (t.x, t.y, _yaw_from_quat(tf.transform.rotation))
            self._pose_src = "tf"
            self._pose_mono = time.monotonic()
            return
        except Exception:
            pass
        if self._amcl is not None:
            self.pose = self._amcl
            self._pose_src = "amcl"
            self._pose_mono = self._amcl_mono

    def _pose_ok(self):
        return (
            self.pose is not None
            and (time.monotonic() - self._pose_mono) < self.cfg["init"]["pose_stale_s"]
        )

    # ------------------------------------------------------------------
    # State machine core
    # ------------------------------------------------------------------

    def _goto(self, state: S):
        self.get_logger().info(f"State: {self._state.name} -> {state.name}")
        self._state = state
        self._state_t0 = time.monotonic()
        self._phase = 0
        self._phase_t0 = time.monotonic()
        self._phase_entered = False
        self._anchor = None
        self._ramp_window = deque(maxlen=int(self.cfg["ramp"]["found_window_frames"]))
        self._ramp_last_seq = None
        self._ramp_last_counted = False
        self._ramp_last_hit = False
        self._classify_samples = []
        self._knob_invalid_t0 = None
        self._servo_settle_t0 = None
        self._stuck_anchor = None
        self._return_exit_mode = None
        self._grasp_verify_depth0 = None
        self._grasp_verify_seq0 = None
        self._grasp_verify_samples = []
        self._grasp_verify_last_seq = None
        self._grasp_verify_probe_m = 0.0
        self._grasp_verify_lock = None
        self._auto_grab_triggered = False
        self._auto_grab_t0 = None
        self._auto_grab_marker = None
        self._bear_grab_snapshot = None
        self._bear_commit_t0 = None
        self._bear_last_settle_depth = None
        self._bear_depth_jump_rejected = False
        self._clear_lost_state = None
        self._return_reverse_wp = None
        self._return_reverse_start_dist = None
        self._return_reverse_wp_label = None
        self._return_exit_stuck_anchor = None
        self._charge_anchor = None
        # Re-arm the one-time GRASP_RAMP_BEAR entry backup on every fresh state
        # entry (i.e. a fresh charge→grasp). Retries use _phase_goto, not _goto,
        # so they don't re-trigger it.
        self._ramp_grasp_backup_done = False
        self._ramp_grasp_backup_anchor = None
        self._ramp_grasp_backup_t0 = None

    def _phase_goto(self, phase: int):
        self.get_logger().info(f"[{self._state.name}] phase {self._phase} -> {phase}")
        self._phase = phase
        self._phase_t0 = time.monotonic()
        self._phase_entered = False
        self._anchor = self.pose[:2] if self.pose else None
        self._auto_grab_triggered = False
        self._auto_grab_t0 = None
        self._auto_grab_marker = None
        self._bear_commit_t0 = None
        self._bear_last_settle_depth = None
        self._bear_depth_jump_rejected = False

    def _fail(self, reason: str):
        self._fail_reason = reason
        self.get_logger().error(f"MISSION FAILED: {reason}")
        self.car.stop()
        self._goto(S.FAILED)

    def _tick(self):
        self._update_pose()
        if self._sea_guard():
            return
        s = self._state

        if s == S.INIT:
            self._state_init()
        elif s == S.TASK3_ROUTE_SEGMENT_1:
            # Outbound leg doubles as a ramp pre-check (watch_ramp): seeing the
            # ramp here means the bridge is horizontal → skip the long side later.
            self._state_route(self.cfg["route"]["turn_point"], S.TASK3_TURN_LEFT, watch_ramp=True)
        elif s == S.TASK3_TURN_LEFT:
            # left 90° from the measured heading at turn_point — not a hardcoded yaw
            target = _norm_ang(self.cfg["route"]["turn_point"]["yaw"] + math.pi / 2.0)
            self._state_turn_to(target, S.TASK3_ROUTE_SEGMENT_2)
        elif s == S.TASK3_ROUTE_SEGMENT_2:
            self._state_route(self.cfg["route"]["door_lane_point"], S.TASK3_TURN_RIGHT_TO_DOOR)
        elif s == S.TASK3_TURN_RIGHT_TO_DOOR:
            self._state_turn_to(self.cfg["route"]["door_front"]["yaw"], S.TASK3_ROUTE_SEGMENT_3)
        elif s == S.TASK3_ROUTE_SEGMENT_3:
            self._state_route(self.cfg["route"]["door_front"], S.TASK3_KNOB_SERVO)
        elif s == S.TASK3_KNOB_SERVO:
            self._state_knob_servo(S.TASK3_DOOR_PRESS_COMMIT)
        elif s == S.TASK3_DOOR_PRESS_COMMIT:
            self._state_door_press(S.TASK3_TURN_FORWARD)
        elif s == S.TASK3_TURN_FORWARD:
            # The arc push leaves the car yawed off; rotate back to face forward
            # (map +x, yaw=0) before reversing for the ramp search. Watchdog so a
            # car still jammed on the door doesn't spin here forever.
            self._state_turn_to(
                0.0, S.TASK3_ARM_SAFE_BEFORE_BACKUP,
                timeout_s=self.cfg["post_task3"]["turn_forward_timeout_s"],
            )
        elif s == S.TASK3_ARM_SAFE_BEFORE_BACKUP:
            self._state_task3_arm_safe_before_backup(S.BACK_UP_AFTER_TASK3)
        elif s == S.BACK_UP_AFTER_TASK3:
            # Bridge known horizontal from the outbound leg → skip long-side
            # observe and route right-angle (via long_to_short_corner, the outer
            # perimeter) straight to the short-side observe chain.
            after_backup = (
                S.MOVE_TO_LONG_SHORT_CORNER if self._bridge_horizontal
                else S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE
            )
            self._state_back_up(
                self.cfg["post_task3"]["backup_target_x"],
                self.cfg["post_task3"]["backup_speed"],
                after_backup,
                self.cfg["post_task3"].get("backup_max_travel_m"),
            )
        elif s == S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE:
            self._state_move_observe(self.cfg["route"]["long_side_observe"], S.RAMP_SCAN_LONG_SIDE)
        elif s == S.RAMP_SCAN_LONG_SIDE:
            self._state_ramp_scan(self.cfg["route"]["long_side_observe"],
                                  move_state=S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE,
                                  found_next=S.RAMP_BEAR_CLASSIFY,
                                  exhausted_next=S.MOVE_TO_LONG_SHORT_CORNER)
        elif s == S.MOVE_TO_LONG_SHORT_CORNER:
            # Route around the outer perimeter instead of diagonally across the
            # inner corner (which crosses the bridge). _state_route drives straight
            # to the corner; the right-edge→bottom-edge legs fall out of the
            # heading-then-forward control in _drive_to_point.
            self._state_route(self.cfg["route"]["long_to_short_corner"],
                              S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE)
        elif s == S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE:
            self._state_move_observe(self.cfg["route"]["short_side_observe"], S.RAMP_SCAN_SHORT_SIDE)
        elif s == S.RAMP_SCAN_SHORT_SIDE:
            self._state_ramp_scan(self.cfg["route"]["short_side_observe"],
                                  move_state=S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE,
                                  found_next=S.RAMP_BEAR_CLASSIFY,
                                  exhausted_next=None)
        elif s == S.RAMP_ALIGN_BOTTOM:
            self._state_ramp_align_bottom(S.RAMP_APPROACH)
        elif s == S.RAMP_APPROACH:
            self._state_ramp_approach(S.RAMP_BEAR_CLASSIFY)
        elif s == S.RAMP_BEAR_CLASSIFY:
            self._state_bear_classify()
        elif s == S.CLEAR_BLOCKING_BEAR:
            # After clearing a blocking bear the car has moved/pushed, so the old
            # ramp-bottom alignment is stale: re-align to the ramp bottom edge
            # before re-approaching and re-classifying (RAMP_ALIGN_BOTTOM ->
            # RAMP_APPROACH -> RAMP_BEAR_CLASSIFY).
            self._state_clear_blocking_bear(S.RAMP_ALIGN_BOTTOM)
        elif s == S.GRASP_RAMP_BEAR:
            self._state_grasp_ramp_bear(S.RETURN_ORIGIN)
        elif s == S.RETURN_ORIGIN:
            # After delivering the ramp bear, optionally run the post-mission
            # cleanup patrol (drive once more to turn_point and grab a ground
            # bear that is not on the ramp). Disabled → finish as before.
            after_return = (
                S.PATROL_TO_TURN
                if (self.cfg.get("patrol") or {}).get("enabled", False)
                else S.DONE
            )
            self._state_return_origin(after_return)
        elif s == S.PATROL_TO_TURN:
            self._state_patrol_to_turn(S.PATROL_GRAB, S.DONE)
        elif s == S.PATROL_GRAB:
            self._state_patrol_grab(S.PATROL_RETURN, S.DONE)
        elif s == S.PATROL_RETURN:
            self._state_patrol_return(S.DONE)
        elif s == S.DONE:
            self.car.stop()
            if not self._done_logged:
                self.get_logger().info("Mission complete.")
                self._done_logged = True
        elif s == S.FAILED:
            self.car.stop()

    # ------------------------------------------------------------------
    # Motion primitives (tick-based; return True when finished)
    # ------------------------------------------------------------------

    def _rotate_dir(self, direction, speed=None):
        """direction > 0 = CCW/left, < 0 = CW/right (only the sign is used)."""
        s = speed if speed is not None else self.cfg["control"]["turn_speed"]
        if direction > 0:
            self.car.publish_velocities(-s, s)   # CCW / left
        else:
            self.car.publish_velocities(s, -s)   # CW / right

    def _turn_speed_for(self, err_rad):
        """Slow down near the target yaw so the 10 Hz bang-bang loop doesn't
        overshoot through the tolerance window and oscillate.
        Must stay above the Unity car's in-place rotation threshold
        (150 stalls — see turn_slow_speed in scripted_mission.yaml)."""
        c = self.cfg["control"]
        if abs(math.degrees(err_rad)) < c["turn_slowdown_deg"]:
            return c.get("turn_slow_speed", c["turn_speed"])
        return c["turn_speed"]

    def _turn_to_yaw(self, target_yaw) -> bool:
        _, _, yaw = self.pose
        err = _norm_ang(target_yaw - yaw)
        if abs(math.degrees(err)) < self.cfg["control"]["yaw_tolerance_deg"]:
            self.car.stop()
            return True
        speed = self._turn_speed_for(err)
        self.get_logger().info(
            f"[{self._state.name}] yaw={math.degrees(yaw):.0f}°/{self._pose_src}  "
            f"target={math.degrees(target_yaw):.0f}°  err={math.degrees(err):.0f}° "
            f"→ ROTATE@{speed:.0f}"
        )
        self._rotate_dir(err, speed)
        return False

    def _stuck_check(self) -> bool:
        """Call while commanding forward.  True when a recovery backup was
        triggered (caller should skip its forward command this tick)."""
        c = self.cfg["stuck"]
        now = time.monotonic()
        x, y, _ = self.pose
        if self._stuck_anchor is None:
            self._stuck_anchor = (x, y, now)
            return False
        ax, ay, at = self._stuck_anchor
        if math.hypot(x - ax, y - ay) > c["move_threshold_m"]:
            self._stuck_anchor = (x, y, now)
            return False
        if now - at > c["timeout_s"]:
            self.get_logger().warn(
                f"[{self._state.name}] STUCK — no movement for {now - at:.1f}s, "
                f"backing up {c['recover_backward_s']:.1f}s"
            )
            self._recover_until = now + c["recover_backward_s"]
            self._stuck_anchor = None
            return True
        return False

    def _drive_to_point(self, wp, speed=None) -> bool:
        # stuck recovery takes priority over everything
        if time.monotonic() < self._recover_until:
            s = self.cfg["control"]["slow_speed"]
            self.car.publish_velocities(-s, -s)
            return False

        x, y = wp["x"], wp["y"]
        tol = wp.get("tolerance", self.cfg["control"]["xy_tolerance"])
        px, py, yaw = self.pose
        dist = math.hypot(x - px, y - py)
        if dist < tol:
            self.car.stop()
            self._stuck_anchor = None
            return True
        bearing = math.atan2(y - py, x - px)
        err = _norm_ang(bearing - yaw)
        if abs(math.degrees(err)) > self.cfg["control"]["forward_angle_deg"]:
            action = "ROTATE"
            self._stuck_anchor = None   # rotation barely translates — don't count it
            self._rotate_dir(err, self._turn_speed_for(err))
        else:
            action = "FORWARD"
            if self._stuck_check():
                return False
            v = speed if speed is not None else self.cfg["control"]["medium_speed"]
            self.car.publish_velocities(v, v)
        self.get_logger().info(
            f"[{self._state.name}] pose=({px:.2f},{py:.2f},{math.degrees(yaw):.0f}°/{self._pose_src})  "
            f"target=({x:.2f},{y:.2f})  dist={dist:.2f}m  heading_err={math.degrees(err):.0f}° → {action}"
        )
        return False

    def _require_pose(self) -> bool:
        """Stop and complain when localization is lost; True when pose usable."""
        if self._pose_ok():
            return True
        self.car.stop()
        self.get_logger().error(
            f"[{self._state.name}] localization pose lost "
            f"(last={time.monotonic() - self._pose_mono:.1f}s ago, src={self._pose_src}) — holding"
        )
        return False

    def _sea_guard(self) -> bool:
        """Keep the car out of the +x sea. Returns True when the guard is driving
        (caller must skip the normal state this tick).

        A blind reverse is unsafe: depending on yaw it can back the car *further*
        into the sea. Instead, turn to face -x (away from the +x sea) and drive
        forward, which reduces x regardless of the starting heading, until x is
        below (limit − sea_clear_margin). Door states are exempt (see _SEA_EXEMPT).
        """
        cfg = self.cfg.get("safety") or {}
        limit = cfg.get("sea_x_limit")
        if limit is None or self._state in self._SEA_EXEMPT or not self._pose_ok():
            self._sea_escaping = False
            return False

        x, _, _ = self.pose
        margin = cfg.get("sea_clear_margin", 0.15)

        if not self._sea_escaping:
            if x <= limit:
                return False
            self._sea_escaping = True
            self.get_logger().error(
                f"[SEA_GUARD] x={x:.2f} > {limit} (sea) in {self._state.name} "
                f"— turning to face -x and escaping"
            )
        elif x <= limit - margin:
            self._sea_escaping = False
            self.car.stop()
            self.get_logger().warn(
                f"[SEA_GUARD] cleared (x={x:.2f}) — resuming {self._state.name}"
            )
            return True   # settle this tick; the state runs again next tick

        # Face -x (map west, away from the +x sea), then drive forward to cut x.
        if not self._turn_to_yaw(math.pi):
            return True   # still rotating toward -x
        self.car.publish_velocities(
            self.cfg["control"]["slow_speed"], self.cfg["control"]["slow_speed"]
        )
        return True

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def _state_init(self):
        missing = []
        if self.pose is None:
            missing.append("pose (TF map->base_footprint or /amcl_pose)")
        if not self.yolo.knob_topic_alive():
            missing.append("/yolo/knob_info")
        if not self.yolo.bear_topic_alive():
            missing.append("/yolo/bear_info")
        if not self.yolo.ramp_topic_alive():
            missing.append("/yolo/bridge_info (ramp seg)")

        if not missing:
            x, y, yaw = self.pose
            self.get_logger().info(
                f"[INIT] ready  pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}°) src={self._pose_src}"
            )
            if self._start_state is not S.TASK3_ROUTE_SEGMENT_1:
                self.get_logger().warn(
                    f"[INIT] debug.start_state set — jumping to {self._start_state.name} "
                    f"(earlier states skipped; mission context like a held bear is on you)"
                )
            self._goto(self._start_state)
            return

        elapsed = time.monotonic() - self._state_t0
        if elapsed > self.cfg["init"]["wait_timeout_s"]:
            self._fail(f"INIT timeout after {elapsed:.0f}s — missing: {', '.join(missing)}")
            return
        self.get_logger().info(f"[INIT] waiting ({elapsed:.1f}s)  missing: {', '.join(missing)}")

    # ------------------------------------------------------------------
    # Task 3 — route, knob servo, door press
    # ------------------------------------------------------------------

    def _accumulate_ramp_hits(self, area_threshold=None, require_center_overlap=False) -> int:
        """Fold this side's confirmed ramp seg inferences into _ramp_window and
        return the running hit count. Counts distinct seg messages only (seg
        publishes ~1 Hz while this loop ticks at 10 Hz, so a sticky message must
        not accumulate repeat hits). Shared by SEGMENT_1's flag check and the
        ramp scan states; area_threshold defaults to the close-range
        found_area_threshold, overridden looser for the far outbound pre-check."""
        c = self.cfg["ramp"]
        thr = area_threshold if area_threshold is not None else c["found_area_threshold"]
        self._ramp_last_counted = False
        self._ramp_last_hit = False

        seq = self.yolo.ramp_seq()
        if seq <= 0 or seq == self._ramp_last_seq:
            return sum(self._ramp_window)

        self._ramp_last_seq = seq
        age = self.yolo.ramp_age_s()
        fresh = age is not None and age <= c.get("info_stale_s", 1.5)
        center_ok = (
            not require_center_overlap
            or self.yolo.ramp_center_overlap_ratio() >= c.get("center_overlap_threshold", 0.01)
        )
        found = (
            fresh
            and self.yolo.ramp_visible()
            and self.yolo.ramp_area_ratio() >= thr
            and center_ok
        )
        self._ramp_window.append(1 if found else 0)
        self._ramp_last_counted = True
        self._ramp_last_hit = found
        return sum(self._ramp_window)

    def _state_route(self, wp, next_state, watch_ramp=False):
        if not self._require_pose():
            return
        if watch_ramp and not self._bridge_horizontal:
            # Outbound leg: if the ramp is already in view here the bridge lies
            # horizontal — flag it so the post-door route skips the long side.
            # Looser thresholds (outbound_*) since the ramp is far and small here.
            r = self.cfg["ramp"]
            if self._accumulate_ramp_hits(r["outbound_area_threshold"]) >= r["outbound_required_frames"]:
                self._bridge_horizontal = True
                self.get_logger().info(
                    "[SEGMENT_1] ramp seen on the outbound leg → bridge is HORIZONTAL; "
                    "post-door route will skip the long side and go straight to the short side"
                )
        if self._drive_to_point(wp):
            self._goto(next_state)

    def _state_turn_to(self, target_yaw, next_state, timeout_s=None):
        """Turn in place to an absolute map-frame yaw (from route config).

        timeout_s (optional): give up and advance after this long instead of
        spinning forever if the car can't reach the heading (e.g. still jammed
        on the door). None = no watchdog (route turns must complete)."""
        if not self._require_pose():
            return
        if self._anchor is None:
            self._anchor = _norm_ang(target_yaw)
            self.get_logger().info(
                f"[{self._state.name}] target yaw={math.degrees(self._anchor):.0f}°"
            )
        if self._turn_to_yaw(self._anchor):
            self._goto(next_state)
        elif timeout_s is not None and time.monotonic() - self._state_t0 > timeout_s:
            self.get_logger().warn(
                f"[{self._state.name}] turn watchdog ({timeout_s:.0f}s) — "
                "could not reach heading; advancing anyway"
            )
            self.car.stop()
            self._goto(next_state)

    # Knob source indirection — door_test overrides these to fall back to
    # /yolo/target_info when /yolo/knob_info is not being published.
    def _knob_visible(self):
        return self.yolo.knob_visible()

    def _knob_dx(self):
        return self.yolo.knob_delta_x()

    def _knob_depth(self):
        return self.yolo.knob_distance()

    def _state_knob_servo(self, next_state):
        if self._knob_servo_step():
            self._press_attempts = 0
            self._goto(next_state)

    def _knob_servo_step(self) -> bool:
        """Stop-and-look knob servo, True once the door press pose is ready.

        The knob detector has enough latency that continuous turn corrections
        overshoot the centre line.  This mirrors the bear servo pattern: stop,
        settle, sample one fresh dx/depth, then execute one bounded open-loop
        burst before measuring again.
        """
        c = self.cfg["knob_servo"]
        slow = self.cfg["control"]["slow_speed"]
        turn = c.get("align_turn_speed", self.cfg["control"]["turn_slow_speed"])
        elapsed = time.monotonic() - self._state_t0
        if elapsed > c["max_seconds"]:
            self._fail(f"knob servo timeout after {elapsed:.0f}s (knob_visible={self._knob_visible()})")
            return False

        if not self._phase_entered:
            self._phase_entered = True
            self._knob_invalid_t0 = None
            self._servo_settle_t0 = None
            self._servo_enter_burst("SETTLE", 0.0, 0.0)

        if self._servo_burst_active():
            return False

        sub_elapsed = time.monotonic() - self._servo_sub_t0
        self.car.stop()
        if sub_elapsed < c.get("align_settle_s", 0.4):
            return False

        if not self._knob_visible():
            self._knob_invalid_t0 = None
            self._servo_settle_t0 = None
            dur = c.get("align_search_burst_s", 0.3)
            self.get_logger().info(
                f"[KNOB_SERVO] no knob after settle ({elapsed:.1f}s) → SEARCH {dur:.2f}s @{turn:.0f}"
            )
            self._servo_enter_burst("SEARCH", dur, turn)
            return False

        dx = self._knob_dx()
        depth = self._knob_depth()

        if abs(dx) > c["center_threshold_px"]:
            self._knob_invalid_t0 = None
            self._servo_settle_t0 = None
            dur = min(c.get("align_turn_max_s", 0.5),
                      max(c.get("align_turn_min_s", 0.15),
                          abs(dx) * c.get("align_turn_s_per_px", 0.0015)))
            cmd = math.copysign(turn, dx)  # dx>0 = knob right of centre → CW
            self.get_logger().info(
                f"[KNOB_SERVO] settled dx={dx:.0f}px depth={depth:.2f}m → TURN {dur:.2f}s @{cmd:+.0f}"
            )
            self._servo_enter_burst("TURN", dur, cmd)
            return False

        # centered — close in on depth
        if depth <= 0:
            now = time.monotonic()
            self._servo_settle_t0 = None
            if self._knob_invalid_t0 is None:
                self._knob_invalid_t0 = now
            self.car.stop()
            held = now - self._knob_invalid_t0
            self.get_logger().warn(f"[KNOB_SERVO] knob centered but depth invalid ({held:.1f}s)")
            if held > c["invalid_depth_fail_s"]:
                self._fail("knob centered but depth stayed invalid — cannot range the door")
            return False
        self._knob_invalid_t0 = None

        if depth > c["target_depth_m"]:
            self._servo_settle_t0 = None
            dur = c.get("align_forward_s", 0.15)
            self.get_logger().info(
                f"[KNOB_SERVO] settled dx={dx:.0f}px  depth={depth:.2f}m > "
                f"{c['target_depth_m']}m → FWD {dur:.2f}s @{slow:.0f}"
            )
            self._servo_enter_burst("FORWARD", dur, slow)
            return False

        # Too close — the press pose is calibrated at target_depth_m, so back
        # up until depth is inside [target - tol, target] before committing.
        if depth < c["target_depth_m"] - c["depth_tolerance_m"]:
            self._servo_settle_t0 = None
            dur = c.get("align_forward_s", 0.15)
            self.get_logger().info(
                f"[KNOB_SERVO] settled dx={dx:.0f}px  depth={depth:.2f}m < "
                f"{c['target_depth_m'] - c['depth_tolerance_m']:.2f}m → BACK {dur:.2f}s @{slow:.0f}"
            )
            self._servo_enter_burst("FORWARD", dur, -slow)
            return False

        # In the window — stop and settle before committing: the depth message
        # lags the camera and the car coasts after stop, so the first in-window
        # reading can be a few cm stale (occasional press overshoot).
        self.car.stop()
        now = time.monotonic()
        if self._servo_settle_t0 is None:
            self._servo_settle_t0 = now
            self.get_logger().info(
                f"[KNOB_SERVO] in window  dx={dx:.0f}px  depth={depth:.2f}m → settling"
            )
            return False
        if now - self._servo_settle_t0 < c["commit_settle_s"]:
            return False

        self.get_logger().info(
            f"[KNOB_SERVO] aligned  dx={dx:.0f}px  depth={depth:.2f}m (settled) "
            "→ commit (camera goes blind now)"
        )
        self._press_commit_depth = depth
        self.car.stop()
        return True

    def _dist_from_anchor(self):
        if self._anchor is None or self.pose is None:
            return 0.0
        return math.hypot(self.pose[0] - self._anchor[0], self.pose[1] - self._anchor[1])

    def _commit_forward_done(self, target_m) -> bool:
        """Open-loop commit forward: pose distance preferred, phase timeout as
        watchdog (the arm may block the camera and skid can degrade pose)."""
        elapsed = time.monotonic() - self._phase_t0
        if self._pose_ok() and self._dist_from_anchor() >= target_m:
            return True
        if elapsed > self.cfg["door_press"]["phase_timeout_s"]:
            self.get_logger().warn(
                f"[DOOR_PRESS] phase watchdog ({elapsed:.1f}s) — advancing "
                f"(moved={self._dist_from_anchor():.2f}m of {target_m:.2f}m)"
            )
            return True
        return False

    def _state_door_press(self, next_state):
        """Scripted open-loop commit — no knob vision from here on.

        Flow: raise claw in place → forward (servo commit depth −
        press_standoff_m) → swing down to the calibrated hit pose → continue
        through to the low pose (full unlock) → back to the hold pose so the
        handle stays pressed (top/bottom rod lock re-engages if released) →
        arc push toward front-right following the door's swing.
        """
        c = self.cfg["door_press"]
        elapsed = time.monotonic() - self._phase_t0

        if self._phase == 0:       # raise claw to highest, in place
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info(f"[DOOR_PRESS] raise arm {c['arm_raise_deg']}")
                self.arm.set_angles_deg(*c["arm_raise_deg"])
            self.car.stop()
            if elapsed > c["arm_settle_s"]:
                self._phase_goto(1)

        elif self._phase == 1:     # forward from the servo stop to press range
            if not self._phase_entered:
                self._phase_entered = True
                if self._press_commit_depth is not None:
                    # Dynamic leg: actual measured gap at commit − standoff,
                    # instead of a fixed 0.20 that assumes the servo stopped
                    # exactly at target_depth_m.
                    self._press_forward_m = max(
                        self._press_commit_depth - c["press_standoff_m"], 0.0
                    )
                    self.get_logger().info(
                        f"[DOOR_PRESS] forward {self._press_forward_m:.2f}m "
                        f"(commit depth {self._press_commit_depth:.2f}m − "
                        f"standoff {c['press_standoff_m']:.2f}m)"
                    )
                else:  # debug jump straight into DOOR_PRESS — no servo depth
                    self._press_forward_m = c["forward_before_press_m"]
                    self.get_logger().info(
                        f"[DOOR_PRESS] forward {self._press_forward_m:.2f}m (fixed fallback)"
                    )
            self.car.publish_velocities(c["push_speed"], c["push_speed"])
            if self._commit_forward_done(self._press_forward_m):
                self.car.stop()
                self._phase_goto(2)

        elif self._phase == 2:     # swing down to the calibrated hit pose
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info(
                    f"[DOOR_PRESS] press attempt {self._press_attempts + 1}: hit pose {c['arm_press_deg']}"
                )
                self.arm.set_angles_deg(*c["arm_press_deg"])
            self.car.stop()
            if elapsed > c["arm_settle_s"]:
                self._phase_goto(3)

        elif self._phase == 3:     # continue through the hit pose to the low pose
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info(
                    f"[DOOR_PRESS] press through to low pose {c['arm_press_low_deg']}"
                )
                self.arm.set_angles_deg(*c["arm_press_low_deg"])
            self.car.stop()
            if elapsed > c["arm_settle_s"]:
                self._phase_goto(4)

        elif self._phase == 4:     # re-seat on the handle for the push
            if not self._phase_entered:
                self._phase_entered = True
                # The door has a top/bottom rod lock: the handle must STAY
                # pressed while the door moves or it re-locks at a small
                # angle. The low pose slides off the handle once the car
                # starts moving — hold at the hit-pose height instead.
                self.get_logger().info(
                    f"[DOOR_PRESS] hold handle for push: {c['arm_push_hold_deg']}"
                )
                self.arm.set_angles_deg(*c["arm_push_hold_deg"])
            self.car.stop()
            if elapsed > c["arm_settle_s"]:
                self._phase_goto(5)

        elif self._phase == 5:     # arc push toward front-right, claw holding the handle
            # Differential speeds (left > right) follow the door's swing arc
            # instead of pushing straight into the door edge.
            self.car.publish_velocities(c["push_speed_left"], c["push_speed_right"])
            # TIME-bounded, not distance-bounded. The claw pins the door, so the
            # base barely translates (door resists + wheels skid) and pose distance
            # never reaches push_after_unlock_m — the old _commit_forward_done() ran
            # every time to the 13 s phase watchdog, shoving the door far past 90°.
            # Door angle = push_speed × push_seconds now; calibrate push_seconds.
            # Distance still ends it early if the base ever does travel the target
            # (e.g. door already swung free), as a sanity cap.
            target = c["forward_during_press_m"] + c["push_after_unlock_m"]
            if elapsed >= c["push_seconds"] or self._commit_forward_done(target):
                self.car.stop()
                self._press_attempts += 1
                if self._press_attempts <= c["retry_count"]:
                    # No door-open feedback exists (camera blocked by the arm):
                    # retry_count > 0 means unconditional scripted re-press.
                    self.get_logger().warn(
                        "[DOOR_PRESS] scripted retry — raise arm, back up, re-press"
                    )
                    self._phase_goto(6)
                else:
                    self._phase_goto(8)

        elif self._phase == 6:     # retry: raise the claw before reversing so it
            if not self._phase_entered:   # doesn't drag across the knob/door
                self._phase_entered = True
                self.arm.set_angles_deg(*c["arm_raise_deg"])
            self.car.stop()
            if elapsed > c["arm_settle_s"]:
                self._phase_goto(7)

        elif self._phase == 7:     # retry: back up the push distance, then re-press
            self.car.publish_velocities(-c["push_speed"], -c["push_speed"])
            target = c["forward_during_press_m"] + c["push_after_unlock_m"]
            if self._commit_forward_done(target):
                self.car.stop()
                self._phase_goto(2)

        elif self._phase == 8:     # retract arm and finish
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info("[DOOR_PRESS] retract arm → stow")
                self.arm.set_angles_deg(*c["arm_raise_deg"])
            self.car.stop()
            if elapsed > c["arm_settle_s"]:
                # Stow, not reset: RESET_POS blocks the camera and the ramp
                # scan states that follow need a clear view.
                self.arm.stow()
                self.get_logger().info(
                    "[DOOR_PRESS] commit done "
                    "(TODO: no door-open verification possible — camera was blocked)"
                )
                self._goto(next_state)

    def _state_task3_arm_safe_before_backup(self, next_state):
        c = self.cfg["door_press"]
        elapsed = time.monotonic() - self._phase_t0
        if not self._phase_entered:
            self._phase_entered = True
            self.get_logger().info(
                f"[TASK3_ARM_SAFE] raise arm + close claw before backup: {c['arm_raise_deg']}"
            )
            self.arm.set_angles_deg_closed(*c["arm_raise_deg"])
        self.car.stop()
        if elapsed > c["arm_settle_s"]:
            self._goto(next_state)

    def _state_back_up(self, target_x, speed, next_state, max_travel_m=None):
        """Reverse straight until the map-frame x drops to `target_x` (absolute,
        pose-measured) before continuing. Run after the door opens to clear the
        doorway so the ramp comes into view for the observe/scan chain. The car
        faces +x here (TURN_FORWARD), so reversing lowers x toward target_x.
        `max_travel_m` is a pose-measured runaway guard (not a timer) for a
        mis-yawed car that would otherwise reverse without ever reaching x."""
        if not self._require_pose():
            return
        px, py, _ = self.pose
        if self._anchor is None:
            self._anchor = (px, py)
            self.get_logger().info(
                f"[BACK_UP_AFTER_TASK3] reversing to x={target_x:.2f} (from x={px:.2f})"
            )
        ax, ay = self._anchor
        travelled = math.hypot(px - ax, py - ay)
        reached = px <= target_x
        capped = max_travel_m is not None and travelled >= max_travel_m
        if reached or capped:
            self.car.stop()
            self.arm.stow()
            why = "reached x target" if reached else f"travel cap {max_travel_m:.1f}m"
            self.get_logger().info(
                f"[BACK_UP_AFTER_TASK3] {why}: x={px:.2f} "
                f"(travelled {travelled:.2f}m) → {next_state.name}"
            )
            self._goto(next_state)
            return
        self.car.publish_velocities(-speed, -speed)
        self.get_logger().info(
            f"[BACK_UP_AFTER_TASK3] reversing: x={px:.2f} → {target_x:.2f} "
            f"(travelled {travelled:.2f}m)"
        )

    # ------------------------------------------------------------------
    # Ramp — observe, scan, approach
    # ------------------------------------------------------------------

    def _state_move_observe(self, chain, next_state):
        wp = chain[self._observe_idx]
        if not self._require_pose():
            return
        if self._phase == 0:
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info(
                    f"[{self._state.name}] observe point "
                    f"{self._observe_idx + 1}/{len(chain)}: "
                    f"x={wp['x']:.3f} y={wp['y']:.3f} yaw={math.degrees(wp['yaw']):.1f}°"
                )
            if self._drive_to_point(wp):
                self._phase_goto(1)
        elif self._phase == 1:
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info(
                    f"[{self._state.name}] observe point "
                    f"{self._observe_idx + 1}/{len(chain)}: turn to "
                    f"{math.degrees(wp['yaw']):.1f}°"
                )
            if self._turn_to_yaw(wp["yaw"]):
                self._goto(next_state)

    def _state_ramp_scan(self, chain, move_state, found_next, exhausted_next):
        c = self.cfg["ramp"]
        self.car.stop()
        idx = self._observe_idx

        hits = self._accumulate_ramp_hits(require_center_overlap=True)
        elapsed = time.monotonic() - self._state_t0
        age = self.yolo.ramp_age_s()
        age_text = "n/a" if age is None else f"{age:.1f}s"
        sample_text = (
            "new-hit" if self._ramp_last_counted and self._ramp_last_hit
            else "new-miss" if self._ramp_last_counted
            else "same-seq"
        )

        self.get_logger().info(
            f"[{self._state.name}] point {idx + 1}/{len(chain)}  "
            f"hits={hits}/{c['found_required_frames']} "
            f"(seg msgs seen={len(self._ramp_window)}, {sample_text})  "
            f"seq={self.yolo.ramp_seq()} found={int(self.yolo.ramp_visible())} age={age_text}  "
            f"area={self.yolo.ramp_area_ratio():.3f} "
            f"(bottom={self.yolo.ramp_bottom_area_ratio():.3f}, full={self.yolo.ramp_full_area_ratio():.3f}, "
            f"center={self.yolo.ramp_center_overlap_ratio():.3f})  "
            f"elapsed={elapsed:.1f}/{c['scan_seconds']:.0f}s"
        )

        if hits >= c["found_required_frames"]:
            self._ramp_reacquire_state = move_state
            self.get_logger().info(f"[{self._state.name}] ramp confirmed → {found_next.name}")
            self._goto(found_next)
            return

        if elapsed > c["scan_seconds"]:
            # Walk the rest of this side's observe chain before giving up on the side.
            if idx + 1 < len(chain):
                self._observe_idx = idx + 1
                self.get_logger().warn(
                    f"[{self._state.name}] ramp not seen at point {idx + 1} "
                    f"→ next observe point {idx + 2}/{len(chain)}"
                )
                self._goto(move_state)
            elif exhausted_next is not None:
                self._observe_idx = 0   # fresh chain for the next side
                self.get_logger().warn(
                    f"[{self._state.name}] ramp not seen from this side → {exhausted_next.name}"
                )
                self._goto(exhausted_next)
            else:
                self._fail("ramp not found from any observation point")

    def _return_to_ramp_observe(self, label, reason):
        move_state = self._ramp_reacquire_state
        if move_state is None:
            self.get_logger().warn(
                f"[{label}] {reason}; no saved observe state, staying put"
            )
            self.car.stop()
            return
        self.car.stop()
        self.get_logger().warn(
            f"[{label}] {reason} → return to observe point "
            f"{self._observe_idx + 1} ({move_state.name})"
        )
        self._goto(move_state)

    def _ramp_align_fresh(self, c) -> bool:
        age = self.yolo.align_age_s()
        return (
            self.yolo.align_visible()
            and age is not None
            and age <= c.get("align_info_stale_s", c.get("info_stale_s", 1.5))
        )

    def _ramp_turn_burst(self, label, err, scale):
        c = self.cfg["ramp"]
        turn = c.get("align_turn_speed", self.cfg["control"]["turn_slow_speed"])
        dur = min(c.get("align_turn_max_s", 0.5),
                  max(c.get("align_turn_min_s", 0.15), abs(err) * scale))
        cmd = math.copysign(turn, err if err != 0.0 else 1.0)
        self.get_logger().info(f"[{label}] TURN {dur:.2f}s @{cmd:+.0f}")
        self._servo_enter_burst("TURN", dur, cmd)

    def _state_ramp_align_bottom(self, next_state):
        """Visual servo the ramp mask's near edge down to the bottom of the
        camera frame before the approach. Center on the ramp (dx), square up
        with /yolo/bridge_align skew, then creep forward until the bottom edge
        reaches the target. A cropped or skewed mask is not allowed to launch."""
        c = self.cfg["ramp"]
        turn = c.get("align_turn_speed", self.cfg["control"]["turn_slow_speed"])
        elapsed = time.monotonic() - self._state_t0

        if not self._phase_entered:
            self._phase_entered = True
            self._servo_enter_burst("SETTLE", 0.0, 0.0)

        if self._servo_burst_active():
            return

        sub_elapsed = time.monotonic() - self._servo_sub_t0
        self.car.stop()
        if sub_elapsed < c.get("align_settle_s", 0.4):
            return

        if not self.yolo.ramp_visible():
            self._return_to_ramp_observe(
                "RAMP_ALIGN_BOTTOM",
                f"ramp lost after settle ({elapsed:.1f}s)",
            )
            return

        align_fresh = self._ramp_align_fresh(c)
        dx = self.yolo.align_center_dx() if align_fresh else self.yolo.ramp_delta_x()
        edge = self.yolo.ramp_bottom_edge_ratio()
        shape_conf = self.yolo.align_shape_conf() if align_fresh else 0.0
        skew = self.yolo.align_skew() if align_fresh else 0.0
        angle = self.yolo.align_angle_hint() if align_fresh else 0.0

        align_center_threshold = c.get(
            "align_center_threshold_px",
            c["center_threshold_px"],
        )
        if abs(dx) > align_center_threshold:
            self.get_logger().info(
                f"[RAMP_ALIGN_BOTTOM] settled dx={dx:.0f}px edge={edge:.2f} "
                f"shape={shape_conf:.2f} skew={skew:.3f} → center"
            )
            self._ramp_turn_burst(
                "RAMP_ALIGN_BOTTOM",
                dx,
                c.get("align_turn_s_per_px", 0.0015),
            )
            return

        min_shape = c.get("align_min_shape_conf", 0.4)
        skew_tol = c.get("align_skew_tolerance", 0.08)
        if align_fresh and shape_conf >= min_shape and abs(skew) > skew_tol:
            self.get_logger().info(
                f"[RAMP_ALIGN_BOTTOM] centered edge={edge:.2f} shape={shape_conf:.2f} "
                f"skew={skew:.3f} angle={angle:+.3f} → square"
            )
            self._ramp_turn_burst(
                "RAMP_ALIGN_BOTTOM",
                angle if abs(angle) > 0.001 else skew,
                c.get("align_turn_s_per_skew", 4.0),
            )
            return

        if edge < c["align_bottom_target_ratio"]:
            dur = c.get("align_forward_s", 0.2)
            self.get_logger().info(
                f"[RAMP_ALIGN_BOTTOM] centered edge={edge:.2f} < "
                f"{c['align_bottom_target_ratio']} shape={shape_conf:.2f} "
                f"skew={skew:.3f} → FWD {dur:.2f}s @{c['align_speed']:.0f}"
            )
            self._servo_enter_burst("FORWARD", dur, c["align_speed"])
            return

        if not align_fresh:
            dur = c.get("align_search_burst_s", 0.3)
            self.get_logger().info(
                f"[RAMP_ALIGN_BOTTOM] bridge_align stale/missing at launch gate → SEARCH {dur:.2f}s @{turn:.0f}"
            )
            self._servo_enter_burst("SEARCH", dur, turn)
            return

        if shape_conf < min_shape:
            dur = c.get("align_search_burst_s", 0.3)
            hint = dx if abs(dx) > 1.0 else (angle if abs(angle) > 0.001 else 1.0)
            cmd = math.copysign(turn, hint)
            self.get_logger().info(
                f"[RAMP_ALIGN_BOTTOM] edge ready but mask cropped/weak "
                f"shape={shape_conf:.2f} < {min_shape:.2f} → TURN {dur:.2f}s @{cmd:+.0f}"
            )
            self._servo_enter_burst("TURN", dur, cmd)
            return

        self.get_logger().info(
            f"[RAMP_ALIGN_BOTTOM] launch gate OK edge={edge:.2f} "
            f"shape={shape_conf:.2f} skew={skew:.3f} → {next_state.name}"
        )
        self.car.stop()
        self._ramp_aligned = True
        self._goto(next_state)

    def _state_ramp_approach(self, next_state):
        c = self.cfg["ramp"]
        b = self.cfg["bear"]
        elapsed = time.monotonic() - self._state_t0
        if elapsed > c["approach_timeout_s"]:
            self._fail(f"ramp approach timeout after {elapsed:.0f}s")
            return

        # Aggressive charge: once committed to the ramp, drive straight up a fixed
        # pose-measured distance instead of settling + re-observing every tick.
        # Climbing tilts the camera and drops the ramp mask; the stop-and-look path
        # below treats that as "ramp lost" and reverses to the observe point, so the
        # car oscillates back and forth on the bridge. Charging through fixes it
        # (knocking a bear off is acceptable per mission). See ramp.charge_* config.
        if c.get("charge_mode", False):
            self._ramp_charge(c, b, next_state)
            return

        if not self._phase_entered:
            self._phase_entered = True
            self._servo_enter_burst("SETTLE", 0.0, 0.0)

        if self._servo_burst_active():
            return

        sub_elapsed = time.monotonic() - self._servo_sub_t0
        self.car.stop()
        if sub_elapsed < c.get("align_settle_s", 0.4):
            return

        if not self.yolo.ramp_visible():
            self._return_to_ramp_observe(
                "RAMP_APPROACH",
                f"ramp lost after settle ({elapsed:.1f}s)",
            )
            return

        # A close bear takes priority over ramp alignment only while the ramp
        # mask is still visible; otherwise re-acquire from the observe chain.
        if self.yolo.bear_visible() and 0 < self.yolo.bear_distance() < b["blocking_depth_threshold_m"]:
            self.get_logger().info(
                f"[RAMP_APPROACH] bear at {self.yolo.bear_distance():.2f}m → classify"
            )
            self.car.stop()
            self._goto(S.RAMP_BEAR_CLASSIFY)
            return

        area = self.yolo.ramp_area_ratio()
        dx = self.yolo.align_center_dx() if self._ramp_align_fresh(c) else self.yolo.ramp_delta_x()

        if area >= c["approach_done_area"]:
            self.get_logger().info(
                f"[RAMP_APPROACH] area={area:.3f} ≥ {c['approach_done_area']} → {next_state.name}"
            )
            self.car.stop()
            self._approach_done = True
            self._goto(next_state)
            return

        if abs(dx) > c["center_threshold_px"]:
            self.get_logger().info(f"[RAMP_APPROACH] settled dx={dx:.0f}px area={area:.3f} → center")
            self._ramp_turn_burst(
                "RAMP_APPROACH",
                dx,
                c.get("align_turn_s_per_px", 0.0015),
            )
            return

        dur = c.get("approach_forward_s", c.get("align_forward_s", 0.2))
        self.get_logger().info(
            f"[RAMP_APPROACH] settled aligned area={area:.3f} → FWD {dur:.2f}s @{c['approach_speed']:.0f}"
        )
        self._servo_enter_burst("FORWARD", dur, c["approach_speed"])

    def _ramp_charge(self, c, b, next_state):
        """Charge-mode RAMP_APPROACH: commit and drive straight up the ramp.

        Continuous forward at charge_speed (no per-tick settle/stop) until one of:
          * ramp mask area ≥ approach_done_area  → reached the top, classify
          * a close bear is in view              → classify (grasp / clear)
          * charge_distance_m covered            → blind commit done, classify
        A lost ramp mask while climbing is EXPECTED here and is NOT a reason to
        reverse — that was the oscillation bug. Light centering only kicks in when
        the heading error is large (charge_center_threshold_px), reusing the same
        turn-burst sign convention as the normal approach."""
        if not self._require_pose():
            return

        # Let any in-flight centering turn-burst finish before re-reading vision.
        if self._servo_burst_active():
            return

        speed = c.get("charge_speed", c["approach_speed"])
        dist_target = c.get("charge_distance_m", 1.2)
        center_th = c.get("charge_center_threshold_px", 90.0)

        px, py, _ = self.pose
        if self._charge_anchor is None:
            self._charge_anchor = (px, py)
            # 上衝點: persist the ramp foot so RETURN can re-square to it (X for
            # horizontal exits, Y for vertical) before leaving the bridge.
            self._ramp_charge_point = (px, py)
            self.get_logger().info(
                f"[RAMP_CHARGE] commit → unconditional charge {dist_target:.2f}m @{speed:.0f} "
                f"(charge point {px:.2f},{py:.2f} saved for RETURN)"
            )
        ax, ay = self._charge_anchor
        travelled = math.hypot(px - ax, py - ay)

        ramp_seen = self.yolo.ramp_visible()
        area = self.yolo.ramp_area_ratio() if ramp_seen else 0.0

        # 2026-06-14 (Mr.Kung): unconditional charge from the ramp foot. The area≥done
        # and close-bear checks both ended the charge at travelled≈0 (mask fills the
        # view / a far bear reads close), so the car backed up without ever climbing.
        # Only the pose-measured charge_distance_m ends the charge now.
        if travelled >= dist_target:
            self.get_logger().info(
                f"[RAMP_CHARGE] charged {travelled:.2f}m ≥ {dist_target:.2f}m → {next_state.name}"
            )
            self.car.stop()
            self._approach_done = True
            self._goto(next_state)
            return

        # Light centering: only a large heading error interrupts the charge.
        if ramp_seen:
            dx = self.yolo.align_center_dx() if self._ramp_align_fresh(c) else self.yolo.ramp_delta_x()
            if abs(dx) > center_th:
                self.get_logger().info(
                    f"[RAMP_CHARGE] dx={dx:.0f}px > {center_th:.0f} "
                    f"(travelled={travelled:.2f}m) → re-center"
                )
                self._ramp_turn_burst(
                    "RAMP_CHARGE", dx, c.get("align_turn_s_per_px", 0.0015)
                )
                return

        self.get_logger().info(
            f"[RAMP_CHARGE] FWD travelled={travelled:.2f}/{dist_target:.2f}m "
            f"ramp={'Y' if ramp_seen else 'N'} area={area:.3f} @{speed:.0f}"
        )
        self.car.publish_velocities(speed, speed)

    # ------------------------------------------------------------------
    # Bear classification / handling
    # ------------------------------------------------------------------

    def _state_bear_classify(self, grasp_state=S.GRASP_RAMP_BEAR):
        b = self.cfg["bear"]
        self.car.stop()

        if self._phase == 0:
            self._classify_entries += 1
            self._phase_goto(1)
            return

        if self.yolo.bear_visible():
            self._classify_samples.append(
                (self.yolo.bear_distance(), self.yolo.bear_pixel_y(),
                 self.yolo.bear_on_ramp(),
                 self.yolo.blocking_bear_visible(), self.yolo.ramp_bear_visible())
            )

        elapsed = time.monotonic() - self._phase_t0
        if elapsed < b["classify_observe_seconds"]:
            return

        if not self._classify_samples:
            if not self._ramp_aligned:
                # Triage classify (before any align): no bear in the way → proceed
                # to the normal ramp align/approach.
                self.get_logger().info(
                    "[CLASSIFY] triage: no bear samples → RAMP_ALIGN_BOTTOM"
                )
                self._goto(S.RAMP_ALIGN_BOTTOM)
            elif self._approach_done:
                self._fail(
                    "ramp reached but no bear visible — "
                    "TODO: extend search pattern around the ramp"
                )
            else:
                self.get_logger().warn("[CLASSIFY] no bear samples yet → resume RAMP_APPROACH")
                self._goto(S.RAMP_APPROACH)
            return

        n = len(self._classify_samples)
        avg_d = sum(s[0] for s in self._classify_samples) / n
        avg_py = sum(s[1] for s in self._classify_samples) / n

        # Prefer the per-group bears (deterministic each frame, robust to two bears
        # in view): if a blocking bear was seen in most samples, clear it first;
        # else if an on-ramp bear was seen in most samples, grasp it. Fall back to
        # the single on_ramp vote, then to the depth/pixel_y heuristic, when the
        # ramp is out of view so the publisher can't classify either group.
        blk_seen = sum(1 for s in self._classify_samples if len(s) > 3 and s[3])
        rmp_seen = sum(1 for s in self._classify_samples if len(s) > 4 and s[4])
        ramp_votes = [s[2] for s in self._classify_samples if s[2] >= 0.0]
        if blk_seen > n / 2:
            blocking = True
            basis = f"blocking group {blk_seen}/{n} samples"
        elif rmp_seen > n / 2:
            blocking = False
            basis = f"ramp group {rmp_seen}/{n} samples"
        elif ramp_votes:
            on_ramp = sum(ramp_votes) / len(ramp_votes) >= 0.5
            blocking = not on_ramp
            basis = f"on_ramp_votes={len(ramp_votes)}/{n} mean={sum(ramp_votes)/len(ramp_votes):.2f}"
        else:
            blocking = (
                avg_py > b["blocking_pixel_y_threshold"]
                and avg_d < b["blocking_depth_threshold_m"]
            )
            basis = f"fallback depth={avg_d:.2f}m pixel_y={avg_py:.0f}"
        if blocking and self._ramp_aligned:
            # Post-align we've committed to grasping the ramp bear; clearing is a
            # pre-align concern. At close range the bridge bear looks "blocking"
            # (near + low in frame / center_y below a stale ramp line), so any
            # blocking verdict here is a misclassification — treat it as the ramp
            # bear instead of looping back through CLEAR_BLOCKING_BEAR.
            blocking = False
            basis += " (post-align→ramp, blocking suppressed)"
        if blocking:
            decision = "BLOCKING_BEAR"
            next_state = S.CLEAR_BLOCKING_BEAR
        elif not self._ramp_aligned:
            # Triage classify (before any align): nothing blocking the path →
            # proceed to the normal ramp align/approach before grasping.
            decision = "RAMP_BEAR (triage → align first)"
            next_state = S.RAMP_ALIGN_BOTTOM
        else:
            decision = "RAMP_BEAR"
            next_state = grasp_state
        self.get_logger().info(
            f"[CLASSIFY] n={n}  {basis}  → {decision}"
        )
        self._goto(next_state)

    def _bear_visible(self, group=None):
        if group == "blocking":
            return self.yolo.blocking_bear_visible()
        if group == "ramp":
            return self.yolo.ramp_bear_visible()
        return self.yolo.bear_visible()

    def _bear_distance(self, group=None):
        if group == "blocking":
            return self.yolo.blocking_bear_distance()
        if group == "ramp":
            return self.yolo.ramp_bear_distance()
        return self.yolo.bear_distance()

    def _bear_delta_x(self, group=None):
        if group == "blocking":
            return self.yolo.blocking_bear_delta_x()
        if group == "ramp":
            return self.yolo.ramp_bear_delta_x()
        return self.yolo.bear_delta_x()

    def _bear_pixel_x(self, group=None):
        if group == "blocking":
            return self.yolo.blocking_bear_pixel_x()
        if group == "ramp":
            return self.yolo.ramp_bear_pixel_x()
        return self.yolo.bear_pixel_x()

    def _bear_pixel_y(self, group=None):
        if group == "blocking":
            return self.yolo.blocking_bear_pixel_y()
        if group == "ramp":
            return self.yolo.ramp_bear_pixel_y()
        return self.yolo.bear_pixel_y()

    def _grasp_ramp_target_group_or_interrupt(self):
        """Pick the current ramp-bear target group, or interrupt for a blocker."""
        b = self.cfg["bear"]
        age = self.yolo.bear_age_s()
        if age is None or age > b.get("grab_info_stale_s", 0.7):
            return None, False

        # Once aligned on the ramp, the blocking/ramp split is unreliable at
        # close range: the bridge bear drops low in frame and the ramp mask may
        # be stale/out of view, so the publisher mislabels it as a blocking bear
        # and the grab keeps diverting to CLEAR_BLOCKING_BEAR (GRASP↔CLEAR loop).
        # We've already committed to this ramp bear, so target the ramp group if
        # present, else the overall-nearest bear, and never interrupt to clear.
        if self._ramp_aligned:
            return ("ramp" if self.yolo.ramp_bear_visible() else None), False

        if self.yolo.blocking_bear_visible():
            self.car.stop()
            self.get_logger().warn(
                "[GRASP_RAMP_BEAR] blocking bear visible during ramp grab "
                f"(d={self.yolo.blocking_bear_distance():.2f}m "
                f"dx={self.yolo.blocking_bear_delta_x():.0f}px "
                f"py={self.yolo.blocking_bear_pixel_y():.0f}) → CLEAR_BLOCKING_BEAR"
            )
            self._goto(S.CLEAR_BLOCKING_BEAR)
            return None, True

        if self.yolo.ramp_bear_visible():
            return "ramp", False

        # Backward-compatible fallback for older /yolo/bear_info without group fields.
        if self.yolo.bear_visible() and self.yolo.bear_on_ramp() == 0.0:
            self.car.stop()
            self.get_logger().warn(
                "[GRASP_RAMP_BEAR] legacy on_ramp=0 during ramp grab "
                "→ CLEAR_BLOCKING_BEAR"
            )
            self._goto(S.CLEAR_BLOCKING_BEAR)
            return None, True

        return None, False

    def _bear_grab_align_band_px(self, d):
        """Depth-scaled grab band: physical lateral tolerance converted to pixels."""
        b = self.cfg["bear"]
        cap = float(b.get("grab_align_threshold_px", b["align_threshold_px"]))
        info = self._camera_info
        if info is None or d is None or not math.isfinite(d) or d <= 0.0:
            return cap

        fx = float(info.k[0]) if len(info.k) > 0 else 0.0
        tol_m = float(b.get("grab_align_tol_m", 0.0))
        if fx <= 0.0 or tol_m <= 0.0:
            return cap

        min_px = float(b.get("grab_align_min_px", 0.0))
        return min(cap, max(min_px, tol_m * fx / d))

    def _bear_forward_burst_duration(self, d, wheel_speed, cap_s):
        """Bound a creep burst by the measured distance gap to grab_target."""
        b = self.cfg["bear"]
        min_s = float(b.get("align_forward_min_s", 0.1))
        cap_s = float(cap_s)
        if cap_s <= 0.0:
            return 0.0, float("nan"), float("nan")
        min_s = min(min_s, cap_s)

        if d is None or not math.isfinite(d) or d <= 0.0:
            return min_s, float("nan"), float("nan")

        grab_target = b.get("grab_target_distance_m", b["grab_distance_m"])
        gap = max(0.0, d - grab_target)
        creep_mps = float(b.get("grab_creep_mps", 0.0))
        commit_speed = abs(float(b.get("grab_commit_forward_speed", wheel_speed)))
        if creep_mps <= 0.0 or commit_speed <= 0.0 or wheel_speed == 0.0:
            return min_s, gap, float("nan")

        scaled_mps = creep_mps * abs(float(wheel_speed)) / commit_speed
        if scaled_mps <= 0.0:
            return min_s, gap, float("nan")
        dur = min(cap_s, max(min_s, gap / scaled_mps))
        return dur, gap, scaled_mps

    def _bear_grab_ready(self, group=None):
        b = self.cfg["bear"]
        age = self.yolo.bear_age_s()
        visible = self._bear_visible(group)
        d = self._bear_distance(group)
        dx = self._bear_delta_x(group)
        fresh = age is not None and age <= b.get("grab_info_stale_s", 0.7)
        grab_target = b.get("grab_target_distance_m", b["grab_distance_m"])
        at_depth = d is not None and 0.0 < d <= grab_target
        band = self._bear_grab_align_band_px(d) if at_depth else b["align_threshold_px"]
        ready = (
            visible
            and fresh
            and d is not None
            and math.isfinite(d)
            and at_depth
            and abs(dx) <= band
        )
        return ready, d, dx, age

    def _target_marker_ready(self, group=None):
        marker = self.yolo.marker(group)
        if marker is None:
            return False, None
        add_action = getattr(marker, "ADD", 0)
        if marker.action != add_action:
            return False, marker
        return True, marker

    def _synth_bear_marker_from_current_view(self, group=None, d=None):
        """Build a camera-frame marker from the current bear pixel/depth reading."""
        info = self._camera_info
        if info is None or len(info.k) < 6:
            return None
        if d is None or not math.isfinite(d) or d <= 0.0:
            return None

        fx = float(info.k[0])
        fy = float(info.k[4])
        cx = float(info.k[2])
        cy = float(info.k[5])
        if fx <= 0.0 or fy <= 0.0:
            return None

        px = float(self._bear_pixel_x(group))
        py = float(self._bear_pixel_y(group))
        if px <= 0.0:
            px = cx + float(self._bear_delta_x(group))
        if py <= 0.0:
            py = cy

        marker = Marker()
        marker.header.frame_id = self.cfg["bear"].get("camera_frame", "camera_optical_frame")
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{group or 'overall'}_bear_grab_snapshot"
        marker.id = 0
        marker.action = Marker.ADD
        marker.type = Marker.SPHERE
        marker.pose.position.x = (px - cx) * d / fx
        marker.pose.position.y = (py - cy) * d / fy
        marker.pose.position.z = d
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 1.0
        marker.color.g = 0.5
        marker.color.b = 0.0
        marker.color.a = 0.8
        return marker

    def _cache_bear_grab_snapshot(self, group, d, dx):
        marker_ok, marker = self._target_marker_ready(group)
        source = "live"
        if not marker_ok:
            marker = self._synth_bear_marker_from_current_view(group, d)
            source = "synthetic" if marker is not None else "missing"
        if marker is None:
            self._bear_grab_snapshot = None
            return None

        snap = {
            "group": group or "overall",
            "marker": marker,
            "source": source,
            "d": d,
            "dx": dx,
            "t": time.monotonic(),
        }
        self._bear_grab_snapshot = snap
        return snap

    def _current_bear_grab_snapshot(self, group=None):
        snap = self._bear_grab_snapshot
        if not snap or snap.get("group") != (group or "overall"):
            return None
        max_age = float(self.cfg["bear"].get("grab_snapshot_max_age_s", 2.0))
        if time.monotonic() - snap.get("t", 0.0) > max_age:
            return None
        return snap

    def _grab_step(self, label: str, group=None) -> bool:
        """Dispatch to the configured grab: direct close-gripper (default) or IK."""
        if self.cfg["bear"].get("grab_direct_close", True):
            return self._direct_grab_step(label, group)
        return self._auto_grab_precondition_step(label, group)

    def _direct_grab_step(self, label: str, group=None) -> bool:
        """Direct close-gripper grab for a bear the chassis has driven into (no IK).

        Mr.Kung 2026-06-14: at close range the car bumps the bear, so it is already
        at the gripper. Skip the IK raise+descend ("舉高爪子再夾"); lower the arm onto
        the bumped bear with the gripper open, close it, then raise to the same lift
        pose holding it. Robust where IK on a synthetic/blind-zone marker is not.

        Returns True when the grab+lift is complete (and back-dates _auto_grab_t0 so
        the caller's grab_wait gate clears immediately — the grab already finished).
        """
        b = self.cfg["bear"]
        grab_pose = b.get("grab_direct_pose_deg", [108.0, 120.0, 90.0])
        lift_pose = b.get(
            "grab_direct_lift_deg", b.get("safe_pre_arm_pose_deg", [0.0, 141.0, 90.0])
        )

        if not self._phase_entered:
            self._phase_entered = True
            self._direct_grab_phase = 0
            self._direct_grab_t0 = time.monotonic()
            self.car.stop()
            # Lower/extend onto the bumped bear with the gripper OPEN (no high raise).
            self.arm.open_gripper()
            self.arm.set_angles_deg(*grab_pose)
            self.get_logger().info(
                f"[{label}] direct grab: arm→{grab_pose} gripper open "
                "(bear bumped, no IK)"
            )
            return False

        elapsed = time.monotonic() - self._direct_grab_t0

        if self._direct_grab_phase == 0:       # let the arm reach the grab pose
            if elapsed < b.get("grab_direct_reach_s", 1.5):
                return False
            self.arm.close_gripper()
            self.get_logger().info(f"[{label}] direct grab: close gripper")
            self._direct_grab_phase = 1
            self._direct_grab_t0 = time.monotonic()
            return False

        if self._direct_grab_phase == 1:       # let the gripper close on the bear
            if elapsed < b.get("grab_direct_close_s", 1.0):
                return False
            self.arm.set_angles_deg_closed(*lift_pose)
            self.get_logger().info(
                f"[{label}] direct grab: raise to {lift_pose} holding bear"
            )
            self._direct_grab_phase = 2
            self._direct_grab_t0 = time.monotonic()
            return False

        # phase 2: let the lift complete, then report done.
        if elapsed < b.get("grab_direct_lift_s", 1.0):
            return False
        self._auto_grab_t0 = time.monotonic() - b["grab_wait_seconds"]
        return True

    def _auto_grab_precondition_step(self, label: str, group=None):
        b = self.cfg["bear"]
        settle_s = b.get("pre_arm_settle_s", 0.5)
        hold_speed = b.get("grab_hold_forward_speed", 0.0)

        if not self._phase_entered:
            ready, d, dx, age = self._bear_grab_ready(group)
            marker_ok, marker = self._target_marker_ready(group)
            snap = self._current_bear_grab_snapshot(group)
            if (not ready or not marker_ok) and snap is not None:
                ready = True
                marker_ok = True
                marker = snap["marker"]
                d = snap["d"]
                dx = snap["dx"]
            if not ready or not marker_ok:
                age_text = "none" if age is None else f"{age:.2f}s"
                marker_state = "ok" if marker_ok else "missing/delete"
                self.get_logger().warn(
                    f"[{label}] grab gate rejected "
                    f"(group={group or 'overall'} visible={self._bear_visible(group)} d={d:.2f}m "
                    f"dx={dx:.0f}px age={age_text} marker={marker_state}) → re-servo"
                )
                self._phase_goto(0)
                return False

            self._phase_entered = True
            self._auto_grab_triggered = False
            self._auto_grab_t0 = None
            self._auto_grab_marker = marker
            if hold_speed > 0.0:
                self.car.publish_velocities(hold_speed, hold_speed)
            else:
                self.car.stop()
            wrist_offset = b.get("grab_wrist_forward_offset_deg")
            self.arm.set_auto_grab_wrist_offset_deg(wrist_offset)
            self.arm.safe_pre_arm_pose(b.get("safe_pre_arm_pose_deg"))
            self.get_logger().info(
                f"[{label}] SAFE_PRE_ARM_POSE → wait {settle_s:.1f}s, "
                f"then publish target_point and trigger auto_arm "
                f"(d={d:.2f}m dx={dx:.0f}px frame={marker.header.frame_id} "
                f"wrist_offset={wrist_offset})"
            )
            return False

        if not self._auto_grab_triggered:
            if hold_speed > 0.0:
                self.car.publish_velocities(hold_speed, hold_speed)
            else:
                self.car.stop()
            if time.monotonic() - self._phase_t0 < settle_s:
                return False
            marker_ok, marker = self._target_marker_ready(group)
            if not marker_ok:
                snap = self._current_bear_grab_snapshot(group)
                if snap is not None:
                    marker = snap["marker"]
                    self.get_logger().warn(
                        f"[{label}] {group or 'overall'} target marker missing at trigger "
                        f"→ use {snap['source']} snapshot"
                    )
                elif self._auto_grab_marker is not None:
                    marker = self._auto_grab_marker
                    self.get_logger().warn(
                        f"[{label}] {group or 'overall'} target marker missing at trigger "
                        "→ use pre-arm marker"
                    )
                else:
                    self.get_logger().warn(
                        f"[{label}] {group or 'overall'} target marker missing at trigger → re-servo"
                    )
                    self._phase_goto(0)
                    return False
            self._auto_grab_marker = marker
            self.arm.set_auto_grab_wrist_offset_deg(
                b.get("grab_wrist_forward_offset_deg")
            )
            self.arm.auto_grab_marker(marker, b.get("grab_z_offset_m", 0.0))
            self._auto_grab_triggered = True
            self._auto_grab_t0 = time.monotonic()
            self.get_logger().info(f"[{label}] published target_point via /clicked_point")
            return False

        if hold_speed > 0.0:
            self.car.publish_velocities(hold_speed, hold_speed)
        return True

    def _servo_enter_burst(self, sub, dur, cmd):
        """Arm a sub-state and ACTUATE it immediately.

        The control loop ticks at 10 Hz (0.1 s). A burst whose duration is shorter
        than one tick would otherwise never publish — by the next tick sub_elapsed
        already exceeds it, so the burst block transitions straight back to SETTLE
        without ever commanding the wheels (the car never turns). Publishing here,
        at arm time, guarantees at least one actuation regardless of duration."""
        self._servo_sub = sub
        self._servo_sub_t0 = time.monotonic()
        self._servo_burst_s = dur
        self._servo_burst_cmd = cmd
        if sub in ("FORWARD", "CRUISE"):
            self.car.publish_velocities(cmd, cmd)
        elif sub in ("TURN", "SEARCH"):
            self.car.publish_velocities(cmd, -cmd)
        else:  # SETTLE
            self.car.stop()

    def _servo_burst_active(self) -> bool:
        """Run an armed open-loop burst.

        Returns True when the caller should skip perception work this tick.  A
        finished burst is converted back to SETTLE and also returns True, so the
        next tick gets a full settle interval before reading vision.
        """
        if self._servo_sub not in ("TURN", "FORWARD", "SEARCH"):
            return False

        sub_elapsed = time.monotonic() - self._servo_sub_t0
        if sub_elapsed < self._servo_burst_s:
            if self._servo_sub == "FORWARD":
                self.car.publish_velocities(self._servo_burst_cmd, self._servo_burst_cmd)
            else:
                self.car.publish_velocities(self._servo_burst_cmd, -self._servo_burst_cmd)
            return True

        self._servo_enter_burst("SETTLE", 0.0, 0.0)
        return True

    def _bear_servo_step(self, group=None) -> bool:
        """Align + close on the bear, True at the grab pose. Stop-and-look stepper.

        The camera→YOLO chain lags ~0.3 s, so any move-while-watching loop (bang-
        bang OR PD) commands against a stale dx and limit-cycles — it can't be
        tuned away, the deadtime is too large. This samples vision ONLY while
        stationary and settled (so the reading is current), then commits ONE
        open-loop burst (rotate toward the bear, or a short forward creep), stops,
        and re-measures. No feedback while moving → immune to the latency. Worst-
        case forward overshoot is bounded by a single creep step, not by the lag.
        Calls _fail on servo timeout."""
        b = self.cfg["bear"]
        slow = self.cfg["control"]["slow_speed"]
        turn = self.cfg["control"]["turn_slow_speed"]
        align_th = b["align_threshold_px"]
        grab_target = b.get("grab_target_distance_m", b["grab_distance_m"])
        # The ramp bear gets its own (closer) servo stop distance.
        if self._state == S.GRASP_RAMP_BEAR:
            grab_target = b.get("ramp_grab_target_distance_m", 0.35)

        if time.monotonic() - self._phase_t0 > b["servo_timeout_s"]:
            self._fail(f"bear servo timeout in {self._state.name}")
            return False

        # One-shot sub-FSM init on phase entry — begin by settling.
        if not self._phase_entered:
            self._phase_entered = True
            self._servo_enter_burst("SETTLE", 0.0, 0.0)

        sub_elapsed = time.monotonic() - self._servo_sub_t0

        # --- open-loop bursts: drive blind for the armed duration, no vision read ---
        if self._servo_sub in ("TURN", "FORWARD", "SEARCH"):
            if sub_elapsed < self._servo_burst_s:
                if self._servo_sub == "FORWARD":
                    self.car.publish_velocities(self._servo_burst_cmd, self._servo_burst_cmd)
                else:  # TURN / SEARCH are in-place rotations (left = -right)
                    self.car.publish_velocities(self._servo_burst_cmd, -self._servo_burst_cmd)
                return False
            self._servo_enter_burst("SETTLE", 0.0, 0.0)  # burst done → re-settle
            return False

        # --- CRUISE: continuous far approach. Unlike the blind bursts above, this
        # re-reads the (lagged) bear every tick to decide WHEN to stop — not for
        # tight alignment — so the car drives straight at a far bear with no stop-
        # and-look gaps. Stops to re-settle once near (hand to the careful stepper),
        # if the bear is lost, or if heading drifts past the wide cruise band. ---
        if self._servo_sub == "CRUISE":
            if not self._bear_visible(group) or self.yolo.bear_age_s() is None:
                self.car.stop()
                self._servo_enter_burst("SETTLE", 0.0, 0.0)
                return False
            dx = self._bear_delta_x(group)
            d = self._bear_distance(group)
            cruise_until = b.get("cruise_until_m", b["grab_distance_m"])
            cruise_band = b.get("cruise_align_band_px", b["align_threshold_px"] * 3.0)
            near_now = d is not None and math.isfinite(d) and 0.0 < d <= cruise_until
            if near_now or abs(dx) > cruise_band:
                if d is not None and math.isfinite(d) and d > 0.0:
                    self._bear_last_settle_depth = d
                d_text = f"{d:.2f}m" if (d is not None and math.isfinite(d)) else "n/a"
                why = "near" if near_now else f"off-heading dx={dx:.0f}px"
                self.get_logger().info(
                    f"[{self._state.name}] cruise stop ({why}, d={d_text}) → settle"
                )
                self.car.stop()
                self._servo_enter_burst("SETTLE", 0.0, 0.0)
                return False
            self.car.publish_velocities(self._servo_burst_cmd, self._servo_burst_cmd)
            return False

        # --- SETTLE: stop, wait for the image to catch up, then read & decide ---
        self.car.stop()
        if sub_elapsed < b.get("align_settle_s", 0.4):
            return False

        if not self._bear_visible(group) or self.yolo.bear_age_s() is None:
            self.get_logger().info(
                f"[{self._state.name}] {group or 'overall'} bear not visible after settle → search"
            )
            self._servo_enter_burst("SEARCH", b.get("align_search_burst_s", 0.3), turn)
            return False

        # On a grab retry the committed bear is locked to a world (map) spot.
        # Track THAT bear, not the new overall-nearest, so the servo can't swing
        # toward a different bear. Only constrains the overall stream; the blocking
        # / ramp groups are already deterministic. If the locked bear can't be
        # re-found this settle, re-search rather than chase the wrong one.
        locked_view = self._locked_bear_view() if group is None else None
        if (
            group is None
            and locked_view is None
            and self._grasp_verify_lock
            and self._grasp_verify_lock.get("map_pos") is not None
        ):
            self.get_logger().info(
                f"[{self._state.name}] locked bear not at map spot after settle → search"
            )
            self._servo_enter_burst("SEARCH", b.get("align_search_burst_s", 0.3), turn)
            return False

        if locked_view is not None:
            dx, d = locked_view
        else:
            dx = self._bear_delta_x(group)
            d = self._bear_distance(group)

        prev_d = self._bear_last_settle_depth
        prev_valid = prev_d is not None and math.isfinite(prev_d) and prev_d > 0.0
        was_recently_near = prev_valid and prev_d < b["grab_distance_m"]
        jumped_from_near = False
        d_for_step = d
        ramp_pick_near = False
        if d is not None and math.isfinite(d) and d > 0.0:
            jump_reject_m = float(b.get("depth_jump_reject_m", 0.0))
            jumped_from_near = (
                was_recently_near
                and jump_reject_m > 0.0
                and (d - prev_d) > jump_reject_m
            )
            if jumped_from_near:
                d_for_step = prev_d
                if self._state == S.GRASP_RAMP_BEAR:
                    # The ramp bear is stationary; a frame that suddenly reads
                    # far after a near lock is a bad depth sample. Always trust
                    # the previous near depth and let the grab proceed instead of
                    # rejecting/re-settling, which limit-cycles forever.
                    ramp_pick_near = True
                    self.get_logger().warn(
                        f"[{self._state.name}] depth jump {prev_d:.2f}→{d:.2f}m; "
                        f"always using near depth {prev_d:.2f}m"
                    )
                elif not self._bear_depth_jump_rejected:
                    self._bear_depth_jump_rejected = True
                    self.get_logger().warn(
                        f"[{self._state.name}] depth jump {prev_d:.2f}→{d:.2f}m "
                        "near bear; reject frame and re-settle"
                    )
                    self._servo_enter_burst("SETTLE", 0.0, 0.0)
                    return False
                else:
                    self.get_logger().warn(
                        f"[{self._state.name}] repeated depth jump {prev_d:.2f}→{d:.2f}m; "
                        f"using previous near depth {prev_d:.2f}m for creep sizing"
                    )
            else:
                self._bear_depth_jump_rejected = False
                self._bear_last_settle_depth = d
        # On the ramp, drive the grab/align gate off the trusted near depth, not
        # the rejected far reading, so at_depth can actually open.
        d_eff = d_for_step if ramp_pick_near else d
        at_depth = d_eff is not None and 0 < d_eff <= grab_target
        band = self._bear_grab_align_band_px(d_eff) if at_depth else align_th

        # 1) Outside the band → one rotate burst toward the bear, sized by |dx|.
        if abs(dx) > band:
            dur = min(b.get("align_turn_max_s", 0.5),
                      max(b.get("align_turn_min_s", 0.15),
                          abs(dx) * b.get("align_turn_s_per_px", 0.0015)))
            cmd = math.copysign(b.get("align_turn_speed", 260.0), dx)
            self.get_logger().info(
                f"[{self._state.name}] settled dx={dx:.0f}px d={d_eff:.2f}m "
                f"band={band:.0f}px → TURN {dur:.2f}s @{cmd:+.0f}"
            )
            self._servo_enter_burst("TURN", dur, cmd)
            return False

        # 2) Within band + at grab depth → done.
        if at_depth:
            snap = self._cache_bear_grab_snapshot(group, d_eff, dx)
            snap_source = snap["source"] if snap is not None else "none"
            self.get_logger().info(
                f"[{self._state.name}] settled aligned dx={dx:.0f}px d={d_eff:.2f}m "
                f"band={band:.0f}px marker={snap_source} → stop & grab"
            )
            self._bear_commit_t0 = None
            self.car.stop()
            return True

        # 3) Aligned but not close (or depth invalid) → one short forward creep.
        if group == "blocking":
            near = False
            branch = "blocking"
            base = b.get(
                "clear_blocking_forward_speed",
                b.get("grab_commit_forward_speed", slow),
            )
            cap = b.get("clear_blocking_forward_s", b.get("align_forward_min_s", 0.1))
            # A blocking bear creeps into the depth cam's blind zone and vanishes
            # ~1cm short of grab_target, so the at_depth snapshot above never
            # fires. Cache the most recent near (aligned) view each settle so
            # CLEAR can still grab it once it's lost — otherwise the approach
            # never converges and CLEAR loops approach<->backup forever.
            if d is not None and math.isfinite(d) and 0.0 < d <= b["grab_distance_m"]:
                self._cache_bear_grab_snapshot(group, d, dx)
        else:
            near = (d is not None and 0 < d < b["grab_distance_m"]) or was_recently_near
            branch = "near" if near else "far"
            # Far + aligned → drive CONTINUOUSLY at the bear instead of the pulsed
            # stop-and-look creep. Precise alignment doesn't matter this far out, so
            # the camera lag is harmless; CRUISE re-reads the (lagged) bear each
            # tick only to know WHEN to hand back to the careful near stepper.
            if not near and b.get("far_continuous", True):
                speed = b.get(
                    "cruise_forward_speed", b.get("climb_forward_speed", slow)
                )
                self.get_logger().info(
                    f"[{self._state.name}] settled dx={dx:.0f}px d={d:.2f}m "
                    f"(far) → CRUISE @{speed:.0f}"
                )
                self._servo_enter_burst("CRUISE", 0.0, speed)
                return False
            # far creep doubles as the ramp climb; let it run faster than the global
            # slow_speed (gravity raises the effective stall floor on the incline)
            # without disturbing visual-servo / final-approach / probe speeds.
            base = b.get("grab_commit_forward_speed", slow) if near else b.get("climb_forward_speed", slow)
            cap = b.get("align_forward_near_s", 0.15) if near else b.get("align_forward_far_s", 0.3)
        dur, gap, mps = self._bear_forward_burst_duration(d_for_step, base, cap)
        gap_text = "n/a" if not math.isfinite(gap) else f"{gap:.2f}m"
        rate_text = "n/a" if not math.isfinite(mps) else f"{mps:.2f}m/s"
        self.get_logger().info(
            f"[{self._state.name}] settled dx={dx:.0f}px d={d_eff:.2f}m "
            f"({branch}, gap={gap_text}, rate={rate_text}) → FWD {dur:.2f}s @{base:.0f}"
        )
        self._servo_enter_burst("FORWARD", dur, base)
        return False

    def _state_clear_blocking_bear(self, next_state, lost_state=S.RAMP_BEAR_CLASSIFY):
        b = self.cfg["bear"]
        elapsed = time.monotonic() - self._phase_t0

        if self._phase == 0:       # servo to the blocking bear
            if self._phase_entered and self._servo_sub == "SETTLE":
                sub_elapsed = time.monotonic() - self._servo_sub_t0
                if sub_elapsed >= b.get("align_settle_s", 0.4):
                    if not self._bear_visible("blocking") or self.yolo.bear_age_s() is None:
                        self.car.stop()
                        prev_d = self._bear_last_settle_depth
                        near = (
                            prev_d is not None and math.isfinite(prev_d)
                            and 0.0 < prev_d <= b["grab_distance_m"]
                        )
                        snap = self._current_bear_grab_snapshot("blocking")
                        if near and snap is not None:
                            # Bear vanished at close range → it's right at the
                            # gripper (inside the depth cam's blind zone), not
                            # lost. Grab it with the cached close-range snapshot
                            # instead of backing up, else CLEAR loops forever.
                            self.get_logger().info(
                                "[CLEAR_BEAR] blocking bear lost at close range "
                                f"(last d={prev_d:.2f}m, snap={snap['source']}) → GRAB"
                            )
                            self._phase_goto(1)
                            return
                        self._clear_lost_state = lost_state
                        self.get_logger().warn(
                            "[CLEAR_BEAR] blocking bear not visible after settle "
                            f"→ BACKUP then {lost_state.name}"
                        )
                        self._phase_goto(6)
                        return
            if self._bear_servo_step("blocking"):
                self._phase_goto(1)

        elif self._phase == 1:     # pre-arm, target_point, auto-arm grab (async)
            if self._grab_step("CLEAR_BEAR", "blocking"):
                elapsed_after_trigger = time.monotonic() - self._auto_grab_t0
            else:
                return

            if elapsed_after_trigger > b["grab_wait_seconds"]:
                self._phase_goto(3)   # no backup: deposit by turning 180° in place

        elif self._phase == 3:     # turn 180° in place to face away from the ramp
            if not self._require_pose():
                return
            if not self._phase_entered:
                self._phase_entered = True
                _, _, yaw = self.pose
                self._clear_turn_target = _norm_ang(yaw + math.pi)
                self.get_logger().info(
                    "[CLEAR_BEAR] no backup — turning 180° in place → target "
                    f"{math.degrees(self._clear_turn_target):.0f}°"
                )
            if self._turn_to_yaw(self._clear_turn_target):
                self._phase_goto(4)

        elif self._phase == 4:     # drop the bear behind, off the ramp path
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info("[CLEAR_BEAR] dropping bear behind")
                self.arm.open_gripper()
            if elapsed > 1.0:
                # Stow, not reset: the ramp re-approach after this needs the
                # camera unobstructed.
                self.arm.stow()
                self._phase_goto(5)

        elif self._phase == 5:     # turn 180° back to face the ramp again
            if not self._require_pose():
                return
            if not self._phase_entered:
                self._phase_entered = True
                _, _, yaw = self.pose
                self._clear_turn_target = _norm_ang(yaw + math.pi)
                self.get_logger().info(
                    "[CLEAR_BEAR] turning 180° back toward the ramp → target "
                    f"{math.degrees(self._clear_turn_target):.0f}°"
                )
            if self._turn_to_yaw(self._clear_turn_target):
                self.car.stop()
                self.get_logger().info("[CLEAR_BEAR] cleared → back to ramp approach")
                self._goto(next_state)

        elif self._phase == 6:     # blocking bear lost during approach: undo forward drift
            speed = b.get("clear_lost_backup_speed", self.cfg["control"]["slow_speed"])
            self.car.publish_velocities(-speed, -speed)
            if elapsed > b.get("clear_lost_backup_seconds", 0.8):
                self.car.stop()
                target = self._clear_lost_state or lost_state
                self._clear_lost_state = None
                self.get_logger().info(
                    f"[CLEAR_BEAR] lost-target backup done → {target.name}"
                )
                self._goto(target)

    def _fresh_bear_depth(self):
        b = self.cfg["bear"]
        age = self.yolo.bear_age_s()
        visible = self.yolo.bear_visible()
        d = self.yolo.bear_distance()
        py = self.yolo.bear_pixel_y()
        seq = self.yolo.bear_seq()
        fresh = age is not None and age <= b.get("verify_info_stale_s", 0.7)
        valid = visible and fresh and math.isfinite(d) and d > 0.0
        return valid, seq, d, py, age

    def _bear_candidate_map_pos(self, cand):
        """Back-project a bear candidate (pixel + depth) into the map frame.

        Returns (x, y) in map, or None when intrinsics / TF / depth are missing.
        Pose-invariant: a stationary (ground) bear maps to the same world point
        no matter where the car is, which is what lets a retry re-lock the same
        bear instead of the new overall-nearest one."""
        info = self._camera_info
        if info is None:
            return None
        depth = cand.get("distance")
        if depth is None or not math.isfinite(depth) or depth <= 0.0:
            return None
        K = info.k
        fx, fy, cx, cy = K[0], K[4], K[2], K[5]
        if fx == 0.0 or fy == 0.0:
            return None
        px = cand.get("pixel_x", 0.0)
        py = cand.get("pixel_y", 0.0)
        pt = PointStamped()
        pt.header.frame_id = self.cfg["bear"].get("camera_frame", "camera_optical_frame")
        pt.header.stamp = self.get_clock().now().to_msg()
        pt.point.x = (px - cx) * depth / fx
        pt.point.y = (py - cy) * depth / fy
        pt.point.z = float(depth)
        try:
            tf = self.tf_buffer.lookup_transform(
                "map", pt.header.frame_id, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            mp = tf2_geometry_msgs.do_transform_point(pt, tf)
            return (mp.point.x, mp.point.y)
        except Exception:
            return None

    def _locked_bear_view(self):
        """On a grab retry, return the live candidate nearest the locked map spot
        as (delta_x, distance), else None.

        Lets the servo re-acquire the SAME ground bear it committed to, instead of
        swinging to whatever is now overall-nearest (the wrong-way-turn bug)."""
        lock = self._grasp_verify_lock
        if not lock or lock.get("map_pos") is None:
            return None
        lx, ly = lock["map_pos"]
        best = None
        best_d = self.cfg["bear"].get("verify_lock_map_radius_m", 0.30)
        for c in self.yolo.bear_candidates():
            mp = self._bear_candidate_map_pos(c)
            if mp is None:
                continue
            dist = math.hypot(mp[0] - lx, mp[1] - ly)
            if dist <= best_d:
                best_d = dist
                best = c
        if best is None:
            return None
        return best["delta_x"], best["distance"]

    def _set_grasp_verify_lock_from_current_bear(self, group=None):
        d = self._bear_distance(group)
        dx = self._bear_delta_x(group)
        py = self._bear_pixel_y(group)
        if math.isfinite(d) and d > 0.0:
            map_pos = self._bear_candidate_map_pos({
                "distance": d,
                "pixel_x": self._bear_pixel_x(group),
                "pixel_y": py,
            })
            self._grasp_verify_lock = {
                "distance": d, "delta_x": dx, "pixel_y": py, "map_pos": map_pos,
            }
            mp_text = "n/a" if map_pos is None else f"({map_pos[0]:.2f},{map_pos[1]:.2f})"
            self.get_logger().info(
                f"[GRASP_RAMP_BEAR] verify lock group={group or 'overall'} "
                f"d={d:.2f}m dx={dx:.0f}px py={py:.0f} map={mp_text}"
            )

    def _candidate_matches_verify_lock(self, cand, lock):
        b = self.cfg["bear"]
        dx_tol = b.get("verify_lock_dx_tolerance_px", 120.0)
        py_tol = b.get("verify_lock_py_tolerance_px", 180.0)
        d_tol = b.get("verify_lock_depth_tolerance_m", 0.35)
        return (
            abs(cand["delta_x"] - lock["delta_x"]) <= dx_tol
            and abs(cand["pixel_y"] - lock["pixel_y"]) <= py_tol
            and abs(cand["distance"] - lock["distance"]) <= d_tol
        )

    def _fresh_locked_bear_depth(self):
        b = self.cfg["bear"]
        age = self.yolo.bear_age_s()
        seq = self.yolo.bear_seq()
        fresh = age is not None and age <= b.get("verify_info_stale_s", 0.7)
        if not fresh:
            return False, seq, float("inf"), 0.0, 0.0, age, "stale"

        candidates = [
            c for c in self.yolo.bear_candidates()
            if math.isfinite(c["distance"]) and c["distance"] > 0.0
        ]
        if not candidates:
            return False, seq, float("inf"), 0.0, 0.0, age, "no candidates"

        lock = self._grasp_verify_lock
        if lock is None:
            chosen = min(candidates, key=lambda c: (c["distance"], abs(c["delta_x"])))
        else:
            matching = [c for c in candidates if self._candidate_matches_verify_lock(c, lock)]
            if not matching:
                return False, seq, float("inf"), 0.0, 0.0, age, "no locked candidate"
            chosen = min(
                matching,
                key=lambda c: (
                    abs(c["delta_x"] - lock["delta_x"]),
                    abs(c["pixel_y"] - lock["pixel_y"]),
                    abs(c["distance"] - lock["distance"]),
                ),
            )

        return (
            True,
            seq,
            chosen["distance"],
            chosen["pixel_y"],
            chosen["delta_x"],
            age,
            "locked",
        )

    def _retry_grasp_or_fail(self, reason: str, restore_probe: bool):
        b = self.cfg["bear"]
        if not b.get("verify_retry_on_ground", True):
            self._fail(f"grab verify failed — {reason}")
            return

        self._grab_retries += 1
        max_retries = int(b.get("grab_retry_max", 2))
        # max_retries < 0 cancels the cap: verify retries forever instead of
        # failing the mission (the servo_timeout_s budget still bounds a run
        # where the bear is lost entirely).
        if max_retries >= 0 and self._grab_retries > max_retries:
            self._fail(
                f"grab failed after {self._grab_retries} failed verifies — {reason}"
            )
            return

        cap_text = "∞" if max_retries < 0 else str(max_retries)
        self.get_logger().warn(
            f"[GRASP_RAMP_BEAR] {reason} — retry "
            f"{self._grab_retries}/{cap_text}"
        )
        self._phase_goto(5 if restore_probe else 0)

    def _ramp_grasp_entry_backup(self, b) -> bool:
        """One-time backup on GRASP_RAMP_BEAR entry. Returns True while this tick is
        owned by the backup (caller must return), False once the backup is done or
        skipped so the normal servo can run. One-shot per state entry (see _goto)."""
        if self._ramp_grasp_backup_done:
            return False

        trig = b.get("ramp_grasp_backup_trigger_m", 1.0)
        back_m = b.get("ramp_grasp_backup_distance_m", 0.30)
        speed = b.get("ramp_grasp_backup_speed", self.cfg["control"]["slow_speed"])
        settle_s = b.get("ramp_grasp_backup_settle_s", 0.4)
        decide_s = b.get("ramp_grasp_backup_decide_s", 1.5)
        timeout_s = b.get("ramp_grasp_backup_timeout_s", 2.0)

        if self._ramp_grasp_backup_t0 is None:
            self._ramp_grasp_backup_t0 = time.monotonic()
        waited = time.monotonic() - self._ramp_grasp_backup_t0

        # Phase A: decide. Stop, let the camera/YOLO catch up, then read bear depth.
        if self._ramp_grasp_backup_anchor is None:
            self.car.stop()
            if waited < settle_s:
                return True
            visible = self.yolo.bear_visible() and self.yolo.bear_age_s() is not None
            d = self.yolo.bear_distance() if visible else None
            if visible and d is not None and math.isfinite(d) and d > trig:
                self._ramp_grasp_backup_anchor = self.pose[:2] if self.pose else (0.0, 0.0)
                self._ramp_grasp_backup_t0 = time.monotonic()  # restart for reverse watchdog
                self.get_logger().info(
                    f"[GRASP_RAMP_BEAR] entry bear d={d:.2f}m > {trig:.2f}m "
                    f"→ BACKUP {back_m:.2f}m @{speed:.0f} before servo"
                )
                return True
            if visible or waited > decide_s:
                why = f"d={d:.2f}m ≤ {trig:.2f}m" if visible else "no fresh bear read"
                self.get_logger().info(
                    f"[GRASP_RAMP_BEAR] entry backup skipped ({why}) → servo"
                )
                self._ramp_grasp_backup_done = True
                return False
            return True  # keep waiting briefly for a first fresh read

        # Phase B: reverse a pose-measured distance (elapsed watchdog as fallback).
        ax, ay = self._ramp_grasp_backup_anchor
        travelled = (
            math.hypot(self.pose[0] - ax, self.pose[1] - ay)
            if self.pose is not None else 0.0
        )
        if travelled >= back_m or waited > timeout_s:
            self.car.stop()
            self.get_logger().info(
                f"[GRASP_RAMP_BEAR] entry backup done {travelled:.2f}/{back_m:.2f}m → servo"
            )
            self._ramp_grasp_backup_done = True
            return False
        self.car.publish_velocities(-speed, -speed)
        return True

    def _state_grasp_ramp_bear(self, next_state):
        b = self.cfg["bear"]
        elapsed = time.monotonic() - self._phase_t0

        # One-time entry backup: the charge can stop with the bear too far ahead;
        # reverse once before the careful servo if the first read exceeds the
        # trigger. No-op once fired / when close enough (see _ramp_grasp_entry_backup).
        if self._ramp_grasp_entry_backup(b):
            return

        if self._phase == 0:       # servo to the ramp bear
            group, interrupted = self._grasp_ramp_target_group_or_interrupt()
            if interrupted:
                return
            if self._bear_servo_step(group):
                self._set_grasp_verify_lock_from_current_bear(group)
                self._phase_goto(1)

        elif self._phase == 1:     # pre-arm, target_point, auto-arm grab (async)
            group, interrupted = self._grasp_ramp_target_group_or_interrupt()
            if interrupted:
                return
            if self._grab_step("GRASP_RAMP_BEAR", group):
                elapsed_after_trigger = time.monotonic() - self._auto_grab_t0
            else:
                return

            if elapsed_after_trigger > b["grab_wait_seconds"]:
                # Temporary: skip the probe/depth verify and return straight home
                # once the grab wait completes. Toggle bear.skip_grasp_verify=false
                # to restore the full verify chain (phases 2-4).
                if b.get("skip_grasp_verify", False):
                    self.car.stop()
                    self.get_logger().info(
                        "[GRASP_RAMP_BEAR] skip_grasp_verify=true → return without verify"
                    )
                    self._grab_retries = 0
                    self._goto(next_state)
                    return
                self._phase_goto(2)

        elif self._phase == 2:     # verify: capture depth before a small probe move
            self.car.stop()
            if not self._phase_entered:
                self._phase_entered = True
                self._grasp_verify_depth0 = None
                self._grasp_verify_seq0 = None
                self._grasp_verify_samples = []
                self._grasp_verify_last_seq = None
                self._grasp_verify_probe_m = 0.0
                self.get_logger().info("[GRASP_RAMP_BEAR] verify depth baseline")

            if elapsed < b.get("verify_baseline_observe_seconds", 0.3):
                return

            valid, seq, d, py, dx, age, reason = self._fresh_locked_bear_depth()
            if not valid:
                age_text = "none" if age is None else f"{age:.2f}s"
                self._retry_grasp_or_fail(
                    f"no locked bear depth before probe "
                    f"({reason}, age={age_text})",
                    restore_probe=False,
                )
                return

            self._grasp_verify_depth0 = d
            self._grasp_verify_seq0 = seq
            self.get_logger().info(
                f"[GRASP_RAMP_BEAR] baseline locked d={d:.2f}m dx={dx:.0f}px "
                f"py={py:.0f} seq={seq} "
                "→ small BACKWARD probe"
            )
            self._phase_goto(3)

        elif self._phase == 3:     # verify: move a short distance on the bridge
            target = b.get("verify_probe_distance_m", 0.12)
            speed = b.get("verify_probe_speed", self.cfg["control"]["slow_speed"])
            timeout = b.get("verify_probe_timeout_s", b.get("verify_backup_seconds", 0.5))
            travelled = self._dist_from_anchor()
            if travelled >= target or elapsed > timeout:
                self.car.stop()
                self._grasp_verify_probe_m = travelled
                self.get_logger().info(
                    f"[GRASP_RAMP_BEAR] probe done {travelled:.2f}/{target:.2f}m "
                    "→ observe depth change"
                )
                self._phase_goto(4)
                return

            self.car.publish_velocities(-speed, -speed)
            self.get_logger().info(
                f"[GRASP_RAMP_BEAR] probe BACKWARD {travelled:.2f}/{target:.2f}m"
            )

        elif self._phase == 4:     # verify: compare fresh depth after the probe
            self.car.stop()
            if not self._phase_entered:
                self._phase_entered = True
                self._grasp_verify_samples = []
                self._grasp_verify_last_seq = None

            valid, seq, d, py, dx, _age, _reason = self._fresh_locked_bear_depth()
            if (
                valid
                and self._grasp_verify_seq0 is not None
                and seq > self._grasp_verify_seq0
                and seq != self._grasp_verify_last_seq
            ):
                self._grasp_verify_samples.append((d, py, dx, seq))
                self._grasp_verify_last_seq = seq

            if elapsed < b["verify_observe_seconds"]:
                return

            min_probe = b.get("verify_probe_min_distance_m", 0.06)
            if self._pose_ok() and self._grasp_verify_probe_m < min_probe:
                self._retry_grasp_or_fail(
                    f"probe movement too small "
                    f"({self._grasp_verify_probe_m:.2f}m < {min_probe:.2f}m)",
                    restore_probe=True,
                )
                return

            min_frames = int(b.get("verify_depth_min_frames", 1))
            if len(self._grasp_verify_samples) < min_frames:
                self._retry_grasp_or_fail(
                    f"not enough fresh depth samples after probe "
                    f"({len(self._grasp_verify_samples)}/{min_frames})",
                    restore_probe=True,
                )
                return

            depth0 = self._grasp_verify_depth0
            avg_d = sum(sample[0] for sample in self._grasp_verify_samples) / len(
                self._grasp_verify_samples
            )
            avg_py = sum(sample[1] for sample in self._grasp_verify_samples) / len(
                self._grasp_verify_samples
            )
            avg_dx = sum(sample[2] for sample in self._grasp_verify_samples) / len(
                self._grasp_verify_samples
            )
            delta = abs(avg_d - depth0)
            tolerance = b.get("verify_depth_stable_tolerance_m", 0.05)
            arm_depth_th = b.get("verify_arm_depth_threshold", 0.35)

            if delta <= tolerance and avg_d < arm_depth_th:
                self.get_logger().info(
                    f"[GRASP_RAMP_BEAR] grasp OK — locked bear stable after probe "
                    f"(d0={depth0:.2f}m d1={avg_d:.2f}m Δ={delta:.2f}m "
                    f"dx={avg_dx:.0f}px py={avg_py:.0f}) → return"
                )
                self._grab_retries = 0
                self._goto(next_state)
            elif delta <= tolerance:
                self._retry_grasp_or_fail(
                    f"locked bear depth stable but not in arm range "
                    f"(d1={avg_d:.2f}m >= {arm_depth_th:.2f}m, dx={avg_dx:.0f}px, py={avg_py:.0f})",
                    restore_probe=True,
                )
            else:
                self._retry_grasp_or_fail(
                    f"locked bear depth changed after probe "
                    f"(d0={depth0:.2f}m d1={avg_d:.2f}m Δ={delta:.2f}m "
                    f"> {tolerance:.2f}m, dx={avg_dx:.0f}px, py={avg_py:.0f})",
                    restore_probe=True,
                )

        elif self._phase == 5:     # retry only: return to the pre-probe grasp pose
            target = b.get("verify_probe_distance_m", 0.12)
            speed = b.get("verify_probe_speed", self.cfg["control"]["slow_speed"])
            timeout = b.get("verify_restore_timeout_s", b.get("verify_probe_timeout_s", 0.5))
            travelled = self._dist_from_anchor()
            if travelled >= target or elapsed > timeout:
                self.car.stop()
                self.get_logger().info(
                    f"[GRASP_RAMP_BEAR] probe restored {travelled:.2f}/{target:.2f}m "
                    "→ retry servo"
                )
                self._phase_goto(0)
                return

            self.car.publish_velocities(speed, speed)
            self.get_logger().info(
                f"[GRASP_RAMP_BEAR] restore FORWARD {travelled:.2f}/{target:.2f}m"
            )

    # ------------------------------------------------------------------
    # Return / drop
    # ------------------------------------------------------------------

    def _select_return_exit_mode(self):
        """Choose how to leave the bridge before routing home.

        Code mode names follow the robot yaw:
        - "horizontal": yaw near ±180°, keep the legacy forward exit.
        - "vertical": yaw near +90°, reverse straight back to the observe point
          used to enter the ramp.
        Fall back to the outbound orientation flag when yaw is not close to
        either expected heading.
        """
        cfg = self.cfg["return"]
        if self.pose is None:
            return "horizontal" if self._bridge_horizontal else "vertical"

        yaw = self.pose[2]
        horizontal_err = abs(_norm_ang(yaw - math.pi))
        vertical_err = abs(_norm_ang(yaw - math.pi / 2.0))
        threshold = math.radians(cfg.get("bridge_yaw_select_threshold_deg", 55.0))

        if horizontal_err <= vertical_err and horizontal_err <= threshold:
            return "horizontal"
        if vertical_err < horizontal_err and vertical_err <= threshold:
            return "vertical"
        return "horizontal" if self._bridge_horizontal else "vertical"

    def _return_observe_waypoint(self):
        move_state = self._ramp_reacquire_state
        if move_state == S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE:
            chain = self.cfg["route"]["long_side_observe"]
            side = "long"
        elif move_state == S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE:
            chain = self.cfg["route"]["short_side_observe"]
            side = "short"
        else:
            return None, "unknown"

        if not chain:
            return None, side
        idx = max(0, min(self._observe_idx, len(chain) - 1))
        return chain[idx], f"{side} observe point {idx + 1}/{len(chain)}"

    def _reverse_straight_to_point(self, wp, label, speed):
        x, y = wp["x"], wp["y"]
        tol = wp.get("tolerance", self.cfg["control"]["xy_tolerance"])
        px, py, yaw = self.pose
        dist = math.hypot(x - px, y - py)
        travelled = self._dist_from_anchor()

        if dist <= tol:
            self.car.stop()
            self.get_logger().info(
                f"[RETURN] reached {label} ({dist:.2f}m <= {tol:.2f}m) → origin route"
            )
            return True

        if self._return_reverse_start_dist is not None:
            max_extra = self.cfg["return"].get("observe_reverse_max_extra_m", 0.30)
            if travelled > self._return_reverse_start_dist + max_extra:
                self.car.stop()
                self.get_logger().warn(
                    f"[RETURN] reverse to {label} overshot guard "
                    f"(travelled={travelled:.2f}m, start_dist={self._return_reverse_start_dist:.2f}m) "
                    "→ origin route"
                )
                return True

        bearing = math.atan2(y - py, x - px)
        reverse_err = _norm_ang(bearing - (yaw + math.pi))
        self.car.publish_velocities(-abs(speed), -abs(speed))
        self.get_logger().info(
            f"[RETURN] reverse to {label} "
            f"pose=({px:.2f},{py:.2f},{math.degrees(yaw):.0f}°/{self._pose_src}) "
            f"target=({x:.2f},{y:.2f}) dist={dist:.2f}m travelled={travelled:.2f}m "
            f"rev_heading_err={math.degrees(reverse_err):.0f}° → BACKWARD"
        )
        return False

    def _align_axis_to_charge(self, mode) -> bool:
        """Drive forward/back along the (already-confirmed) standard heading so the
        along-exit-axis coordinate matches the recorded charge-up point (上衝點):
        horizontal mode (faces ±x) aligns X, vertical mode (faces ±y) aligns Y.
        Returns True when within tolerance (and stops)."""
        r = self.cfg["return"]
        px, py, yaw = self.pose
        cx, cy = self._ramp_charge_point
        tol = r.get("bridge_exit_align_tol_m", 0.10)
        speed = r.get("bridge_exit_align_speed", r["bridge_exit_speed"])
        if mode == "horizontal":
            target_c, cur_c, fwd, axis = cx, px, math.cos(yaw), "x"
        else:
            target_c, cur_c, fwd, axis = cy, py, math.sin(yaw), "y"
        err = target_c - cur_c
        if abs(err) <= tol:
            self.car.stop()
            return True
        # Drive forward when forward motion (sign = fwd) reduces |err|, else reverse.
        v = abs(speed) if (err * fwd) > 0 else -abs(speed)
        self.car.publish_velocities(v, v)
        self.get_logger().info(
            f"[RETURN] charge-align {axis}: cur={cur_c:.2f} target={target_c:.2f} "
            f"err={err:.2f}m → {'FWD' if v > 0 else 'REV'}"
        )
        return False

    def _state_return_origin(self, next_state):
        r = self.cfg["return"]
        elapsed = time.monotonic() - self._phase_t0

        if self._phase == 0:       # ALIGN: standard yaw + along-axis coord to charge point
            if not self._require_pose():
                return
            if self._return_exit_mode is None:
                # Exit direction comes PURELY from yaw — never from XY map coords
                # (absolute XY drifts and wrongly reads "off-bridge", which once
                # sent the car into an in-place rotate on the ramp). RETURN always
                # follows a ramp grab, so always re-square to the canonical heading.
                self._return_exit_mode = self._select_return_exit_mode()
                _, _, yaw0 = self.pose
                self.get_logger().info(
                    f"[RETURN] align: exit mode={self._return_exit_mode} "
                    f"yaw={math.degrees(yaw0):.0f}° (yaw-based; XY ignored)"
                )

            # Standard yaw for this orientation: horizontal (faces ±x) ≈ -180°,
            # vertical (faces ±y) ≈ +90°.
            if self._return_exit_mode == "horizontal":
                target_yaw = r.get("bridge_exit_realign_horizontal_yaw", -3.14)
            else:
                target_yaw = r.get("bridge_exit_realign_vertical_yaw", 1.57)
            align_timed_out = elapsed > r.get("bridge_exit_align_timeout_s", 8.0)

            # Step 1 — confirm YAW.
            if not align_timed_out and not self._turn_to_yaw(target_yaw):
                return
            # Step 2 — bring the along-exit-axis coordinate in line with the
            # recorded charge-up point (上衝點): horizontal aligns X, vertical
            # aligns Y. Skipped if no charge point was ever recorded.
            if (not align_timed_out and self._ramp_charge_point is not None
                    and not self._align_axis_to_charge(self._return_exit_mode)):
                return

            self.car.stop()
            self.get_logger().info(
                f"[RETURN] aligned ({'timeout' if align_timed_out else 'ok'}) → bridge exit"
            )
            self._phase_goto(5)
            return

        elif self._phase == 5:     # EXIT: leave the bridge along the standard heading
            if not self._require_pose():
                return
            if not self._phase_entered:
                self._phase_entered = True
                if self._return_exit_mode == "vertical":
                    x, y, _ = self.pose
                    wp, label = self._return_observe_waypoint()
                    self._return_reverse_wp = wp
                    self._return_reverse_wp_label = label
                    if wp is not None:
                        self._return_reverse_start_dist = math.hypot(
                            wp["x"] - x, wp["y"] - y
                        )
                        self.get_logger().info(
                            f"[RETURN] reverse exit target={label} "
                            f"x={wp['x']:.2f} y={wp['y']:.2f} "
                            f"start_dist={self._return_reverse_start_dist:.2f}m"
                        )
                    else:
                        self.get_logger().warn(
                            "[RETURN] reverse exit has no saved observe point; "
                            "falling back to configured reverse distance"
                        )

            # Watchdog: no progress → re-align (yaw + charge axis) and re-judge in
            # phase 0, rather than abandoning the bridge rule.
            now = time.monotonic()
            sx, sy, _ = self.pose
            move_thr = self.cfg["stuck"]["move_threshold_m"]
            stuck_to = r.get("bridge_exit_stuck_timeout_s", 4.0)
            if self._return_exit_stuck_anchor is None:
                self._return_exit_stuck_anchor = (sx, sy, now)
            else:
                ax, ay, at = self._return_exit_stuck_anchor
                if math.hypot(sx - ax, sy - ay) > move_thr:
                    self._return_exit_stuck_anchor = (sx, sy, now)
                elif now - at > stuck_to:
                    self.car.stop()
                    self.get_logger().warn(
                        f"[RETURN] bridge exit STUCK — no movement for {now - at:.1f}s; "
                        f"re-aligning ({self._return_exit_mode}) then re-judging bridge rule"
                    )
                    self._return_exit_stuck_anchor = None
                    self._return_exit_mode = None   # re-judge mode + re-align in phase 0
                    self._return_reverse_wp = None
                    self._return_reverse_start_dist = None
                    self._return_reverse_wp_label = None
                    self._phase_goto(0)
                    return

            if self._return_exit_mode == "horizontal":
                target = r.get("horizontal_forward_m", 1.0)
                speed = r["bridge_exit_speed"]
                action = "FORWARD"
            else:
                if self._return_reverse_wp is not None:
                    if self._reverse_straight_to_point(
                        self._return_reverse_wp,
                        self._return_reverse_wp_label,
                        r["bridge_exit_speed"],
                    ):
                        self._phase_goto(1)
                    return
                target = r.get("vertical_reverse_m", r.get("vertical_forward_m", 1.5))
                speed = -r["bridge_exit_speed"]
                action = "BACKWARD"

            travelled = self._dist_from_anchor()
            if travelled >= target:
                self.car.stop()
                self.get_logger().info(
                    f"[RETURN] bridge exit complete ({travelled:.2f}/{target:.2f}m) → origin route"
                )
                self._phase_goto(1)
                return

            self.car.publish_velocities(speed, speed)
            self.get_logger().info(
                f"[RETURN] bridge exit {action} {travelled:.2f}/{target:.2f}m"
            )

        elif self._phase == 1:     # route to origin after clearing the bridge
            if not self._require_pose():
                return
            if self._drive_to_point(self.cfg["route"]["origin"]):
                if r["drop_at_origin"]:
                    self._phase_goto(2)
                else:
                    self.get_logger().info(
                        "[RETURN] at origin — holding bear (drop_at_origin=false)"
                    )
                    self._goto(next_state)

        elif self._phase == 2:     # drop
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info("[RETURN] dropping bear at origin")
                self.arm.open_gripper()
            self.car.stop()
            if elapsed > r["drop_wait_seconds"]:
                self._phase_goto(3)

        elif self._phase == 3:     # back away and stow
            self.car.publish_velocities(
                -self.cfg["control"]["slow_speed"], -self.cfg["control"]["slow_speed"]
            )
            if elapsed > r["back_away_seconds"]:
                self.car.stop()
                self.arm.stow()   # done with the arm — fold it away
                self._goto(next_state)

    # ------------------------------------------------------------------
    # Post-mission cleanup patrol — drive once more toward turn_point and grab
    # a ground bear (NOT on the ramp), then deliver it to origin and finish.
    # Mr.Kung 2026-06-14: "任務做完回到原點之後再走一次到 turn point 的路，路上看到熊
    # 且不在 ramp mask 之上(比較 y 值)的話就抓回原點." Single bear, no repeat sweep.
    # ------------------------------------------------------------------

    def _patrol_bear_qualifies(self):
        """True when a patrol-grabbable bear is in view: visible, fresh, and NOT
        on the ramp. 'Not on the ramp' reuses the publisher's existing on-ramp
        classification (bbox centre-y vs ramp centre-y) exposed as bear_on_ramp():
        1.0 = on the ramp → skip; 0.0 (ground/blocking) or -1.0 (no ramp mask in
        view) → qualifies. An optional depth gate avoids chasing a far/noisy hit."""
        p = self.cfg.get("patrol") or {}
        if not self.yolo.bear_visible():
            return False
        age = self.yolo.bear_age_s()
        if age is None or age > p.get("bear_info_stale_s", 0.5):
            return False
        if self.yolo.bear_on_ramp() >= 0.5:        # on the ramp → not a ground bear
            return False
        d = self.yolo.bear_distance()
        max_d = p.get("max_bear_distance_m", 2.5)
        if d is not None and math.isfinite(d) and d > 0.0 and d > max_d:
            return False
        return True

    def _state_patrol_to_turn(self, grab_state, done_state):
        """Drive once more from origin toward route.turn_point, watching for a
        ground bear the whole way. A qualifying bear interrupts the drive to grab
        it; reaching turn_point with nothing to grab finishes the mission. Bears
        are ignored until the car is min_grab_from_origin_m away so the just-
        delivered bear sitting at origin is not immediately re-grabbed."""
        p = self.cfg.get("patrol") or {}
        if not self._require_pose():
            return
        px, py, _ = self.pose
        o = self.cfg["route"]["origin"]
        from_origin = math.hypot(px - o["x"], py - o["y"])
        if (
            from_origin >= p.get("min_grab_from_origin_m", 0.8)
            and self._patrol_bear_qualifies()
        ):
            self.car.stop()
            self.get_logger().info(
                f"[PATROL] ground bear (not on ramp) d={self.yolo.bear_distance():.2f}m "
                f"dx={self.yolo.bear_delta_x():.0f}px on_ramp={self.yolo.bear_on_ramp():.0f} "
                f"(from_origin={from_origin:.2f}m) → grab"
            )
            self._goto(grab_state)
            return
        if self._drive_to_point(self.cfg["route"]["turn_point"]):
            self.car.stop()
            self.get_logger().info(
                "[PATROL] reached turn_point with no ground bear → mission complete"
            )
            self._goto(done_state)

    def _state_patrol_grab(self, return_state, abandon_state):
        """Servo to the overall-nearest bear and grab it, reusing the bear servo
        + grab primitives. The main mission already succeeded, so a bear lost for
        too long abandons the grab (→ done) instead of failing the mission."""
        p = self.cfg.get("patrol") or {}
        b = self.cfg["bear"]

        if self._phase == 0:       # servo to the bear
            now = time.monotonic()
            if self.yolo.bear_visible() and self.yolo.bear_age_s() is not None:
                self._patrol_grab_lost_t0 = None
            else:
                if self._patrol_grab_lost_t0 is None:
                    self._patrol_grab_lost_t0 = now
                elif now - self._patrol_grab_lost_t0 > p.get("grab_lost_timeout_s", 6.0):
                    self.car.stop()
                    self.get_logger().warn(
                        "[PATROL] bear lost during grab servo → abandon "
                        "(main mission already complete)"
                    )
                    self._goto(abandon_state)
                    return
            if self._bear_servo_step(None):
                self._patrol_grab_lost_t0 = None
                self._phase_goto(1)

        elif self._phase == 1:     # pre-arm, target_point, auto-arm grab (async)
            if self._grab_step("PATROL_GRAB", None):
                elapsed_after_trigger = time.monotonic() - self._auto_grab_t0
            else:
                return
            if elapsed_after_trigger > b["grab_wait_seconds"]:
                self.car.stop()
                self.get_logger().info("[PATROL] bear grabbed → return to origin")
                self._goto(return_state)

    def _state_patrol_return(self, done_state):
        """Drive straight back to origin (no bridge-exit logic — the patrol stays
        on flat ground), drop the bear, back away, stow, and finish."""
        r = self.cfg["return"]
        elapsed = time.monotonic() - self._phase_t0

        if self._phase == 0:       # route straight to origin
            if not self._require_pose():
                return
            if self._drive_to_point(self.cfg["route"]["origin"]):
                self._phase_goto(1)

        elif self._phase == 1:     # drop the bear at origin
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info("[PATROL] dropping bear at origin")
                self.arm.open_gripper()
            self.car.stop()
            if elapsed > r.get("drop_wait_seconds", 2.0):
                self._phase_goto(2)

        elif self._phase == 2:     # back away and stow
            self.car.publish_velocities(
                -self.cfg["control"]["slow_speed"], -self.cfg["control"]["slow_speed"]
            )
            if elapsed > r.get("back_away_seconds", 1.0):
                self.car.stop()
                self.arm.stow()
                self.get_logger().info(
                    "[PATROL] bear delivered to origin → mission complete"
                )
                self._goto(done_state)


def main(args=None):
    rclpy.init(args=args)
    node = ScriptedFinalMission()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.car.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
