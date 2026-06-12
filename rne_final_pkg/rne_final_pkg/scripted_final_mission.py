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
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import PoseWithCovarianceStamped
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
        self._fail_reason = None
        self._done_logged = False

        self._ramp_window = deque(maxlen=int(self.cfg["ramp"]["found_window_frames"]))
        self._ramp_last_seq = None   # last counted seg message (see yolo_client.ramp_seq)
        self._ramp_last_counted = False
        self._ramp_last_hit = False
        self._observe_idx = 0        # index into the current side's observe chain
                                     # (survives MOVE→SCAN→MOVE; reset only on side switch)
        self._bridge_horizontal = False  # set if the ramp is seen on the outbound
                                     # leg (SEGMENT_1): bridge lies horizontal, so the
                                     # post-door route skips the long side and goes
                                     # straight to the short-side observe chain.
                                     # Persists across _goto (not reset there).
        self._approach_done = False  # ramp approach reached approach_done_area at least once
        self._classify_entries = 0
        self._classify_samples = []
        self._grab_retries = 0
        self._auto_grab_triggered = False
        self._auto_grab_t0 = None
        self._auto_grab_marker = None
        self._bear_commit_t0 = None
        self._grasp_verify_depth0 = None
        self._grasp_verify_seq0 = None
        self._grasp_verify_samples = []
        self._grasp_verify_last_seq = None
        self._grasp_verify_probe_m = 0.0
        self._return_exit_mode = None
        self._press_attempts = 0
        self._knob_invalid_t0 = None
        self._servo_settle_t0 = None     # knob servo: in-window settle timer
        self._press_commit_depth = None  # depth at servo commit → dynamic forward leg

        # stuck detection while driving forward (route legs)
        self._stuck_anchor = None    # (x, y, monotonic) of last confirmed movement
        self._recover_until = 0.0    # while monotonic < this, back up instead of driving

        # sea guard: True while actively escaping the +x sea edge (see _sea_guard)
        self._sea_escaping = False

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
        self.arm.stow()
        self.get_logger().info(f"[STOW] stow pose published (attempt {self._stow_attempts})")
        self._stow_timer.cancel()

    def _amcl_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._amcl = (p.x, p.y, _yaw_from_quat(q))
        self._amcl_mono = time.monotonic()

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
        self._auto_grab_triggered = False
        self._auto_grab_t0 = None
        self._auto_grab_marker = None
        self._bear_commit_t0 = None

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
                0.0, S.BACK_UP_AFTER_TASK3,
                timeout_s=self.cfg["post_task3"]["turn_forward_timeout_s"],
            )
        elif s == S.BACK_UP_AFTER_TASK3:
            # Bridge known horizontal from the outbound leg → skip long-side
            # observe and route right-angle (via long_to_short_corner, the outer
            # perimeter) straight to the short-side observe chain.
            after_backup = (
                S.MOVE_TO_LONG_SHORT_CORNER if self._bridge_horizontal
                else S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE
            )
            self._state_back_up(
                self.cfg["post_task3"]["backup_distance_m"],
                self.cfg["post_task3"]["backup_speed"],
                after_backup,
            )
        elif s == S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE:
            self._state_move_observe(self.cfg["route"]["long_side_observe"], S.RAMP_SCAN_LONG_SIDE)
        elif s == S.RAMP_SCAN_LONG_SIDE:
            self._state_ramp_scan(self.cfg["route"]["long_side_observe"],
                                  move_state=S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE,
                                  found_next=S.RAMP_ALIGN_BOTTOM,
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
                                  found_next=S.RAMP_ALIGN_BOTTOM,
                                  exhausted_next=None)
        elif s == S.RAMP_ALIGN_BOTTOM:
            self._state_ramp_align_bottom(S.RAMP_APPROACH)
        elif s == S.RAMP_APPROACH:
            self._state_ramp_approach(S.RAMP_BEAR_CLASSIFY)
        elif s == S.RAMP_BEAR_CLASSIFY:
            self._state_bear_classify()
        elif s == S.CLEAR_BLOCKING_BEAR:
            self._state_clear_blocking_bear(S.RAMP_APPROACH)
        elif s == S.GRASP_RAMP_BEAR:
            self._state_grasp_ramp_bear(S.RETURN_ORIGIN)
        elif s == S.RETURN_ORIGIN:
            self._state_return_origin(S.DONE)
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

    def _accumulate_ramp_hits(self, area_threshold=None) -> int:
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
        found = fresh and self.yolo.ramp_visible() and self.yolo.ramp_area_ratio() >= thr
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
        c = self.cfg["knob_servo"]
        slow = self.cfg["control"]["slow_speed"]
        # In-place rotation stalls below ~250 in Unity (see turn_slow_speed) —
        # slow_speed (200) is for forward motion only.
        turn = self.cfg["control"]["turn_slow_speed"]
        elapsed = time.monotonic() - self._state_t0
        if elapsed > c["max_seconds"]:
            self._fail(f"knob servo timeout after {elapsed:.0f}s (knob_visible={self._knob_visible()})")
            return

        if not self._knob_visible():
            self._knob_invalid_t0 = None
            self._servo_settle_t0 = None
            self.get_logger().info(f"[KNOB_SERVO] no knob ({elapsed:.1f}s) → rotate CW search")
            self.car.publish_velocities(turn, -turn)
            return

        dx = self._knob_dx()
        depth = self._knob_depth()

        if abs(dx) > c["center_threshold_px"]:
            self._knob_invalid_t0 = None
            self._servo_settle_t0 = None
            self.get_logger().info(f"[KNOB_SERVO] dx={dx:.0f}px depth={depth:.2f}m → rotate")
            self._rotate_dir(-dx, speed=turn)   # dx>0 = knob right of center → CW
            return

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
            return
        self._knob_invalid_t0 = None

        if depth > c["target_depth_m"]:
            self._servo_settle_t0 = None
            self.get_logger().info(
                f"[KNOB_SERVO] centered dx={dx:.0f}px  depth={depth:.2f}m > {c['target_depth_m']}m → FORWARD_SLOW"
            )
            self.car.publish_velocities(slow, slow)
            return

        # Too close — the press pose is calibrated at target_depth_m, so back
        # up until depth is inside [target - tol, target] before committing.
        if depth < c["target_depth_m"] - c["depth_tolerance_m"]:
            self._servo_settle_t0 = None
            self.get_logger().info(
                f"[KNOB_SERVO] centered dx={dx:.0f}px  depth={depth:.2f}m < "
                f"{c['target_depth_m'] - c['depth_tolerance_m']:.2f}m → BACKWARD_SLOW"
            )
            self.car.publish_velocities(-slow, -slow)
            return

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
            return
        if now - self._servo_settle_t0 < c["commit_settle_s"]:
            return

        self.get_logger().info(
            f"[KNOB_SERVO] aligned  dx={dx:.0f}px  depth={depth:.2f}m (settled) "
            "→ commit (camera goes blind now)"
        )
        self._press_commit_depth = depth
        self._press_attempts = 0
        self._goto(next_state)

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

    def _state_back_up(self, distance_m, speed, next_state):
        """Reverse straight back `distance_m` (pose-measured) before continuing.
        Run after the door opens to clear the doorway so the ramp comes into
        view for the observe/scan chain."""
        if not self._require_pose():
            return
        px, py, _ = self.pose
        if self._anchor is None:
            self._anchor = (px, py)
            self.get_logger().info(
                f"[BACK_UP_AFTER_TASK3] backing up {distance_m:.1f}m before ramp search"
            )
        ax, ay = self._anchor
        travelled = math.hypot(px - ax, py - ay)
        if travelled >= distance_m:
            self.car.stop()
            self.get_logger().info(
                f"[BACK_UP_AFTER_TASK3] backed up {travelled:.2f}m → {next_state.name}"
            )
            self._goto(next_state)
            return
        self.car.publish_velocities(-speed, -speed)
        self.get_logger().info(
            f"[BACK_UP_AFTER_TASK3] backed up {travelled:.2f}/{distance_m:.1f}m"
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

        hits = self._accumulate_ramp_hits()
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
            f"(bottom={self.yolo.ramp_bottom_area_ratio():.3f}, full={self.yolo.ramp_full_area_ratio():.3f})  "
            f"elapsed={elapsed:.1f}/{c['scan_seconds']:.0f}s"
        )

        if hits >= c["found_required_frames"]:
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

    def _state_ramp_align_bottom(self, next_state):
        """Visual servo the ramp mask's near edge down to the bottom of the
        camera frame before the approach. Center on the ramp (dx) first, then
        creep forward until the published bottom-edge ratio reaches the target;
        getting closer pushes the near edge lower in the frame. On timeout,
        proceed to the approach anyway rather than failing — the approach has
        its own search/forward logic."""
        c = self.cfg["ramp"]
        slow = self.cfg["control"]["turn_slow_speed"]
        elapsed = time.monotonic() - self._state_t0

        if not self.yolo.ramp_visible():
            if elapsed > c["align_timeout_s"]:
                self.get_logger().warn(
                    f"[RAMP_ALIGN_BOTTOM] ramp not reacquired in {elapsed:.0f}s → {next_state.name}"
                )
                self.car.stop()
                self._goto(next_state)
                return
            self.get_logger().info(f"[RAMP_ALIGN_BOTTOM] ramp lost ({elapsed:.1f}s) → rotate CW search")
            self.car.publish_velocities(slow, -slow)
            return

        dx = self.yolo.ramp_delta_x()
        edge = self.yolo.ramp_bottom_edge_ratio()

        if abs(dx) > c["center_threshold_px"]:
            self.get_logger().info(
                f"[RAMP_ALIGN_BOTTOM] dx={dx:.0f}px edge={edge:.2f} → rotate to center"
            )
            self._rotate_dir(-dx, speed=slow)
            return

        if edge < c["align_bottom_target_ratio"]:
            self.get_logger().info(
                f"[RAMP_ALIGN_BOTTOM] edge={edge:.2f} < {c['align_bottom_target_ratio']} → forward"
            )
            v = c["align_speed"]
            self.car.publish_velocities(v, v)
            return

        self.get_logger().info(
            f"[RAMP_ALIGN_BOTTOM] near edge at frame bottom (edge={edge:.2f}) → {next_state.name}"
        )
        self.car.stop()
        self._goto(next_state)

    def _state_ramp_approach(self, next_state):
        c = self.cfg["ramp"]
        b = self.cfg["bear"]
        elapsed = time.monotonic() - self._state_t0
        if elapsed > c["approach_timeout_s"]:
            self._fail(f"ramp approach timeout after {elapsed:.0f}s")
            return

        # A close bear takes priority over ramp alignment
        if self.yolo.bear_visible() and 0 < self.yolo.bear_distance() < b["blocking_depth_threshold_m"]:
            self.get_logger().info(
                f"[RAMP_APPROACH] bear at {self.yolo.bear_distance():.2f}m → classify"
            )
            self.car.stop()
            self._goto(S.RAMP_BEAR_CLASSIFY)
            return

        if not self.yolo.ramp_visible():
            self.get_logger().info(f"[RAMP_APPROACH] ramp lost ({elapsed:.1f}s) → rotate CW search")
            s = self.cfg["control"]["turn_slow_speed"]
            self.car.publish_velocities(s, -s)
            return

        area = self.yolo.ramp_area_ratio()
        dx = self.yolo.ramp_delta_x()

        if area >= c["approach_done_area"]:
            self.get_logger().info(f"[RAMP_APPROACH] area={area:.3f} ≥ {c['approach_done_area']} → classify")
            self.car.stop()
            self._approach_done = True
            self._goto(S.RAMP_BEAR_CLASSIFY)
            return

        if abs(dx) > c["center_threshold_px"]:
            self.get_logger().info(f"[RAMP_APPROACH] dx={dx:.0f}px area={area:.3f} → rotate")
            self._rotate_dir(-dx, speed=self.cfg["control"]["turn_slow_speed"])
        else:
            self.get_logger().info(f"[RAMP_APPROACH] aligned  area={area:.3f} → forward")
            v = c["approach_speed"]
            self.car.publish_velocities(v, v)

    # ------------------------------------------------------------------
    # Bear classification / handling
    # ------------------------------------------------------------------

    def _state_bear_classify(self):
        b = self.cfg["bear"]
        self.car.stop()

        if self._phase == 0:
            self._classify_entries += 1
            if self._classify_entries > 5:
                self._fail("bear classification looped 5× without a stable result")
                return
            self._phase_goto(1)
            return

        if self.yolo.bear_visible():
            self._classify_samples.append(
                (self.yolo.bear_distance(), self.yolo.bear_pixel_y(),
                 self.yolo.bear_on_ramp())
            )

        elapsed = time.monotonic() - self._phase_t0
        if elapsed < b["classify_observe_seconds"]:
            return

        if not self._classify_samples:
            if self._approach_done:
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

        # Prefer the seg/det relation (bear bbox vs ramp mask) when the detector
        # reports it: on_ramp==1 → ramp bear, ==0 → blocking. Fall back to the
        # depth/pixel_y heuristic only when every sample is unknown (-1), e.g. no
        # ramp mask in view or an old detector without the on_ramp field.
        ramp_votes = [s[2] for s in self._classify_samples if s[2] >= 0.0]
        if ramp_votes:
            on_ramp = sum(ramp_votes) / len(ramp_votes) >= 0.5
            blocking = not on_ramp
            basis = f"on_ramp_votes={len(ramp_votes)}/{n} mean={sum(ramp_votes)/len(ramp_votes):.2f}"
        else:
            blocking = (
                avg_py > b["blocking_pixel_y_threshold"]
                and avg_d < b["blocking_depth_threshold_m"]
            )
            basis = f"fallback depth={avg_d:.2f}m pixel_y={avg_py:.0f}"
        self.get_logger().info(
            f"[CLASSIFY] n={n}  {basis}  "
            f"→ {'BLOCKING_BEAR' if blocking else 'RAMP_BEAR'}"
        )
        self._goto(S.CLEAR_BLOCKING_BEAR if blocking else S.GRASP_RAMP_BEAR)

    def _bear_grab_ready(self):
        b = self.cfg["bear"]
        age = self.yolo.bear_age_s()
        visible = self.yolo.bear_visible()
        d = self.yolo.bear_distance()
        dx = self.yolo.bear_delta_x()
        fresh = age is not None and age <= b.get("grab_info_stale_s", 0.7)
        ready = (
            visible
            and fresh
            and math.isfinite(d)
            and 0.0 < d < b["grab_distance_m"]
            and abs(dx) <= b["align_threshold_px"]
        )
        return ready, d, dx, age

    def _target_marker_ready(self):
        marker = self.yolo.marker()
        if marker is None:
            return False, None
        add_action = getattr(marker, "ADD", 0)
        if marker.action != add_action:
            return False, marker
        return True, marker

    def _auto_grab_precondition_step(self, label: str):
        b = self.cfg["bear"]
        settle_s = b.get("pre_arm_settle_s", 0.5)
        hold_speed = b.get("grab_hold_forward_speed", 0.0)

        if not self._phase_entered:
            ready, d, dx, age = self._bear_grab_ready()
            marker_ok, marker = self._target_marker_ready()
            if not ready or not marker_ok:
                age_text = "none" if age is None else f"{age:.2f}s"
                marker_state = "ok" if marker_ok else "missing/delete"
                self.get_logger().warn(
                    f"[{label}] grab gate rejected "
                    f"(visible={self.yolo.bear_visible()} d={d:.2f}m "
                    f"dx={dx:.0f}px age={age_text} marker={marker_state}) → re-servo"
                )
                self._phase_goto(0)
                return False

            self._phase_entered = True
            self._auto_grab_triggered = False
            self._auto_grab_t0 = None
            self._auto_grab_marker = None
            if hold_speed > 0.0:
                self.car.publish_velocities(hold_speed, hold_speed)
            else:
                self.car.stop()
            self.arm.safe_pre_arm_pose(b.get("safe_pre_arm_pose_deg"))
            self.get_logger().info(
                f"[{label}] SAFE_PRE_ARM_POSE → wait {settle_s:.1f}s, "
                f"then publish target_point and trigger auto_arm "
                f"(d={d:.2f}m dx={dx:.0f}px frame={marker.header.frame_id})"
            )
            return False

        if not self._auto_grab_triggered:
            if hold_speed > 0.0:
                self.car.publish_velocities(hold_speed, hold_speed)
            else:
                self.car.stop()
            if time.monotonic() - self._phase_t0 < settle_s:
                return False
            marker_ok, marker = self._target_marker_ready()
            if not marker_ok:
                self.get_logger().warn(
                    f"[{label}] target marker missing at trigger → re-servo"
                )
                self._phase_goto(0)
                return False
            self._auto_grab_marker = marker
            self.arm.auto_grab_marker(marker, b.get("grab_z_offset_m", 0.0))
            self._auto_grab_triggered = True
            self._auto_grab_t0 = time.monotonic()
            self.get_logger().info(f"[{label}] published target_point via /clicked_point")
            return False

        if hold_speed > 0.0:
            self.car.publish_velocities(hold_speed, hold_speed)
        return True

    def _bear_servo_step(self) -> bool:
        """Align + close on the bear; True when at grab distance.
        Calls _fail on servo timeout."""
        b = self.cfg["bear"]
        slow = self.cfg["control"]["slow_speed"]
        turn = self.cfg["control"]["turn_slow_speed"]
        elapsed = time.monotonic() - self._phase_t0

        if elapsed > b["servo_timeout_s"]:
            self._fail(f"bear servo timeout after {elapsed:.0f}s in {self._state.name}")
            return False

        if not self.yolo.bear_visible():
            self._bear_commit_t0 = None
            self.get_logger().info(
                f"[{self._state.name}] bear lost ({elapsed:.1f}s) → rotate CW search"
            )
            self.car.publish_velocities(turn, -turn)
            return False

        age = self.yolo.bear_age_s()
        if age is None or age > b.get("grab_info_stale_s", 0.7):
            self._bear_commit_t0 = None
            age_text = "none" if age is None else f"{age:.2f}s"
            self.get_logger().info(
                f"[{self._state.name}] bear info stale (age={age_text}) → rotate CW search"
            )
            self.car.publish_velocities(turn, -turn)
            return False

        dx = self.yolo.bear_delta_x()
        d = self.yolo.bear_distance()

        if abs(dx) > b["align_threshold_px"]:
            self._bear_commit_t0 = None
            self.get_logger().info(f"[{self._state.name}] dx={dx:.0f}px d={d:.2f}m → rotate")
            self._rotate_dir(-dx, speed=turn)
            return False
        if 0 < d < b["grab_distance_m"]:
            commit_s = b.get("grab_commit_forward_seconds", 0.0)
            if commit_s > 0.0:
                if self._bear_commit_t0 is None:
                    self._bear_commit_t0 = time.monotonic()
                    self.get_logger().info(
                        f"[{self._state.name}] grab range d={d:.2f}m -> "
                        f"commit forward {commit_s:.1f}s"
                    )
                commit_elapsed = time.monotonic() - self._bear_commit_t0
                if commit_elapsed < commit_s:
                    v = b.get("grab_commit_forward_speed", slow)
                    self.car.publish_velocities(v, v)
                    self.get_logger().info(
                        f"[{self._state.name}] commit forward "
                        f"{commit_elapsed:.1f}/{commit_s:.1f}s d={d:.2f}m"
                    )
                    return False
            self.car.stop()
            return True
        self._bear_commit_t0 = None
        self.get_logger().info(f"[{self._state.name}] aligned d={d:.2f}m → forward slow")
        self.car.publish_velocities(slow, slow)
        return False

    def _state_clear_blocking_bear(self, next_state):
        b = self.cfg["bear"]
        elapsed = time.monotonic() - self._phase_t0

        if self._phase == 0:       # servo to the blocking bear
            if self._bear_servo_step():
                self._phase_goto(1)

        elif self._phase == 1:     # pre-arm, target_point, auto-arm grab (async)
            if self._auto_grab_precondition_step("CLEAR_BEAR"):
                elapsed_after_trigger = time.monotonic() - self._auto_grab_t0
            else:
                return

            if elapsed_after_trigger > b["grab_wait_seconds"]:
                self._phase_goto(2)

        elif self._phase == 2:     # back away from the ramp entrance
            self.car.publish_velocities(
                -self.cfg["control"]["slow_speed"], -self.cfg["control"]["slow_speed"]
            )
            if elapsed > b["clear_backward_seconds"]:
                self._phase_goto(3)

        elif self._phase == 3:     # turn aside (CW)
            s = self.cfg["control"]["turn_speed"]
            self.car.publish_velocities(s, -s)
            if elapsed > b["clear_rotate_seconds"]:
                self.car.stop()
                self._phase_goto(4)

        elif self._phase == 4:     # drop the bear beside the path
            if not self._phase_entered:
                self._phase_entered = True
                self.get_logger().info("[CLEAR_BEAR] dropping bear aside")
                self.arm.open_gripper()
            if elapsed > 1.0:
                # Stow, not reset: the ramp re-approach after this needs the
                # camera unobstructed.
                self.arm.stow()
                self._phase_goto(5)

        elif self._phase == 5:     # turn back toward the ramp (CCW)
            s = self.cfg["control"]["turn_speed"]
            self.car.publish_velocities(-s, s)
            if elapsed > b["clear_rotate_seconds"]:
                self.car.stop()
                self.get_logger().info("[CLEAR_BEAR] cleared → back to ramp approach")
                self._goto(next_state)

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

    def _retry_grasp_or_fail(self, reason: str, restore_probe: bool):
        b = self.cfg["bear"]
        if not b.get("verify_retry_on_ground", True):
            self._fail(f"grab verify failed — {reason}")
            return

        self._grab_retries += 1
        max_retries = int(b.get("grab_retry_max", 2))
        if self._grab_retries > max_retries:
            self._fail(
                f"grab failed after {self._grab_retries} failed verifies — {reason}"
            )
            return

        self.get_logger().warn(
            f"[GRASP_RAMP_BEAR] {reason} — retry "
            f"{self._grab_retries}/{max_retries}"
        )
        self._phase_goto(5 if restore_probe else 0)

    def _state_grasp_ramp_bear(self, next_state):
        b = self.cfg["bear"]
        elapsed = time.monotonic() - self._phase_t0

        if self._phase == 0:       # servo to the ramp bear
            if self._bear_servo_step():
                self._phase_goto(1)

        elif self._phase == 1:     # pre-arm, target_point, auto-arm grab (async)
            if self._auto_grab_precondition_step("GRASP_RAMP_BEAR"):
                elapsed_after_trigger = time.monotonic() - self._auto_grab_t0
            else:
                return

            if elapsed_after_trigger > b["grab_wait_seconds"]:
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

            valid, seq, d, py, age = self._fresh_bear_depth()
            if not valid:
                age_text = "none" if age is None else f"{age:.2f}s"
                self._retry_grasp_or_fail(
                    f"no valid bear depth before probe "
                    f"(visible={self.yolo.bear_visible()} d={d:.2f}m age={age_text})",
                    restore_probe=False,
                )
                return

            self._grasp_verify_depth0 = d
            self._grasp_verify_seq0 = seq
            self.get_logger().info(
                f"[GRASP_RAMP_BEAR] baseline d={d:.2f}m py={py:.0f} seq={seq} "
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

            valid, seq, d, py, _age = self._fresh_bear_depth()
            if (
                valid
                and self._grasp_verify_seq0 is not None
                and seq > self._grasp_verify_seq0
                and seq != self._grasp_verify_last_seq
            ):
                self._grasp_verify_samples.append((d, py, seq))
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
            delta = abs(avg_d - depth0)
            tolerance = b.get("verify_depth_stable_tolerance_m", 0.05)
            arm_depth_th = b.get("verify_arm_depth_threshold", 0.35)

            if delta <= tolerance and avg_d < arm_depth_th:
                self.get_logger().info(
                    f"[GRASP_RAMP_BEAR] grasp OK — depth stable after probe "
                    f"(d0={depth0:.2f}m d1={avg_d:.2f}m Δ={delta:.2f}m "
                    f"py={avg_py:.0f}) → return"
                )
                self._grab_retries = 0
                self._goto(next_state)
            elif delta <= tolerance:
                self._retry_grasp_or_fail(
                    f"depth stable but not in arm range "
                    f"(d1={avg_d:.2f}m >= {arm_depth_th:.2f}m)",
                    restore_probe=True,
                )
            else:
                self._retry_grasp_or_fail(
                    f"depth changed after probe "
                    f"(d0={depth0:.2f}m d1={avg_d:.2f}m Δ={delta:.2f}m "
                    f"> {tolerance:.2f}m)",
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

        Horizontal bridge: robot yaw is near ±180° after the grab; drive forward
        down the bridge first. Vertical bridge: yaw is near +90°; reverse down
        the bridge first. Fall back to the outbound orientation flag when yaw is
        not close to either expected heading.
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

    def _state_return_origin(self, next_state):
        r = self.cfg["return"]
        elapsed = time.monotonic() - self._phase_t0

        if self._phase == 0:       # first leave the bridge, then route home
            if not self._require_pose():
                return
            if self._anchor is None:
                self._anchor = self.pose[:2]
                self._return_exit_mode = self._select_return_exit_mode()
                self.get_logger().info(
                    f"[RETURN] bridge exit mode={self._return_exit_mode} "
                    f"yaw={math.degrees(self.pose[2]):.0f}°"
                )

            if self._return_exit_mode == "horizontal":
                target = r.get("horizontal_forward_m", r.get("horizontal_reverse_m", 1.0))
                speed = r["bridge_exit_speed"]
                action = "FORWARD"
            else:
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
