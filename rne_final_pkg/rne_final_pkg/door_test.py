"""Door-only calibration node — Task 3 knob servo + door press, nothing else.

Subclasses ScriptedFinalMission and reuses its helpers and state handlers;
only the INIT requirements and the state flow differ:

    INIT (pose + /yolo/knob_info, /yolo/target_info fallback)
      → TASK3_KNOB_SERVO → TASK3_DOOR_PRESS_COMMIT → TASK3_DOOR_EXIT_WAIT → DONE

No Nav2.  Params come from scripted_mission.yaml (knob_servo / door_press).

Run:  ros2 run rne_final_pkg door_test
"""

import math
import time

import rclpy

from rne_final_pkg.scripted_final_mission import ScriptedFinalMission, S

# Seconds to keep waiting for /yolo/knob_info before settling for the legacy
# /yolo/target_info fallback.  target_info is any-class / last-box-wins, so
# the fallback is only trustworthy when the knob is the sole object in view.
_KNOB_GRACE_S = 5.0


class DoorTest(ScriptedFinalMission):
    def __init__(self):
        super().__init__(node_name="door_test")
        self._use_target_fallback = False

    # ── knob source indirection (overrides base) ─────────────────────

    def _knob_visible(self):
        if self._use_target_fallback:
            return self.yolo.is_visible()
        return self.yolo.knob_visible()

    def _knob_dx(self):
        if self._use_target_fallback:
            return self.yolo.delta_x()
        return self.yolo.knob_delta_x()

    def _knob_depth(self):
        if self._use_target_fallback:
            return self.yolo.distance()
        return self.yolo.knob_distance()

    # ── reduced flow ──────────────────────────────────────────────────

    def _state_init(self):
        elapsed = time.monotonic() - self._state_t0
        missing = []
        if self.pose is None:
            missing.append("pose (TF map->base_footprint or /amcl_pose)")

        if not self.yolo.knob_topic_alive() and not self._use_target_fallback:
            if self.yolo.target_topic_alive() and elapsed > _KNOB_GRACE_S:
                self._use_target_fallback = True
                self.get_logger().warn(
                    "[INIT] /yolo/knob_info absent — falling back to /yolo/target_info "
                    "(any-class detection: make sure only the knob is in view)"
                )
            else:
                missing.append(f"/yolo/knob_info (target_info fallback after {_KNOB_GRACE_S:.0f}s)")

        if not missing:
            # Arm stow is handled by the inherited startup stow timer
            # (_stow_arm_once) — calling arm.reset() here would un-stow it
            # (RESET_POS raises the arm into the camera view).
            x, y, yaw = self.pose
            src = "target_info fallback" if self._use_target_fallback else "knob_info"
            self.get_logger().info(
                f"[INIT] ready  pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}°) "
                f"pose_src={self._pose_src}  knob_src={src}"
            )
            self._goto(S.TASK3_KNOB_SERVO)
            return

        if elapsed > self.cfg["init"]["wait_timeout_s"]:
            self._fail(f"INIT timeout after {elapsed:.0f}s — missing: {', '.join(missing)}")
            return
        self.get_logger().info(f"[INIT] waiting ({elapsed:.1f}s)  missing: {', '.join(missing)}")

    def _tick(self):
        self._update_pose()
        s = self._state

        if s == S.INIT:
            self._state_init()
        elif s == S.TASK3_KNOB_SERVO:
            self._state_knob_servo(S.TASK3_DOOR_PRESS_COMMIT)
        elif s == S.TASK3_DOOR_PRESS_COMMIT:
            self._state_door_press(S.TASK3_DOOR_EXIT_WAIT)
        elif s == S.TASK3_DOOR_EXIT_WAIT:
            self._state_door_exit_wait(S.DONE)
        elif s == S.DONE:
            self.car.stop()
            if not self._done_logged:
                self.get_logger().info("Door test complete.")
                self._done_logged = True
        elif s == S.FAILED:
            self.car.stop()
        else:
            self._fail(f"door_test does not handle state {s.name}")


def main(args=None):
    rclpy.init(args=args)
    node = DoorTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.car.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
