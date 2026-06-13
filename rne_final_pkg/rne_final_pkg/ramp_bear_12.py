"""Ramp bear flow that starts directly at short-side observe point 3.

Like ramp_bear, this subclasses ScriptedFinalMission and reuses its ramp,
bear, grasp, and return handlers.  The difference: it skips the whole route +
long-side scan.  From INIT it jumps straight to the short-side observe chain's
THIRD point (route.short_side_observe[2]) and scans the ramp from there:

    INIT (pose + /yolo/bear_info + /yolo/bridge_info)
      -> short_side_observe point 3 -> RAMP_SCAN_SHORT_SIDE
      -> bear classify (blocking -> clear -> re-classify)
      -> ramp align bottom -> ramp approach -> grasp -> return origin

    Note: classify runs *before* align here, so a blocking bear is cleared
    before any ramp-bottom alignment; only a confirmed ramp bear proceeds to
    align/approach/grasp.

Run:  ros2 run rne_final_pkg ramp_bear_12
"""

import math
import time

import rclpy

from rne_final_pkg.scripted_final_mission import ScriptedFinalMission, S


# Short-side observe point 3 (1-based) -> zero-based index into short_side_observe.
_SHORT_SIDE_OBSERVE_POINT = 2


class RampBear12(ScriptedFinalMission):
    def __init__(self):
        super().__init__(node_name="ramp_bear_12")

    def _state_init(self):
        missing = []
        if self.pose is None:
            missing.append("pose (TF map->base_footprint or /amcl_pose)")
        if not self.yolo.bear_topic_alive():
            missing.append("/yolo/bear_info")
        if not self.yolo.ramp_topic_alive():
            missing.append("/yolo/bridge_info (ramp seg)")

        if not missing:
            x, y, yaw = self.pose
            self.get_logger().info(
                f"[INIT] ready  pose=({x:.2f},{y:.2f},{math.degrees(yaw):.0f}deg) "
                f"src={self._pose_src}"
            )
            # debug.start_state still wins (one-phase calibration); otherwise go
            # straight to the short-side observe chain's third point.
            if self._start_state is not S.TASK3_ROUTE_SEGMENT_1:
                self.get_logger().warn(
                    f"[INIT] debug.start_state set - jumping to {self._start_state.name} "
                    "in ramp_bear_12 flow"
                )
                self._goto(self._start_state)
                return
            chain = self.cfg["route"]["short_side_observe"]
            self._observe_idx = min(_SHORT_SIDE_OBSERVE_POINT, len(chain) - 1)
            self.get_logger().info(
                f"[INIT] starting directly at short-side observe point "
                f"{self._observe_idx + 1}/{len(chain)}"
            )
            self._goto(S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE)
            return

        elapsed = time.monotonic() - self._state_t0
        if elapsed > self.cfg["init"]["wait_timeout_s"]:
            self._fail(
                f"INIT timeout after {elapsed:.0f}s - missing: {', '.join(missing)}"
            )
            return
        self.get_logger().info(
            f"[INIT] waiting ({elapsed:.1f}s)  missing: {', '.join(missing)}"
        )

    def _tick(self):
        self._update_pose()
        if self._sea_guard():
            return
        s = self._state

        if s == S.INIT:
            self._state_init()
        elif s == S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE:
            self._state_move_observe(
                self.cfg["route"]["short_side_observe"], S.RAMP_SCAN_SHORT_SIDE
            )
        elif s == S.RAMP_SCAN_SHORT_SIDE:
            self._state_ramp_scan(
                self.cfg["route"]["short_side_observe"],
                move_state=S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE,
                found_next=S.RAMP_BEAR_CLASSIFY,
                exhausted_next=None,
            )
        elif s == S.RAMP_BEAR_CLASSIFY:
            # Classify the bear before aligning: a blocking bear is cleared and
            # re-classified first; only a confirmed ramp bear proceeds to align.
            self._state_bear_classify(grasp_state=S.RAMP_ALIGN_BOTTOM)
        elif s == S.CLEAR_BLOCKING_BEAR:
            self._state_clear_blocking_bear(
                S.RAMP_BEAR_CLASSIFY,
                lost_state=S.RAMP_BEAR_CLASSIFY,
            )
        elif s == S.RAMP_ALIGN_BOTTOM:
            self._state_ramp_align_bottom(S.RAMP_APPROACH)
        elif s == S.RAMP_APPROACH:
            self._state_ramp_approach(S.GRASP_RAMP_BEAR)
        elif s == S.GRASP_RAMP_BEAR:
            self._state_grasp_ramp_bear(S.RETURN_ORIGIN)
        elif s == S.RETURN_ORIGIN:
            self._state_return_origin(S.DONE)
        elif s == S.DONE:
            self.car.stop()
            if not self._done_logged:
                self.get_logger().info("Ramp bear 12 flow complete.")
                self._done_logged = True
        elif s == S.FAILED:
            self.car.stop()
        else:
            self._fail(f"ramp_bear_12 does not handle state {s.name}")


def main(args=None):
    rclpy.init(args=args)
    node = RampBear12()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.car.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
