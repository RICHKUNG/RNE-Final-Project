"""Task 1+2 ramp bear flow without the Task 3 door sequence.

Subclasses ScriptedFinalMission and reuses its route, ramp, bear, grasp, and
return handlers.  The flow is:

    INIT (pose + /yolo/bear_info + /yolo/bridge_info)
      -> route.turn_point while watching for ramp
      -> long/short observe chain selected by the outbound ramp flag
      -> ramp approach -> bear classify/grasp -> return origin

Run:  ros2 run rne_final_pkg ramp_bear
"""

import math
import time

import rclpy

from rne_final_pkg.scripted_final_mission import ScriptedFinalMission, S


class RampBear(ScriptedFinalMission):
    def __init__(self):
        super().__init__(node_name="ramp_bear")

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
            if self._start_state is not S.TASK3_ROUTE_SEGMENT_1:
                self.get_logger().warn(
                    f"[INIT] debug.start_state set - jumping to {self._start_state.name} "
                    "in ramp_bear flow"
                )
            self._goto(self._start_state)
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

    def _state_route_to_turn_point(self):
        if not self._require_pose():
            return

        if not self._bridge_horizontal:
            ramp = self.cfg["ramp"]
            hits = self._accumulate_ramp_hits(ramp["outbound_area_threshold"])
            if hits >= ramp["outbound_required_frames"]:
                self._bridge_horizontal = True
                self.get_logger().info(
                    "[RAMP_BEAR] ramp seen before turn_point -> bridge is HORIZONTAL; "
                    "using the short-side observe chain"
                )

        if self._drive_to_point(self.cfg["route"]["turn_point"]):
            if self._bridge_horizontal:
                self.get_logger().info(
                    "[RAMP_BEAR] turn_point reached with horizontal flag -> short side"
                )
                self._goto(S.MOVE_TO_LONG_SHORT_CORNER)
            else:
                self.get_logger().info(
                    "[RAMP_BEAR] turn_point reached without ramp pre-check -> long side"
                )
                self._goto(S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE)

    def _tick(self):
        self._update_pose()
        if self._sea_guard():
            return
        s = self._state

        if s == S.INIT:
            self._state_init()
        elif s == S.TASK3_ROUTE_SEGMENT_1:
            self._state_route_to_turn_point()
        elif s == S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE:
            self._state_move_observe(
                self.cfg["route"]["long_side_observe"], S.RAMP_SCAN_LONG_SIDE
            )
        elif s == S.RAMP_SCAN_LONG_SIDE:
            self._state_ramp_scan(
                self.cfg["route"]["long_side_observe"],
                move_state=S.MOVE_TO_RAMP_OBSERVE_LONG_SIDE,
                found_next=S.RAMP_APPROACH,
                exhausted_next=S.MOVE_TO_LONG_SHORT_CORNER,
            )
        elif s == S.MOVE_TO_LONG_SHORT_CORNER:
            self._state_route(
                self.cfg["route"]["long_to_short_corner"],
                S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE,
            )
        elif s == S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE:
            self._state_move_observe(
                self.cfg["route"]["short_side_observe"], S.RAMP_SCAN_SHORT_SIDE
            )
        elif s == S.RAMP_SCAN_SHORT_SIDE:
            self._state_ramp_scan(
                self.cfg["route"]["short_side_observe"],
                move_state=S.MOVE_TO_RAMP_OBSERVE_SHORT_SIDE,
                found_next=S.RAMP_APPROACH,
                exhausted_next=None,
            )
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
                self.get_logger().info("Ramp bear flow complete.")
                self._done_logged = True
        elif s == S.FAILED:
            self.car.stop()
        else:
            self._fail(f"ramp_bear does not handle state {s.name}")


def main(args=None):
    rclpy.init(args=args)
    node = RampBear()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.car.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
