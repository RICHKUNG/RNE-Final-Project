"""bridge_align — shape-based ramp/bridge entry alignment (SKELETON).

Drives the car to square up to the ramp/bridge entrance using the segmentation
mask SHAPE, not just its centroid:

  1. CENTER  — rotate so the mask centroid sits on the image centre (kills the
               lateral-offset component of the apparent skew).
  2. DEYAW   — with the centroid centred, the residual centreline skew is mostly
               body yaw; micro-rotate in place to null it.
  3. ADVANCE — centred + de-yawed: creep forward with arc steering, falling back
               to CENTER/DEYAW (with hysteresis) if either error re-opens.

Why this order: a monocular mask's centreline shift conflates lateral offset and
yaw — they are not separately observable from one number.  Centring first removes
the lateral term, so the leftover skew is a usable yaw proxy.  See the bridge-align
design notes for the full argument.

Input  : /yolo/bridge_align  [found, center_delta_x(px), skew_score,
                              full_area_ratio, angle_hint, shape_conf]
Output : car_C_rear_wheel / car_C_front_wheel  (via CarDriver)

STATUS: skeleton. The state machine and transitions are real; every threshold
lives in config/bridge_align.yaml and is an UNCALIBRATED starting guess. The DONE
hand-off (what happens after alignment) is a stub — wire it to CROSS_BRIDGE.
"""

import os
import time

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from rne_final_pkg.car_driver import CarDriver


# /yolo/bridge_align field indices.
F_FOUND = 0
F_CENTER_DX = 1
F_SKEW = 2
F_AREA = 3
F_ANGLE = 4
F_CONF = 5


class BridgeAlign(Node):
    def __init__(self):
        super().__init__("bridge_align")

        share = get_package_share_directory("rne_final_pkg")
        cfg_path = os.path.join(share, "config", "bridge_align.yaml")
        with open(cfg_path, "r") as f:
            self.p = yaml.safe_load(f)

        self.car = CarDriver(self)
        self.create_subscription(
            Float32MultiArray, "/yolo/bridge_align", self._align_cb, 10
        )

        # Latest raw message + bookkeeping.
        self._raw = None
        self._seq = 0
        self._last_mono = None

        # EMA-smoothed signals (seg is ~1 Hz, so smoothing is light).
        # _skew is the SIGNED angle_hint (+ve = mask top leans right of bottom);
        # its magnitude is compared against skew_enter/skew_exit, its sign sets
        # the de-yaw rotation direction.
        self._center_dx = 0.0
        self._skew = 0.0
        self._conf = 0.0
        self._valid_streak = 0   # consecutive valid seg messages
        self._last_processed_seq = 0

        self.state = "SEARCH"
        self._state_since = time.monotonic()
        self._start_mono = time.monotonic()

        # 10 Hz control loop (faster than seg; it re-acts on the latched signals).
        self.create_timer(0.1, self._tick)
        self.get_logger().info("bridge_align started — state=SEARCH")

    # ── perception intake ────────────────────────────────────────────────
    def _align_cb(self, msg):
        self._raw = list(msg.data)
        self._seq += 1
        self._last_mono = time.monotonic()

    def _ingest(self):
        """Fold the newest seg message into the EMA signals + validity streak.
        Runs at most once per new message (seg << control rate)."""
        if self._raw is None or self._seq == self._last_processed_seq:
            return
        self._last_processed_seq = self._seq
        d = self._raw

        per = self.p["perception"]
        found = len(d) > F_FOUND and d[F_FOUND] == 1.0
        area = d[F_AREA] if len(d) > F_AREA else 0.0
        conf = d[F_CONF] if len(d) > F_CONF else 0.0
        usable = found and area >= per["min_full_area_ratio"]

        if not usable:
            self._valid_streak = 0
            self._conf = 0.0
            return

        ema = per["ema"]
        self._center_dx = ema * d[F_CENTER_DX] + (1 - ema) * self._center_dx
        # Fold in the SIGNED angle_hint (skew_score is its magnitude); only when
        # the shape is trustworthy (mask not cropped/short) so a bad frame can't
        # drag the smoothed yaw estimate.
        if conf >= per["min_shape_conf"] and len(d) > F_ANGLE:
            self._skew = ema * d[F_ANGLE] + (1 - ema) * self._skew
        self._conf = conf
        self._valid_streak += 1

    def _info_fresh(self):
        per = self.p["perception"]
        return (
            self._last_mono is not None
            and (time.monotonic() - self._last_mono) <= per["info_stale_s"]
        )

    def _confirmed(self):
        return self._valid_streak >= self.p["perception"]["valid_required_frames"]

    def _skew_trusted(self):
        return self._conf >= self.p["perception"]["min_shape_conf"]

    # ── control loop ─────────────────────────────────────────────────────
    def _set_state(self, new):
        if new != self.state:
            self.get_logger().info(f"{self.state} → {new}")
            self.state = new
            self._state_since = time.monotonic()

    def _tick(self):
        self._ingest()
        c = self.p["control"]

        if self.state in ("DONE", "FAILED"):
            self.car.stop()
            return

        # Lost the mask (stale or unconfirmed) → SEARCH, unless already there.
        have_signal = self._info_fresh() and self._confirmed()
        if not have_signal and self.state != "SEARCH":
            self._set_state("SEARCH")

        elapsed = time.monotonic() - self._start_mono
        if self.state != "SEARCH" and elapsed > self.p["timeouts"]["align_s"]:
            self._fail("alignment timed out")
            return

        handler = {
            "SEARCH": self._do_search,
            "CENTER": self._do_center,
            "DEYAW": self._do_deyaw,
            "ADVANCE": self._do_advance,
        }[self.state]
        handler(c)

    def _do_search(self, c):
        if self._info_fresh() and self._confirmed():
            self._set_state("CENTER")
            return
        if time.monotonic() - self._state_since > self.p["timeouts"]["search_s"]:
            self._fail("no usable bridge mask found")
            return
        # Slow scan rotation (TODO: bias toward the last-seen side).
        s = self.p["timeouts"]["search_turn_speed"]
        self.car.publish_velocities(-s, s)   # CCW in place

    def _do_center(self, c):
        dx = self._center_dx
        if abs(dx) <= c["center_px"]:
            self._set_state("DEYAW")
            return
        s = c["center_turn_speed"]
        # dx > 0 = mask right of centre → rotate CW to bring it back.
        if dx > 0:
            self.car.publish_velocities(s, -s)
        else:
            self.car.publish_velocities(-s, s)

    def _do_deyaw(self, c):
        # Re-open the centre error? Hand back to CENTER (with hysteresis).
        if abs(self._center_dx) > c["center_px"] + c["center_hysteresis_px"]:
            self._set_state("CENTER")
            return
        if not self._skew_trusted():
            # Mask cropped/short: can't measure yaw → just creep forward.
            self._set_state("ADVANCE")
            return
        if abs(self._skew) <= c["skew_exit"]:
            self._set_state("ADVANCE")
            return
        if abs(self._skew) < c["skew_enter"]:
            # Inside the hysteresis band — hold, don't chatter.
            self.car.stop()
            return
        s = c["deyaw_turn_speed"]
        # _skew is the smoothed signed angle_hint: +ve = mask top leans right of
        # bottom (car yawed left of the entrance) → rotate CW to square up.
        if self._skew > 0:
            self.car.publish_velocities(s, -s)
        else:
            self.car.publish_velocities(-s, s)

    def _do_advance(self, c):
        if self._raw is not None and self._raw[F_AREA] >= self.p["done"]["full_area_ratio"]:
            self._set_state("DONE")
            self.get_logger().info("bridge_align DONE — hand off to CROSS (stub)")
            return
        # Re-open either error → fall back.
        if abs(self._center_dx) > c["center_px"] + c["center_hysteresis_px"]:
            self._set_state("CENTER")
            return
        if self._skew_trusted() and abs(self._skew) > c["skew_enter"]:
            self._set_state("DEYAW")
            return
        # Arc steering: creep forward, steer proportional to residual skew.
        base = c["advance_speed"]
        steer = 0.0
        if self._skew_trusted():
            steer = c["advance_steer_gain"] * self._skew
            steer = max(-c["advance_steer_max"], min(c["advance_steer_max"], steer))
        self.car.publish_velocities(base + steer, base - steer)

    def _fail(self, why):
        self.get_logger().error(f"bridge_align FAILED: {why}")
        self.car.stop()
        self._set_state("FAILED")


def main(args=None):
    rclpy.init(args=args)
    node = BridgeAlign()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.car.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
