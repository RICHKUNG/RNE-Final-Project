import time

from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker


class YoloClient:
    def __init__(self, node):
        self._info = None
        self._marker = None
        self._blocking_marker = None
        self._ramp_marker = None
        self._bridge_info = None
        self._bridge_seq = 0
        self._bridge_mono = None
        self._bridge_align_info = None
        self._bridge_align_seq = 0
        self._bridge_align_mono = None
        self._bear_info = None
        self._bear_seq = 0
        self._bear_mono = None
        self._knob_info = None

        node.create_subscription(Float32MultiArray, "/yolo/target_info", self._info_cb, 10)
        node.create_subscription(Marker, "/yolo/target_marker", self._marker_cb, 10)
        node.create_subscription(Marker, "/yolo/blocking_bear_marker", self._blocking_marker_cb, 10)
        node.create_subscription(Marker, "/yolo/ramp_bear_marker", self._ramp_marker_cb, 10)
        node.create_subscription(Float32MultiArray, "/yolo/bridge_info", self._bridge_cb, 10)
        node.create_subscription(Float32MultiArray, "/yolo/bridge_align", self._bridge_align_cb, 10)
        node.create_subscription(Float32MultiArray, "/yolo/bear_info", self._bear_cb, 10)
        node.create_subscription(Float32MultiArray, "/yolo/knob_info", self._knob_cb, 10)

    def _info_cb(self, msg):
        self._info = list(msg.data)

    def _marker_cb(self, msg):
        self._marker = msg

    def _blocking_marker_cb(self, msg):
        self._blocking_marker = msg

    def _ramp_marker_cb(self, msg):
        self._ramp_marker = msg

    def _bridge_cb(self, msg):
        self._bridge_info = list(msg.data)
        self._bridge_seq += 1
        self._bridge_mono = time.monotonic()

    def _bridge_align_cb(self, msg):
        self._bridge_align_info = list(msg.data)
        self._bridge_align_seq += 1
        self._bridge_align_mono = time.monotonic()

    def _bear_cb(self, msg):
        self._bear_info = list(msg.data)
        self._bear_seq += 1
        self._bear_mono = time.monotonic()

    def _knob_cb(self, msg):
        self._knob_info = list(msg.data)

    def target_topic_alive(self):
        return self._info is not None

    def is_visible(self):
        return self._info is not None and len(self._info) >= 1 and self._info[0] == 1.0

    def distance(self):
        return self._info[1] if self._info and len(self._info) >= 2 else float("inf")

    def delta_x(self):
        return self._info[2] if self._info and len(self._info) >= 3 else 0.0

    def marker(self, group=None):
        if group == "blocking":
            return self._blocking_marker
        if group == "ramp":
            return self._ramp_marker
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

    # ── ramp accessors ────────────────────────────────────────────────
    # The seg model now detects the ramp face; data still arrives on the
    # legacy /yolo/bridge_info topic. New publishers send:
    # [legacy_bottom_found, dx, bottom_area_ratio, full_area_ratio,
    #  bottom_edge_ratio, center_box_overlap_ratio].

    def ramp_topic_alive(self):
        return self._bridge_info is not None

    def ramp_seq(self):
        """Monotonic count of seg messages received. Seg is timer-driven and
        slower than the 10 Hz control loop, so confirmation logic must count
        distinct messages, not control ticks re-reading the same sticky value."""
        return self._bridge_seq

    def ramp_age_s(self):
        return time.monotonic() - self._bridge_mono if self._bridge_mono is not None else None

    def ramp_visible(self):
        return self.bridge_visible() or self.ramp_full_area_ratio() > 0.0

    def ramp_delta_x(self):
        return self.bridge_delta_x()

    def ramp_bottom_area_ratio(self):
        return self.bridge_area_ratio()

    def ramp_full_area_ratio(self):
        return self._bridge_info[3] if self._bridge_info and len(self._bridge_info) >= 4 else 0.0

    def ramp_bottom_edge_ratio(self):
        """Ramp mask's lowest row, normalised to image height (0=top, 1=frame
        bottom). 0.0 when the publisher predates this field (4-field message)."""
        return self._bridge_info[4] if self._bridge_info and len(self._bridge_info) >= 5 else 0.0

    def ramp_center_overlap_ratio(self):
        return self._bridge_info[5] if self._bridge_info and len(self._bridge_info) >= 6 else 0.0

    def ramp_area_ratio(self):
        # New ramp publishers append full-frame mask area at index 3.  Keep old
        # three-field messages usable by falling back to the legacy bottom-half
        # area at index 2.
        return max(self.ramp_bottom_area_ratio(), self.ramp_full_area_ratio())

    # ── bridge_align accessors ────────────────────────────────────────
    # /yolo/bridge_align:
    # [found, center_delta_x, skew_score, full_area_ratio, angle_hint, shape_conf]

    def align_topic_alive(self):
        return self._bridge_align_info is not None

    def align_seq(self):
        return self._bridge_align_seq

    def align_age_s(self):
        if self._bridge_align_mono is None:
            return None
        return time.monotonic() - self._bridge_align_mono

    def align_visible(self):
        return (
            self._bridge_align_info is not None
            and len(self._bridge_align_info) >= 1
            and self._bridge_align_info[0] == 1.0
        )

    def align_center_dx(self):
        if self._bridge_align_info and len(self._bridge_align_info) >= 2:
            return self._bridge_align_info[1]
        return self.ramp_delta_x()

    def align_skew(self):
        return (
            self._bridge_align_info[2]
            if self._bridge_align_info and len(self._bridge_align_info) >= 3
            else 0.0
        )

    def align_full_area_ratio(self):
        if self._bridge_align_info and len(self._bridge_align_info) >= 4:
            return self._bridge_align_info[3]
        return self.ramp_full_area_ratio()

    def align_angle_hint(self):
        return (
            self._bridge_align_info[4]
            if self._bridge_align_info and len(self._bridge_align_info) >= 5
            else 0.0
        )

    def align_shape_conf(self):
        return (
            self._bridge_align_info[5]
            if self._bridge_align_info and len(self._bridge_align_info) >= 6
            else 0.0
        )

    # ── bear_info accessors ───────────────────────────────────────────
    # /yolo/bear_info: [found, distance, delta_x, pixel_x, pixel_y, on_ramp]

    def bear_topic_alive(self):
        return self._bear_info is not None

    def bear_seq(self):
        return self._bear_seq

    def bear_age_s(self):
        return time.monotonic() - self._bear_mono if self._bear_mono is not None else None

    def bear_visible(self):
        return self._bear_info is not None and len(self._bear_info) >= 1 and self._bear_info[0] == 1.0

    def bear_distance(self):
        return self._bear_info[1] if self._bear_info and len(self._bear_info) >= 2 else float("inf")

    def bear_delta_x(self):
        return self._bear_info[2] if self._bear_info and len(self._bear_info) >= 3 else 0.0

    def bear_pixel_y(self):
        return self._bear_info[4] if self._bear_info and len(self._bear_info) >= 5 else 0.0

    def bear_on_ramp(self):
        """1.0 on ramp, 0.0 blocking, -1.0 unknown (no ramp mask / old detector)."""
        return self._bear_info[5] if self._bear_info and len(self._bear_info) >= 6 else -1.0

    # Per-group bears (publisher splits by bbox centre-y vs ramp centre-y).
    # data[6:11] = nearest blocking (under-bridge) bear: found, dist, delta_x, px, py
    # data[11:16]= nearest on-ramp (on-bridge) bear:     found, dist, delta_x, px, py
    # These are deterministic per frame, so a consumer can target one group
    # without the overall-nearest target flipping between two visible bears.
    def _group(self, base):
        if self._bear_info and len(self._bear_info) >= base + 5:
            return self._bear_info[base:base + 5]
        return None

    def blocking_bear_visible(self):
        g = self._group(6)
        return g is not None and g[0] == 1.0

    def blocking_bear_distance(self):
        g = self._group(6)
        return g[1] if g and g[0] == 1.0 else float("inf")

    def blocking_bear_delta_x(self):
        g = self._group(6)
        return g[2] if g and g[0] == 1.0 else 0.0

    def blocking_bear_pixel_y(self):
        g = self._group(6)
        return g[4] if g and g[0] == 1.0 else 0.0

    def ramp_bear_visible(self):
        g = self._group(11)
        return g is not None and g[0] == 1.0

    def ramp_bear_distance(self):
        g = self._group(11)
        return g[1] if g and g[0] == 1.0 else float("inf")

    def ramp_bear_delta_x(self):
        g = self._group(11)
        return g[2] if g and g[0] == 1.0 else 0.0

    def ramp_bear_pixel_y(self):
        g = self._group(11)
        return g[4] if g and g[0] == 1.0 else 0.0

    # ── knob_info accessors ───────────────────────────────────────────
    # /yolo/knob_info: [found, distance, delta_x, pixel_x, pixel_y, area, conf]
    # distance = -1.0 when depth is invalid.

    def knob_topic_alive(self):
        return self._knob_info is not None

    def knob_visible(self):
        return self._knob_info is not None and len(self._knob_info) >= 1 and self._knob_info[0] == 1.0

    def knob_distance(self):
        return self._knob_info[1] if self._knob_info and len(self._knob_info) >= 2 else -1.0

    def knob_delta_x(self):
        return self._knob_info[2] if self._knob_info and len(self._knob_info) >= 3 else 0.0
