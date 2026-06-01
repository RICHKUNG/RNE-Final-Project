import math


class WallFollower:
    """
    Right-hand wall following using LaserScan.
    Returns [left, right] wheel speed pairs; caller publishes to both
    front and rear topics.
    """

    FRONT_SAFE   = 0.55   # m: turn away if closer
    RIGHT_MIN    = 0.40   # m: too close, veer left
    RIGHT_MAX    = 0.75   # m: too far, veer right
    LEFT_MIN     = 0.35   # m: don't curve left if left wall too close

    SPEED      = 220.0   # straight forward
    CURVE_DIFF = 70.0    # gentle differential — was 130 (caused hard left crash)
    ROT_SPEED  = 300.0   # in-place rotation

    def compute(self, scan):
        """
        scan: sensor_msgs/LaserScan
        Returns (wheels, wheels, debug_str) or None if scan is empty.
        wheels = [left, right]
        """
        ranges = list(scan.ranges)
        if not ranges:
            return None

        # Filter out self-hit readings below the sensor's stated minimum range
        rmin = scan.range_min
        ranges = [r if r >= rmin else float('inf') for r in ranges]

        a0  = scan.angle_min
        inc = scan.angle_increment

        front = self._sector_min(ranges, a0, inc, -20,  20)
        right = self._sector_min(ranges, a0, inc, -110, -70)
        left  = self._sector_min(ranges, a0, inc,   70, 110)

        if front < self.FRONT_SAFE:
            # Rotate toward the more open side
            if left >= right:
                v      = [-self.ROT_SPEED, self.ROT_SPEED]
                action = f"ROTATE_LEFT  front={front:.2f}m  l={left:.2f} r={right:.2f}"
            else:
                v      = [self.ROT_SPEED, -self.ROT_SPEED]
                action = f"ROTATE_RIGHT front={front:.2f}m  l={left:.2f} r={right:.2f}"
        elif right > self.RIGHT_MAX:
            v      = [self.SPEED, self.SPEED - self.CURVE_DIFF]
            action = f"CURVE_RIGHT  right={right:.2f}m > {self.RIGHT_MAX}m"
        elif right < self.RIGHT_MIN:
            if left < self.LEFT_MIN:
                # Both sides blocked — go straight to avoid crashing left
                v      = [self.SPEED, self.SPEED]
                action = f"STRAIGHT(narrow) r={right:.2f}m l={left:.2f}m"
            else:
                v      = [self.SPEED - self.CURVE_DIFF, self.SPEED]
                action = f"CURVE_LEFT   right={right:.2f}m  l={left:.2f}m"
        else:
            v      = [self.SPEED, self.SPEED]
            action = f"STRAIGHT     right={right:.2f}m front={front:.2f}m"

        return v, v, action

    def _sector_min(self, ranges, angle_min, angle_inc, deg_from, deg_to):
        """Minimum valid range in the given degree arc."""
        n  = len(ranges)
        i0 = max(0, min(n - 1, round((math.radians(deg_from) - angle_min) / angle_inc)))
        i1 = max(0, min(n - 1, round((math.radians(deg_to)   - angle_min) / angle_inc)))
        if i0 > i1:
            i0, i1 = i1, i0
        valid = [
            ranges[i] for i in range(i0, i1 + 1)
            if not (math.isnan(ranges[i]) or math.isinf(ranges[i])) and ranges[i] > 0
        ]
        return min(valid) if valid else 10.0
