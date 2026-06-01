class BridgeAvoider:
    """
    Generates a wheel-speed override when a bridge is detected ahead.
    bridge_info: [found, delta_x, area_ratio, ...]  (from /yolo/bridge_info)
      found      : 1.0 = bridge visible
      delta_x    : pixels, positive = bridge right of image center
      area_ratio : bridge mask area / bottom-half image area
    Returns ([left, right], [left, right]) or None (no override).
    """

    AREA_WARNING = 0.04   # bridge visible but far — start steering
    AREA_DANGER  = 0.10   # bridge close — stop + hard turn

    DELTA_BAND   = 30.0   # px deadband for steering direction
    SLOW_SPEED   = 150.0
    TURN_SPEED   = 300.0

    def compute(self, bridge_info):
        if not bridge_info or len(bridge_info) < 3:
            return None
        if bridge_info[0] < 0.5:
            return None

        delta_x    = bridge_info[1]
        area_ratio = bridge_info[2]

        if area_ratio < self.AREA_WARNING:
            return None   # too far to worry about

        # Steer away from bridge side
        # delta_x > 0 → bridge right → turn left  (left=-T, right=+T)
        # delta_x < 0 → bridge left  → turn right (left=+T, right=-T)
        if delta_x >= self.DELTA_BAND:
            turn = [-self.TURN_SPEED, self.TURN_SPEED]
        else:
            turn = [self.TURN_SPEED, -self.TURN_SPEED]

        if area_ratio >= self.AREA_DANGER:
            # Immediate in-place rotation
            return turn, turn

        # Gradual: slow forward + bias toward turn direction
        v = [self.SLOW_SPEED + turn[0] * 0.5, self.SLOW_SPEED + turn[1] * 0.5]
        return v, v
