"""
Run once to verify all required topics and TF frames are available.
Usage: ros2 run rne_final_pkg topic_check
"""
import rclpy
from rclpy.node import Node
import subprocess


REQUIRED_TOPICS = [
    "/car_C_rear_wheel",
    "/car_C_front_wheel",
    "/robot_arm",
    "/goal_pose",
    "/amcl_pose",
    "/plan",
    "/yolo/target_info",
    "/yolo/detection/compressed",
    "/camera/image/compressed",
    "/camera/depth/image_raw",
]

REQUIRED_TF_PAIRS = [
    ("map", "base_footprint"),
    ("map", "camera_optical_frame"),
]

CAMERA_INFO_CANDIDATES = [
    "/camera/image/camera_info",
    "/camera/color/camera_info",
    "/camera/depth/camera_info",
]


def check_topics():
    result = subprocess.run(["ros2", "topic", "list"], capture_output=True, text=True)
    available = set(result.stdout.splitlines())

    print("\n=== Topic Check ===")
    for t in REQUIRED_TOPICS:
        status = "OK" if t in available else "MISSING"
        print(f"  [{status}] {t}")

    print("\n=== CameraInfo candidates ===")
    for t in CAMERA_INFO_CANDIDATES:
        status = "FOUND" if t in available else "not present"
        print(f"  [{status}] {t}")


def check_tf():
    print("\n=== TF Frame Check ===")
    for source, target in REQUIRED_TF_PAIRS:
        proc = subprocess.run(
            ["ros2", "run", "tf2_ros", "tf2_echo", source, target, "--timeout", "2.0"],
            capture_output=True, text=True, timeout=5,
        )
        ok = "Translation" in proc.stdout
        status = "OK" if ok else "MISSING"
        print(f"  [{status}] {source} -> {target}")

    # arm_ik_base is optional but good to know
    proc = subprocess.run(
        ["ros2", "run", "tf2_ros", "tf2_echo", "map", "arm_ik_base", "--timeout", "2.0"],
        capture_output=True, text=True, timeout=5,
    )
    ok = "Translation" in proc.stdout
    status = "OK" if ok else "NOT FOUND (TF grab disabled)"
    print(f"  [{status}] map -> arm_ik_base")


class TopicCheckNode(Node):
    def __init__(self):
        super().__init__("topic_check")


def main(args=None):
    rclpy.init(args=args)
    node = TopicCheckNode()

    check_topics()
    try:
        check_tf()
    except Exception as e:
        print(f"\nTF check failed: {e}")

    print("\nDone.")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
