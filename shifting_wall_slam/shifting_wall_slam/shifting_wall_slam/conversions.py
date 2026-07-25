"""Small pose/quaternion/time conversions shared across nodes."""
import math


def yaw_from_quat(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def quat_from_yaw(yaw):
    """Return (x, y, z, w) for a rotation about Z."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def stamp_sec(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9
