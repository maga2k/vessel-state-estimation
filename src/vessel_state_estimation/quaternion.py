"""Quaternion utilities. Hamilton convention, scalar first: q = [w, x, y, z].

A quaternion ``q`` represents the rotation BODY -> NED:   v_ned = R(q) v_body.
Euler angles are aerospace ZYX (yaw psi, pitch theta, roll phi):  R = Rz(psi) Ry(theta) Rx(phi).
Frames: NED navigation frame, FRD body frame (x forward, y starboard, z down).
"""

from __future__ import annotations

import numpy as np


def quat_mult(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a (x) b (apply b first, then a, when used as rotations)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_from_euler(roll, pitch, yaw) -> np.ndarray:
    """ZYX Euler angles [rad] -> quaternion (vectorised: arrays in, (..., 4) out)."""
    cr, sr = np.cos(0.5 * np.asarray(roll)), np.sin(0.5 * np.asarray(roll))
    cp, sp = np.cos(0.5 * np.asarray(pitch)), np.sin(0.5 * np.asarray(pitch))
    cy, sy = np.cos(0.5 * np.asarray(yaw)), np.sin(0.5 * np.asarray(yaw))
    return np.stack([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], axis=-1)


def euler_from_quat(q: np.ndarray) -> np.ndarray:
    """Quaternion -> [roll, pitch, yaw] (ZYX), shape (..., 3)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.stack([roll, pitch, yaw], axis=-1)


def rot_from_quat(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R_nb (body -> NED) of a single unit quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def quat_from_rotvec(v: np.ndarray) -> np.ndarray:
    """Exponential map: rotation vector [rad] -> unit quaternion."""
    angle = float(np.linalg.norm(v))
    if angle < 1e-9:
        return quat_normalize(np.array([1.0, *(0.5 * v)]))
    return np.array([np.cos(0.5 * angle), *(np.sin(0.5 * angle) * v / angle)])


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
    """Logarithm map: unit quaternion -> rotation vector, shortest rotation (angle in [0, pi])."""
    q = q if q[0] >= 0.0 else -q
    vec = q[1:]
    s = float(np.linalg.norm(vec))
    if s < 1e-12:
        return 2.0 * vec
    return 2.0 * np.arctan2(s, q[0]) * vec / s


def skew(v: np.ndarray) -> np.ndarray:
    """Matrix [v]x such that [v]x u = v x u."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])