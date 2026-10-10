import numpy as np

from vessel_state_estimation.quaternion import (euler_from_quat, quat_conj, quat_from_euler,
                                                quat_from_rotvec, quat_mult, quat_normalize,
                                                rot_from_quat, rotvec_from_quat, skew)


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def random_q(rng):
    return quat_normalize(rng.normal(size=4))


def test_euler_round_trip_and_zyx_rotation_matrix():
    rng = np.random.default_rng(0)
    for _ in range(50):
        roll, pitch, yaw = rng.uniform(-3.0, 3.0), rng.uniform(-1.4, 1.4), rng.uniform(-3.0, 3.0)
        q = quat_from_euler(roll, pitch, yaw)
        np.testing.assert_allclose(euler_from_quat(q), [roll, pitch, yaw], atol=1e-9)
        np.testing.assert_allclose(rot_from_quat(q), Rz(yaw) @ Ry(pitch) @ Rx(roll), atol=1e-12)


def test_rotation_matrix_is_orthonormal_and_composition_matches():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = random_q(rng), random_q(rng)
        R = rot_from_quat(a)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12
        np.testing.assert_allclose(rot_from_quat(quat_mult(a, b)), rot_from_quat(a) @ rot_from_quat(b),
                                   atol=1e-12)
        np.testing.assert_allclose(rot_from_quat(quat_conj(a)), R.T, atol=1e-12)


def test_exp_log_round_trip_and_rodrigues():
    rng = np.random.default_rng(2)
    for angle in (1e-10, 1e-4, 0.3, 1.5, 3.0):
        axis = rng.normal(size=3)
        v = angle * axis / np.linalg.norm(axis)
        q = quat_from_rotvec(v)
        np.testing.assert_allclose(rotvec_from_quat(q), v, atol=1e-9)
        np.testing.assert_allclose(rotvec_from_quat(-q), v, atol=1e-9)  # q and -q: same rotation
        K = skew(v / max(angle, 1e-30)) if angle > 1e-6 else np.zeros((3, 3))
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        if angle > 1e-6:
            np.testing.assert_allclose(rot_from_quat(q), R, atol=1e-10)


def test_skew_is_the_cross_product():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(skew(a) @ b, np.cross(a, b), atol=1e-14)