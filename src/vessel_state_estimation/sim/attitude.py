"""Synthetic 3D attitude truth: planar maneuver (heading) + wave-induced roll / pitch / accelerations.

The 3-DOF planar model has no roll, pitch or heave. Here they are added as narrowband random
processes (sums of sinusoids), so the attitude and the accelerations are known ANALYTICALLY: no
numerical differentiation is involved in the ground truth.

Produced for every sample of a planar ``Trajectory``:
    q        body -> NED quaternion (ZYX Euler: heading from the maneuver, roll/pitch from waves)
    omega_b  true angular velocity in body axes [rad/s]
    f_b      true specific force in body axes [m/s^2] = R^T (a_ned - g_ned),  g_ned = [0, 0, +g]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import WaveConfig
from ..quaternion import euler_from_quat, quat_from_euler
from .truth import Trajectory

G = 9.81  # [m/s^2]


class Narrowband:
    """x(t) = sum_k a sin(w_k t + phi_k): zero mean, variance sigma^2, energy around ``period``."""

    def __init__(self, sigma: float, period: float, bandwidth: float, n: int,
                 rng: np.random.Generator) -> None:
        w0 = 2.0 * np.pi / period
        self.w = w0 * (1.0 + bandwidth * (2.0 * rng.random(n) - 1.0))  # always 2n draws
        self.phi = 2.0 * np.pi * rng.random(n)
        self.a = sigma * np.sqrt(2.0 / n)

    def value(self, t: np.ndarray) -> np.ndarray:
        return self.a * np.sin(np.outer(t, self.w) + self.phi).sum(axis=1)

    def rate(self, t: np.ndarray) -> np.ndarray:
        return self.a * (self.w * np.cos(np.outer(t, self.w) + self.phi)).sum(axis=1)


@dataclass(frozen=True)
class AttitudeSample:
    t: float
    q: np.ndarray  # (4,) body -> NED
    omega_b: np.ndarray  # (3,) rad/s
    f_b: np.ndarray  # (3,) m/s^2


@dataclass
class AttitudeTruth:
    t: np.ndarray  # (n,)
    q: np.ndarray  # (n, 4)
    euler: np.ndarray  # (n, 3)  [roll, pitch, yaw]
    omega_b: np.ndarray  # (n, 3)
    f_b: np.ndarray  # (n, 3)
    a_ned: np.ndarray  # (n, 3)  linear acceleration

    def __len__(self) -> int:
        return len(self.t)

    def sample(self, k: int) -> AttitudeSample:
        return AttitudeSample(float(self.t[k]), self.q[k], self.omega_b[k], self.f_b[k])


def generate_attitude(traj: Trajectory, cfg: WaveConfig, rng: np.random.Generator) -> AttitudeTruth:
    t, psi, r = traj.array("t"), traj.array("psi"), traj.array("r")
    args = (cfg.peak_period, cfg.bandwidth, cfg.n_components)
    roll_p = Narrowband(np.radians(cfg.roll_sigma_deg), *args, rng)
    pitch_p = Narrowband(np.radians(cfg.pitch_sigma_deg), *args, rng)
    acc_p = [Narrowband(cfg.accel_sigma, *args, rng) for _ in range(3)]  # N, E, D

    phi, dphi = roll_p.value(t), roll_p.rate(t)
    theta, dtheta = pitch_p.value(t), pitch_p.rate(t)
    dpsi = r  # yaw rate of the planar maneuver

    # Euler rates -> body angular velocity (ZYX)
    cphi, sphi, cth, sth = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta)
    omega_b = np.stack([dphi - dpsi * sth,
                        dtheta * cphi + dpsi * cth * sphi,
                        -dtheta * sphi + dpsi * cth * cphi], axis=1)

    q = quat_from_euler(phi, theta, psi)

    # linear acceleration in NED: waves + (optionally) the horizontal acceleration of the maneuver
    a_ned = np.stack([p.value(t) for p in acc_p], axis=1)
    if cfg.include_maneuver_accel:
        f_x = traj.array("u_dot") - traj.array("v") * r
        f_y = traj.array("v_dot") + traj.array("u") * r
        a_ned[:, 0] += np.cos(psi) * f_x - np.sin(psi) * f_y
        a_ned[:, 1] += np.sin(psi) * f_x + np.cos(psi) * f_y

    # specific force in body axes: f_b = R^T (a - g),  g = [0, 0, +G] (down)
    f_ned = a_ned - np.array([0.0, 0.0, G])
    sy, cy = np.sin(psi), np.cos(psi)
    R = np.empty((len(t), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 0, 2] = cy * cth, cy * sth * sphi - sy * cphi, cy * sth * cphi + sy * sphi
    R[:, 1, 0], R[:, 1, 1], R[:, 1, 2] = sy * cth, sy * sth * sphi + cy * cphi, sy * sth * cphi - cy * sphi
    R[:, 2, 0], R[:, 2, 1], R[:, 2, 2] = -sth, cth * sphi, cth * cphi
    f_b = np.einsum("nji,nj->ni", R, f_ned)  # R^T f_ned
    return AttitudeTruth(t, q, euler_from_quat(q), omega_b, f_b, a_ned)