"""Multiplicative EKF (error-state) for attitude: gyroscope + accelerometer.

Nominal state:  q (unit quaternion, body -> NED),  b (gyro bias, body axes)
Error state (6): dx = [dtheta, db]   with   q_true = q (x) dq(dtheta),  dq ~ [1, dtheta / 2]
    dtheta  small rotation expressed in BODY axes;  b_true = b + db

The covariance lives in the 6-dim error space, so it stays well-conditioned while the 4-dim
quaternion keeps its unit norm (a plain EKF on q would need a normalisation hack).

Propagation (gyro omega_m = omega + b + n_g):
    q  <- q (x) exp((omega_m - b) dt)                   exact for a constant rate
    dtheta[k+1] = R(w dt)^T dtheta[k] - db dt - n_g dt   w = omega_m - b
    db[k+1] = db[k] + n_b
Accelerometer update (gravity direction), f_m = -R^T g_ned + linear acceleration + noise
(the centripetal part omega x v of the linear acceleration is compensated with the GPS speed):
    f_hat = -G * R(q)^T e3,   y = f_m - f_hat,   H = [ [f_hat]x , 0 ]
Yaw is NOT observable from gravity: its covariance grows until a heading sensor is added.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..config import ImuConfig, MekfConfig
from ..quaternion import (quat_from_rotvec, quat_mult, quat_normalize, rot_from_quat,
                          skew)
from ..sensors.base import Measurement
from ..sim.attitude import G
from .base import Innovation

ATT, BIAS = slice(0, 3), slice(3, 6)


def initial_covariance(cfg: MekfConfig) -> np.ndarray:
    return np.diag([cfg.p0_att**2] * 3 + [cfg.p0_gyro_bias**2] * 3)


def draw_initial_estimate(q_true: np.ndarray, P0: np.ndarray,
                          rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """q0 = q_true (x) exp(error), error ~ N(0, P0_att) in body axes; bias estimate starts at 0."""
    w = rng.standard_normal(6)  # always 6 draws
    dtheta = np.sqrt(np.diag(P0)[:3]) * w[:3]
    return quat_normalize(quat_mult(q_true, quat_from_rotvec(dtheta))), np.zeros(3)


class Mekf:
    def __init__(self, cfg: MekfConfig, imu_assumed: ImuConfig, q0: np.ndarray, b0: np.ndarray,
                 P0: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.imu = imu_assumed  # what the filter BELIEVES about the gyro noise
        self.q = quat_normalize(q0.astype(float))
        self.b = b0.astype(float).copy()
        self.P = initial_covariance(cfg) if P0 is None else P0.astype(float).copy()
        self._gate_cache: dict[int, float] = {}
        self._q_cache: dict[float, np.ndarray] = {}  # Q depends only on dt

    def _process_noise(self, dt: float) -> np.ndarray:
        key = round(dt, 9)
        Q = self._q_cache.get(key)
        if Q is None:
            if len(self._q_cache) > 64:
                self._q_cache.clear()
            qs = self.cfg.q_scale
            Q = np.zeros((6, 6))
            Q[ATT, ATT] = (qs * self.imu.gyro_noise_density) ** 2 * dt * np.eye(3)
            Q[BIAS, BIAS] = (qs * self.imu.gyro_bias_rw) ** 2 * dt * np.eye(3)
            self._q_cache[key] = Q
        return Q

    # ------------------------------------------------------------------ prediction
    def error_transition(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        """F of the error state [dtheta, db] over one step (checked against finite differences)."""
        R_dq = rot_from_quat(quat_from_rotvec((gyro - self.b) * dt))  # rotation by w dt
        F = np.eye(6)
        F[ATT, ATT] = R_dq.T
        F[ATT, BIAS] = -dt * np.eye(3)
        return F

    def predict(self, gyro: np.ndarray, dt: float) -> None:
        F = self.error_transition(gyro, dt)
        self.q = quat_normalize(quat_mult(self.q, quat_from_rotvec((gyro - self.b) * dt)))
        P = F @ self.P @ F.T + self._process_noise(dt)
        self.P = 0.5 * (P + P.T)

    # ------------------------------------------------------------------ accelerometer update
    def expected_specific_force(self) -> np.ndarray:
        return -G * rot_from_quat(self.q).T @ np.array([0.0, 0.0, 1.0])

    def update_accel(self, f_meas: np.ndarray, sigma_noise: float, gyro: np.ndarray | None = None,
                     speed: float = 0.0) -> Innovation:
        """Fuse the accelerometer. If ``turn_compensation`` is on and a gyro sample and the ground
        speed are given, the centripetal acceleration of the turn, omega x [SOG, 0, 0], is removed
        first: a vessel in a turn would otherwise read a tilted gravity (coordinated-turn error).
        """
        if self.cfg.turn_compensation and gyro is not None and speed > 0.0:
            w = gyro - self.b
            f_meas = f_meas - np.array([0.0, w[2] * speed, -w[1] * speed])
        f_hat = self.expected_specific_force()
        y = f_meas - f_hat
        H = np.zeros((3, 6))
        H[:, ATT] = skew(f_hat)
        R = (sigma_noise**2 + self.cfg.accel_lin_sigma**2) * np.eye(3)
        S = H @ self.P @ H.T + R
        if self.cfg.gate_prob > 0.0:
            if 3 not in self._gate_cache:
                self._gate_cache[3] = float(chi2.ppf(self.cfg.gate_prob, 3))
            if float(y @ np.linalg.solve(S, y)) > self._gate_cache[3]:
                return Innovation(y, S, accepted=False)
        K = np.linalg.solve(S, H @ self.P).T
        dx = K @ y
        IKH = np.eye(6) - K @ H
        P = IKH @ self.P @ IKH.T + K @ R @ K.T
        # inject the correction into the nominal state, then RESET the error state to zero
        self.q = quat_normalize(quat_mult(self.q, quat_from_rotvec(dx[ATT])))
        self.b = self.b + dx[BIAS]
        Greset = np.eye(6)
        Greset[ATT, ATT] = np.eye(3) - skew(0.5 * dx[ATT])
        P = Greset @ P @ Greset.T
        self.P = 0.5 * (P + P.T)
        return Innovation(y, S)