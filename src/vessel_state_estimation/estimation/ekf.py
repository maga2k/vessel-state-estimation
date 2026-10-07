"""Planar EKF: IMU as input (prediction), GPS and compass as measurements (update).

State (8):  x = [N, E, psi, vN, vE, b_ax, b_ay, b_g]
    N, E        position, NED [m]
    psi         heading [rad]
    vN, vE      ground velocity, NED [m/s]
    b_ax, b_ay  accelerometer bias, body frame [m/s^2]
    b_g         gyro bias [rad/s]

Input u = IMU = [f_x, f_y, omega_z]  (specific force in body axes, yaw rate).

Continuous model (n_* = IMU white noise, w_* = bias random walks):
    psi_dot = (omega - b_g) - n_g
    a_N = cos(psi) (f_x - b_ax - n_ax) - sin(psi) (f_y - b_ay - n_ay)
    a_E = sin(psi) (f_x - b_ax - n_ax) + cos(psi) (f_y - b_ay - n_ay)
    N_dot = vN,  E_dot = vE,  vN_dot = a_N,  vE_dot = a_E,  b_dot = w

Discretisation: constant acceleration over the step, rotated with the heading at the middle
of the step (second order in dt; a frozen start-of-step heading drifts in fast turns).
"""

from __future__ import annotations

import numpy as np

from ..angles import wrap_angle
from ..config import EkfConfig, ImuConfig
from ..sensors.base import Measurement
from ..sim.truth import TruthSample
from .base import Estimator, Innovation

POS_N, POS_E, PSI, VEL_N, VEL_E, B_AX, B_AY, B_G = range(8)
NX = 8


def _corrected_acceleration(x: np.ndarray, u: np.ndarray, dt: float):
    """Bias-corrected specific force rotated to NED with the MID-STEP heading.

    Returns (a_N, a_E, cos, sin, w) where cos/sin are of psi_mid = psi + w dt / 2.
    """
    f_x, f_y, w = u[0] - x[B_AX], u[1] - x[B_AY], u[2] - x[B_G]
    psi_mid = x[PSI] + 0.5 * w * dt
    c, s = np.cos(psi_mid), np.sin(psi_mid)
    return c * f_x - s * f_y, s * f_x + c * f_y, c, s, w


def propagate(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """Noise-free discrete state transition x[k+1] = f(x[k], u[k])."""
    a_n, a_e, _, _, w = _corrected_acceleration(x, u, dt)
    xn = x.copy()
    xn[POS_N] = x[POS_N] + x[VEL_N] * dt + 0.5 * a_n * dt * dt
    xn[POS_E] = x[POS_E] + x[VEL_E] * dt + 0.5 * a_e * dt * dt
    xn[VEL_N] = x[VEL_N] + a_n * dt
    xn[VEL_E] = x[VEL_E] + a_e * dt
    xn[PSI] = wrap_angle(x[PSI] + w * dt)
    return xn


def transition_jacobian(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
    """F = d f / d x, derived by hand (checked against finite differences in the tests)."""
    a_n, a_e, c, s, _ = _corrected_acceleration(x, u, dt)  # c, s: mid-step heading
    h = 0.5 * dt * dt
    F = np.eye(NX)
    F[POS_N, VEL_N] = dt
    F[POS_E, VEL_E] = dt
    # d a / d psi (psi_mid moves 1:1 with psi):  d a_N = -a_E,  d a_E = +a_N
    F[POS_N, PSI], F[POS_E, PSI] = -h * a_e, h * a_n
    F[VEL_N, PSI], F[VEL_E, PSI] = -dt * a_e, dt * a_n
    # d a / d b_a:  d a_N = (-c, +s),  d a_E = (-s, -c)
    F[POS_N, B_AX], F[POS_N, B_AY] = -h * c, h * s
    F[POS_E, B_AX], F[POS_E, B_AY] = -h * s, -h * c
    F[VEL_N, B_AX], F[VEL_N, B_AY] = -dt * c, dt * s
    F[VEL_E, B_AX], F[VEL_E, B_AY] = -dt * s, -dt * c
    # d a / d b_g: psi_mid = psi + (omega - b_g) dt / 2  ->  d psi_mid / d b_g = -dt / 2
    F[POS_N, B_G], F[POS_E, B_G] = 0.5 * h * dt * a_e, -0.5 * h * dt * a_n
    F[VEL_N, B_G], F[VEL_E, B_G] = 0.5 * dt * dt * a_e, -0.5 * dt * dt * a_n
    F[PSI, B_G] = -dt
    return F


def process_noise(dt: float, imu: ImuConfig, q_scale: float = 1.0) -> np.ndarray:
    """Discrete process noise Q = G diag(sigma^2) G^T + bias random walks.

    Per-sample IMU noise: sigma = density / sqrt(dt). Because the noise enters through a rotation
    matrix and both accelerometer axes share the same sigma, Q does not depend on psi.
    The (tiny) coupling of gyro noise into velocity through the mid-step heading is neglected.
    """
    sa2 = (q_scale * imu.accel_noise_density) ** 2 / dt
    sg2 = (q_scale * imu.gyro_noise_density) ** 2 / dt
    Q = np.zeros((NX, NX))
    pos, vel = [POS_N, POS_E], [VEL_N, VEL_E]
    Q[np.ix_(pos, pos)] = 0.25 * dt**4 * sa2 * np.eye(2)
    Q[np.ix_(pos, vel)] = 0.5 * dt**3 * sa2 * np.eye(2)
    Q[np.ix_(vel, pos)] = 0.5 * dt**3 * sa2 * np.eye(2)
    Q[np.ix_(vel, vel)] = dt**2 * sa2 * np.eye(2)
    Q[PSI, PSI] = dt**2 * sg2
    Q[B_AX, B_AX] = Q[B_AY, B_AY] = (q_scale * imu.accel_bias_rw) ** 2 * dt
    Q[B_G, B_G] = (q_scale * imu.gyro_bias_rw) ** 2 * dt
    return Q


def initial_covariance(cfg: EkfConfig) -> np.ndarray:
    return np.diag([cfg.p0_pos**2] * 2 + [cfg.p0_psi**2] + [cfg.p0_vel**2] * 2
                   + [cfg.p0_accel_bias**2] * 2 + [cfg.p0_gyro_bias**2])


def draw_initial_estimate(truth0: TruthSample, P0: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """Initial estimate: truth + error drawn from N(0, P0); biases start at 0 (their prior mean)."""
    w = rng.standard_normal(NX)  # always 8 draws
    x0 = np.array([truth0.n, truth0.e, truth0.psi, truth0.vn, truth0.ve, 0.0, 0.0, 0.0])
    err = np.sqrt(np.diag(P0)) * w
    err[[B_AX, B_AY, B_G]] = 0.0
    x0 = x0 + err
    x0[PSI] = wrap_angle(x0[PSI])
    return x0


class PlanarEkf(Estimator):
    def __init__(self, cfg: EkfConfig, imu_assumed: ImuConfig, x0: np.ndarray,
                 P0: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.imu = imu_assumed  # what the filter BELIEVES about the IMU noise
        self._x = x0.astype(float).copy()
        self._P = initial_covariance(cfg) if P0 is None else P0.astype(float).copy()

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def P(self) -> np.ndarray:
        return self._P

    def predict(self, u: np.ndarray, dt: float) -> None:
        F = transition_jacobian(self._x, u, dt)
        Q = process_noise(dt, self.imu, self.cfg.q_scale)
        self._x = propagate(self._x, u, dt)
        P = F @ self._P @ F.T + Q
        self._P = 0.5 * (P + P.T)

    def update(self, meas: Measurement) -> Innovation:
        H = np.zeros((len(meas.z), NX))
        if meas.sensor == "gps":  # z = [N, E, vN, vE]
            H[0, POS_N], H[1, POS_E], H[2, VEL_N], H[3, VEL_E] = 1.0, 1.0, 1.0, 1.0
            y = meas.z - H @ self._x
        elif meas.sensor == "compass":  # z = [psi]
            H[0, PSI] = 1.0
            y = np.array([float(wrap_angle(meas.z[0] - self._x[PSI]))])
        else:
            raise ValueError(f"EKF cannot use sensor '{meas.sensor}'")
        R = self.cfg.r_scale**2 * meas.R
        S = H @ self._P @ H.T + R
        K = np.linalg.solve(S, H @ self._P).T  # P H^T S^-1  (S, P symmetric)
        self._x = self._x + K @ y
        self._x[PSI] = wrap_angle(self._x[PSI])
        IKH = np.eye(NX) - K @ H
        P = IKH @ self._P @ IKH.T + K @ R @ K.T  # Joseph form: stays symmetric positive definite
        self._P = 0.5 * (P + P.T)
        return Innovation(y, S)