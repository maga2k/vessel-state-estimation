"""Planar IMU: two accelerometers (body x, y) and one gyro (body z).

    f_x   = u_dot - v r          specific force = inertial acceleration in the body frame
    f_y   = v_dot + u r          (horizontal motion: no gravity component)
    z     = [f_x, f_y, r] + bias + white noise
    bias  : random walk,  b[k+1] = b[k] + rw * sqrt(dt) * w,   b[0] ~ N(0, init_std^2)
    white : sigma = noise_density / sqrt(dt)

The specific-force expressions neglect the time derivative of the current/wind velocity
(tiny compared with the accelerometer noise).
"""

from __future__ import annotations

import numpy as np

from ..config import ImuConfig
from ..sim.truth import TruthSample
from .base import Measurement, Sensor


class PlanarImu(Sensor):
    name = "imu"

    def __init__(self, cfg: ImuConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.rate_hz, rng)
        self.cfg = cfg
        dt = 1.0 / cfg.rate_hz
        self._sig_a = cfg.accel_noise_density / np.sqrt(dt)
        self._sig_g = cfg.gyro_noise_density / np.sqrt(dt)
        self.R = np.diag([self._sig_a**2, self._sig_a**2, self._sig_g**2])
        self.accel_bias = cfg.accel_bias_init_std * rng.standard_normal(2)
        self.gyro_bias = cfg.gyro_bias_init_std * rng.standard_normal()
        self._started = False

    def measure(self, t: float, truth: TruthSample) -> Measurement | None:
        if not self._due(t):
            return None
        c, dt = self.cfg, 1.0 / self.rate_hz
        w = self.rng.standard_normal(6)  # always 6 draws per sample: stream use is config-independent
        if self._started:  # bias random walk (not before the first sample)
            self.accel_bias = self.accel_bias + c.accel_bias_rw * np.sqrt(dt) * w[0:2]
            self.gyro_bias = self.gyro_bias + c.gyro_bias_rw * np.sqrt(dt) * w[2]
        self._started = True

        f_x = truth.u_dot - truth.v * truth.r
        f_y = truth.v_dot + truth.u * truth.r
        z = np.array([
            f_x + self.accel_bias[0] + self._sig_a * w[3],
            f_y + self.accel_bias[1] + self._sig_a * w[4],
            truth.r + self.gyro_bias + self._sig_g * w[5],
        ])
        bias = np.array([self.accel_bias[0], self.accel_bias[1], self.gyro_bias])
        return Measurement(t, self.name, z, self.R.copy(), {"bias": bias})