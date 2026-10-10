"""3-axis IMU for the attitude study: gyroscope + accelerometer in body axes.

    z = [f_b + b_a + n_a ,  omega_b + b_g + n_g]     (6,)
    bias: random walk per axis;  white noise: sigma = density / sqrt(dt)

Uses the same ImuConfig as the planar IMU (per axis) and its own random stream.
"""

from __future__ import annotations

import numpy as np

from ..config import ImuConfig
from ..sim.attitude import AttitudeSample
from .base import Measurement, Sensor


class Imu3d(Sensor):
    name = "imu3d"

    def __init__(self, cfg: ImuConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.rate_hz, rng)
        self.cfg = cfg
        dt = 1.0 / cfg.rate_hz
        self._sig_a = cfg.accel_noise_density / np.sqrt(dt)
        self._sig_g = cfg.gyro_noise_density / np.sqrt(dt)
        self.R = np.diag([self._sig_a**2] * 3 + [self._sig_g**2] * 3)
        self.accel_bias = cfg.accel_bias_init_std * rng.standard_normal(3)
        self.gyro_bias = cfg.gyro_bias_init_std * rng.standard_normal(3)
        self._started = False

    def measure(self, t: float, truth: AttitudeSample) -> Measurement | None:
        if not self._due(t):
            return None
        c, dt = self.cfg, 1.0 / self.rate_hz
        w = self.rng.standard_normal(12)  # always 12 draws per sample
        if self._started:
            self.accel_bias = self.accel_bias + c.accel_bias_rw * np.sqrt(dt) * w[0:3]
            self.gyro_bias = self.gyro_bias + c.gyro_bias_rw * np.sqrt(dt) * w[3:6]
        self._started = True
        z = np.concatenate([truth.f_b + self.accel_bias + self._sig_a * w[6:9],
                            truth.omega_b + self.gyro_bias + self._sig_g * w[9:12]])
        bias = np.concatenate([self.accel_bias, self.gyro_bias])
        return Measurement(t, self.name, z, self.R.copy(), {"bias": bias})


def simulate_imu3d(truth, sensor: Imu3d) -> list[Measurement]:
    """IMU3D measurements over an ``AttitudeTruth`` (time ordered)."""
    out = []
    for k in range(len(truth)):
        m = sensor.measure(float(truth.t[k]), truth.sample(k))
        if m is not None:
            out.append(m)
    return out