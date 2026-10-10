"""Run the MEKF over IMU3D measurements and log its output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sensors.base import Measurement
from .base import Innovation
from .mekf import Mekf


@dataclass
class MekfLog:
    t: np.ndarray  # (n,)
    q: np.ndarray  # (n, 4)
    b: np.ndarray  # (n, 3)
    P: np.ndarray  # (n, 6, 6)
    true_bias: np.ndarray  # (n, 3) true gyro bias: EVALUATION ONLY
    innovations: list[tuple[float, Innovation]] = field(default_factory=list)


def run_mekf(mekf: Mekf, measurements: list[Measurement], imu_rate_hz: float,
             log_every: int = 10) -> MekfLog:
    """Gyro samples drive the prediction (trapezoidal rule over consecutive samples); the
    accelerometer is fused at ``cfg.accel_update_hz``. GPS fixes in the list only provide the ground
    speed for the turn compensation. Logs after the update of each logged step.
    """
    every = max(1, int(round(imu_rate_hz / mekf.cfg.accel_update_hz)))
    sigma_noise = mekf.imu.accel_noise_density * np.sqrt(imu_rate_hz)  # per-sample accel noise
    ts, qs, bs, Ps, tb, innovations = [], [], [], [], [], []
    t_prev, g_prev, bias_prev, count, speed = None, None, None, 0, 0.0

    def snapshot() -> None:
        ts.append(t_prev)
        qs.append(mekf.q.copy())
        bs.append(mekf.b.copy())
        Ps.append(mekf.P.copy())
        tb.append(bias_prev)

    for m in measurements:
        if m.sensor == "gps":
            speed = float(np.hypot(m.z[2], m.z[3]))  # speed over ground from the GPS velocity
            continue
        if m.sensor != "imu3d":
            continue
        gyro = m.z[3:]
        if t_prev is not None:
            if count % log_every == 0:
                snapshot()
            count += 1
            mekf.predict(0.5 * (g_prev + gyro), m.t - t_prev)
            if count % every == 0:
                innovations.append((m.t, mekf.update_accel(m.z[:3], sigma_noise, gyro, speed)))
        t_prev, g_prev = m.t, gyro
        bias_prev = np.asarray(m.info.get("bias", np.full(6, np.nan)))[3:]
    if t_prev is not None and count % log_every == 0:
        snapshot()
    return MekfLog(np.array(ts), np.array(qs), np.array(bs), np.array(Ps), np.array(tb), innovations)