"""Run a filter over a time-ordered list of measurements and log its output."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sensors.base import Measurement
from .base import Estimator, Innovation
from .ekf import PlanarEkf


@dataclass
class FilterLog:
    t: np.ndarray  # (n,) logged times
    x: np.ndarray  # (n, nx) estimates
    P: np.ndarray  # (n, nx, nx) covariances
    true_bias: np.ndarray  # (n, 3) true IMU bias [b_ax, b_ay, b_g]: EVALUATION ONLY
    innovations: list[tuple[float, str, Innovation]] = field(default_factory=list)
    # same order as ``innovations``: was that measurement a true outlier? EVALUATION ONLY
    outlier_flags: list[bool] = field(default_factory=list)


def run_filter(ekf: PlanarEkf, measurements: list[Measurement], log_every: int = 10) -> FilterLog:
    """IMU samples drive the prediction (over the interval since the previous IMU sample);
    GPS / compass samples are fused as they arrive. States are logged every ``log_every`` IMU
    steps, AFTER the updates that share the same timestamp.
    """
    use = {"gps"} | ({"compass"} if ekf.cfg.use_compass else set())
    ts: list[float] = []
    xs: list[np.ndarray] = []
    Ps: list[np.ndarray] = []
    bs: list[np.ndarray] = []
    innovations: list[tuple[float, str, Innovation]] = []
    flags: list[bool] = []
    t_prev, bias_prev, count, z_prev = None, None, 0, None

    def snapshot() -> None:
        ts.append(t_prev)
        xs.append(ekf.x.copy())
        Ps.append(ekf.P.copy())
        bs.append(bias_prev)

    for m in measurements:
        if m.sensor == "imu":
            if t_prev is not None:
                if count % log_every == 0:
                    snapshot()
                count += 1
                ekf.predict(0.5 * (z_prev + m.z), m.t - t_prev)  # trapezoidal rule on the IMU samples
            t_prev, z_prev = m.t, m.z
            bias_prev = np.asarray(m.info.get("bias", np.full(3, np.nan)))
        elif m.sensor in use:
            innovations.append((m.t, m.sensor, ekf.update(m)))
            flags.append(bool(m.info.get("outlier", False)))
    if t_prev is not None and count % log_every == 0:
        snapshot()
    return FilterLog(np.array(ts), np.array(xs), np.array(Ps), np.array(bs), innovations, flags)