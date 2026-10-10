"""GPS: low-rate position + ground-velocity fixes with white noise and dropouts.

    z = [N, E, vN, vE] + noise,   R = diag(pos_sigma^2, pos_sigma^2, vel_sigma^2, vel_sigma^2)
    (times ``r_report_scale``^2: a sensor that reports a smaller R than its real noise is optimistic)

Dropouts: fixes are lost inside the configured windows and, independently, with probability
``dropout_prob``. The random draws are made at every fix, lost or not, so the noise sequence does
not depend on the dropout settings.

Outliers: with probability ``outlier_prob`` the position carries an extra jump N(0, outlier_sigma^2)
per axis (multipath). They use their own random stream, so nominal fixes are unchanged by the
outlier settings. ``Measurement.info["outlier"]`` tells which fixes are outliers (evaluation only).
"""

from __future__ import annotations

import numpy as np

from ..config import GpsConfig
from ..sim.truth import TruthSample
from .base import Measurement, Sensor


class Gps(Sensor):
    name = "gps"

    def __init__(self, cfg: GpsConfig, rng: np.random.Generator,
                 outlier_rng: np.random.Generator | None = None) -> None:
        super().__init__(cfg.rate_hz, rng)
        self.cfg = cfg
        self.outlier_rng = rng if outlier_rng is None else outlier_rng
        k = cfg.r_report_scale  # R the sensor REPORTS (normally equal to the true noise)
        self.R = np.diag([(k * cfg.pos_sigma) ** 2] * 2 + [(k * cfg.vel_sigma) ** 2] * 2)

    def in_dropout_window(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.cfg.dropout_windows)

    def measure(self, t: float, truth: TruthSample) -> Measurement | None:
        if not self._due(t):
            return None
        c = self.cfg
        w = self.rng.standard_normal(4)
        u = self.rng.random()
        u_out = self.outlier_rng.random()  # drawn at every fix, like the others
        jump = self.outlier_rng.standard_normal(2)
        if self.in_dropout_window(t) or u < c.dropout_prob:
            return None
        z = np.array([
            truth.n + c.pos_sigma * w[0],
            truth.e + c.pos_sigma * w[1],
            truth.vn + c.vel_sigma * w[2],
            truth.ve + c.vel_sigma * w[3],
        ])
        is_outlier = bool(u_out < c.outlier_prob)
        if is_outlier:
            z[:2] += c.outlier_sigma * jump
        return Measurement(t, self.name, z, self.R.copy(), {"outlier": is_outlier})