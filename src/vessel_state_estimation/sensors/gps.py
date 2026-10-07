"""GPS: low-rate position + ground-velocity fixes with white noise and dropouts.

    z = [N, E, vN, vE] + noise,   R = diag(pos_sigma^2, pos_sigma^2, vel_sigma^2, vel_sigma^2)

Dropouts: fixes are lost inside the configured windows and, independently, with probability
``dropout_prob``. The random draws are made at every fix, lost or not, so the noise sequence does
not depend on the dropout settings.
"""

from __future__ import annotations

import numpy as np

from ..config import GpsConfig
from ..sim.truth import TruthSample
from .base import Measurement, Sensor


class Gps(Sensor):
    name = "gps"

    def __init__(self, cfg: GpsConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.rate_hz, rng)
        self.cfg = cfg
        self.R = np.diag([cfg.pos_sigma**2] * 2 + [cfg.vel_sigma**2] * 2)

    def in_dropout_window(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.cfg.dropout_windows)

    def measure(self, t: float, truth: TruthSample) -> Measurement | None:
        if not self._due(t):
            return None
        c = self.cfg
        w = self.rng.standard_normal(4)
        u = self.rng.random()
        if self.in_dropout_window(t) or u < c.dropout_prob:
            return None
        z = np.array([
            truth.n + c.pos_sigma * w[0],
            truth.e + c.pos_sigma * w[1],
            truth.vn + c.vel_sigma * w[2],
            truth.ve + c.vel_sigma * w[3],
        ])
        return Measurement(t, self.name, z, self.R.copy())