"""Compass (heading sensor): z = wrap(psi + bias + noise).

Why it exists: with only IMU + GPS the heading of a vessel in steady motion is weakly observable
(GPS gives course over ground, which differs from heading when there is current or leeway).
"""

from __future__ import annotations

import numpy as np

from ..angles import wrap_angle
from ..config import CompassConfig
from ..sim.truth import TruthSample
from .base import Measurement, Sensor


class Compass(Sensor):
    name = "compass"

    def __init__(self, cfg: CompassConfig, rng: np.random.Generator) -> None:
        super().__init__(cfg.rate_hz, rng)
        self.cfg = cfg
        self.R = np.array([[cfg.sigma**2]])

    def measure(self, t: float, truth: TruthSample) -> Measurement | None:
        if not self._due(t):
            return None
        w = self.rng.standard_normal()
        z = np.array([float(wrap_angle(truth.psi + self.cfg.bias + self.cfg.sigma * w))])
        return Measurement(t, self.name, z, self.R.copy())