"""Common sensor interface.

A sensor looks at the ground truth at time ``t`` and returns either a noisy
``Measurement`` or ``None`` (not due yet, or dropout). The estimator never sees the truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..sim.truth import TruthSample


@dataclass(frozen=True)
class Measurement:
    t: float
    sensor: str  # "imu", "gps", "compass", ...
    z: np.ndarray  # measurement vector
    R: np.ndarray  # nominal covariance the filter should assume
    # Ground-truth side information (e.g. true IMU bias) for evaluation only.
    # Estimators must NEVER read it.
    info: dict = field(default_factory=dict)


class Sensor(ABC):
    name: str
    rate_hz: float

    def __init__(self, rate_hz: float, rng: np.random.Generator) -> None:
        self.rate_hz = rate_hz
        self.rng = rng
        self._last_tick = -1

    def _due(self, t: float) -> bool:
        """True once per sensor period (ticks at t = 0, 1/rate, 2/rate, ...)."""
        tick = int(np.floor(t * self.rate_hz + 1e-9))
        if tick > self._last_tick:
            self._last_tick = tick
            return True
        return False

    @abstractmethod
    def measure(self, t: float, truth: TruthSample) -> Measurement | None:
        """Return a noisy measurement of the truth, or None."""