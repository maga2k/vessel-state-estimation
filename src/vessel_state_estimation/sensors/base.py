"""Common sensor interface.

A sensor looks at the ground truth at time ``t`` and returns either a noisy
``Measurement`` or ``None`` (not due yet, or dropout). The estimator never sees the truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..sim.truth import TruthSample


@dataclass(frozen=True)
class Measurement:
    t: float
    sensor: str  # e.g. "gps", "imu"
    z: np.ndarray  # measurement vector
    R: np.ndarray  # nominal covariance the filter should assume


class Sensor(ABC):
    name: str
    rate_hz: float

    def __init__(self, rate_hz: float, rng: np.random.Generator) -> None:
        self.rate_hz = rate_hz
        self.rng = rng

    @abstractmethod
    def measure(self, t: float, truth: TruthSample) -> Measurement | None:
        """Return a noisy measurement of the truth, or None."""
