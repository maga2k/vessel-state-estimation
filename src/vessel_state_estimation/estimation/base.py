"""Common estimator interface shared by EKF and MEKF."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..sensors.base import Measurement


class Estimator(ABC):
    @property
    @abstractmethod
    def x(self) -> np.ndarray:
        """Current state estimate."""

    @property
    @abstractmethod
    def P(self) -> np.ndarray:
        """Current state covariance."""

    @abstractmethod
    def predict(self, u: np.ndarray, dt: float) -> None:
        """Propagate state and covariance with input ``u`` (e.g. IMU) over ``dt``."""

    @abstractmethod
    def update(self, meas: Measurement) -> np.ndarray:
        """Fuse one measurement; return the innovation (needed for NIS)."""
