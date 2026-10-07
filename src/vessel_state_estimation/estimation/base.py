"""Common estimator interface shared by EKF and MEKF."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from ..sensors.base import Measurement


class Innovation(NamedTuple):
    """Result of a measurement update: innovation y = z - h(x-) and its covariance S."""

    y: np.ndarray
    S: np.ndarray

    @property
    def nis(self) -> float:
        """Normalized innovation squared  y^T S^-1 y  (chi-square with len(y) dof if consistent)."""
        return float(self.y @ np.linalg.solve(self.S, self.y))


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
    def update(self, meas: Measurement) -> Innovation:
        """Fuse one measurement; return the innovation and its covariance."""