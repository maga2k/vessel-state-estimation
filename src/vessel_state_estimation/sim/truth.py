"""Ground-truth containers produced by the simulator and consumed by sensors / evaluation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TruthSample:
    """True vessel state at one instant. NED navigation frame, FRD body frame, SI units."""

    t: float
    n: float  # north position [m]
    e: float  # east position [m]
    psi: float  # heading [rad], wrapped to [-pi, pi)
    u: float  # surge speed through water [m/s]
    v: float  # sway speed through water [m/s]
    r: float  # yaw rate [rad/s]
    delta: float  # actual rudder angle [rad]
    u_dot: float  # d(u)/dt [m/s^2]  (body-frame velocity derivatives, NOT specific force)
    v_dot: float  # d(v)/dt [m/s^2]
    r_dot: float  # d(r)/dt [rad/s^2]
    vn: float  # ground velocity, north component [m/s] (includes current)
    ve: float  # ground velocity, east component [m/s] (includes current)
    current_n: float = 0.0  # current, north [m/s]
    current_e: float = 0.0  # current, east [m/s]
    wind_n: float = 0.0  # wind velocity (towards), north [m/s]
    wind_e: float = 0.0  # wind velocity (towards), east [m/s]


class Trajectory:
    """Time-ordered list of ``TruthSample`` with convenient array access."""

    def __init__(self, samples: list[TruthSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> TruthSample:
        return self.samples[i]

    def array(self, name: str) -> np.ndarray:
        """All values of one field as a 1-D array, e.g. ``traj.array("psi")``."""
        return np.array([getattr(s, name) for s in self.samples])

    def save(self, path: str | Path) -> None:
        """Save every field as an array in a ``.npz`` file."""
        np.savez(path, **{f.name: self.array(f.name) for f in fields(TruthSample)})

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        d = np.load(path)
        names = [f.name for f in fields(TruthSample)]
        return cls([TruthSample(**{k: float(d[k][i]) for k in names}) for i in range(len(d["t"]))])
