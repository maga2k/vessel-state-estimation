"""Estimation errors against the ground truth."""

from __future__ import annotations

import numpy as np

from ..angles import wrap_angle
from ..estimation.ekf import B_AX, B_AY, B_G, PSI
from ..estimation.runner import FilterLog
from ..sim.truth import Trajectory


def true_states(traj: Trajectory, log: FilterLog) -> np.ndarray:
    """True state vector [N, E, psi, vN, vE, b_ax, b_ay, b_g] at the logged times (nearest sample)."""
    dt = traj[1].t - traj[0].t
    idx = np.clip(np.round(log.t / dt).astype(int), 0, len(traj) - 1)
    x = np.column_stack([traj.array(name)[idx] for name in ("n", "e", "psi", "vn", "ve")])
    return np.column_stack([x, log.true_bias[:, :3]])


def state_errors(log: FilterLog, x_true: np.ndarray) -> np.ndarray:
    """Estimate minus truth, with the heading error wrapped to [-pi, pi)."""
    err = log.x - x_true
    err[:, PSI] = wrap_angle(err[:, PSI])
    return err


def rms(err: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(err**2, axis=0))


__all__ = ["B_AX", "B_AY", "B_G", "rms", "state_errors", "true_states"]