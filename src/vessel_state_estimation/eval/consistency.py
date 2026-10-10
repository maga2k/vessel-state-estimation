"""Filter consistency: NEES / NIS and their chi-square acceptance regions.

If the filter is consistent, the errors really are N(0, P):
  NEES  e^T P^-1 e         ~ chi2(nx)   for the state error e = x_hat - x_true
  NIS   y^T S^-1 y         ~ chi2(m)    for an innovation y of dimension m
Averaging over N independent runs gives N * mean ~ chi2(N * dof), hence tighter bounds.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from ..estimation.runner import FilterLog


def nees(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """NEES at every logged time. ``err``: (n, nx), ``P``: (n, nx, nx). Wrap angle errors first."""
    z = np.linalg.solve(P, err[..., None])[..., 0]
    return np.einsum("ij,ij->i", err, z)


def normalized_squared_errors(err: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Per-state e_i^2 / P_ii, shape (n, nx): each column should average to 1 (marginal check)."""
    return err**2 / np.einsum("ijj->ij", P)


def nis_by_sensor(log: FilterLog) -> dict[str, tuple[np.ndarray, np.ndarray, int]]:
    """sensor -> (times, NIS values, dof) from the innovations stored in a filter log."""
    acc: dict[str, tuple[list, list, int]] = {}
    for t, sensor, inn in log.innovations:
        times, values, _ = acc.setdefault(sensor, ([], [], len(inn.y)))
        times.append(t)
        values.append(inn.nis)
    return {k: (np.array(a), np.array(b), d) for k, (a, b, d) in acc.items()}


def chi2_bounds(dof: float, n_runs: int = 1, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided (1 - alpha) acceptance region for the MEAN of ``n_runs`` chi2(dof) values."""
    lo, hi = chi2.interval(1.0 - alpha, n_runs * dof)
    return lo / n_runs, hi / n_runs