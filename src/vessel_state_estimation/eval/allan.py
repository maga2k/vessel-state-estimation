"""Allan deviation: the standard tool to read noise parameters off an inertial sensor record."""

from __future__ import annotations

import numpy as np


def allan_deviation(x: np.ndarray, fs: float, taus: np.ndarray) -> np.ndarray:
    """Overlapping Allan deviation of a rate signal ``x`` sampled at ``fs`` Hz.

    Integrates ``x`` to an "angle" theta and uses the second difference over clusters of length
    tau:  sigma^2(tau) = 1 / (2 tau^2 (N - 2m)) * sum (theta[k+2m] - 2 theta[k+m] + theta[k])^2.
    Entries are NaN where the record is too short for that tau.

    Reading the log-log curve for a gyro:
      white noise (density N):   sigma(tau) = N / sqrt(tau)        slope -1/2
      bias random walk (K):      sigma(tau) = K sqrt(tau / 3)      slope +1/2
    """
    theta = np.cumsum(x) / fs
    out = np.full(len(taus), np.nan)
    for i, tau in enumerate(taus):
        m = int(round(tau * fs))
        if m < 1 or 2 * m >= len(theta):
            continue
        d = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        out[i] = np.sqrt(0.5 * np.mean(d**2)) / (m / fs)
    return out