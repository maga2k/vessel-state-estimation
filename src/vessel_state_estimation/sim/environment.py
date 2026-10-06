"""Environment: current and wind as time series (mean + Gauss-Markov fluctuation).

The whole series is pre-generated from a seeded stream, so a run is reproducible and the
environment can be logged next to the truth. Within one integration step it is held constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import EnvConfig


@dataclass(frozen=True)
class EnvState:
    current: tuple[float, float] = (0.0, 0.0)  # water velocity over ground, NED [m/s]
    wind: tuple[float, float] = (0.0, 0.0)  # air velocity over ground, NED [m/s] (towards)


def _gauss_markov(mean: np.ndarray, sigma: float, tau: float, n: int, dt: float,
                  rng: np.random.Generator) -> np.ndarray:
    """(n, 2) series: constant ``mean`` + zero-mean first-order Gauss-Markov process.

    x[k] = a x[k-1] + b w[k],  a = exp(-dt/tau),  b = sigma sqrt(1 - a^2),  w ~ N(0, 1)
    gives a stationary process with std ``sigma`` and autocorrelation exp(-|lag|/tau).
    """
    w = rng.standard_normal((n, 2))  # always drawn: stream use does not depend on sigma
    a = np.exp(-dt / tau)
    b = sigma * np.sqrt(1.0 - a * a)
    x = np.empty((n, 2))
    x[0] = sigma * w[0]  # start in the stationary distribution
    for k in range(1, n):
        x[k] = a * x[k - 1] + b * w[k]
    return mean + x


class Environment:
    def __init__(self, cfg: EnvConfig, rng: np.random.Generator, dt: float, duration: float):
        n = int(round(duration / dt)) + 1
        self.dt = dt
        c = np.radians(cfg.current_dir_deg)
        current_mean = cfg.current_speed * np.array([np.cos(c), np.sin(c)])  # flows towards
        w = np.radians(cfg.wind_dir_deg)
        wind_mean = -cfg.wind_speed * np.array([np.cos(w), np.sin(w)])  # blows FROM -> negate
        self._current = _gauss_markov(current_mean, cfg.current_sigma, cfg.current_tau, n, dt, rng)
        self._wind = _gauss_markov(wind_mean, cfg.wind_sigma, cfg.wind_tau, n, dt, rng)

    def at(self, t: float) -> EnvState:
        k = min(int(round(t / self.dt)), len(self._current) - 1)
        c, w = self._current[k], self._wind[k]
        return EnvState((float(c[0]), float(c[1])), (float(w[0]), float(w[1])))