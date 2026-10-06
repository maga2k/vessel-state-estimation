"""Simulation loop: vessel + command function + environment -> ground truth."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .environment import Environment, EnvState
from .truth import Trajectory, TruthSample
from .vessel import DELTA, E, N, PSI, R, U, V, Command, Vessel

CommandFn = Callable[[float, np.ndarray], Command]  # (t, state) -> Command


def simulate(vessel: Vessel, x0: np.ndarray, command_fn: CommandFn, duration: float, dt: float,
             environment: Environment | None = None) -> Trajectory:
    """Integrate the vessel for ``duration`` seconds and log the truth at every step.

    ``command_fn`` receives the current state, so the same hook serves open-loop maneuvers
    (ignore the state) and closed-loop heading controllers (use the state).
    ``environment=None`` means calm water and no wind.
    """
    n_steps = int(round(duration / dt))
    x = x0.copy()
    samples: list[TruthSample] = []
    for k in range(n_steps + 1):
        t = k * dt
        cmd = command_fn(t, x)
        env = environment.at(t) if environment is not None else EnvState()
        drift = vessel.drift_velocity(env)
        xd = vessel.derivatives(x, cmd.speed, drift)
        samples.append(TruthSample(
            t=t, n=float(x[N]), e=float(x[E]), psi=float(x[PSI]),
            u=float(x[U]), v=float(x[V]), r=float(x[R]), delta=float(x[DELTA]),
            u_dot=float(xd[U]), v_dot=float(xd[V]), r_dot=float(xd[R]),
            vn=float(xd[N]), ve=float(xd[E]),
            current_n=env.current[0], current_e=env.current[1],
            wind_n=env.wind[0], wind_e=env.wind[1],
        ))
        if k < n_steps:
            x = vessel.step(x, cmd, dt, drift)
    return Trajectory(samples)