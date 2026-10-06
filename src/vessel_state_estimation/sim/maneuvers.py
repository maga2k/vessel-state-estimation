"""Maneuver library: functions ``(t, state) -> Command`` usable with ``simulate``."""

from __future__ import annotations

import numpy as np

from ..angles import wrap_angle
from .vessel import PSI, R, Command, Vessel


def constant_command(rudder: float, speed: float):
    """Open-loop maneuver: constant rudder and speed command. ``rudder=0`` is a straight course."""
    cmd = Command(rudder=rudder, speed=speed)

    def command_fn(t: float, x: np.ndarray) -> Command:
        return cmd

    return command_fn


class ZigZag:
    """IMO-style zig-zag test (e.g. 10/10): rudder +/-``rudder``, reversed when the heading
    reaches +/-``heading_change`` from the initial heading ``psi0``. Stateful: one instance per run.

    The heading keeps changing after the reversal (inertia): that overshoot angle is the classic
    way to identify the Nomoto indices K and T from sea trials.
    """

    def __init__(self, rudder: float, heading_change: float, speed: float, psi0: float = 0.0):
        self.rudder, self.heading_change, self.speed, self.psi0 = rudder, heading_change, speed, psi0
        self._sign = 1.0

    def __call__(self, t: float, x: np.ndarray) -> Command:
        err = wrap_angle(x[PSI] - self.psi0)
        if self._sign > 0 and err >= self.heading_change:
            self._sign = -1.0
        elif self._sign < 0 and err <= -self.heading_change:
            self._sign = 1.0
        return Command(self._sign * self.rudder, self.speed)


class HeadingAutopilot:
    """PD heading controller:  delta = kp * wrap(psi_ref - psi) - kd * r.

    Gains from pole placement on the Nomoto plant psi/delta = K / (s (T s + 1)). The closed loop is
    T s^2 + (1 + K kd) s + K kp = 0; choosing wn = 1/T and damping zeta gives
    kp = 1/(K T),  kd = (2 zeta - 1) / K.  Gains are fixed at the design speed.
    Uses the TRUE state for now; a later session feeds it the EKF estimate instead.

    ``schedule``: list of (t_start, psi_ref [rad]); the last entry with t_start <= t applies.
    """

    def __init__(self, vessel: Vessel, speed: float, schedule: list[tuple[float, float]],
                 zeta: float = 0.9):
        K, T = vessel.nomoto(speed)
        self.kp = 1.0 / (K * T)
        self.kd = (2.0 * zeta - 1.0) / K
        self.speed = speed
        self.schedule = sorted(schedule)

    def reference(self, t: float) -> float:
        ref = self.schedule[0][1]
        for t0, psi in self.schedule:
            if t >= t0:
                ref = psi
        return ref

    def __call__(self, t: float, x: np.ndarray) -> Command:
        e = wrap_angle(self.reference(t) - x[PSI])
        return Command(float(self.kp * e - self.kd * x[R]), self.speed)  # step() saturates