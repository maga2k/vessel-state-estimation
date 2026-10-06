"""3-DOF vessel model: surge / sway / yaw with a first-order Nomoto yaw response.

State vector (index constants below):  x = [N, E, psi, u, v, r, delta]

    N, E   position in the NED frame [m]
    psi    heading [rad]
    u, v   surge / sway speed through water, body frame [m/s]
    r      yaw rate [rad/s]
    delta  rudder angle [rad]; positive delta -> positive yaw rate (turn to starboard)

Model (details and derivation in docs/theory/01-nomoto-3dof.md):

    N_dot   = u cos(psi) - v sin(psi) + Vc_N          (kinematics + current)
    E_dot   = u sin(psi) + v cos(psi) + Vc_E
    psi_dot = r
    T r_dot + r = K delta        (Nomoto; K = K' U / L,  T = T' L / U)
    v_dot   = (-x_p r - v) / T_sway                   (sway follows the pivot-point geometry)
    u_dot   = (u_cmd - u) / T_surge                   (first-order speed response)

The rudder is not part of the ODE: saturation and rate limit are applied in ``step``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..angles import wrap_angle
from ..config import VesselParams
from .environment import EnvState

N, E, PSI, U, V, R, DELTA = range(7)

U_MIN = 0.5  # [m/s] floor on the speed used to scale K and T (avoids T -> infinity at rest)


@dataclass(frozen=True)
class Command:
    rudder: float  # commanded rudder angle [rad], > 0 turns to starboard
    speed: float  # commanded surge speed [m/s]


class SteadyTurn(NamedTuple):
    yaw_rate: float  # r_ss [rad/s]
    sway_speed: float  # v_ss [m/s]
    radius: float  # radius of the CG path [m] (inf for zero rudder)


class Vessel:
    def __init__(self, params: VesselParams) -> None:
        self.p = params

    # ------------------------------------------------------------------ helpers
    def initial_state(self, n: float = 0.0, e: float = 0.0, psi: float = 0.0,
                      speed: float | None = None, rudder: float = 0.0) -> np.ndarray:
        u = self.p.nominal_speed if speed is None else speed
        return np.array([n, e, psi, u, 0.0, 0.0, rudder])

    def nomoto(self, u: float) -> tuple[float, float]:
        """Dimensional Nomoto gain K [1/s] and time constant T [s] at surge speed ``u``."""
        ue = max(abs(u), U_MIN)
        return self.p.K_prime * ue / self.p.length, self.p.T_prime * self.p.length / ue

    def drift_velocity(self, env: EnvState) -> tuple[float, float]:
        """Velocity (NED) added to the through-water motion: current + leeway from the wind."""
        k = self.p.leeway_coeff
        return (env.current[0] + k * env.wind[0], env.current[1] + k * env.wind[1])

    def steady_turn(self, rudder: float, speed: float | None = None) -> SteadyTurn:
        """Theoretical steady turn for a constant rudder (used to validate the simulation)."""
        u = self.p.nominal_speed if speed is None else speed
        delta = float(np.clip(rudder, -self.p.rudder_max, self.p.rudder_max))
        K, _ = self.nomoto(u)
        r_ss = K * delta
        v_ss = -self.p.pivot_dist * r_ss
        radius = float("inf") if r_ss == 0.0 else float(np.hypot(u, v_ss) / abs(r_ss))
        return SteadyTurn(r_ss, v_ss, radius)

    # ------------------------------------------------------------------ dynamics
    def derivatives(self, x: np.ndarray, speed_cmd: float,
                    current: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Time derivative of the state. ``current`` is the NED current velocity [m/s]."""
        p = self.p
        psi, u, v, r, delta = x[PSI], x[U], x[V], x[R], x[DELTA]
        K, T = self.nomoto(u)
        cp, sp = np.cos(psi), np.sin(psi)

        xd = np.zeros(7)
        xd[N] = u * cp - v * sp + current[0]
        xd[E] = u * sp + v * cp + current[1]
        xd[PSI] = r
        xd[U] = (speed_cmd - u) / p.T_surge
        xd[V] = (-p.pivot_dist * r - v) / p.T_sway
        xd[R] = (K * delta - r) / T
        return xd  # xd[DELTA] = 0: rudder handled in step()

    def step(self, x: np.ndarray, cmd: Command, dt: float,
             current: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        """Advance the state by ``dt`` (RK4, rudder held at its mid-step value)."""
        p = self.p
        delta_cmd = np.clip(cmd.rudder, -p.rudder_max, p.rudder_max)  # saturation
        max_move = p.rudder_rate * dt  # rate limit
        delta_new = x[DELTA] + np.clip(delta_cmd - x[DELTA], -max_move, max_move)

        xm = x.copy()
        xm[DELTA] = 0.5 * (x[DELTA] + delta_new)

        def f(s: np.ndarray) -> np.ndarray:
            return self.derivatives(s, cmd.speed, current)

        k1 = f(xm)
        k2 = f(xm + 0.5 * dt * k1)
        k3 = f(xm + 0.5 * dt * k2)
        k4 = f(xm + dt * k3)
        x_new = xm + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_new[DELTA] = delta_new
        x_new[PSI] = wrap_angle(x_new[PSI])
        return x_new
