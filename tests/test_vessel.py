from dataclasses import replace
from pathlib import Path

import numpy as np

from vessel_state_estimation.config import EnvConfig, get_vessel
from vessel_state_estimation.eval.geometry import fit_circle
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sim import Environment, Vessel, constant_command, simulate

VESSELS = Path(__file__).resolve().parents[1] / "configs" / "vessels.yaml"
NAMES = ("merchant_ship", "sailboat", "small_boat")


def make(name, **overrides):
    params = get_vessel(name, VESSELS)
    return Vessel(replace(params, **overrides) if overrides else params)


def test_straight_course_stays_straight():
    for name in NAMES:
        v = make(name)
        u0, psi0, t_end = v.p.nominal_speed, 0.7, 60.0
        traj = simulate(v, v.initial_state(psi=psi0), constant_command(0.0, u0), t_end, 0.05)
        last = traj[-1]
        np.testing.assert_allclose(traj.array("psi"), psi0, atol=1e-12)
        np.testing.assert_allclose(last.n, u0 * np.cos(psi0) * t_end, rtol=1e-9)
        np.testing.assert_allclose(last.e, u0 * np.sin(psi0) * t_end, rtol=1e-9)


def test_current_adds_drift():
    v = make("merchant_ship")
    u0, t_end, dt = v.p.nominal_speed, 100.0, 0.05
    cfg = EnvConfig(current_speed=0.5, current_dir_deg=90.0)  # 0.5 m/s towards East, no noise
    env = Environment(cfg, make_rngs(1)["disturbance"], dt, t_end)
    traj = simulate(v, v.initial_state(), constant_command(0.0, u0), t_end, dt, environment=env)
    np.testing.assert_allclose(traj[-1].n, u0 * t_end, rtol=1e-9)
    np.testing.assert_allclose(traj[-1].e, 0.5 * t_end, rtol=1e-9)

def test_yaw_step_response_matches_first_order_lag():
    for name in NAMES:
        v = make(name)
        u0, delta = v.p.nominal_speed, 0.1
        K, T = v.nomoto(u0)
        # Rudder already at its command at t=0, so the response is a pure first-order lag.
        x0 = v.initial_state(rudder=delta)
        traj = simulate(v, x0, constant_command(delta, u0), 5 * T, 0.02)
        t, r = traj.array("t"), traj.array("r")
        np.testing.assert_allclose(r, K * delta * (1.0 - np.exp(-t / T)), atol=1e-6)
        # 63.2 % of the steady yaw rate at t = T
        r_at_T = np.interp(T, t, r)
        np.testing.assert_allclose(r_at_T / (K * delta), 1.0 - np.exp(-1.0), atol=1e-4)


def test_steady_turn_radius_matches_theory():
    for name in NAMES:
        v = make(name)
        u0 = v.p.nominal_speed
        rudder = np.radians(v.p.maneuvers.turn_rudder_deg)
        turn = v.steady_turn(rudder, u0)
        _, T = v.nomoto(u0)
        t_settle = 8 * T
        duration = t_settle + 1.3 * 2 * np.pi * turn.radius / np.hypot(u0, turn.sway_speed)
        traj = simulate(v, v.initial_state(), constant_command(rudder, u0), duration, 0.05)
        steady = traj.array("t") >= t_settle
        _, _, radius = fit_circle(traj.array("n")[steady], traj.array("e")[steady])
        np.testing.assert_allclose(radius, turn.radius, rtol=5e-3)
        np.testing.assert_allclose(traj[-1].r, turn.yaw_rate, rtol=1e-3)
        np.testing.assert_allclose(traj[-1].v, turn.sway_speed, rtol=1e-3)


def test_turn_to_starboard_gives_negative_sway():
    v = make("merchant_ship")
    turn = v.steady_turn(np.radians(15.0))
    assert turn.yaw_rate > 0.0 and turn.sway_speed < 0.0


def test_rudder_saturation_and_rate_limit():
    v = make("merchant_ship")
    traj = simulate(v, v.initial_state(), constant_command(3.0, v.p.nominal_speed), 30.0, 0.1)
    delta, t = traj.array("delta"), traj.array("t")
    i5 = int(np.argmin(np.abs(t - 5.0)))
    np.testing.assert_allclose(delta[i5], v.p.rudder_rate * 5.0, rtol=1e-9)
    np.testing.assert_allclose(delta.max(), v.p.rudder_max, rtol=1e-12)


def test_heading_is_wrapped():
    v = make("small_boat")
    rudder = np.radians(20.0)
    traj = simulate(v, v.initial_state(psi=3.0), constant_command(rudder, v.p.nominal_speed),
                    30.0, 0.02)
    psi = traj.array("psi")
    assert psi.min() >= -np.pi and psi.max() < np.pi
    assert psi.min() < -2.0  # it did cross +/- pi
