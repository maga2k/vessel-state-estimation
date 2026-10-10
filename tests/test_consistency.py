import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from vessel_state_estimation.config import Config, get_vessel
from vessel_state_estimation.estimation import FilterLog, Innovation
from vessel_state_estimation.eval.consistency import (chi2_bounds, nees, nis_by_sensor,
                                                      normalized_squared_errors)
from vessel_state_estimation.eval.montecarlo import evaluate, generate_runs
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sim import Environment, HeadingAutopilot, Vessel, simulate

ROOT = Path(__file__).resolve().parents[1]


def test_nees_of_gaussian_errors_averages_to_dimension():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8))
    P = A @ A.T + 8 * np.eye(8)
    L = np.linalg.cholesky(P)
    n = 20000
    err = (L @ rng.normal(size=(8, n))).T
    values = nees(err, np.broadcast_to(P, (n, 8, 8)))
    assert abs(values.mean() / 8 - 1.0) < 0.03
    np.testing.assert_allclose(normalized_squared_errors(err, np.broadcast_to(P, (n, 8, 8))).mean(axis=0),
                               1.0, atol=0.05)


def test_chi2_bounds_contain_dof_and_shrink_with_runs():
    lo1, hi1 = chi2_bounds(8, n_runs=1)
    lo20, hi20 = chi2_bounds(8, n_runs=20)
    assert (lo1, hi1) == chi2.interval(0.95, 8)
    assert lo20 < 8 < hi20
    assert (hi20 - lo20) < (hi1 - lo1) / 3


def test_nis_by_sensor_groups_values_and_dof():
    log = FilterLog(np.zeros(1), np.zeros((1, 8)), np.zeros((1, 8, 8)), np.zeros((1, 3)), [
        (0.0, "gps", Innovation(np.array([1.0, 2.0, 0.0, 0.0]), np.eye(4))),
        (0.1, "compass", Innovation(np.array([3.0]), np.array([[9.0]]))),
        (1.0, "gps", Innovation(np.array([0.0, 0.0, 1.0, 0.0]), 2.0 * np.eye(4))),
    ])
    out = nis_by_sensor(log)
    t, nis, dof = out["gps"]
    np.testing.assert_allclose(t, [0.0, 1.0])
    np.testing.assert_allclose(nis, [5.0, 0.5])
    assert dof == 4 and out["compass"][2] == 1
    np.testing.assert_allclose(out["compass"][1], [1.0])


def _monte_carlo_setup(n_runs=10, duration=150.0, dt=0.02):
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    cfg = replace(cfg, sim=replace(cfg.sim, dt=dt),
                  sensors=replace(cfg.sensors, imu=replace(cfg.sensors.imu, rate_hz=1.0 / dt)))
    v = Vessel(get_vessel("small_boat", ROOT / "configs" / "vessels.yaml"))
    marks = [(0.0, 0.0), (0.2, 40.0), (0.5, -30.0), (0.75, 60.0)]
    ap = HeadingAutopilot(v, v.p.nominal_speed, [(f * duration, math.radians(h)) for f, h in marks])
    env = Environment(cfg.env, make_rngs(cfg.sim.seed)["disturbance"], dt, duration)
    traj = simulate(v, v.initial_state(), ap, duration, dt, environment=env)
    return cfg, traj, generate_runs(cfg, traj, n_runs)


def test_monte_carlo_tells_a_consistent_filter_from_mistuned_ones():
    cfg, traj, runs = _monte_carlo_setup()
    imu = cfg.sensors.imu
    good = evaluate(runs, traj, cfg.ekf, imu)
    overconfident = evaluate(runs, traj, replace(cfg.ekf, q_scale=0.25), imu)
    conservative = evaluate(runs, traj, replace(cfg.ekf, q_scale=4.0), imu)
    bad_r = evaluate(runs, traj, replace(cfg.ekf, r_scale=0.5), imu)

    assert 5.5 < good.anees() < 11.0  # ideal: 8 (state dimension)
    assert abs(good.mean_nis("gps") / 4 - 1.0) < 0.1 and abs(good.mean_nis("compass") - 1.0) < 0.1
    assert overconfident.anees() > 1.5 * good.anees()  # Q too small: errors exceed the covariance
    assert conservative.anees() < 0.8 * good.anees()  # Q too big: covariance exceeds the errors
    assert bad_r.mean_nis("gps") > 8.0  # R too small: NIS flags it (twice the ideal 4 or more)
    # ...while the accuracy of the estimate hardly changes: consistency is not accuracy
    assert abs(overconfident.rms_error()[0] / good.rms_error()[0] - 1.0) < 0.15