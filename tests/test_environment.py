import tempfile
from pathlib import Path

import numpy as np

from vessel_state_estimation.config import EnvConfig, get_vessel
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sim import Environment, Vessel, constant_command, simulate

VESSELS = Path(__file__).resolve().parents[1] / "configs" / "vessels.yaml"


def series(env, n, dt):
    return np.array([env.at(k * dt).current for k in range(n)])


def test_same_seed_same_series_different_seed_differs():
    cfg = EnvConfig(current_speed=0.5, current_sigma=0.1, current_tau=50.0)
    a = Environment(cfg, make_rngs(3)["disturbance"], 0.1, 100.0)
    b = Environment(cfg, make_rngs(3)["disturbance"], 0.1, 100.0)
    c = Environment(cfg, make_rngs(4)["disturbance"], 0.1, 100.0)
    assert np.array_equal(series(a, 1000, 0.1), series(b, 1000, 0.1))
    assert not np.array_equal(series(a, 1000, 0.1), series(c, 1000, 0.1))


def test_gauss_markov_statistics():
    dt, tau, sigma = 0.1, 20.0, 0.2
    cfg = EnvConfig(current_speed=0.5, current_dir_deg=0.0, current_sigma=sigma, current_tau=tau)
    env = Environment(cfg, make_rngs(5)["disturbance"], dt, 20000.0)
    x = series(env, 200000, dt)
    assert abs(x[:, 0].mean() - 0.5) < 0.05  # mean north = current speed
    assert abs(x[:, 0].std() / sigma - 1.0) < 0.1
    assert abs(x[:, 1].mean()) < 0.05
    # one-step autocorrelation of a Gauss-Markov process is exp(-dt/tau)
    d = x[:, 0] - x[:, 0].mean()
    rho = (d[1:] * d[:-1]).mean() / d.var()
    assert abs(rho - np.exp(-dt / tau)) < 0.01


def test_constant_environment_when_sigma_is_zero():
    cfg = EnvConfig(current_speed=0.5, current_dir_deg=90.0, wind_speed=10.0, wind_dir_deg=0.0)
    env = Environment(cfg, make_rngs(1)["disturbance"], 0.1, 10.0)
    assert env.at(0.0) == env.at(9.0)


def test_direction_conventions():
    cfg = EnvConfig(current_speed=1.0, current_dir_deg=90.0, wind_speed=10.0, wind_dir_deg=0.0)
    s = Environment(cfg, make_rngs(1)["disturbance"], 0.1, 1.0).at(0.0)
    np.testing.assert_allclose(s.current, (0.0, 1.0), atol=1e-12)  # flows towards East
    np.testing.assert_allclose(s.wind, (-10.0, 0.0), atol=1e-12)  # wind FROM North blows South


def test_cog_differs_from_heading_with_cross_current():
    v = Vessel(get_vessel("merchant_ship", VESSELS))
    u0, dt = v.p.nominal_speed, 0.05
    cfg = EnvConfig(current_speed=1.0, current_dir_deg=90.0)
    env = Environment(cfg, make_rngs(1)["disturbance"], dt, 20.0)
    traj = simulate(v, v.initial_state(psi=0.0), constant_command(0.0, u0), 20.0, dt, environment=env)
    last = traj[-1]
    assert abs(last.psi) < 1e-12  # heading North
    np.testing.assert_allclose(np.arctan2(last.ve, last.vn), np.arctan2(1.0, u0), rtol=1e-9)


def test_wind_leeway():
    v = Vessel(get_vessel("small_boat", VESSELS))
    dt = 0.05
    cfg = EnvConfig(wind_speed=10.0, wind_dir_deg=270.0)  # from West -> blows towards East
    env = Environment(cfg, make_rngs(1)["disturbance"], dt, 10.0)
    traj = simulate(v, v.initial_state(psi=0.0), constant_command(0.0, v.p.nominal_speed), 10.0, dt,
                    environment=env)
    np.testing.assert_allclose(traj[-1].ve, v.p.leeway_coeff * 10.0, rtol=1e-9)


def test_trajectory_save_load_roundtrip():
    v = Vessel(get_vessel("sailboat", VESSELS))
    traj = simulate(v, v.initial_state(), constant_command(0.1, v.p.nominal_speed), 5.0, 0.05)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "traj.npz"
        traj.save(path)
        back = type(traj).load(path)
    assert len(back) == len(traj)
    assert back[-1] == traj[-1]