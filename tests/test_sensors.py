from dataclasses import replace
from pathlib import Path

import numpy as np

from vessel_state_estimation.angles import wrap_angle
from vessel_state_estimation.config import (CompassConfig, GpsConfig, ImuConfig, SensorsConfig,
                                            get_vessel)
from vessel_state_estimation.eval.allan import allan_deviation
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sensors import Compass, Gps, PlanarImu, build_sensors, simulate_sensors
from vessel_state_estimation.sim import Vessel, ZigZag, constant_command, simulate
from vessel_state_estimation.sim.truth import TruthSample

VESSELS = Path(__file__).resolve().parents[1] / "configs" / "vessels.yaml"
NO_NOISE_IMU = ImuConfig(gyro_noise_density=0.0, gyro_bias_rw=0.0, gyro_bias_init_std=0.0,
                         accel_noise_density=0.0, accel_bias_rw=0.0, accel_bias_init_std=0.0)


def truth(**kw) -> TruthSample:
    base = dict(t=0.0, n=0.0, e=0.0, psi=0.0, u=0.0, v=0.0, r=0.0, delta=0.0,
                u_dot=0.0, v_dot=0.0, r_dot=0.0, vn=0.0, ve=0.0)
    base.update(kw)
    return TruthSample(**base)


def test_imu_specific_force_formula():
    imu = PlanarImu(NO_NOISE_IMU, make_rngs(1)["imu"])
    m = imu.measure(0.0, truth(u=7.0, v=-0.3, r=0.02, u_dot=0.1, v_dot=0.05))
    np.testing.assert_allclose(m.z, [0.1 + 0.3 * 0.02, 0.05 + 7.0 * 0.02, 0.02], atol=1e-12)


def test_imu_matches_finite_difference_of_ground_velocity():
    """f_body = R(psi)^T * d(v_ground_NED)/dt must equal the formula used by the IMU."""
    v = Vessel(get_vessel("small_boat", VESSELS))
    u0, dt = v.p.nominal_speed, 0.01
    traj = simulate(v, v.initial_state(), ZigZag(0.26, 0.26, u0), 20.0, dt)
    imu = PlanarImu(replace(NO_NOISE_IMU, rate_hz=100.0), make_rngs(1)["imu"])
    vn, ve, psi = traj.array("vn"), traj.array("ve"), traj.array("psi")
    for k in range(10, len(traj) - 10, 50):
        a_n = (vn[k + 1] - vn[k - 1]) / (2 * dt)
        a_e = (ve[k + 1] - ve[k - 1]) / (2 * dt)
        f_x = np.cos(psi[k]) * a_n + np.sin(psi[k]) * a_e
        f_y = -np.sin(psi[k]) * a_n + np.cos(psi[k]) * a_e
        imu._last_tick = -1
        m = imu.measure(traj[k].t, traj[k])
        np.testing.assert_allclose(m.z[:2], [f_x, f_y], atol=2e-3)


def test_imu_allan_deviation_recovers_noise_parameters():
    N, K = 2e-4, 2e-4
    cfg = ImuConfig(rate_hz=10.0, gyro_noise_density=N, gyro_bias_rw=K, gyro_bias_init_std=0.0,
                    accel_noise_density=0.0, accel_bias_rw=0.0, accel_bias_init_std=0.0)
    imu = PlanarImu(cfg, make_rngs(11)["imu"])
    still = truth()
    gyro = np.array([imu.measure(k / cfg.rate_hz, still).z[2] for k in range(200_000)])  # 20000 s
    taus = np.array([1.0, 100.0])
    adev = allan_deviation(gyro, cfg.rate_hz, taus)
    # The Allan variances of the two noise terms add: s^2(tau) = N^2/tau + K^2 tau/3
    model = np.sqrt(N**2 / taus + K**2 * taus / 3.0)
    assert abs(adev[0] / model[0] - 1.0) < 0.03  # short tau: many clusters, tight
    assert abs(adev[1] / model[1] - 1.0) < 0.2  # long tau: few clusters, looser


def test_imu_bias_random_walk_variance_grows_linearly():
    cfg = ImuConfig(rate_hz=10.0, gyro_noise_density=0.0, gyro_bias_rw=1e-3, gyro_bias_init_std=0.0,
                    accel_noise_density=0.0, accel_bias_rw=0.0, accel_bias_init_std=0.0)
    finals = []
    for seed in range(200):
        imu = PlanarImu(cfg, make_rngs(seed)["imu"])
        for k in range(1001):  # 100 s
            m = imu.measure(k / cfg.rate_hz, truth())
        finals.append(m.info["bias"][2])
    assert abs(np.std(finals) / (cfg.gyro_bias_rw * np.sqrt(100.0)) - 1.0) < 0.15


def test_sensor_rates():
    v = Vessel(get_vessel("sailboat", VESSELS))
    traj = simulate(v, v.initial_state(), constant_command(0.0, v.p.nominal_speed), 10.0, 0.01)
    meas = simulate_sensors(traj, build_sensors(SensorsConfig(), make_rngs(2)))
    count = {name: sum(m.sensor == name for m in meas) for name in ("imu", "gps", "compass")}
    assert count == {"imu": 1001, "gps": 11, "compass": 101}  # ticks at t = 0 and t = 10 included
    assert all(a.t <= b.t for a, b in zip(meas, meas[1:]))


def test_gps_noise_statistics():
    cfg = GpsConfig(rate_hz=1.0, pos_sigma=2.5, vel_sigma=0.1)
    gps = Gps(cfg, make_rngs(3)["gps"])
    tr = truth(n=100.0, e=-50.0, vn=3.0, ve=1.0)
    z = np.array([gps.measure(float(k), tr).z for k in range(4000)])
    err = z - np.array([100.0, -50.0, 3.0, 1.0])
    assert np.all(np.abs(err.mean(axis=0)) < 0.1 * np.array([2.5, 2.5, 0.1, 0.1]) + 1e-12)
    np.testing.assert_allclose(err.std(axis=0), [2.5, 2.5, 0.1, 0.1], rtol=0.05)


def test_gps_dropout_window_and_independent_noise_stream():
    base = GpsConfig(rate_hz=1.0)
    win = GpsConfig(rate_hz=1.0, dropout_windows=((20.0, 50.0),))
    a, b = Gps(base, make_rngs(4)["gps"]), Gps(win, make_rngs(4)["gps"])
    tr = truth()
    za = {k: a.measure(float(k), tr) for k in range(100)}
    zb = {k: b.measure(float(k), tr) for k in range(100)}
    assert all(zb[k] is None for k in range(20, 50)) and all(zb[k] is not None for k in range(50, 100))
    for k in range(50, 100):  # noise after the window is unchanged by the window
        np.testing.assert_array_equal(za[k].z, zb[k].z)


def test_gps_random_dropout_rate():
    gps = Gps(GpsConfig(rate_hz=1.0, dropout_prob=0.3), make_rngs(5)["gps"])
    got = [gps.measure(float(k), truth()) is not None for k in range(5000)]
    assert abs(np.mean(got) - 0.7) < 0.03


def test_compass_wraps_and_applies_bias():
    cfg = CompassConfig(sigma=0.0, bias=0.2)
    c = Compass(cfg, make_rngs(6)["mag"])
    z = c.measure(0.0, truth(psi=np.radians(179.0))).z[0]
    np.testing.assert_allclose(z, wrap_angle(np.radians(179.0) + 0.2), atol=1e-12)
    assert -np.pi <= z < np.pi


def test_same_seed_same_measurements():
    def run(seed):
        sensors = build_sensors(SensorsConfig(), make_rngs(seed))
        return np.array([sensors[0].measure(k * 0.01, truth()).z for k in range(100)])
    assert np.array_equal(run(9), run(9)) and not np.array_equal(run(9), run(10))