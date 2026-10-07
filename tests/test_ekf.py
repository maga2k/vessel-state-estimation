import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from vessel_state_estimation.config import Config, EkfConfig, ImuConfig, get_vessel
from vessel_state_estimation.estimation import (PlanarEkf, draw_initial_estimate,
                                                initial_covariance, run_filter)
from vessel_state_estimation.estimation.ekf import (B_AX, B_AY, B_G, NX, POS_E, POS_N, PSI, VEL_E,
                                                    VEL_N, process_noise, propagate,
                                                    transition_jacobian)
from vessel_state_estimation.eval.errors import rms, state_errors, true_states
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sensors import Measurement, PlanarImu, build_sensors, simulate_sensors
from vessel_state_estimation.sim import (Environment, HeadingAutopilot, Vessel, ZigZag, simulate)

ROOT = Path(__file__).resolve().parents[1]
VESSELS = ROOT / "configs" / "vessels.yaml"
IMU = ImuConfig()


def random_state_and_input(seed: int):
    r = np.random.default_rng(seed)
    x = np.concatenate([r.normal(0, 100, 2), [r.uniform(-2.0, 2.0)], r.normal(0, 5, 2),
                        r.normal(0, 0.05, 2), [r.normal(0, 0.01)]])
    u = np.array([r.normal(0, 1), r.normal(0, 1), r.normal(0, 0.2)])
    return x, u


def test_jacobian_matches_finite_differences():
    for seed in range(5):
        x, u = random_state_and_input(seed)
        dt, eps = 0.1, 1e-6
        F = transition_jacobian(x, u, dt)
        F_num = np.zeros((NX, NX))
        for j in range(NX):
            d = np.zeros(NX)
            d[j] = eps
            F_num[:, j] = (propagate(x + d, u, dt) - propagate(x - d, u, dt)) / (2 * eps)
        np.testing.assert_allclose(F, F_num, atol=1e-7)


def test_process_noise_matches_accelerometer_noise_propagation():
    """Q's position/velocity blocks = G S G^T with G = d f / d (accel input); Q is independent of psi."""
    dt = 0.01
    sa2 = IMU.accel_noise_density**2 / dt
    sg2 = IMU.gyro_noise_density**2 / dt
    Q = process_noise(dt, IMU)
    for psi in (-2.0, 0.3, 1.7):
        x = np.array([0.0, 0.0, psi, 3.0, -1.0, 0.0, 0.0, 0.0])
        u = np.array([0.5, -0.2, 0.1])
        G = np.zeros((NX, 2))
        for j in range(2):  # accelerometer columns; propagate is linear in u
            du = np.zeros(3)
            du[j] = 1.0
            G[:, j] = propagate(x, u + du, dt) - propagate(x, u, dt)
        np.testing.assert_allclose(G @ (sa2 * np.eye(2)) @ G.T, Q * _mask_accel(), rtol=1e-6, atol=1e-20)
    assert abs(Q[PSI, PSI] - dt**2 * sg2) < 1e-20
    assert abs(Q[B_G, B_G] - IMU.gyro_bias_rw**2 * dt) < 1e-24


def _mask_accel():
    m = np.zeros((NX, NX))
    idx = [POS_N, POS_E, VEL_N, VEL_E]
    m[np.ix_(idx, idx)] = 1.0
    return m


def test_compass_innovation_is_wrapped():
    x0 = np.zeros(NX)
    x0[PSI] = 3.1
    ekf = PlanarEkf(EkfConfig(), IMU, x0)
    meas = Measurement(0.0, "compass", np.array([-3.1]), np.array([[0.0175**2]]))
    y = ekf.update(meas).y[0]
    np.testing.assert_allclose(y, 2 * math.pi - 6.2, atol=1e-9)  # +0.083 rad, not -6.2
    assert -math.pi <= ekf.x[PSI] < math.pi


def test_update_keeps_covariance_symmetric_psd_and_reduces_uncertainty():
    x0 = np.zeros(NX)
    ekf = PlanarEkf(EkfConfig(), IMU, x0)
    P_before = ekf.P.copy()
    R = np.diag([2.5**2, 2.5**2, 0.1**2, 0.1**2])
    inn = ekf.update(Measurement(0.0, "gps", np.array([1.0, -1.0, 0.1, 0.0]), R))
    P = ekf.P
    np.testing.assert_allclose(P, P.T, atol=1e-15)
    assert np.linalg.eigvalsh(P).min() > 0.0
    assert P[POS_N, POS_N] < P_before[POS_N, POS_N] and P[VEL_E, VEL_E] < P_before[VEL_E, VEL_E]
    assert inn.nis > 0.0 and inn.S.shape == (4, 4)


def test_dead_reckoning_with_ideal_imu_tracks_truth():
    """Prediction only, noise-free IMU, exact initial state: validates f and the integration."""
    ideal = replace(IMU, rate_hz=50.0, gyro_noise_density=0.0, gyro_bias_rw=0.0,
                    gyro_bias_init_std=0.0, accel_noise_density=0.0, accel_bias_rw=0.0,
                    accel_bias_init_std=0.0)
    for name in ("merchant_ship", "sailboat", "small_boat"):
        v = Vessel(get_vessel(name, VESSELS))
        m, u0, dt = v.p.maneuvers, v.p.nominal_speed, 0.02
        zz = ZigZag(math.radians(m.zigzag_rudder_deg), math.radians(m.zigzag_heading_deg), u0)
        traj = simulate(v, v.initial_state(), zz, 60.0, dt)
        meas = simulate_sensors(traj, [PlanarImu(ideal, make_rngs(1)["imu"])])
        t0 = traj[0]
        x0 = np.array([t0.n, t0.e, t0.psi, t0.vn, t0.ve, 0.0, 0.0, 0.0])
        log = run_filter(PlanarEkf(EkfConfig(), ideal, x0), meas, log_every=1)
        err = state_errors(log, true_states(traj, log))
        assert np.hypot(err[-1, POS_N], err[-1, POS_E]) < 0.1  # was 3.8 m with a frozen heading


def test_pipeline_beats_raw_gps_and_compass_fixes_heading():
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    imu = replace(cfg.sensors.imu, rate_hz=50.0)
    sensors_cfg = replace(cfg.sensors, imu=imu)
    v = Vessel(get_vessel("small_boat", VESSELS))
    dt, d, u0 = 0.02, 300.0, v.p.nominal_speed
    rngs = make_rngs(cfg.sim.seed)
    marks = [(0.0, 0.0), (0.2, 40.0), (0.5, -30.0), (0.75, 60.0)]
    ap = HeadingAutopilot(v, u0, [(f * d, math.radians(h)) for f, h in marks])
    env = Environment(cfg.env, rngs["disturbance"], dt, d)
    traj = simulate(v, v.initial_state(), ap, d, dt, environment=env)
    meas = simulate_sensors(traj, build_sensors(sensors_cfg, rngs))

    P0 = initial_covariance(cfg.ekf)
    x0 = draw_initial_estimate(traj[0], P0, rngs["init"])
    results = {}
    for use_compass in (True, False):
        ekf = PlanarEkf(replace(cfg.ekf, use_compass=use_compass), imu, x0, P0)
        log = run_filter(ekf, meas)
        results[use_compass] = rms(state_errors(log, true_states(traj, log)))

    gps = [m for m in meas if m.sensor == "gps"]
    t = np.array([m.t for m in gps])
    z = np.array([m.z for m in gps])
    raw = np.sqrt(np.mean((z[:, 0] - np.interp(t, traj.array("t"), traj.array("n"))) ** 2
                          + (z[:, 1] - np.interp(t, traj.array("t"), traj.array("e"))) ** 2))
    with_c, without_c = results[True], results[False]
    assert np.hypot(with_c[POS_N], with_c[POS_E]) < 0.5 * raw  # fusion beats the raw GPS
    assert math.degrees(with_c[PSI]) < 1.0
    assert without_c[PSI] > 3.0 * with_c[PSI]  # heading is poorly observable without the compass