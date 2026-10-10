import math
from dataclasses import replace
from pathlib import Path

import numpy as np

from vessel_state_estimation.config import Config, ImuConfig, MekfConfig, WaveConfig, get_vessel
from vessel_state_estimation.estimation.mekf import ATT, BIAS, Mekf, initial_covariance
from vessel_state_estimation.estimation.mekf_runner import run_mekf
from vessel_state_estimation.eval.attitude import (attitude_error, evaluate_attitude,
                                                   generate_attitude_runs)
from vessel_state_estimation.quaternion import (quat_conj, quat_from_euler, quat_from_rotvec,
                                                quat_mult, rot_from_quat, rotvec_from_quat, skew)
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sensors.imu3d import Imu3d, simulate_imu3d
from vessel_state_estimation.sim import (Environment, HeadingAutopilot, Vessel, constant_command,
                                         simulate)
from vessel_state_estimation.sim.attitude import G, AttitudeTruth, generate_attitude

ROOT = Path(__file__).resolve().parents[1]
VESSELS = ROOT / "configs" / "vessels.yaml"
NO_NOISE = ImuConfig(gyro_noise_density=0.0, gyro_bias_rw=0.0, gyro_bias_init_std=0.0,
                     accel_noise_density=0.0, accel_bias_rw=0.0, accel_bias_init_std=0.0)


def planar_traj(duration=60.0, dt=0.02, rudder=0.0):
    v = Vessel(get_vessel("small_boat", VESSELS))
    return v, simulate(v, v.initial_state(), constant_command(rudder, v.p.nominal_speed), duration, dt)


def test_specific_force_and_kinematics_of_the_synthetic_truth():
    v, traj = planar_traj(rudder=0.2)
    truth = generate_attitude(traj, WaveConfig(), make_rngs(5)["waves"])
    dt = truth.t[1] - truth.t[0]
    for k in range(0, len(truth) - 1, 97):
        R = rot_from_quat(truth.q[k])
        # independent path: f_b = R^T (a - g), with R built from the quaternion
        np.testing.assert_allclose(truth.f_b[k], R.T @ (truth.a_ned[k] - np.array([0, 0, G])), atol=1e-9)
        # the body angular velocity (from Euler rates) must be the one that rotates q to q[k+1]
        q_next = quat_mult(truth.q[k], quat_from_rotvec(truth.omega_b[k] * dt))
        err = rotvec_from_quat(quat_mult(quat_conj(q_next), truth.q[k + 1]))
        assert np.linalg.norm(err) < 2e-4


def test_wave_statistics_and_options():
    v, traj = planar_traj(duration=600.0)
    cfg = WaveConfig(roll_sigma_deg=4.0, pitch_sigma_deg=2.0, accel_sigma=0.5, include_maneuver_accel=False)
    truth = generate_attitude(traj, cfg, make_rngs(6)["waves"])
    assert abs(np.degrees(truth.euler[:, 0].std()) / 4.0 - 1.0) < 0.15
    assert abs(np.degrees(truth.euler[:, 1].std()) / 2.0 - 1.0) < 0.15
    assert abs(truth.a_ned[:, 2].std() / 0.5 - 1.0) < 0.15
    calm = generate_attitude(traj, WaveConfig(roll_sigma_deg=0, pitch_sigma_deg=0, accel_sigma=0.0,
                                              include_maneuver_accel=False), make_rngs(6)["waves"])
    np.testing.assert_allclose(calm.f_b, np.tile([0.0, 0.0, -G], (len(calm), 1)), atol=1e-12)


def test_imu3d_noise_free_returns_the_truth():
    v, traj = planar_traj(duration=5.0)
    truth = generate_attitude(traj, WaveConfig(), make_rngs(7)["waves"])
    meas = simulate_imu3d(truth, Imu3d(replace(NO_NOISE, rate_hz=50.0), make_rngs(7)["imu3d"]))
    assert len(meas) == len(truth)
    np.testing.assert_allclose(meas[10].z, np.concatenate([truth.f_b[10], truth.omega_b[10]]), atol=1e-12)


def test_mekf_jacobians_match_numerical_derivatives():
    rng = np.random.default_rng(8)
    q = quat_from_rotvec(rng.normal(size=3) * 0.5)
    b = rng.normal(size=3) * 1e-3
    gyro, dt, eps = np.array([0.2, -0.1, 0.3]), 0.02, 1e-6

    # propagation: error of a perturbed state w.r.t. the nominal one, after the same gyro input
    def propagated_error(dx):
        nom = Mekf(MekfConfig(), NO_NOISE, q, b)
        pert = Mekf(MekfConfig(), NO_NOISE, quat_mult(q, quat_from_rotvec(dx[ATT])), b + dx[BIAS])
        nom.predict(gyro, dt)
        pert.predict(gyro, dt)
        return np.concatenate([rotvec_from_quat(quat_mult(quat_conj(nom.q), pert.q)), pert.b - nom.b])

    F_num = np.zeros((6, 6))
    for j in range(6):
        d = np.zeros(6)
        d[j] = eps
        F_num[:, j] = (propagated_error(d) - propagated_error(-d)) / (2 * eps)
    F = Mekf(MekfConfig(), NO_NOISE, q, b).error_transition(gyro, dt)
    np.testing.assert_allclose(F, F_num, atol=5e-4)  # the -dt*I block is first order in dt

    # measurement: f_hat(q (x) exp(dtheta)) = f_hat + [f_hat]x dtheta
    f0 = Mekf(MekfConfig(), NO_NOISE, q, b).expected_specific_force()
    H_num = np.zeros((3, 3))
    for j in range(3):
        d = np.zeros(3)
        d[j] = eps
        up = Mekf(MekfConfig(), NO_NOISE, quat_mult(q, quat_from_rotvec(d)), b).expected_specific_force()
        dn = Mekf(MekfConfig(), NO_NOISE, quat_mult(q, quat_from_rotvec(-d)), b).expected_specific_force()
        H_num[:, j] = (up - dn) / (2 * eps)
    np.testing.assert_allclose(H_num, skew(f0), atol=1e-6)


def test_gate_rejects_a_violent_accelerometer_sample():
    mk = Mekf(MekfConfig(gate_prob=0.999), NO_NOISE, np.array([1.0, 0, 0, 0]), np.zeros(3))
    inn = mk.update_accel(np.array([30.0, 0.0, -G]), 0.01)
    assert not inn.accepted
    assert mk.update_accel(np.array([0.1, 0.0, -G]), 0.01).accepted


def _static_truth(n=3000, dt=0.02, roll=0.15, pitch=-0.08, yaw=0.5):
    t = np.arange(n) * dt
    q = np.tile(quat_from_euler(roll, pitch, yaw), (n, 1))
    R = rot_from_quat(q[0])
    f_b = np.tile(R.T @ np.array([0.0, 0.0, -G]), (n, 1))
    return AttitudeTruth(t, q, np.tile([roll, pitch, yaw], (n, 1)), np.zeros((n, 3)), f_b, np.zeros((n, 3)))


def test_mekf_static_convergence_and_unobservable_yaw():
    truth = _static_truth()
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    imu = replace(cfg.sensors.imu, rate_hz=50.0)
    meas = simulate_imu3d(truth, Imu3d(imu, make_rngs(9)["imu3d"]))
    P0 = initial_covariance(cfg.mekf)
    q0 = quat_mult(truth.q[0], quat_from_rotvec(np.array([0.2, -0.15, 0.25])))  # ~19 deg off
    log = run_mekf(Mekf(cfg.mekf, imu, q0, np.zeros(3), P0), meas, imu.rate_hz)

    dth = attitude_error(log, np.tile(truth.q[0], (len(log.t), 1)))
    R_end = rot_from_quat(log.q[-1])
    assert np.degrees(np.linalg.norm((R_end @ dth[-1])[:2])) < 0.5  # roll / pitch converge
    assert np.all(np.abs(log.b[-1, :2] - log.true_bias[-1, :2]) < 1.5e-3)  # x, y gyro biases too

    # heading is unobservable from gravity: its uncertainty does not shrink, tilt's does
    R0 = rot_from_quat(log.q[0])
    P_n0 = R0 @ log.P[0, :3, :3] @ R0.T
    P_n1 = R_end @ log.P[-1, :3, :3] @ R_end.T
    assert P_n1[2, 2] > P_n0[2, 2] and P_n1[0, 0] < 0.1 * P_n0[0, 0]


def test_turn_compensation_removes_the_coordinated_turn_error():
    v, traj = planar_traj(duration=60.0, rudder=math.radians(15.0))
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    dt = 0.02
    cfg = replace(cfg, sim=replace(cfg.sim, dt=dt), sensors=replace(
        cfg.sensors, imu=replace(cfg.sensors.imu, rate_hz=1 / dt)), waves=WaveConfig(
        roll_sigma_deg=0.0, pitch_sigma_deg=0.0, accel_sigma=0.0, include_maneuver_accel=True))
    truth = generate_attitude(traj, cfg.waves, make_rngs(10)["waves"])
    runs = generate_attitude_runs(cfg, traj, truth, 2)
    out = {}
    for comp in (True, False):
        res = evaluate_attitude(runs, truth, replace(cfg.mekf, turn_compensation=comp),
                                cfg.sensors.imu, cfg.sensors.imu.rate_hz)
        late = res.t > 30.0  # after the filter has converged, steady turn
        out[comp] = np.degrees(np.abs(res.euler_err[:, late, 0]).mean())
    assert out[False] > 3.0  # the tilted apparent gravity of the turn is read as real roll
    assert out[True] < 1.0


def test_monte_carlo_attitude_is_consistent_with_waves_and_maneuvers():
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    dt, d = 0.02, 150.0
    cfg = replace(cfg, sim=replace(cfg.sim, dt=dt),
                  sensors=replace(cfg.sensors, imu=replace(cfg.sensors.imu, rate_hz=1 / dt)))
    v = Vessel(get_vessel("small_boat", VESSELS))
    marks = [(0.0, 0.0), (0.2, 40.0), (0.5, -30.0), (0.75, 60.0)]
    ap = HeadingAutopilot(v, v.p.nominal_speed, [(f * d, math.radians(h)) for f, h in marks])
    env = Environment(cfg.env, make_rngs(cfg.sim.seed)["disturbance"], dt, d)
    traj = simulate(v, v.initial_state(), ap, d, dt, environment=env)
    truth = generate_attitude(traj, cfg.waves, make_rngs(cfg.sim.seed)["waves"])
    runs = generate_attitude_runs(cfg, traj, truth, 6)
    res = evaluate_attitude(runs, truth, cfg.mekf, cfg.sensors.imu, cfg.sensors.imu.rate_hz)

    roll_rms, pitch_rms, _ = res.rms_deg()
    assert roll_rms < 1.0 and pitch_rms < 0.6  # waves of 4 deg / 2 deg rms are tracked
    assert 1.0 < res.nees_tilt.mean() < 3.5  # ideal 2: consistent roll / pitch
    assert 3.5 < res.nees_full.mean() < 9.0  # ideal 6: consistent, yaw drift included
    assert res.yaw_sigma[:, -1].mean() > 3.0 * res.yaw_sigma[:, 0].mean()  # yaw uncertainty grows