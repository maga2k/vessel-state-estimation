import math
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from vessel_state_estimation.config import Config, EkfConfig, GpsConfig, ImuConfig, get_vessel
from vessel_state_estimation.estimation import PlanarEkf
from vessel_state_estimation.estimation.ekf import NX
from vessel_state_estimation.eval.montecarlo import evaluate, generate_runs
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sensors import Gps, Measurement
from vessel_state_estimation.sim import Environment, HeadingAutopilot, Vessel, simulate
from vessel_state_estimation.sim.truth import TruthSample

ROOT = Path(__file__).resolve().parents[1]
IMU = ImuConfig()


def truth() -> TruthSample:
    return TruthSample(t=0.0, n=0.0, e=0.0, psi=0.0, u=0.0, v=0.0, r=0.0, delta=0.0,
                       u_dot=0.0, v_dot=0.0, r_dot=0.0, vn=0.0, ve=0.0)


def test_gps_outliers_have_expected_rate_and_size():
    cfg = GpsConfig(rate_hz=1.0, pos_sigma=2.5, outlier_prob=0.05, outlier_sigma=30.0)
    rngs = make_rngs(7)
    gps = Gps(cfg, rngs["gps"], rngs["outlier"])
    ms = [gps.measure(float(k), truth()) for k in range(20000)]
    flags = np.array([m.info["outlier"] for m in ms])
    assert abs(flags.mean() - 0.05) < 0.01
    err = np.array([m.z[:2] for m in ms])
    assert abs(err[flags].std() / math.hypot(30.0, 2.5) - 1.0) < 0.1  # nominal noise + jump
    assert abs(err[~flags].std() / 2.5 - 1.0) < 0.05


def test_outlier_settings_do_not_change_nominal_fixes():
    def run(p):
        rngs = make_rngs(8)
        gps = Gps(GpsConfig(outlier_prob=p), rngs["gps"], rngs["outlier"])
        return [gps.measure(float(k), truth()) for k in range(300)]

    clean, dirty = run(0.0), run(0.5)
    n_checked = 0
    for a, b in zip(clean, dirty):
        if not b.info["outlier"]:
            np.testing.assert_array_equal(a.z, b.z)
            n_checked += 1
    assert n_checked > 100


def test_gps_reports_scaled_covariance():
    gps = Gps(GpsConfig(pos_sigma=4.0, vel_sigma=0.2, r_report_scale=0.5), make_rngs(1)["gps"])
    m = gps.measure(0.0, truth())
    np.testing.assert_allclose(np.diag(m.R), [2.0**2, 2.0**2, 0.1**2, 0.1**2])


def test_gate_rejects_outlier_and_keeps_state_unchanged():
    x0 = np.zeros(NX)
    R = np.diag([2.5**2, 2.5**2, 0.1**2, 0.1**2])
    far = Measurement(0.0, "gps", np.array([100.0, 0.0, 0.0, 0.0]), R)
    near = Measurement(0.0, "gps", np.array([1.0, -1.0, 0.05, 0.0]), R)

    gated = PlanarEkf(EkfConfig(gate_prob=0.999), IMU, x0)
    assert abs(gated._gate_threshold(4) - chi2.ppf(0.999, 4)) < 1e-9
    x_before, P_before = gated.x.copy(), gated.P.copy()
    inn = gated.update(far)
    assert not inn.accepted and inn.nis > chi2.ppf(0.999, 4)
    np.testing.assert_array_equal(gated.x, x_before)
    np.testing.assert_array_equal(gated.P, P_before)
    assert gated.update(near).accepted

    ungated = PlanarEkf(EkfConfig(gate_prob=0.0), IMU, x0)
    assert ungated.update(far).accepted and ungated.x[0] > 10.0  # swallowed: the state jumps


def test_config_validation_of_new_fields():
    for bad in (lambda: EkfConfig(gate_prob=1.0), lambda: GpsConfig(outlier_prob=2.0),
                lambda: GpsConfig(r_report_scale=0.0)):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def _setup(sensors_mod=None, n_runs=6, duration=150.0, dt=0.02):
    cfg = Config.from_yaml(ROOT / "configs" / "default.yaml")
    cfg = replace(cfg, sim=replace(cfg.sim, dt=dt),
                  sensors=replace(cfg.sensors, imu=replace(cfg.sensors.imu, rate_hz=1.0 / dt)))
    v = Vessel(get_vessel("small_boat", ROOT / "configs" / "vessels.yaml"))
    marks = [(0.0, 0.0), (0.2, 40.0), (0.5, -30.0), (0.75, 60.0)]
    ap = HeadingAutopilot(v, v.p.nominal_speed, [(f * duration, math.radians(h)) for f, h in marks])
    env = Environment(cfg.env, make_rngs(cfg.sim.seed)["disturbance"], dt, duration)
    return cfg, simulate(v, v.initial_state(), ap, duration, dt, environment=env), n_runs


def test_robustness_scenarios_degrade_in_the_expected_way():
    cfg, traj, n = _setup()
    s, win = cfg.sensors, (60.0, 120.0)  # the 60 s outage window: it contains two turns (75, 112 s)

    def run(sensors, ekf=cfg.ekf):
        res = evaluate(generate_runs(replace(cfg, sensors=sensors), traj, n), traj, ekf, s.imu)
        pe = res.position_error()
        res.pos_all = float(np.sqrt(np.mean(pe**2)))
        res.pos_win = float(np.sqrt(np.mean(pe[:, res.window_mask(*win)] ** 2)))
        return res

    def gps(**kw):
        return replace(s, gps=replace(s.gps, **kw))

    base = run(s)
    outage = run(gps(dropout_windows=(win,)))
    outage_nc = run(gps(dropout_windows=(win,)), replace(cfg.ekf, use_compass=False))
    out_raw = run(gps(outlier_prob=0.03))
    out_gated = run(gps(outlier_prob=0.03), replace(cfg.ekf, gate_prob=0.999))
    comp = run(replace(s, compass=replace(s.compass, bias=math.radians(3.0))))
    imu3 = run(replace(s, imu=replace(s.imu, gyro_noise_density=3 * s.imu.gyro_noise_density,
                                      gyro_bias_rw=3 * s.imu.gyro_bias_rw,
                                      accel_noise_density=3 * s.imu.accel_noise_density,
                                      accel_bias_rw=3 * s.imu.accel_bias_rw)))

    # outage: error grows, but the covariance grows with it (the filter stays honest)
    assert outage.pos_win > 2.0 * base.pos_win
    assert outage.anees() < 1.5 * base.anees()
    # without the compass the dead-reckoning in the turn is much worse
    assert outage_nc.pos_win > 2.0 * outage.pos_win
    # outliers hurt; the gate removes most of the damage and almost never rejects good fixes
    assert out_raw.pos_all > 1.3 * base.pos_all and out_gated.pos_all < 1.15 * base.pos_all
    g = out_gated.gate
    assert g["outliers_rejected"] / g["outliers"] > 0.7
    assert g["nominal_rejected"] / g["nominal"] < 0.02
    # unmodeled compass bias: the heading error equals the bias, and NEES flags it loudly
    assert abs(math.degrees(comp.rms_error()[2]) / 3.0 - 1.0) < 0.2
    assert comp.anees() > 10.0 * base.anees()
    # IMU noise worse than assumed: consistency is lost, accuracy hardly changes
    assert imu3.anees() > 2.0 * base.anees()
    assert abs(imu3.pos_all / base.pos_all - 1.0) < 0.25