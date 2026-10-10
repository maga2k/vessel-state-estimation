#!/usr/bin/env python
"""Robustness study: the same EKF under GPS outages, outliers, unmodeled biases and mismatch.

Every scenario is a Monte Carlo (same true trajectory, independent noise). The EKF always ASSUMES
the nominal sensor and IMU parameters; only the data (and, where stated, the filter settings) change.

Examples:
    python scripts/robustness.py --runs 15
    python scripts/robustness.py --runs 15 --outage-start 100
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np

from vessel_state_estimation.config import Config, EkfConfig, SensorsConfig, get_vessel
from vessel_state_estimation.eval.montecarlo import evaluate, generate_runs
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sim import Environment, HeadingAutopilot, Vessel, simulate

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Scenario:
    name: str
    sensors: SensorsConfig  # what really happens in the data
    ekf: EkfConfig  # how the filter is set up


def scale_imu(imu, k: float):
    return replace(imu, gyro_noise_density=k * imu.gyro_noise_density, gyro_bias_rw=k * imu.gyro_bias_rw,
                   accel_noise_density=k * imu.accel_noise_density, accel_bias_rw=k * imu.accel_bias_rw)


def build_scenarios(cfg: Config, outage_start: float) -> list[Scenario]:
    s, e = cfg.sensors, cfg.ekf
    gps = s.gps

    def with_gps(**kw):
        return replace(s, gps=replace(gps, **kw))

    return [
        Scenario("baseline", s, e),
        Scenario("GPS outage 30 s", with_gps(dropout_windows=((outage_start, outage_start + 30.0),)), e),
        Scenario("GPS outage 60 s", with_gps(dropout_windows=((outage_start, outage_start + 60.0),)), e),
        Scenario("outliers, no gate", with_gps(outlier_prob=0.02), replace(e, gate_prob=0.0)),
        Scenario("outliers, gate 99.9%", with_gps(outlier_prob=0.02), replace(e, gate_prob=0.999)),
        Scenario("compass bias 3 deg", replace(s, compass=replace(s.compass, bias=math.radians(3.0))), e),
        Scenario("IMU noise x3", replace(s, imu=scale_imu(s.imu, 3.0)), e),
        # real GPS noise doubles, but the sensor keeps reporting the nominal R
        Scenario("GPS noise x2", with_gps(pos_sigma=2 * gps.pos_sigma, vel_sigma=2 * gps.vel_sigma,
                                          r_report_scale=0.5), e),
        Scenario("outage 60 s, no compass", with_gps(dropout_windows=((outage_start, outage_start + 60.0),)),
                 replace(e, use_compass=False)),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    p.add_argument("--vessels", default=ROOT / "configs" / "vessels.yaml")
    p.add_argument("--vessel")
    p.add_argument("--runs", type=int, default=15)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--outage-start", type=float, default=120.0)
    p.add_argument("--out", default=ROOT / "results")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config.from_yaml(args.config)
    cfg = replace(cfg, sim=replace(cfg.sim, dt=args.dt),
                  sensors=replace(cfg.sensors, imu=replace(cfg.sensors.imu, rate_hz=1.0 / args.dt)))
    params = get_vessel(args.vessel or cfg.vessel, args.vessels)
    vessel, d = Vessel(params), args.duration
    marks = [(0.0, 0.0), (0.2, 40.0), (0.5, -30.0), (0.75, 60.0)]
    autopilot = HeadingAutopilot(vessel, params.nominal_speed,
                                 [(f * d, math.radians(h)) for f, h in marks])
    env = Environment(cfg.env, make_rngs(cfg.sim.seed)["disturbance"], args.dt, d)
    traj = simulate(vessel, vessel.initial_state(), autopilot, d, args.dt, environment=env)

    win = (args.outage_start, args.outage_start + 60.0)  # common comparison window (the 60 s outage)
    print(f"{params.name}: {args.runs} runs x {d:.0f} s; comparison window {win[0]:.0f}-{win[1]:.0f} s "
          f"(turn at {0.5 * d:.0f} s)")
    header = (f"{'scenario':<22}{'pos rms':>8}{'in win':>8}{'peak':>8}{'psi rms':>9}{'ANEES':>8}"
              f"{'NIS gps':>9}  gate (detected / false)")
    print(header)
    print("-" * len(header))

    curves = {}
    for sc in build_scenarios(cfg, args.outage_start):
        runs = generate_runs(replace(cfg, sensors=sc.sensors), traj, args.runs)
        res = evaluate(runs, traj, sc.ekf, cfg.sensors.imu)  # filter ASSUMES the nominal IMU
        pe = res.position_error()
        mask = res.window_mask(*win)
        rms_t = np.sqrt(np.mean(pe**2, axis=0))
        g = res.gate
        gate = ""
        if sc.ekf.gate_prob > 0.0 and g["outliers"]:
            gate = (f"{g['outliers_rejected']}/{g['outliers']} outliers, "
                    f"{g['nominal_rejected']}/{g['nominal']} nominal")
        print(f"{sc.name:<22}{np.sqrt(np.mean(pe**2)):8.2f}{np.sqrt(np.mean(pe[:, mask]**2)):8.2f}"
              f"{rms_t.max():8.1f}{math.degrees(res.rms_error()[2]):9.2f}{res.anees():8.1f}"
              f"{res.mean_nis('gps'):9.1f}  {gate}")
        curves[sc.name] = (res.t, rms_t, np.degrees(np.sqrt(np.mean(res.errors[:, :, 2] ** 2, axis=0))))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, (t, pos, psi) in curves.items():
        ax[0].semilogy(t, pos, label=name)
        ax[1].semilogy(t, psi, label=name)
    for a, title, unit in ((ax[0], "Position error (rms over runs)", "m"),
                           (ax[1], "Heading error (rms over runs)", "deg")):
        a.axvspan(*win, color="r", alpha=0.07)
        a.set(xlabel="t [s]", ylabel=unit, title=title)
        a.grid(True, which="both")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{params.name}: robustness scenarios (red band: 60 s outage window)")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{params.name}_robustness.png", dpi=130)
    print(f"saved {out / (params.name + '_robustness.png')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()