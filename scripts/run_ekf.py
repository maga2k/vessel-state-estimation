#!/usr/bin/env python
"""Full pipeline: simulate -> sensors -> EKF (with and without compass) -> errors and plots."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from vessel_state_estimation.config import Config, get_vessel
from vessel_state_estimation.estimation import (PlanarEkf, draw_initial_estimate,
                                                initial_covariance, run_filter)
from vessel_state_estimation.estimation.ekf import B_AX, B_G, PSI, POS_N
from vessel_state_estimation.eval.errors import rms, state_errors, true_states
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sensors import build_sensors, simulate_sensors
from vessel_state_estimation.sim import Environment, HeadingAutopilot, Vessel, simulate

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    p.add_argument("--vessels", default=ROOT / "configs" / "vessels.yaml")
    p.add_argument("--vessel")
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument("--dropout", type=float, nargs=2, metavar=("T0", "T1"),
                   help="GPS outage window [s]")
    p.add_argument("--out", default=ROOT / "results")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config.from_yaml(args.config)
    params = get_vessel(args.vessel or cfg.vessel, args.vessels)
    vessel = Vessel(params)
    dt, u0, d = cfg.sim.dt, params.nominal_speed, args.duration
    rngs = make_rngs(cfg.sim.seed)

    # mostly straight legs with three course changes (heading is weakly observable on the legs)
    marks = [(0.0, 0.0), (0.2, 40.0), (0.5, -30.0), (0.75, 60.0)]
    autopilot = HeadingAutopilot(vessel, u0, [(f * d, math.radians(h)) for f, h in marks])
    env = Environment(cfg.env, rngs["disturbance"], dt, d)
    traj = simulate(vessel, vessel.initial_state(), autopilot, d, dt, environment=env)

    sensors_cfg = cfg.sensors
    if args.dropout:
        sensors_cfg = replace(sensors_cfg, gps=replace(sensors_cfg.gps,
                                                       dropout_windows=(tuple(args.dropout),)))
    meas = simulate_sensors(traj, build_sensors(sensors_cfg, rngs))

    P0 = initial_covariance(cfg.ekf)
    x0 = draw_initial_estimate(traj[0], P0, rngs["init"])

    logs = {}
    for name, use_compass in (("with compass", True), ("without compass", False)):
        ekf_cfg = replace(cfg.ekf, use_compass=use_compass)
        ekf = PlanarEkf(ekf_cfg, cfg.sensors.imu, x0, P0)  # assumed IMU noise = the true one
        logs[name] = run_filter(ekf, meas)

    # ---- numbers
    gps = [m for m in meas if m.sensor == "gps"]
    t_gps = np.array([m.t for m in gps])
    z = np.array([m.z for m in gps])
    raw = np.hypot(z[:, 0] - np.interp(t_gps, traj.array("t"), traj.array("n")),
                   z[:, 1] - np.interp(t_gps, traj.array("t"), traj.array("e")))
    print(f"{params.name}, {d:.0f} s" + (f", GPS outage {args.dropout[0]:.0f}-{args.dropout[1]:.0f} s"
                                         if args.dropout else ""))
    print(f"  raw GPS position error rms: {np.sqrt(np.mean(raw**2)):.2f} m")
    for name, log in logs.items():
        err = state_errors(log, true_states(traj, log))
        r = rms(err)
        print(f"  EKF {name:<16}: position {np.hypot(r[0], r[1]):.2f} m, velocity "
              f"{np.hypot(r[3], r[4]):.3f} m/s, heading {math.degrees(r[PSI]):.2f} deg, "
              f"gyro bias {math.degrees(r[B_G]) * 3600:.0f} deg/h")

    # ---- plots (errors with +/-3 sigma from the filter's own covariance)
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    full = logs["with compass"]
    err = state_errors(full, true_states(traj, full))
    sig = np.sqrt(np.array([np.diag(P) for P in full.P]))

    ax[0, 0].plot(full.t, err[:, POS_N], label="error (N)")
    ax[0, 0].fill_between(full.t, -3 * sig[:, POS_N], 3 * sig[:, POS_N], alpha=0.2, label="+/- 3 sigma")
    ax[0, 0].set(xlabel="t [s]", ylabel="m", title="North position error")

    for name, color in (("with compass", "C0"), ("without compass", "C3")):
        log = logs[name]
        e = state_errors(log, true_states(traj, log))
        s = np.sqrt(np.array([P[PSI, PSI] for P in log.P]))
        ax[0, 1].plot(log.t, np.degrees(e[:, PSI]), color=color, label=f"error, {name}")
        ax[0, 1].fill_between(log.t, -3 * np.degrees(s), 3 * np.degrees(s), color=color, alpha=0.12)
    ax[0, 1].set(xlabel="t [s]", ylabel="deg", title="Heading error (bands: +/- 3 sigma)")

    ax[1, 0].plot(full.t, np.degrees(full.x[:, B_G]), label="estimated")
    ax[1, 0].plot(full.t, np.degrees(full.true_bias[:, 2]), "k--", label="true")
    ax[1, 0].set(xlabel="t [s]", ylabel="deg/s", title="Gyro bias")

    ax[1, 1].plot(full.t, full.x[:, B_AX], label="estimated")
    ax[1, 1].plot(full.t, full.true_bias[:, 0], "k--", label="true")
    ax[1, 1].set(xlabel="t [s]", ylabel="m/s^2", title="Accelerometer x bias")
    for a in ax.flat:
        a.grid(True)
        a.legend()
    if args.dropout:
        for a in ax.flat:
            a.axvspan(*args.dropout, color="r", alpha=0.08)
    fig.suptitle(f"{params.name}: EKF IMU + GPS + compass")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{params.name}_ekf.png", dpi=130)
    print(f"  saved {out / (params.name + '_ekf.png')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()