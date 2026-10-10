#!/usr/bin/env python
"""Attitude MEKF (gyro + accelerometer) on a synthetic 3D attitude with waves.

Pipeline: planar maneuver -> synthetic roll/pitch/accelerations (waves) -> IMU3D (+ GPS speed) -> MEKF.
With --runs N > 1 the consistency statistics (NEES / NIS) are computed over N noise realisations.

Examples:
    python scripts/run_mekf.py
    python scripts/run_mekf.py --runs 20
    python scripts/run_mekf.py --runs 20 --no-compensation
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from vessel_state_estimation.config import Config, get_vessel
from vessel_state_estimation.estimation.mekf import Mekf, initial_covariance
from vessel_state_estimation.estimation.mekf_runner import run_mekf
from vessel_state_estimation.eval.attitude import (attitude_error, evaluate_attitude,
                                                   generate_attitude_runs, truth_at)
from vessel_state_estimation.quaternion import euler_from_quat, rot_from_quat
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sim import Environment, HeadingAutopilot, Vessel, simulate
from vessel_state_estimation.sim.attitude import generate_attitude

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    p.add_argument("--vessels", default=ROOT / "configs" / "vessels.yaml")
    p.add_argument("--vessel")
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.02, help="sim step = 1 / IMU rate")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--no-compensation", action="store_true", help="do not compensate omega x v")
    p.add_argument("--out", default=ROOT / "results")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config.from_yaml(args.config)
    cfg = replace(cfg, sim=replace(cfg.sim, dt=args.dt),
                  sensors=replace(cfg.sensors, imu=replace(cfg.sensors.imu, rate_hz=1.0 / args.dt)),
                  mekf=replace(cfg.mekf, turn_compensation=not args.no_compensation))
    params = get_vessel(args.vessel or cfg.vessel, args.vessels)
    vessel, d = Vessel(params), args.duration

    marks = [(0.0, 0.0), (0.2, 40.0), (0.5, -30.0), (0.75, 60.0)]
    autopilot = HeadingAutopilot(vessel, params.nominal_speed,
                                 [(f * d, math.radians(h)) for f, h in marks])
    rngs = make_rngs(cfg.sim.seed)
    env = Environment(cfg.env, rngs["disturbance"], args.dt, d)
    traj = simulate(vessel, vessel.initial_state(), autopilot, d, args.dt, environment=env)
    truth = generate_attitude(traj, cfg.waves, rngs["waves"])
    runs = generate_attitude_runs(cfg, traj, truth, max(1, args.runs))

    rate = cfg.sensors.imu.rate_hz
    res = evaluate_attitude(runs, truth, cfg.mekf, cfg.sensors.imu, rate)
    roll, pitch, yaw = res.rms_deg()
    print(f"{params.name}: {len(runs)} run(s) x {d:.0f} s, IMU {rate:.0f} Hz, "
          f"turn compensation {'on' if cfg.mekf.turn_compensation else 'OFF'}")
    print(f"  true roll {np.degrees(truth.euler[:, 0].std()):.2f} deg rms, "
          f"pitch {np.degrees(truth.euler[:, 1].std()):.2f} deg rms")
    print(f"  error rms: roll {roll:.2f} deg, pitch {pitch:.2f} deg, yaw {yaw:.1f} deg (unobservable)")
    print(f"  NEES tilt {res.nees_tilt.mean():.2f} (ideal 2) | full {res.nees_full.mean():.2f} (ideal 6) | "
          f"NIS accel {res.nis.mean():.2f} (ideal 3)")

    # ---- plots for run 0
    P0 = initial_covariance(cfg.mekf)
    run = runs[0]
    log = run_mekf(Mekf(cfg.mekf, cfg.sensors.imu, run.q0, run.b0, P0), run.measurements, rate)
    q_true, e_true = truth_at(truth, log)
    dth = attitude_error(log, q_true)
    est = euler_from_quat(log.q)
    err_n, sig_n = [], []
    for k in range(len(log.t)):
        R = rot_from_quat(log.q[k])
        err_n.append(R @ dth[k])
        sig_n.append(np.sqrt(np.diag(R @ log.P[k, :3, :3] @ R.T)))
    err_n, sig_n = np.degrees(np.array(err_n)), np.degrees(np.array(sig_n))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    w = log.t <= 60.0
    ax[0, 0].plot(log.t[w], np.degrees(e_true[w, 0]), "k", label="roll true")
    ax[0, 0].plot(log.t[w], np.degrees(est[w, 0]), "C0--", label="roll estimate")
    ax[0, 0].plot(log.t[w], np.degrees(e_true[w, 1]), "gray", label="pitch true")
    ax[0, 0].plot(log.t[w], np.degrees(est[w, 1]), "C1--", label="pitch estimate")
    ax[0, 0].set(xlabel="t [s]", ylabel="deg", title="Roll and pitch (first 60 s)")

    for i, name in enumerate(("about North", "about East")):
        ax[0, 1].plot(log.t, err_n[:, i], lw=0.8, label=f"tilt error {name}")
    ax[0, 1].fill_between(log.t, -3 * sig_n[:, 0], 3 * sig_n[:, 0], color="C0", alpha=0.15, label="+/- 3 sigma")
    ax[0, 1].set(xlabel="t [s]", ylabel="deg", title="Roll / pitch-like error (NED axes)")

    ax[1, 0].plot(log.t, err_n[:, 2], label="yaw error")
    ax[1, 0].fill_between(log.t, -3 * sig_n[:, 2], 3 * sig_n[:, 2], color="C0", alpha=0.15, label="+/- 3 sigma")
    ax[1, 0].set(xlabel="t [s]", ylabel="deg", title="Yaw error: unobservable, grows and the filter knows it")

    for i, name in enumerate("xyz"):
        ax[1, 1].plot(log.t, np.degrees(log.b[:, i]), f"C{i}", label=f"b_{name} est.")
        ax[1, 1].plot(log.t, np.degrees(log.true_bias[:, i]), f"C{i}--", lw=0.8)
    ax[1, 1].set(xlabel="t [s]", ylabel="deg/s", title="Gyro bias (dashed: true; z is unobservable)")
    for a in ax.flat:
        a.grid(True)
        a.legend(fontsize=8)
    fig.suptitle(f"{params.name}: attitude MEKF (gyro + accelerometer)")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{params.name}_mekf.png", dpi=130)
    print(f"  saved {out / (params.name + '_mekf.png')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()