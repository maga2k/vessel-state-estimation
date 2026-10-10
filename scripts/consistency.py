#!/usr/bin/env python
"""Monte Carlo consistency of the EKF: NEES / NIS with chi-square bounds, plus tuning sweeps.

Many noise realisations over the SAME true trajectory; the data are generated once and every
tuning is evaluated on identical measurements.

Examples:
    python scripts/consistency.py --runs 20
    python scripts/consistency.py --sweep-q 0.25 0.5 1 2 4
    python scripts/consistency.py --sweep-vel 0 0.003 0.006 0.009
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from vessel_state_estimation.config import Config, get_vessel
from vessel_state_estimation.eval.consistency import chi2_bounds
from vessel_state_estimation.eval.montecarlo import evaluate, generate_runs
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sim import Environment, HeadingAutopilot, Vessel, simulate

ROOT = Path(__file__).resolve().parents[1]
STATES = ["N", "E", "psi", "vN", "vE", "b_ax", "b_ay", "b_g"]


def expected_vel_noise(cfg: Config, leeway: float) -> float:
    """Velocity random-walk density produced by the simulated wind and current fluctuations.

    An Ornstein-Uhlenbeck process (sigma, tau) has d/dt-noise intensity 2 sigma^2 / tau. The wind
    enters ground velocity through the leeway coefficient. (A real filter would NOT know this:
    there you tune vel_noise from the data; here it is a check.)
    """
    e = cfg.env
    return math.sqrt(leeway**2 * 2 * e.wind_sigma**2 / e.wind_tau
                     + 2 * e.current_sigma**2 / e.current_tau)


def row(label: str, res) -> str:
    lo, hi = res.anees_bounds()
    r = res.rms_error()
    return (f"{label:<18} ANEES {res.anees():5.2f} (ideal {res.nx}, bounds {lo:.1f}-{hi:.1f}) "
            f"inside {100 * res.fraction_anees_inside():3.0f}% | NIS gps {res.mean_nis('gps'):.2f}/4 "
            f"compass {res.mean_nis('compass'):.2f}/1 | pos {math.hypot(r[0], r[1]):.2f} m "
            f"psi {math.degrees(r[2]):.2f} deg")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    p.add_argument("--vessels", default=ROOT / "configs" / "vessels.yaml")
    p.add_argument("--vessel")
    p.add_argument("--runs", type=int, default=20)
    p.add_argument("--duration", type=float, default=300.0)
    p.add_argument("--dt", type=float, default=0.02, help="sim step = 1 / IMU rate (50 Hz: faster)")
    p.add_argument("--sweep-q", type=float, nargs="+", help="q_scale values to compare")
    p.add_argument("--sweep-r", type=float, nargs="+", help="r_scale values to compare")
    p.add_argument("--sweep-vel", type=float, nargs="+", help="vel_noise values to compare")
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
    vessel = Vessel(params)
    d, u0 = args.duration, params.nominal_speed

    marks = [(0.0, 0.0), (0.2, 40.0), (0.5, -30.0), (0.75, 60.0)]
    autopilot = HeadingAutopilot(vessel, u0, [(f * d, math.radians(h)) for f, h in marks])
    env = Environment(cfg.env, make_rngs(cfg.sim.seed)["disturbance"], args.dt, d)
    traj = simulate(vessel, vessel.initial_state(), autopilot, d, args.dt, environment=env)
    runs = generate_runs(cfg, traj, args.runs)

    print(f"{params.name}: {args.runs} runs x {d:.0f} s, IMU {1 / args.dt:.0f} Hz")
    print(f"vel_noise implied by the simulated wind/current: "
          f"{expected_vel_noise(cfg, params.leeway_coeff):.4f} m/s/sqrt(s)  (config: {cfg.ekf.vel_noise})")

    sweeps = [("q_scale", args.sweep_q), ("r_scale", args.sweep_r), ("vel_noise", args.sweep_vel)]
    if any(v for _, v in sweeps):
        for name, values in sweeps:
            for v in values or []:
                res = evaluate(runs, traj, replace(cfg.ekf, **{name: v}), cfg.sensors.imu)
                print(row(f"{name}={v:g}", res))
        return

    res = evaluate(runs, traj, cfg.ekf, cfg.sensors.imu)
    print(row("config", res))
    print("per-state mean e^2/P (ideal 1): " +
          ", ".join(f"{n} {v:.2f}" for n, v in zip(STATES, res.per_state_ratio())))

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    lo, hi = res.anees_bounds()
    ax[0, 0].plot(res.t, res.anees_t(), label="ANEES")
    ax[0, 0].axhline(res.nx, color="k", ls="--", label=f"ideal ({res.nx})")
    ax[0, 0].axhspan(lo, hi, color="g", alpha=0.15, label="95% bounds")
    ax[0, 0].set(xlabel="t [s]", ylabel="NEES", title=f"NEES averaged over {res.n_runs} runs")

    for sensor, color in (("gps", "C0"), ("compass", "C1")):
        dof = res.nis_dof[sensor]
        lo_s, hi_s = chi2_bounds(dof, res.n_runs)
        ax[0, 1].plot(res.nis_t[sensor], res.nis[sensor].mean(axis=0) / dof, color=color,
                      lw=0.6, label=f"{sensor} (dof {dof})")
        ax[0, 1].axhspan(lo_s / dof, hi_s / dof, color=color, alpha=0.12)
    ax[0, 1].axhline(1.0, color="k", ls="--")
    ax[0, 1].set(xlabel="t [s]", ylabel="mean NIS / dof", title="NIS averaged over runs (ideal 1)")

    ax[1, 0].bar(STATES, res.per_state_ratio())
    ax[1, 0].axhline(1.0, color="k", ls="--")
    ax[1, 0].set(ylabel="mean e^2 / P", title="Per-state consistency (> 1: overconfident)")

    for i in range(res.n_runs):
        ax[1, 1].plot(res.t, res.errors[i, :, 0], lw=0.5, color="C0", alpha=0.5)
    ax[1, 1].plot(res.t, 3 * res.sigma[0, :, 0], "r", label="+/- 3 sigma (run 0)")
    ax[1, 1].plot(res.t, -3 * res.sigma[0, :, 0], "r")
    ax[1, 1].set(xlabel="t [s]", ylabel="m", title="North position error, all runs")
    for a in ax.flat:
        a.grid(True)
    for a in (ax[0, 0], ax[0, 1], ax[1, 1]):
        a.legend()
    fig.suptitle(f"{params.name}: EKF consistency (vel_noise = {cfg.ekf.vel_noise})")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{params.name}_consistency.png", dpi=130)
    print(f"saved {out / (params.name + '_consistency.png')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()