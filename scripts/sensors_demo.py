#!/usr/bin/env python
"""Simulate a maneuver, generate IMU/GPS/compass data and show why fusion is needed.

Panels: true IMU gyro bias, GPS fixes with a dropout window, and the error of a naive
dead-reckoning (integrating the raw IMU) that starts from the exact initial state.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from vessel_state_estimation.config import Config, get_vessel
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sensors import build_sensors, simulate_sensors
from vessel_state_estimation.sim import Environment, Vessel, ZigZag, simulate

ROOT = Path(__file__).resolve().parents[1]


def dead_reckon(imu, truth0, dt):
    """Planar strapdown: integrate gyro -> heading, rotate accel to NED, integrate twice."""
    psi, vn, ve, n, e = truth0.psi, truth0.vn, truth0.ve, truth0.n, truth0.e
    out = []
    for m in imu:
        f_x, f_y, gz = m.z
        psi += gz * dt
        a_n = np.cos(psi) * f_x - np.sin(psi) * f_y
        a_e = np.sin(psi) * f_x + np.cos(psi) * f_y
        vn, ve = vn + a_n * dt, ve + a_e * dt
        n, e = n + vn * dt, e + ve * dt
        out.append((m.t, n, e))
    return np.array(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    p.add_argument("--vessels", default=ROOT / "configs" / "vessels.yaml")
    p.add_argument("--vessel")
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument("--out", default=ROOT / "results")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config.from_yaml(args.config)
    params = get_vessel(args.vessel or cfg.vessel, args.vessels)
    vessel = Vessel(params)
    dt, u0 = cfg.sim.dt, params.nominal_speed
    m = params.maneuvers
    rngs = make_rngs(cfg.sim.seed)

    env = Environment(cfg.env, rngs["disturbance"], dt, args.duration)
    cmd = ZigZag(np.radians(m.zigzag_rudder_deg), np.radians(m.zigzag_heading_deg), u0)
    traj = simulate(vessel, vessel.initial_state(), cmd, args.duration, dt, environment=env)

    window = (0.4 * args.duration, 0.4 * args.duration + 60.0)
    sensors_cfg = replace(cfg.sensors, gps=replace(cfg.sensors.gps, dropout_windows=(window,)))
    meas = simulate_sensors(traj, build_sensors(sensors_cfg, rngs))
    imu = [x for x in meas if x.sensor == "imu"]
    gps = [x for x in meas if x.sensor == "gps"]

    t_imu = np.array([x.t for x in imu])
    bias = np.array([x.info["bias"] for x in imu])
    dr = dead_reckon(imu, traj[0], 1.0 / cfg.sensors.imu.rate_hz)
    truth_n = np.interp(dr[:, 0], traj.array("t"), traj.array("n"))
    truth_e = np.interp(dr[:, 0], traj.array("t"), traj.array("e"))
    dr_err = np.hypot(dr[:, 1] - truth_n, dr[:, 2] - truth_e)

    t_gps = np.array([x.t for x in gps])
    z_gps = np.array([x.z for x in gps])
    gps_err = np.hypot(z_gps[:, 0] - np.interp(t_gps, traj.array("t"), traj.array("n")),
                       z_gps[:, 1] - np.interp(t_gps, traj.array("t"), traj.array("e")))

    print(f"{params.name}: {len(imu)} IMU, {len(gps)} GPS, "
          f"{sum(x.sensor == 'compass' for x in meas)} compass samples")
    print(f"GPS dropout window: {window[0]:.0f}-{window[1]:.0f} s; "
          f"GPS position error rms {np.sqrt(np.mean(gps_err**2)):.1f} m")
    for tt in (10, 30, 60, 120):
        i = int(np.argmin(np.abs(dr[:, 0] - tt)))
        print(f"  raw dead-reckoning error after {tt:>3d} s: {dr_err[i]:7.1f} m")

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(traj.array("e"), traj.array("n"), label="truth")
    ax[0, 0].plot(z_gps[:, 1], z_gps[:, 0], ".", ms=4, label="GPS fixes")
    ax[0, 0].set(xlabel="East [m]", ylabel="North [m]", title="Path and GPS fixes")
    ax[0, 0].axis("equal")
    ax[0, 0].legend()

    ax[0, 1].plot(t_imu, np.degrees(bias[:, 2]))
    ax[0, 1].set(xlabel="t [s]", ylabel="deg/s", title="True gyro bias (initial value + random walk)")

    ax[1, 0].plot(t_gps, gps_err, ".")
    ax[1, 0].axvspan(*window, color="r", alpha=0.15, label="GPS dropout")
    ax[1, 0].set(xlabel="t [s]", ylabel="m", title="GPS position error (white noise)")
    ax[1, 0].legend()

    ax[1, 1].semilogy(dr[:, 0], dr_err + 1e-3)
    ax[1, 1].set(xlabel="t [s]", ylabel="m", title="Raw dead-reckoning error (no GPS, no filter)")
    for a in ax.flat:
        a.grid(True, which="both")
    fig.suptitle(f"{params.name}: sensor models")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{params.name}_sensors.png", dpi=130)
    print(f"saved {out / (params.name + '_sensors.png')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()