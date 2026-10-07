#!/usr/bin/env python
"""Static IMU record -> Allan deviation, compared with the theoretical curves of the config.

The vessel is not moving, so the gyro/accelerometer output is only bias + noise: exactly what an
Allan analysis needs. Use it to check (and understand) the IMU noise parameters.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np

from vessel_state_estimation.config import Config
from vessel_state_estimation.eval.allan import allan_deviation
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sensors import PlanarImu
from vessel_state_estimation.sim.truth import TruthSample

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    p.add_argument("--hours", type=float, default=3.0)
    p.add_argument("--rate", type=float, default=20.0, help="IMU rate for this study [Hz]")
    p.add_argument("--out", default=ROOT / "results")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config.from_yaml(args.config)
    imu_cfg = replace(cfg.sensors.imu, rate_hz=args.rate, gyro_bias_init_std=0.0,
                      accel_bias_init_std=0.0)
    imu = PlanarImu(imu_cfg, make_rngs(cfg.sim.seed)["imu"])
    still = TruthSample(t=0.0, n=0.0, e=0.0, psi=0.0, u=0.0, v=0.0, r=0.0, delta=0.0,
                        u_dot=0.0, v_dot=0.0, r_dot=0.0, vn=0.0, ve=0.0)
    n = int(args.hours * 3600 * args.rate)
    z = np.array([imu.measure(k / args.rate, still).z for k in range(n)])

    taus = np.logspace(np.log10(2 / args.rate), np.log10(args.hours * 3600 / 9), 40)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, col, label, N, K, unit in (
        (ax[0], 2, "gyro", imu_cfg.gyro_noise_density, imu_cfg.gyro_bias_rw, "rad/s"),
        (ax[1], 0, "accel x", imu_cfg.accel_noise_density, imu_cfg.accel_bias_rw, "m/s^2"),
    ):
        adev = allan_deviation(z[:, col], args.rate, taus)
        a.loglog(taus, adev, "o-", label=f"{label} (simulated)")
        a.loglog(taus, N / np.sqrt(taus), "--", label="N / sqrt(tau)  (white noise)")
        a.loglog(taus, K * np.sqrt(taus / 3), "--", label="K sqrt(tau/3)  (bias random walk)")
        a.loglog(taus, np.sqrt(N**2 / taus + K**2 * taus / 3), "k:", label="sum of the two")
        a.set(xlabel="tau [s]", ylabel=f"Allan deviation [{unit}]", title=label)
        a.grid(True, which="both")
        a.legend()
    fig.suptitle(f"Static IMU, {args.hours:.1f} h at {args.rate:.0f} Hz")
    fig.tight_layout()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "imu_allan.png", dpi=130)
    print(f"saved {out / 'imu_allan.png'}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()