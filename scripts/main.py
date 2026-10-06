#!/usr/bin/env python
"""Simulate one maneuver of the selected vessel (with current and wind), save truth and plot it.

Examples:
    python scripts/main.py --maneuver turning_circle --calm
    python scripts/main.py --vessel sailboat --maneuver zigzag
    python scripts/main.py --vessel small_boat --maneuver course_changes
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from vessel_state_estimation.angles import wrap_angle
from vessel_state_estimation.config import Config, EnvConfig, get_vessel
from vessel_state_estimation.eval.geometry import fit_circle
from vessel_state_estimation.rng import make_rngs
from vessel_state_estimation.sim import (Environment, HeadingAutopilot, Vessel, ZigZag,
                                         constant_command, simulate)

ROOT = Path(__file__).resolve().parents[1]
MANEUVERS = ("straight", "turning_circle", "zigzag", "course_changes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs" / "default.yaml")
    p.add_argument("--vessels", default=ROOT / "configs" / "vessels.yaml")
    p.add_argument("--vessel", help="vessel type (default: 'vessel' key of the config)")
    p.add_argument("--maneuver", choices=MANEUVERS, default="turning_circle")
    p.add_argument("--rudder-deg", type=float, help="turning circle: rudder angle")
    p.add_argument("--duration", type=float, help="seconds (default: depends on the maneuver)")
    p.add_argument("--calm", action="store_true", help="no current, no wind")
    p.add_argument("--out", default=ROOT / "results")
    p.add_argument("--show", action="store_true", help="open the plot window")
    return p.parse_args()


def zigzag_overshoot_deg(psi: np.ndarray, psi0: float, trigger: float) -> float:
    """First overshoot angle: how far the heading goes past the trigger after the 1st reversal."""
    rel = np.degrees(wrap_angle(psi - psi0))
    i1 = int(np.argmax(rel >= math.degrees(trigger)))  # first reversal
    after = rel[i1:]
    back = np.nonzero(after <= 0.0)[0]
    window = after[: back[0]] if back.size else after
    return float(window.max() - math.degrees(trigger))


def main() -> None:
    args = parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = Config.from_yaml(args.config)
    params = get_vessel(args.vessel or cfg.vessel, args.vessels)
    vessel = Vessel(params)
    u0 = params.nominal_speed
    K, T = vessel.nomoto(u0)
    m = params.maneuvers
    dt = cfg.sim.dt

    # ---- maneuver -> (command function, default duration)
    rudder = 0.0
    schedule = None
    if args.maneuver == "straight":
        command_fn, duration = constant_command(0.0, u0), cfg.sim.duration
    elif args.maneuver == "turning_circle":
        rudder = math.radians(args.rudder_deg if args.rudder_deg is not None else m.turn_rudder_deg)
        turn = vessel.steady_turn(rudder, u0)
        t_circle = 2.0 * math.pi * turn.radius / math.hypot(u0, turn.sway_speed)
        command_fn, duration = constant_command(rudder, u0), 8.0 * T + 1.3 * t_circle
    elif args.maneuver == "zigzag":
        trigger = math.radians(m.zigzag_heading_deg)
        command_fn = ZigZag(math.radians(m.zigzag_rudder_deg), trigger, u0)
        duration = 40.0 * T
    else:  # course_changes: absolute headings [deg] at multiples of T
        marks = [(0, 0.0), (8, 45.0), (24, -30.0), (40, 90.0)]
        schedule = [(k * T, math.radians(h)) for k, h in marks]
        command_fn, duration = HeadingAutopilot(vessel, u0, schedule), 56.0 * T
    if args.duration is not None:
        duration = args.duration

    # ---- environment and simulation
    env_cfg = EnvConfig() if args.calm else cfg.env
    env = Environment(env_cfg, make_rngs(cfg.sim.seed)["disturbance"], dt, duration)
    traj = simulate(vessel, vessel.initial_state(), command_fn, duration, dt, environment=env)

    t, n, e = traj.array("t"), traj.array("n"), traj.array("e")
    psi = traj.array("psi")
    cog = np.arctan2(traj.array("ve"), traj.array("vn"))
    crab = np.degrees(wrap_angle(cog - psi))

    print(f"Vessel: {params.name} - {params.description}")
    print(f"  U = {u0:.2f} m/s, K = {K:.4f} 1/s, T = {T:.1f} s")
    print(f"  {args.maneuver}: {duration:.0f} s, {len(traj)} samples, "
          f"{'calm' if args.calm else 'with current and wind'}, seed {cfg.sim.seed}")
    if args.maneuver == "turning_circle":
        print(f"  rudder {math.degrees(rudder):.1f} deg, theoretical radius {turn.radius:.1f} m")
        if args.calm:
            _, _, radius = fit_circle(n[t >= 8.0 * T], e[t >= 8.0 * T])
            print(f"  simulated radius {radius:.1f} m ({100 * (radius / turn.radius - 1):+.2f} %)")
        else:
            print("  radius check skipped: the current turns the circle into a trochoid (use --calm)")
    if args.maneuver == "zigzag":
        print(f"  first overshoot angle: {zigzag_overshoot_deg(psi, 0.0, trigger):.1f} deg")
    if not args.calm:
        print(f"  crab angle (COG - heading): mean {crab.mean():+.1f} deg, "
              f"range [{crab.min():+.1f}, {crab.max():+.1f}] deg")

    # ---- plots
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    ax[0, 0].plot(e, n)
    ax[0, 0].plot(e[0], n[0], "go", label="start")
    ax[0, 0].set(xlabel="East [m]", ylabel="North [m]", title="Path (NED)")
    ax[0, 0].axis("equal")
    ax[0, 0].legend()

    ax[0, 1].plot(t, np.degrees(np.unwrap(psi)), label="heading")
    ax[0, 1].plot(t, np.degrees(np.unwrap(cog)), label="COG", alpha=0.7)
    if schedule is not None:
        ref = [command_fn.reference(ti) for ti in t[::100]]
        ax[0, 1].plot(t[::100], np.degrees(ref), "k--", label="reference")
    ax[0, 1].set(xlabel="t [s]", ylabel="deg", title="Heading and course over ground")
    ax[0, 1].legend()

    ax[1, 0].plot(t, np.degrees(traj.array("delta")))
    ax[1, 0].set(xlabel="t [s]", ylabel="deg", title="Rudder angle")

    ax[1, 1].plot(t, crab)
    ax[1, 1].set(xlabel="t [s]", ylabel="deg", title="Crab angle: COG - heading")
    for a in ax.flat:
        a.grid(True)
    fig.suptitle(f"{params.name} - {args.maneuver}")
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{params.name}_{args.maneuver}"
    fig.savefig(out_dir / f"{stem}.png", dpi=130)
    traj.save(out_dir / f"{stem}.npz")
    print(f"  saved: {out_dir / (stem + '.png')} and {stem}.npz")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()