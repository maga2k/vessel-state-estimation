from pathlib import Path

import numpy as np

from vessel_state_estimation.angles import wrap_angle
from vessel_state_estimation.config import get_vessel
from vessel_state_estimation.sim import HeadingAutopilot, Vessel, ZigZag, simulate

VESSELS = Path(__file__).resolve().parents[1] / "configs" / "vessels.yaml"
NAMES = ("merchant_ship", "sailboat", "small_boat")


def test_zigzag_reverses_and_overshoots():
    for name in NAMES:
        v = Vessel(get_vessel(name, VESSELS))
        m = v.p.maneuvers
        u0 = v.p.nominal_speed
        _, T = v.nomoto(u0)
        rud, dpsi = np.radians(m.zigzag_rudder_deg), np.radians(m.zigzag_heading_deg)
        traj = simulate(v, v.initial_state(), ZigZag(rud, dpsi, u0), 60 * T, 0.05)
        delta = traj.array("delta")
        psi = wrap_angle(traj.array("psi"))
        # the commanded rudder (hence the actual one) changes sign several times
        assert np.sum(np.diff(np.sign(delta)) != 0) >= 4
        # inertia: heading goes beyond the trigger angle on both sides
        assert psi.max() > dpsi and psi.min() < -dpsi


def test_heading_autopilot_reaches_reference_without_overshoot():
    for name in NAMES:
        v = Vessel(get_vessel(name, VESSELS))
        u0 = v.p.nominal_speed
        _, T = v.nomoto(u0)
        step = np.radians(20.0)
        ap = HeadingAutopilot(v, u0, [(0.0, 0.0), (5 * T, step)])
        traj = simulate(v, v.initial_state(), ap, 40 * T, 0.05)
        psi = traj.array("psi")
        assert abs(psi[-1] - step) < np.radians(0.2)
        assert psi.max() < step * 1.02  # zeta = 0.9: practically no overshoot
        assert np.abs(traj.array("delta")).max() <= v.p.rudder_max + 1e-12


def test_heading_autopilot_wraps_across_pi():
    v = Vessel(get_vessel("small_boat", VESSELS))
    u0 = v.p.nominal_speed
    ap = HeadingAutopilot(v, u0, [(0.0, np.radians(179.0))])
    traj = simulate(v, v.initial_state(psi=np.radians(-179.0)), ap, 20.0, 0.02)
    # shortest way is 2 degrees through +/-180, not 358 degrees the long way round
    err = wrap_angle(traj.array("psi") - np.radians(179.0))
    assert abs(err[-1]) < np.radians(0.5)
    assert np.abs(traj.array("delta")).max() < v.p.rudder_max