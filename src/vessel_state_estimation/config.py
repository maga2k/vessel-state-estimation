"""Typed configuration: frozen dataclasses loaded from YAML.

Two files drive a run:

* ``configs/default.yaml``  -> simulation settings and the *base vessel* to use;
* ``configs/vessels.yaml``  -> one parameter set per vessel type (merchant ship, sailboat, ...).

Unknown YAML keys raise on purpose, so typos in config files do not pass silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.01  # integration step [s]
    duration: float = 600.0  # default simulated time [s]
    seed: int = 42  # master seed for all random streams

@dataclass(frozen=True)
class EnvConfig:
    """Environment: mean current / wind plus first-order Gauss-Markov fluctuations.

    Conventions (nautical): current direction is where it flows TOWARDS ("set"), wind direction
    is where it blows FROM. Both in degrees from North, clockwise.
    """

    current_speed: float = 0.0  # [m/s]
    current_dir_deg: float = 0.0
    current_sigma: float = 0.0  # std of the fluctuation, per NED component [m/s]
    current_tau: float = 600.0  # correlation time [s]
    wind_speed: float = 0.0  # [m/s]
    wind_dir_deg: float = 0.0
    wind_sigma: float = 0.0  # gust std, per NED component [m/s]
    wind_tau: float = 60.0  # [s]

    def __post_init__(self) -> None:
        if self.current_tau <= 0.0 or self.wind_tau <= 0.0:
            raise ValueError("Gauss-Markov correlation times must be > 0")


@dataclass(frozen=True)
class ManeuverDefaults:
    """Typical maneuver settings for a vessel type (used by the maneuver library, session 2)."""

    turn_rudder_deg: float  # rudder angle for a "normal" course change / turning circle
    zigzag_rudder_deg: float  # zig-zag test: rudder angle
    zigzag_heading_deg: float  # zig-zag test: heading change that triggers the rudder reversal


@dataclass(frozen=True)
class VesselParams:
    """Parameters of the 3-DOF vessel model (see docs/theory/01-nomoto-3dof.md).

    YAML uses degrees for rudder quantities (human friendly); the ``rudder_*`` properties
    give SI values. Everything else is already SI.
    """

    name: str
    description: str
    length: float  # L, length between perpendiculars [m]
    nominal_speed: float  # cruising surge speed [m/s]
    K_prime: float  # non-dimensional Nomoto gain  (K = K' * U / L)
    T_prime: float  # non-dimensional Nomoto time constant (T = T' * L / U)
    pivot_frac: float  # pivot point distance ahead of the CG, as a fraction of L
    T_sway: float  # sway relaxation time [s]
    T_surge: float  # surge speed response time [s]
    leeway_coeff: float  # wind-induced drift: fraction of the wind velocity added to ground motion
    rudder_max_deg: float  # rudder saturation [deg]
    rudder_rate_deg_s: float  # rudder rate limit [deg/s]
    maneuvers: ManeuverDefaults

    def __post_init__(self) -> None:
        positives = ("length", "nominal_speed", "K_prime", "T_prime", "T_sway", "T_surge",
                     "rudder_max_deg", "rudder_rate_deg_s")
        for key in positives:
            if getattr(self, key) <= 0.0:
                raise ValueError(f"Vessel '{self.name}': {key} must be > 0")

    @property
    def rudder_max(self) -> float:
        return math.radians(self.rudder_max_deg)

    @property
    def rudder_rate(self) -> float:
        return math.radians(self.rudder_rate_deg_s)

    @property
    def pivot_dist(self) -> float:
        return self.pivot_frac * self.length


def load_vessels(path: str | Path) -> dict[str, VesselParams]:
    """Load every vessel type defined in a vessels YAML file."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    vessels: dict[str, VesselParams] = {}
    for name, spec in raw.items():
        spec = dict(spec)
        maneuvers = ManeuverDefaults(**spec.pop("maneuvers"))
        vessels[name] = VesselParams(name=name, maneuvers=maneuvers, **spec)
    return vessels


def get_vessel(name: str, path: str | Path) -> VesselParams:
    vessels = load_vessels(path)
    if name not in vessels:
        raise KeyError(f"Unknown vessel '{name}'. Available: {sorted(vessels)}")
    return vessels[name]


@dataclass(frozen=True)
class Config:
    sim: SimConfig = field(default_factory=SimConfig)
    vessel: str = "merchant_ship"  # base vessel: key in configs/vessels.yaml
    env: EnvConfig = field(default_factory=EnvConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {"sim", "vessel", "env"}
        if unknown:
            raise TypeError(f"Unknown top-level config keys: {sorted(unknown)}")
        return cls(
            sim=SimConfig(**raw.get("sim", {})),
            vessel=raw.get("vessel", "merchant_ship"),
            env=EnvConfig(**raw.get("env", {})),
        )