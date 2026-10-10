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
class ImuConfig:
    """Planar IMU (2 accelerometers + 1 gyro), MEMS-grade defaults. SI units.

    Noise densities are per sqrt(Hz); bias random-walk coefficients are per sqrt(s).
    """

    rate_hz: float = 100.0
    gyro_noise_density: float = 2.0e-4  # [rad/s/sqrt(Hz)]  (angle random walk)
    gyro_bias_rw: float = 1.0e-5  # [rad/s/sqrt(s)]   (rate random walk)
    gyro_bias_init_std: float = 2.0e-3  # [rad/s]    initial bias ~ N(0, std^2)
    accel_noise_density: float = 4.0e-3  # [m/s^2/sqrt(Hz)]  (velocity random walk)
    accel_bias_rw: float = 1.0e-4  # [m/s^2/sqrt(s)]
    accel_bias_init_std: float = 2.0e-2  # [m/s^2]

    def __post_init__(self) -> None:
        if self.rate_hz <= 0.0:
            raise ValueError("IMU rate_hz must be > 0")


@dataclass(frozen=True)
class GpsConfig:
    rate_hz: float = 1.0
    pos_sigma: float = 2.5  # per axis [m]
    vel_sigma: float = 0.1  # per axis [m/s]
    dropout_prob: float = 0.0  # independent loss probability of each fix
    dropout_windows: tuple = ()  # ((t_start, t_end), ...) [s]: no fixes inside
    outlier_prob: float = 0.0  # probability that a fix carries a position jump (multipath)
    outlier_sigma: float = 30.0  # std of that jump, per axis [m]
    r_report_scale: float = 1.0  # reported R std / true noise std (< 1: the sensor is optimistic)

    def __post_init__(self) -> None:
        if self.rate_hz <= 0.0:
            raise ValueError("GPS rate_hz must be > 0")
        if not 0.0 <= self.outlier_prob <= 1.0:
            raise ValueError("GPS outlier_prob must be in [0, 1]")
        if self.r_report_scale <= 0.0:
            raise ValueError("GPS r_report_scale must be > 0")
        windows = tuple((float(a), float(b)) for a, b in self.dropout_windows)
        if any(b <= a for a, b in windows):
            raise ValueError("each GPS dropout window needs t_start < t_end")
        object.__setattr__(self, "dropout_windows", windows)


@dataclass(frozen=True)
class CompassConfig:
    rate_hz: float = 10.0
    sigma: float = 0.0175  # [rad] (~1 deg)
    bias: float = 0.0  # constant deviation [rad]

    def __post_init__(self) -> None:
        if self.rate_hz <= 0.0:
            raise ValueError("compass rate_hz must be > 0")


@dataclass(frozen=True)
class SensorsConfig:
    imu: ImuConfig = field(default_factory=ImuConfig)
    gps: GpsConfig = field(default_factory=GpsConfig)
    compass: CompassConfig = field(default_factory=CompassConfig)

    @classmethod
    def from_dict(cls, raw: dict) -> "SensorsConfig":
        unknown = set(raw) - {"imu", "gps", "compass"}
        if unknown:
            raise TypeError(f"Unknown sensors config keys: {sorted(unknown)}")
        return cls(
            imu=ImuConfig(**raw.get("imu", {})),
            gps=GpsConfig(**raw.get("gps", {})),
            compass=CompassConfig(**raw.get("compass", {})),
        )

@dataclass(frozen=True)
class EkfConfig:
    """EKF tuning. Process noise comes from the ASSUMED IMU parameters (an ImuConfig), scaled here.

    ``q_scale`` / ``r_scale`` multiply the assumed standard deviations (covariances scale with the
    square): 1.0 = the filter knows the true noise, other values = deliberate mismatch.
    """

    q_scale: float = 1.0
    r_scale: float = 1.0
    use_compass: bool = True
    gate_prob: float = 0.0  # chi-square gate on the NIS of each update (0 = off, e.g. 0.999)
    vel_noise: float = 0.0  # extra velocity random walk density [m/s/sqrt(s)] (not seen by the IMU)
    p0_pos: float = 5.0  # initial 1-sigma uncertainties of the estimate
    p0_vel: float = 0.5  # [m/s]
    p0_psi: float = 0.1  # [rad]
    p0_accel_bias: float = 2.0e-2  # [m/s^2]
    p0_gyro_bias: float = 2.0e-3  # [rad/s]

    def __post_init__(self) -> None:
        if self.q_scale <= 0.0 or self.r_scale <= 0.0:
            raise ValueError("q_scale and r_scale must be > 0")
        if self.vel_noise < 0.0:
            raise ValueError("vel_noise must be >= 0")
        if not 0.0 <= self.gate_prob < 1.0:
            raise ValueError("gate_prob must be in [0, 1)")

@dataclass(frozen=True)
class WaveConfig:
    """Synthetic wave-induced motion (the 3-DOF planar model has no roll / pitch / heave).

    Roll, pitch and the linear accelerations are narrowband random processes: sums of sinusoids
    with random phases and frequencies spread around the peak period.
    """

    roll_sigma_deg: float = 4.0  # rms roll [deg]
    pitch_sigma_deg: float = 2.0  # rms pitch [deg]
    peak_period: float = 7.0  # [s]
    bandwidth: float = 0.3  # relative half-width of the frequency band
    accel_sigma: float = 0.5  # rms wave-induced linear acceleration, per NED axis [m/s^2]
    n_components: int = 12
    include_maneuver_accel: bool = True  # add the horizontal accelerations of the planar maneuver

    def __post_init__(self) -> None:
        if self.peak_period <= 0.0 or self.n_components < 1:
            raise ValueError("wave peak_period must be > 0 and n_components >= 1")


@dataclass(frozen=True)
class MekfConfig:
    """MEKF tuning (attitude from gyro + accelerometer)."""

    q_scale: float = 1.0  # scales the assumed gyro noise / bias random walk std devs
    accel_lin_sigma: float = 0.6  # unmodeled linear acceleration seen by the accelerometer [m/s^2]
    turn_compensation: bool = True  # subtract the centripetal term  omega x [SOG, 0, 0]  (GPS speed)
    accel_update_hz: float = 10.0  # the accelerometer is fused at this rate (errors are time-correlated)
    gate_prob: float = 0.0  # chi-square gate on the accelerometer NIS (0 = off)
    p0_att: float = 0.1  # initial attitude uncertainty, per axis [rad]
    p0_gyro_bias: float = 2.0e-3  # [rad/s]

    def __post_init__(self) -> None:
        if self.q_scale <= 0.0 or self.accel_lin_sigma < 0.0 or self.accel_update_hz <= 0.0:
            raise ValueError("invalid MEKF tuning")
        if not 0.0 <= self.gate_prob < 1.0:
            raise ValueError("gate_prob must be in [0, 1)")

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
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    ekf: EkfConfig = field(default_factory=EkfConfig)
    waves: WaveConfig = field(default_factory=WaveConfig)
    mekf: MekfConfig = field(default_factory=MekfConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unknown = set(raw) - {"sim", "vessel", "env", "sensors", "ekf", "waves", "mekf"}
        if unknown:
            raise TypeError(f"Unknown top-level config keys: {sorted(unknown)}")
        return cls(
            sim=SimConfig(**raw.get("sim", {})),
            vessel=raw.get("vessel", "merchant_ship"),
            env=EnvConfig(**raw.get("env", {})),
            sensors=SensorsConfig.from_dict(raw.get("sensors", {})),
            ekf=EkfConfig(**raw.get("ekf", {})),
            waves=WaveConfig(**raw.get("waves", {})),
            mekf=MekfConfig(**raw.get("mekf", {})),
        )