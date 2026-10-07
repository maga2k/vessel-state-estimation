"""Build the sensor set from a config and run it over a ground-truth trajectory."""

from __future__ import annotations

import numpy as np

from ..config import SensorsConfig
from ..sim.truth import Trajectory
from .base import Measurement, Sensor
from .compass import Compass
from .gps import Gps
from .imu import PlanarImu


def build_sensors(cfg: SensorsConfig, rngs: dict[str, np.random.Generator]) -> list[Sensor]:
    """One sensor per type, each on its own random stream (see ``rng.STREAMS``)."""
    return [PlanarImu(cfg.imu, rngs["imu"]), Gps(cfg.gps, rngs["gps"]),
            Compass(cfg.compass, rngs["mag"])]


def simulate_sensors(traj: Trajectory, sensors: list[Sensor]) -> list[Measurement]:
    """All measurements over the trajectory, in time order (list order breaks ties)."""
    out: list[Measurement] = []
    for sample in traj.samples:
        for sensor in sensors:
            m = sensor.measure(sample.t, sample)
            if m is not None:
                out.append(m)
    return out