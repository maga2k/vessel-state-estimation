"""Reproducible, independent random streams.

Each component (disturbances, IMU, GPS, ...) gets its own generator, spawned from a
single master seed. Adding a consumer never changes the draws of the existing ones,
as long as new names are appended at the END of ``STREAMS``.
"""

from __future__ import annotations

import numpy as np

STREAMS: tuple[str, ...] = ("disturbance", "imu", "gps", "mag", "outlier", "init")


def make_rngs(seed: int) -> dict[str, np.random.Generator]:
    """Return one independent ``Generator`` per name in ``STREAMS``."""
    children = np.random.SeedSequence(seed).spawn(len(STREAMS))
    return {name: np.random.default_rng(child) for name, child in zip(STREAMS, children)}
