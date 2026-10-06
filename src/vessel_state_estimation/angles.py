"""Angle helpers. All angles are in radians."""

from __future__ import annotations

import numpy as np


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi). Use it on every angular residual (innovations, errors)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi
