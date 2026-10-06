"""Geometric helpers for validating simulated maneuvers."""

from __future__ import annotations

import numpy as np


def fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares circle fit (Kasa): returns centre ``(cx, cy)`` and radius.

    Solves  x^2 + y^2 = 2 cx x + 2 cy y + c  linearly; exact for points on a circle.
    """
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x**2 + y**2
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(cx), float(cy), float(np.sqrt(c + cx**2 + cy**2))
