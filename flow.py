"""Finite-direction, equal-weight empirical sliced-Wasserstein gradient descent."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlowResult:
    positions: np.ndarray  # (steps+1, n, 2)
    energies: np.ndarray   # (steps+1,)
    times: np.ndarray      # (steps+1,)


def directions_2d(count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("At least two directions are required")
    angles = np.arange(count) * np.pi / count
    return np.column_stack((np.cos(angles), np.sin(angles)))


def sliced_energy(x: np.ndarray, target_sorted: np.ndarray, directions: np.ndarray) -> float:
    p = np.sort(x @ directions.T, axis=0)
    return 0.5 * float(np.mean((p - target_sorted) ** 2))


def simulate(source: np.ndarray, target: np.ndarray, n_directions: int = 384,
             dt: float = 0.4, steps: int = 70) -> FlowResult:
    """Explicit Euler for the empirical W2 gradient of 0.5 * finite-direction SW2^2.

    Equal masses and identical sample sizes are required. Fixed directions and
    target samples are used for the whole run. With a normalized spherical
    measure, no extra dimension factor is included in the gradient.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or source.shape[1] != 2 or target.shape != source.shape:
        raise ValueError("Source and target must have identical (N, 2) shapes")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("All particles must be finite")
    if not (0 < dt <= 1):
        raise ValueError("Use a step size 0 < dt <= 1")
    if not (1 <= steps <= 250):
        raise ValueError("Steps must be between 1 and 250")

    dirs = directions_2d(n_directions)
    target_sorted = np.sort(target @ dirs.T, axis=0)
    positions = np.empty((steps + 1, *source.shape), dtype=np.float64)
    energies = np.empty(steps + 1, dtype=np.float64)
    x = source.copy()
    for k in range(steps + 1):
        positions[k] = x
        p = x @ dirs.T
        order = np.argsort(p, axis=0)
        p_sorted = np.take_along_axis(p, order, axis=0)
        energies[k] = 0.5 * float(np.mean((p_sorted - target_sorted) ** 2))
        if k == steps:
            break
        # q_i(theta) is the target projection at the rank of x_i(theta).
        target_at_particle = np.empty_like(p)
        np.put_along_axis(target_at_particle, order, target_sorted, axis=0)
        velocity = ((target_at_particle - p) @ dirs) / n_directions
        x = x + dt * velocity
    return FlowResult(positions, energies, np.arange(steps + 1) * dt)
