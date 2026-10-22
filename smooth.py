"""Visualization-only smoothed histogram (not a PDE solution)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def smooth_density(points: np.ndarray, bounds: tuple[float, float, float, float],
                   resolution: int = 85, bandwidth: float = 0.20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    if not (xmin < xmax and ymin < ymax and resolution >= 10 and bandwidth > 0):
        raise ValueError("Invalid KDE parameters")
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("Expected a nonempty finite point cloud in R^2")
    dx = (xmax - xmin) / resolution
    dy = (ymax - ymin) / resolution
    histogram, _, _ = np.histogram2d(
        points[:, 1], points[:, 0], bins=resolution, range=((ymin, ymax), (xmin, xmax))
    )
    if histogram.sum() == 0:
        raise ValueError("All particles lie outside the visualization domain")
    values = gaussian_filter(histogram, sigma=(bandwidth / dy, bandwidth / dx), mode="constant")
    values = values / (values.sum() * dx * dy)
    x = xmin + (np.arange(resolution) + 0.5) * dx
    y = ymin + (np.arange(resolution) + 0.5) * dy
    return x, y, values
