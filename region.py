"""Safe, explicit planar regions for characteristic-function factors 1_A."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Region:
    shape: str
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0
    inner_radius: float = 0.5
    radii: tuple[float, float] = (2.0, 1.0)
    rectangle: tuple[float, float, float, float] = (-1.0, 1.0, -1.0, 1.0)
    threshold: float = 0.0

    def validate(self) -> None:
        if self.shape not in {"Disk", "Annulus", "Ellipse", "Rectangle", "Half-plane x ≥ a"}:
            raise ValueError("Unsupported region A")
        if len(self.center) != 2 or not np.isfinite(self.center).all():
            raise ValueError("Region center must be finite")
        if self.shape in {"Disk", "Annulus"}:
            if not np.isfinite(self.radius) or self.radius <= 0:
                raise ValueError("Outer radius must be positive and finite")
        if self.shape == "Annulus" and (
            not np.isfinite(self.inner_radius) or not (0 <= self.inner_radius < self.radius)
        ):
            raise ValueError("Annulus requires 0 ≤ inner radius < outer radius")
        if self.shape == "Ellipse" and (
            len(self.radii) != 2 or not np.isfinite(self.radii).all() or min(self.radii) <= 0
        ):
            raise ValueError("Ellipse semiaxes must be positive and finite")
        if self.shape == "Rectangle":
            if len(self.rectangle) != 4:
                raise ValueError("Rectangle must have four finite bounds")
            xmin, xmax, ymin, ymax = self.rectangle
            if not np.isfinite(self.rectangle).all() or not (xmin < xmax and ymin < ymax):
                raise ValueError("Rectangle must have positive area")
        if self.shape == "Half-plane x ≥ a" and not np.isfinite(self.threshold):
            raise ValueError("Half-plane threshold must be finite")

    def contains(self, x, y):
        self.validate()
        x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        cx, cy = self.center
        if self.shape == "Disk":
            return (x - cx) ** 2 + (y - cy) ** 2 <= self.radius**2
        if self.shape == "Annulus":
            distance_squared = (x - cx) ** 2 + (y - cy) ** 2
            return (distance_squared >= self.inner_radius**2) & (distance_squared <= self.radius**2)
        if self.shape == "Ellipse":
            a, b = self.radii
            return ((x - cx) / a) ** 2 + ((y - cy) / b) ** 2 <= 1
        if self.shape == "Rectangle":
            xmin, xmax, ymin, ymax = self.rectangle
            return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        return x >= self.threshold

    def boundary(self, bounds: tuple[float, float, float, float], count: int = 360) -> list[np.ndarray]:
        """Return Nx2 curves for plotting region boundaries."""
        self.validate()
        cx, cy = self.center
        if self.shape in {"Disk", "Annulus", "Ellipse"}:
            theta = np.linspace(0, 2 * np.pi, count)
            if self.shape == "Ellipse":
                a, b = self.radii
                return [np.column_stack((cx + a * np.cos(theta), cy + b * np.sin(theta)))]
            curves = [np.column_stack((cx + self.radius * np.cos(theta),
                                       cy + self.radius * np.sin(theta)))]
            if self.shape == "Annulus" and self.inner_radius > 0:
                curves.append(np.column_stack((cx + self.inner_radius * np.cos(theta),
                                               cy + self.inner_radius * np.sin(theta))))
            return curves
        if self.shape == "Rectangle":
            xmin, xmax, ymin, ymax = self.rectangle
            return [np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax],
                              [xmin, ymax], [xmin, ymin]])]
        return [np.array([[self.threshold, bounds[2]], [self.threshold, bounds[3]]])]
