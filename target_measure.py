"""Normalized Hausdorff restrictions to supported 0D/1D geometric sets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SingularTarget:
    geometry: str
    points: tuple[tuple[float, float], ...] = ()
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 1.0

    @property
    def dimension(self) -> int:
        return 0 if self.geometry == "Finite points" else 1

    def _vertices(self) -> np.ndarray:
        vertices = np.asarray(self.points, dtype=float)
        if vertices.ndim != 2 or vertices.shape[1] != 2 or not np.isfinite(vertices).all():
            raise ValueError("Provide valid finite 2D coordinates")
        return vertices

    def validate(self) -> None:
        if self.geometry not in {"Finite points", "Segment", "Circle"}:
            raise ValueError("Unsupported singular geometry")
        if self.geometry == "Circle":
            if not (np.isfinite(self.radius) and self.radius > 0 and np.isfinite(self.center).all()):
                raise ValueError("Circle must have finite center and positive radius")
            return
        vertices = self._vertices()
        if self.geometry == "Finite points":
            if len(vertices) < 1 or len(np.unique(vertices, axis=0)) != len(vertices):
                raise ValueError("H^0 requires at least one distinct point; remove duplicates")
        elif len(vertices) != 2 or np.linalg.norm(vertices[1] - vertices[0]) <= 0:
            raise ValueError("A segment requires two distinct endpoints")

    def mass(self) -> float:
        self.validate()
        if self.geometry == "Circle":
            return float(2 * np.pi * self.radius)
        vertices = self._vertices()
        if self.geometry == "Finite points":
            return float(len(vertices))
        return float(np.linalg.norm(vertices[1] - vertices[0]))

    def sample(self, count: int, seed: int) -> np.ndarray:
        self.validate()
        if count < 1:
            raise ValueError("Sample count must be positive")
        rng = np.random.default_rng(seed)
        if self.geometry == "Circle":
            theta = rng.uniform(0, 2 * np.pi, count)
            return np.column_stack((
                self.center[0] + self.radius * np.cos(theta),
                self.center[1] + self.radius * np.sin(theta),
            ))
        vertices = self._vertices()
        if self.geometry == "Finite points":
            indices = np.tile(np.arange(len(vertices)), (count + len(vertices) - 1) // len(vertices))[:count]
            rng.shuffle(indices)
            return vertices[indices].copy()
        fractions = rng.uniform(size=count)
        return vertices[0] + fractions[:, None] * (vertices[1] - vertices[0])

    def support(self, count: int = 300) -> np.ndarray:
        self.validate()
        if self.geometry == "Circle":
            theta = np.linspace(0, 2 * np.pi, count)
            return np.column_stack((
                self.center[0] + self.radius * np.cos(theta),
                self.center[1] + self.radius * np.sin(theta),
            ))
        vertices = self._vertices()
        if self.geometry == "Finite points":
            return vertices.copy()
        fractions = np.linspace(0, 1, count)
        return vertices[0] + fractions[:, None] * (vertices[1] - vertices[0])


def parse_points(text: str) -> tuple[tuple[float, float], ...]:
    """Parse one x,y pair per line without evaluating arbitrary expressions."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or len(lines) > 50:
        raise ValueError("Enter 1 to 50 coordinate pairs, one per line")
    points = []
    for line in lines:
        fields = line.split(",")
        if len(fields) != 2:
            raise ValueError("Each line must have format x,y")
        try:
            point = (float(fields[0]), float(fields[1]))
        except ValueError as exc:
            raise ValueError("Coordinates must be numbers") from exc
        if not np.isfinite(point).all():
            raise ValueError("Coordinates must be finite")
        points.append(point)
    return tuple(points)
