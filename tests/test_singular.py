import numpy as np
import pytest

from flow import simulate
from smooth import smooth_density
from target_measure import SingularTarget, parse_points


def test_h0_equal_masses_and_support():
    target = SingularTarget("Finite points", points=((0.0, 0.0), (2.0, 0.0), (0.0, 3.0)))
    assert target.dimension == 0 and target.mass() == 3
    points = target.sample(999, 2)
    counts = [np.sum(np.all(points == point, axis=1)) for point in target.points]
    assert counts == [333, 333, 333]


def test_h1_segment_and_circle():
    segment = SingularTarget("Segment", points=((0.0, 0.0), (2.0, 0.0)))
    points = segment.sample(500, 1)
    assert np.all(points[:, 1] == 0) and np.all((0 <= points[:, 0]) & (points[:, 0] <= 2))
    assert segment.mass() == pytest.approx(2.0)
    circle = SingularTarget("Circle", center=(2.0, 1.0), radius=1.5)
    samples = circle.sample(1000, 1)
    assert np.allclose(np.linalg.norm(samples - [2.0, 1.0], axis=1), 1.5)
    assert circle.mass() == pytest.approx(3 * np.pi)


def test_reject_invalid_geometry():
    invalid_targets = (
        SingularTarget("Finite points", points=((0.0, 0.0), (0.0, 0.0))),
        SingularTarget("Segment", points=((1.0, 1.0), (1.0, 1.0))),
        SingularTarget("Circle", radius=-1),
    )
    for target in invalid_targets:
        with pytest.raises(ValueError):
            target.validate()
    with pytest.raises(ValueError):
        parse_points("1,2,3")


def test_smooth_nonnegative_normalized():
    points = np.array([[0.0, 0.0]] * 100)
    x, y, values = smooth_density(points, (-2, 2, -2, 2), 70, 0.2)
    assert values.min() >= 0
    assert values.sum() * (x[1] - x[0]) * (y[1] - y[0]) == pytest.approx(1)


def test_singular_target_flow_energy_decrease():
    source = np.random.default_rng(5).normal(size=(150, 2))
    target = SingularTarget("Circle", radius=1.5).sample(150, 7)
    result = simulate(source, target, 96, dt=0.3, steps=25)
    assert result.energies[-1] < result.energies[0]
    assert np.max(np.diff(result.energies)) <= 1e-10
