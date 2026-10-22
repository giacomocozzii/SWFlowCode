import numpy as np
import pytest

from density import DensitySyntaxError, grid_density, sample_density, split_indicator
from flow import simulate
from region import Region

BOUNDS = (-4, 4, -4, 4)


def test_indicator_parsing():
    for formula in (
        r"e^{-(x^2+y^2)}\mathbf{1}_A",
        r"e^{-(x^2+y^2)}\mathbb{1}_{A}",
        r"3\chi_A",
    ):
        expression, flagged = split_indicator(formula)
        assert flagged and expression
    assert split_indicator("x^2+y^2") == ("x^2+y^2", False)
    with pytest.raises(DensitySyntaxError):
        split_indicator(r"\mathbf{1}_A + 2")


def test_masked_rectangle_normalization_and_support():
    region = Region("Rectangle", rectangle=(-1, 1, -2, 2))
    grid = grid_density(r"7\mathbf{1}_A", BOUNDS, 160, region)
    area = 8 * 8 / 160**2
    assert np.isclose(grid.values.sum() * area, 1)
    assert np.isclose(grid.mass_before_normalization, 7 * 8, rtol=1e-12)
    points = sample_density(grid, 2000, 1)
    assert region.contains(points[:, 0], points[:, 1]).all()


def test_masked_disk_gaussian():
    region = Region("Disk", center=(1, 0), radius=1.5)
    grid = grid_density(r"e^{-(x^2+y^2)}\mathbf{1}_{A}", BOUNDS, 200, region)
    points = sample_density(grid, 1000, 10)
    assert region.contains(points[:, 0], points[:, 1]).all()
    assert np.isclose(grid.values.sum() * (8 / 200) ** 2, 1)


def test_masked_annulus_ellipse_and_half_plane():
    regions = (
        Region("Annulus", radius=2, inner_radius=1),
        Region("Ellipse", radii=(2, 1)),
        Region("Half-plane x ≥ a", threshold=0),
    )
    for region in regions:
        grid = grid_density("1", BOUNDS, 120, region)
        points = sample_density(grid, 500, 9)
        assert region.contains(points[:, 0], points[:, 1]).all()


def test_invalid_mask_and_zero_mass():
    with pytest.raises(ValueError, match="select a geometric region"):
        grid_density(r"e^{-x^2-y^2}\mathbf{1}_A", BOUNDS)
    with pytest.raises(ValueError, match="zero"):
        grid_density("1", BOUNDS, 100, Region("Disk", center=(100, 100), radius=1))
    with pytest.raises(ValueError):
        grid_density("1", BOUNDS, 100, Region("Annulus", radius=1, inner_radius=2))


def test_masked_target_in_flow():
    region = Region("Disk", center=(2, 1), radius=1)
    source = sample_density(grid_density("1", BOUNDS, 100), 150, 3)
    target = sample_density(
        grid_density(r"e^{-(x^2+y^2)}\mathbf{1}_A", BOUNDS, 100, region), 150, 4
    )
    result = simulate(source, target, 96, dt=0.3, steps=20)
    assert result.energies[-1] < result.energies[0]
