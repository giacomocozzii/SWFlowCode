import numpy as np
import pytest
from density import COMPACT, INITIAL, MIXTURE, TARGET, DensitySyntaxError, Parser, grid_density, sample_density


def test_formulas_and_normalization():
    for formula in (INITIAL, TARGET, COMPACT, MIXTURE):
        grid = grid_density(formula, (-6, 8, -6, 7), 100)
        area = 14 * 13 / 100**2
        assert abs(grid.values.sum() * area - 1) < 1e-10
        assert np.min(grid.values) >= 0


def test_sample_center_gaussian():
    g = grid_density(INITIAL, (-6, 8, -6, 7))
    pts = sample_density(g, 20000, 42)
    assert np.linalg.norm(pts.mean(axis=0)) < 0.05


def test_reject_unsafe_input():
    for s in ('__import__("os")', r'\input{secret}', 'x.__class__'):
        with pytest.raises(DensitySyntaxError):
            Parser(s).parse()


def test_reject_negative_density():
    with pytest.raises(ValueError, match="negative"):
        grid_density("x", (-2, 2, -2, 2))
