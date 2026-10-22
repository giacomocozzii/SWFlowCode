import numpy as np
from flow import directions_2d, simulate


def test_directions():
    dirs = directions_2d(384)
    assert dirs.shape == (384, 2)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1)


def test_energy_decreases():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(150, 2))
    target = x + [2, 1]
    r = simulate(x, target, 96, dt=0.4, steps=30)
    assert np.max(np.diff(r.energies)) <= 1e-10
    assert r.energies[-1] < r.energies[0] * 0.1


def test_identical_samples_stationary():
    x = np.random.default_rng(1).normal(size=(40, 2))
    r = simulate(x, x.copy(), 32, steps=4)
    assert np.allclose(r.positions, x[None, :, :])
    assert np.allclose(r.energies, 0)
