"""Floor is the lowest wide sheet, not the densest wall-height slab."""

from __future__ import annotations

import numpy as np

from src.floor import select_floor
from src.planes import fit_planes


def _sheet(rng: np.random.Generator, y: float, n: int, x0, x1, z0, z1) -> np.ndarray:
    return np.stack(
        [
            rng.uniform(x0, x1, n),
            np.full(n, y),
            rng.uniform(z0, z1, n),
        ],
        axis=1,
    ) + rng.normal(0.0, 0.01, size=(n, 3))


def test_lowest_wide_sheet_beats_a_denser_wall_slab():
    rng = np.random.default_rng(0)
    floor = _sheet(rng, 0.0, 1800, 0.0, 5.0, 0.0, 6.0)
    # Chest-height horizontal slice through the walls: more points, wrong plane.
    slab = _sheet(rng, 1.35, 5000, 0.0, 5.0, 0.0, 6.0)
    cloud = np.concatenate([floor, slab], axis=0)
    chosen, _rejected, notes = select_floor(cloud, distance_m=0.04)
    assert chosen is not None
    assert abs(float(chosen.mean_xyz[1])) < 0.12
    assert any(n.startswith("floor:") for n in notes)


def test_fit_planes_uses_the_low_floor_for_height():
    rng = np.random.default_rng(1)
    n = 800
    y = rng.uniform(0.0, 2.45, n)
    z = rng.uniform(0.0, 6.0, n)
    x = rng.uniform(0.0, 5.0, n)
    walls = np.concatenate(
        [
            np.stack([np.zeros(n), y, z], axis=1),
            np.stack([np.full(n, 5.0), y, z], axis=1),
            np.stack([x, y, np.zeros(n)], axis=1),
            np.stack([x, y, np.full(n, 6.0)], axis=1),
        ],
        axis=0,
    )
    cloud = np.concatenate(
        [
            _sheet(rng, 0.0, 2000, 0.0, 5.0, 0.0, 6.0),
            _sheet(rng, 1.3, 4000, 0.0, 5.0, 0.0, 6.0),
            _sheet(rng, 2.45, 2200, 0.0, 5.0, 0.0, 6.0),
            walls + rng.normal(0.0, 0.01, size=walls.shape),
        ],
        axis=0,
    )
    result = fit_planes(cloud, voxel_m=0.05, distance_m=0.03)
    assert result.floor is not None
    assert abs(float(result.floor.mean_xyz[1])) < 0.15
    assert result.height_m is not None
    assert abs(result.height_m - 2.45) < 0.2
