"""Ceiling search: a high empty-above sheet counts; a table or a wall slice does not."""

from __future__ import annotations

import numpy as np

from src.ceiling import _reject_reason
from src.export import plan_to_dict
from src.planes import _fit_plane_svd, _to_fitted, fit_planes


def _sheet(rng: np.random.Generator, y: float, n: int, x0, x1, z0, z1) -> np.ndarray:
    return np.stack(
        [
            rng.uniform(x0, x1, n),
            np.full(n, y),
            rng.uniform(z0, z1, n),
        ],
        axis=1,
    ) + rng.normal(0.0, 0.008, size=(n, 3))


def _walls(rng: np.random.Generator, n: int, height: float) -> np.ndarray:
    y = rng.uniform(0.0, height, n)
    z = rng.uniform(0.0, 5.0, n)
    x = rng.uniform(0.0, 4.0, n)
    return np.concatenate(
        [
            np.stack([np.zeros(n), y, z], axis=1),
            np.stack([np.full(n, 4.0), y, z], axis=1),
            np.stack([x, y, np.zeros(n)], axis=1),
            np.stack([x, y, np.full(n, 5.0)], axis=1),
        ],
        axis=0,
    ) + rng.normal(0.0, 0.008, size=(4 * n, 3))


def _room(rng: np.random.Generator, *, ceiling: bool, table: bool) -> np.ndarray:
    parts = [
        _sheet(rng, 0.0, 2500, 0.0, 4.0, 0.0, 5.0),
        _walls(rng, 800, 2.5 if ceiling else 2.2),
    ]
    if ceiling:
        parts.append(_sheet(rng, 2.5, 2500, 0.0, 4.0, 0.0, 5.0))
    if table:
        parts.append(_sheet(rng, 0.75, 1200, 1.0, 2.2, 1.0, 2.4))
    return np.concatenate(parts, axis=0)


def test_table_does_not_hide_a_real_ceiling():
    rng = np.random.default_rng(0)
    result = fit_planes(_room(rng, ceiling=True, table=True), voxel_m=0.05, distance_m=0.03)
    assert result.ceiling is not None
    assert result.height_m is not None
    assert abs(result.height_m - 2.5) < 0.12
    assert abs(result.ceiling.mean_xyz[1] - 2.5) < 0.12
    assert any(note.startswith("ceiling:") for note in result.notes)


def test_walls_and_table_without_ceiling_stay_blocked():
    rng = np.random.default_rng(1)
    result = fit_planes(_room(rng, ceiling=False, table=True), voxel_m=0.05, distance_m=0.03)
    assert result.floor is not None
    assert result.ceiling is None
    assert result.height_m is None
    assert result.rejected_ceilings
    assert any("height left null" in note for note in result.notes)
    payload = plan_to_dict(result, capture="no-ceiling")
    assert payload["height_m"] is None
    assert payload["ceiling"] is None
    assert "rejected_ceilings" in payload


def test_points_above_a_filled_sheet_are_not_a_ceiling():
    rng = np.random.default_rng(4)
    sheet = _sheet(rng, 2.0, 2000, 0.0, 4.0, 0.0, 5.0)
    above = _sheet(rng, 2.3, 2000, 0.0, 4.0, 0.0, 5.0)
    model = _fit_plane_svd(sheet)
    cand = _to_fitted(model, sheet, "ceiling")
    code, _detail = _reject_reason(
        np.concatenate([sheet, above], axis=0), cand, 0.0, 0.04
    )
    assert code == "occupied_above"


def test_small_high_patch_is_not_a_ceiling():
    rng = np.random.default_rng(2)
    cloud = np.concatenate(
        [
            _sheet(rng, 0.0, 2000, 0.0, 4.0, 0.0, 5.0),
            _walls(rng, 600, 2.6),
            _sheet(rng, 2.5, 200, 1.5, 1.9, 1.5, 1.9),
        ],
        axis=0,
    )
    result = fit_planes(cloud, voxel_m=0.05, distance_m=0.03)
    assert result.height_m is None
    assert result.ceiling is None
    assert any(item.code == "sparse_coverage" for item in result.rejected_ceilings)


def test_occupied_sheet_falls_back_to_column_height():
    """A real ceiling with clutter above still yields height from empty mid-band columns."""
    rng = np.random.default_rng(5)
    n = 900
    y = rng.uniform(0.0, 2.5, n)
    z = rng.uniform(0.0, 5.0, n)
    x = rng.uniform(0.0, 4.0, n)
    walls = np.concatenate(
        [
            np.stack([np.zeros(n), y, z], axis=1),
            np.stack([np.full(n, 4.0), y, z], axis=1),
            np.stack([x, y, np.zeros(n)], axis=1),
            np.stack([x, y, np.full(n, 5.0)], axis=1),
        ],
        axis=0,
    ) + rng.normal(0.0, 0.008, size=(4 * n, 3))
    floor = _sheet(rng, 0.0, 2500, 0.0, 4.0, 0.0, 5.0)
    ceiling = _sheet(rng, 2.5, 1800, 0.0, 4.0, 0.0, 5.0)
    clutter = _sheet(rng, 2.75, 1800, 0.0, 4.0, 0.0, 5.0)
    result = fit_planes(
        np.concatenate([floor, walls, ceiling, clutter], axis=0),
        voxel_m=0.05,
        distance_m=0.03,
    )
    assert result.height_m is not None
    assert abs(result.height_m - 2.5) < 0.25
    assert any("bimodal" in note or note.startswith("ceiling:") for note in result.notes)


def test_ceiling_search_is_repeatable():
    rng = np.random.default_rng(3)
    pts = _room(rng, ceiling=True, table=True)
    first = fit_planes(pts, voxel_m=0.05, distance_m=0.03)
    second = fit_planes(pts, voxel_m=0.05, distance_m=0.03)
    assert first.height_m == second.height_m
    assert [item.code for item in first.rejected_ceilings] == [
        item.code for item in second.rejected_ceilings
    ]
