from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cloud import build_cloud
from src.ingest import load_capture
from src.planes import fit_planes, write_plan_preview
from tests.conftest import SINGLE_ROOM

WITH_CEILING = Path(__file__).resolve().parents[1] / "store" / "c7d28f72c6"


def _box_cloud(rng: np.random.Generator) -> np.ndarray:
    """Axis-aligned room: floor Y=0, ceiling Y=2.5, X in [0,4], Z in [0,5]."""
    n = 2500
    noise = 0.008

    def nrm(n3):
        return rng.normal(0.0, noise, size=(n, 3))

    xf = rng.uniform(0, 4, n)
    zf = rng.uniform(0, 5, n)
    floor = np.stack([xf, np.zeros(n), zf], axis=1) + nrm(n)
    ceil = np.stack([xf, np.full(n, 2.5), zf], axis=1) + nrm(n)

    y = rng.uniform(0, 2.5, n)
    z = rng.uniform(0, 5, n)
    wall_x0 = np.stack([np.zeros(n), y, z], axis=1) + nrm(n)
    wall_x1 = np.stack([np.full(n, 4.0), y, z], axis=1) + nrm(n)
    x = rng.uniform(0, 4, n)
    wall_z0 = np.stack([x, y, np.zeros(n)], axis=1) + nrm(n)
    wall_z1 = np.stack([x, y, np.full(n, 5.0)], axis=1) + nrm(n)
    return np.concatenate(
        [floor, ceil, wall_x0, wall_x1, wall_z0, wall_z1], axis=0
    )


def test_synthetic_box_floor_walls_ceiling_and_height(tmp_path: Path):
    rng = np.random.default_rng(0)
    pts = _box_cloud(rng)
    result = fit_planes(pts, voxel_m=0.05, distance_m=0.03, max_planes=8)
    assert result.floor is not None
    assert result.ceiling is not None
    assert abs(result.floor.mean_xyz[1]) < 0.08
    assert abs(result.ceiling.mean_xyz[1] - 2.5) < 0.08
    assert result.height_m is not None
    assert result.height_m == pytest.approx(2.5, abs=0.12)
    assert len(result.walls) >= 3
    for w in result.walls:
        assert abs(w.normal[1]) <= 0.35
    assert result.polygon_xz.shape[0] >= 4
    xz = result.polygon_xz[:-1]
    assert xz[:, 0].min() < 0.6
    assert xz[:, 0].max() > 3.4
    assert xz[:, 1].min() < 0.6
    assert xz[:, 1].max() > 4.4
    png = write_plan_preview(pts, result.polygon_xz, tmp_path / "p.png")
    assert png.stat().st_size > 1000


def test_no_ceiling_is_blocked_not_invented():
    rng = np.random.default_rng(1)
    n = 2000
    floor = np.stack(
        [rng.uniform(0, 3, n), np.zeros(n), rng.uniform(0, 3, n)], axis=1
    )
    y = rng.uniform(0, 1.5, n)
    z = rng.uniform(0, 3, n)
    wall = np.stack([np.zeros(n), y, z], axis=1)
    result = fit_planes(np.concatenate([floor, wall], axis=0), voxel_m=0.05)
    assert result.floor is not None
    assert result.ceiling is None
    assert result.height_m is None


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="founder dump not on disk")
def test_single_room_has_floor_and_walls():
    cap = load_capture(SINGLE_ROOM)
    cloud = build_cloud(cap, frame_stride=40, pixel_stride=8, min_confidence=1)
    result = fit_planes(cloud.points)
    assert result.floor is not None
    assert abs(result.floor.normal[1]) > 0.8
    assert len(result.walls) >= 3
    span = cloud.cloud_span_m
    assert 2.0 < span[0] < 20.0
    assert 2.0 < span[2] < 20.0
    y = result.floor.mean_xyz[1]
    assert -0.8 < y < 1.0


@pytest.mark.skipif(not WITH_CEILING.exists(), reason="ceiling dump not on disk")
def test_with_ceiling_height_is_plausible_not_exploded():
    cap = load_capture(WITH_CEILING)
    cloud = build_cloud(cap, frame_stride=80, pixel_stride=10, min_confidence=1)
    result = fit_planes(cloud.points)
    assert result.floor is not None
    if result.height_m is not None:
        assert 1.8 < result.height_m < 4.5
        assert result.ceiling is not None
    else:
        # still a useful percentile check; must not be inverted ~13 m
        assert result.height_p05_p95_m is not None
        assert result.height_p05_p95_m < 8.0
