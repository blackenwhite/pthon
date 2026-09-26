from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.export import plan_to_dict
from src.planes import FittedPlane, fit_planes
from src.walls import select_walls
from tests.test_planes import _box_cloud

PLY = Path(__file__).resolve().parents[1] / "out" / "c00a170fe1" / "cloud.ply"


def _floor() -> FittedPlane:
    rng = np.random.default_rng(0)
    pts = np.stack(
        [rng.uniform(0, 4, 600), np.zeros(600), rng.uniform(0, 5, 600)],
        axis=1,
    )
    return FittedPlane(np.array([0.0, 1.0, 0.0, 0.0]), pts, 0.01, 0.005, "floor")


def _wall(abcd: list[float], pts: np.ndarray) -> FittedPlane:
    return FittedPlane(np.array(abcd, dtype=np.float64), pts, 0.01, 0.005, "wall")


def _face(
    axis: str,
    value: float,
    *,
    n: int = 200,
    y0: float = 0.0,
    y1: float = 2.4,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = rng.uniform(y0, y1, n)
    if axis == "x":
        along = rng.uniform(0.0, 5.0, n)
        return np.stack([np.full(n, value), y, along], axis=1)
    if axis == "z":
        along = rng.uniform(0.0, 4.0, n)
        return np.stack([along, y, np.full(n, value)], axis=1)
    raise ValueError(axis)


def _room() -> list[FittedPlane]:
    return [
        _wall([1, 0, 0, 0], _face("x", 0.0, seed=1)),
        _wall([1, 0, 0, -4], _face("x", 4.0, seed=2)),
        _wall([0, 0, 1, 0], _face("z", 0.0, seed=3)),
        _wall([0, 0, 1, -5], _face("z", 5.0, seed=4)),
    ]


def test_room_walls_are_kept():
    kept, rejected = select_walls(_room(), _floor())
    assert len(kept) == 4
    assert rejected == []
    assert all(w.confidence == "ok" for w in kept)


def test_short_interior_cabinet_is_rejected_with_a_reason():
    walls = _room()
    cabinet = _wall([1, 0, 0, -1.8], _face("x", 1.8, n=160, y0=0.0, y1=0.7))
    # Narrow the cabinet so it is a patch, not a second full wall.
    cabinet.points[:, 2] = np.linspace(1.6, 2.3, cabinet.points.shape[0])
    walls.append(cabinet)
    kept, rejected = select_walls(walls, _floor())
    assert len(kept) == 4
    assert len(rejected) == 1
    assert rejected[0].code in {"short_vertical_extent", "short_span", "interior_patch"}
    assert "inliers=" in rejected[0].detail


def test_near_parallel_inner_face_is_dropped():
    walls = _room()
    inner = _wall([1, 0, 0, -0.4], _face("x", 0.4))
    walls.append(inner)
    kept, rejected = select_walls(walls, _floor())
    assert len(kept) == 4
    codes = {item.code for item in rejected}
    assert "near_parallel_duplicate" in codes
    means = [float(w.mean_xyz[0]) for w in kept]
    assert min(means) < 0.15


def test_tilted_plane_is_rejected_and_listed_on_the_plan():
    walls = _room()
    normal = np.array([0.15, 0.45, 0.88], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tilted = _wall([*normal.tolist(), 0.0], _face("z", 2.5))
    walls.append(tilted)
    kept, rejected = select_walls(walls, _floor())
    assert any(item.code == "tilted_plane" for item in rejected)
    from src.planes import PlaneResult

    result = PlaneResult(
        floor=_floor(),
        ceiling=None,
        walls=kept,
        others=[],
        polygon_xz=np.zeros((0, 2)),
        height_m=None,
        height_p05_p95_m=None,
        n_points=100,
        n_downsampled=80,
        rejected_walls=rejected,
    )
    payload = plan_to_dict(result, capture="filter")
    reasons = {item["reason"] for item in payload["rejected_walls"]}
    assert "tilted_plane" in reasons
    assert payload["rejected_walls"][0]["detail"]


def test_slightly_tilted_boundary_wall_is_kept_with_low_confidence():
    walls = _room()[:3]
    normal = np.array([0.0, 0.15, 0.988], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    soft = _wall([*normal.tolist(), -5.0], _face("z", 5.0))
    walls.append(soft)
    kept, rejected = select_walls(walls, _floor())
    assert len(kept) == 4
    assert rejected == []
    assert any(w.confidence == "low" for w in kept)


def test_fit_planes_records_the_filter_on_a_box():
    pts = _box_cloud(np.random.default_rng(0))
    result = fit_planes(pts, voxel_m=0.05, distance_m=0.03, max_planes=8)
    assert len(result.walls) >= 3
    assert any(note.startswith("wall filter:") for note in result.notes)
    payload = plan_to_dict(result, capture="box")
    assert "rejected_walls" in payload
    assert all("confidence" in wall for wall in payload["walls"])


@pytest.mark.skipif(not PLY.exists(), reason="frozen cloud not on disk")
def test_repeated_exports_of_the_frozen_cloud_keep_the_same_walls():
    from src.cloud import read_ply

    points, _colors = read_ply(PLY)
    first = fit_planes(points)
    second = fit_planes(points)
    assert len(first.walls) == len(second.walls)
    assert len(first.walls) >= 2
    assert [w.n_inliers for w in first.walls] == [w.n_inliers for w in second.walls]
    assert len(first.rejected_walls) == len(second.rejected_walls)
