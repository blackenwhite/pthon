from __future__ import annotations

import numpy as np

from src.planes import (
    FittedPlane,
    _polygon_self_intersects,
    _within_floor_margin,
    footprint_quality,
    polygon_from_walls,
)


def _floor() -> FittedPlane:
    rng = np.random.default_rng(0)
    pts = np.stack(
        [rng.uniform(0, 4, 400), np.zeros(400), rng.uniform(0, 5, 400)],
        axis=1,
    )
    return FittedPlane(np.array([0.0, 1.0, 0.0, 0.0]), pts, 0.01, 0.005, "floor")


def _wall(abcd: list[float], pts: np.ndarray) -> FittedPlane:
    return FittedPlane(np.array(abcd, dtype=np.float64), pts, 0.01, 0.005, "wall")


def _edge_pts(axis: str, value: float, n: int = 80) -> np.ndarray:
    rng = np.random.default_rng(2)
    y = rng.uniform(0.0, 2.4, n)
    along = rng.uniform(0.0, 5.0 if axis == "x" else 4.0, n)
    if axis == "x":
        return np.stack([np.full(n, value), y, along], axis=1)
    return np.stack([along, y, np.full(n, value)], axis=1)


def _room_walls() -> list[FittedPlane]:
    return [
        _wall([1, 0, 0, 0], _edge_pts("x", 0.0)),
        _wall([-1, 0, 0, 4], _edge_pts("x", 4.0)),
        _wall([0, 0, 1, 0], _edge_pts("z", 0.0)),
        _wall([0, 0, -1, 5], _edge_pts("z", 5.0)),
    ]


def test_rectangle_uses_wall_lines_inside_floor():
    floor = _floor()
    poly, method, note = polygon_from_walls(floor, _room_walls())
    assert method == "wall_lines"
    assert footprint_quality(method, poly) == "ok"
    assert "wall ∩ floor" in note
    assert not _polygon_self_intersects(poly)
    assert poly[:, 0].min() > -0.5
    assert poly[:, 0].max() < 4.5
    assert poly[:, 1].min() > -0.5
    assert poly[:, 1].max() < 5.5


def test_bowtie_is_not_accepted_against_the_floor():
    hull = np.array(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 5.0], [0.0, 5.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    bowtie = np.array(
        [[0.5, 0.5], [3.0, 4.0], [0.5, 4.0], [3.0, 0.5], [0.5, 0.5]],
        dtype=np.float64,
    )
    rect = np.array(
        [[0.2, 0.2], [3.8, 0.2], [3.8, 4.8], [0.2, 4.8], [0.2, 0.2]],
        dtype=np.float64,
    )
    assert _polygon_self_intersects(bowtie)
    assert not _polygon_self_intersects(rect)
    assert not _within_floor_margin(bowtie, hull)
    assert _within_floor_margin(rect, hull)


def test_wall_lines_that_cross_fall_back():
    """Angle order of these four lines is a bowtie inside the floor."""
    floor = _floor()
    specs = [
        ([0.783111, 0.0, -0.621882, 0.105950], [0.5, 0.8], [3.2, 4.2]),
        ([-0.076696, 0.0, 0.997054, -3.942200], [3.2, 4.2], [0.6, 4.0]),
        ([-0.808736, 0.0, -0.588172, 2.837928], [0.6, 4.0], [3.0, 0.7]),
        ([-0.039968, 0.0, -0.999201, 0.819345], [3.0, 0.7], [0.5, 0.8]),
    ]
    walls: list[FittedPlane] = []
    for abcd, a, b in specs:
        pts = []
        for t in np.linspace(0.0, 1.0, 40):
            x = a[0] + t * (b[0] - a[0])
            z = a[1] + t * (b[1] - a[1])
            pts.append([x, 1.0, z])
        walls.append(_wall(abcd, np.array(pts, dtype=np.float64)))
    poly, method, note = polygon_from_walls(floor, walls)
    assert method == "wall_inlier_hull"
    assert "self-intersect" in note
    assert float(poly[:, 0].max()) < 4.5
    assert footprint_quality(method, poly) == "warning"


def test_outline_past_the_floor_margin_uses_the_floor_hull():
    floor = _floor()
    # 4.7 m is inside the wide spike box and outside the 0.5 m floor margin.
    # A wall at 8 m is dropped as a far spike and no longer fails the ring.
    walls = [
        *_room_walls(),
        _wall([-1, 0, 0, 4.7], _edge_pts("x", 4.7)),
    ]
    poly, method, note = polygon_from_walls(floor, walls)
    assert method == "floor_hull"
    assert "outside floor margin" in note
    assert float(poly[:, 0].max()) < 4.6
    assert float(poly[:, 1].max()) < 5.6
    assert footprint_quality(method, poly) == "warning"


def test_no_walls_uses_the_floor_hull():
    floor = _floor()
    poly, method, note = polygon_from_walls(floor, [])
    assert method == "floor_hull"
    assert "no walls" in note
    assert footprint_quality(method, poly) == "warning"
    assert poly.shape[0] >= 4
