from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.planes import FittedPlane, PlaneResult, polygon_from_walls
from src.report import assess, fix_loop
from src.run import main


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
        _wall([1, 0, 0, -4], _edge_pts("x", 4.0)),
        _wall([0, 0, 1, 0], _edge_pts("z", 0.0)),
        _wall([0, 0, 1, -5], _edge_pts("z", 5.0)),
    ]


def _result(
    *,
    floor: FittedPlane | None,
    walls: list[FittedPlane],
    polygon: np.ndarray,
    height: float | None = None,
) -> PlaneResult:
    return PlaneResult(
        floor=floor,
        ceiling=None,
        walls=walls,
        others=[],
        polygon_xz=polygon,
        height_m=height,
        height_p05_p95_m=2.2,
        n_points=1000,
        n_downsampled=400,
    )


def test_near_parallel_wall_does_not_explode_outline():
    floor = _floor()
    # Meets the x=0 wall about 80 m away, and still crosses the room.
    rogue = _wall([26.0, 0.0, 1.0, -80.0], _edge_pts("x", 3.0, n=40))
    poly, method, _how = polygon_from_walls(floor, [*_room_walls(), rogue])
    assert method in {"wall_lines", "wall_support_rect", "wall_inlier_hull", "floor_hull"}
    span_x = float(poly[:, 0].max() - poly[:, 0].min())
    span_z = float(poly[:, 1].max() - poly[:, 1].min())
    assert span_x < 12.0
    assert span_z < 12.0
    assert poly[:, 0].max() < 20.0


def test_exploded_polygon_is_flagged_and_replaced():
    floor = _floor()
    walls = _room_walls()
    exploded = np.array(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 5.0], [100.0, 80.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    raw = _result(floor=floor, walls=walls, polygon=exploded)
    codes = {f.code for f in assess(raw)}
    assert "exploded_polygon" in codes
    report = fix_loop(raw, capture="synthetic")
    assert any("exploded outline" in fix for fix in report.fixes)
    assert "exploded_polygon" not in {f["code"] for f in report.after["findings"]}
    span = report.after["polygon_span_xz_m"]
    assert span is not None
    assert span[0] < 12.0
    assert span[1] < 12.0


def test_duplicate_walls_are_merged():
    floor = _floor()
    walls = _room_walls()
    walls.append(_wall([1, 0, 0, -0.2], _edge_pts("x", 0.2)))
    poly = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 5.0], [0.0, 5.0], [0.0, 0.0]])
    raw = _result(floor=floor, walls=walls, polygon=poly)
    assert any(f.code == "duplicate_walls" for f in assess(raw))
    report = fix_loop(raw, capture="synthetic")
    assert any("near-duplicate" in fix for fix in report.fixes)
    assert report.after["n_walls"] == 4
    assert "duplicate_walls" not in {f["code"] for f in report.after["findings"]}


def test_height_blocked_is_info_and_not_invented():
    floor = _floor()
    poly, _method, _how = polygon_from_walls(floor, _room_walls())
    raw = _result(floor=floor, walls=_room_walls(), polygon=poly, height=None)
    findings = assess(raw)
    blocked = [f for f in findings if f.code == "height_blocked"]
    assert len(blocked) == 1
    assert blocked[0].severity == "info"
    report = fix_loop(raw, capture="floor_only")
    assert report.fixes == []
    assert report.after["height_m"] is None
    assert report.after["height_blocked"] is True


def test_cli_report_writes_json(mini_capture: Path, tmp_path: Path, capsys):
    ply = tmp_path / "cloud.ply"
    rc = main([str(mini_capture), "--report", "--out", str(ply), "--frame-stride", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "damage report:" in out
    payload = json.loads((tmp_path / "report.json").read_text())
    assert payload["schema"] == "cozmo.damage_report.v1"
    assert payload["status"] == "baseline"
    assert payload["input_tier"] == "lidar"
    assert payload["method"] == "reconstruction_health_check"
    assert payload["measurement_status"] == "estimated"
    assert payload["accuracy_status"] == "not_calibrated"
    assert payload["cloud_origin"] == "rebuilt_this_run"
    assert any(name.endswith("odometry.csv") for name in payload["source_files"])
    assert "before" in payload and "after" in payload
    assert "physical room damage" in payload["disclaimer"]
