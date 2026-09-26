from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.export import plan_to_dict
from src.ingest import Pose, load_capture
from src.planes import fit_planes
from src.posegraph import (
    T_to_pose,
    path_length_m,
    pose_to_T,
    refine_pose_graph,
    start_end_m,
)
from src.run import main
from tests.conftest import write_mini_capture
from tests.test_planes import _box_cloud


def _box_with_openings(rng: np.random.Generator) -> np.ndarray:
    pts = _box_cloud(rng)
    # door in wall x=0: z in [1.7, 2.6], y below 2.1
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    door = (np.abs(x) < 0.12) & (z >= 1.7) & (z <= 2.6) & (y < 2.1)
    # window in wall x=4: z in [2.0, 3.2], y in [1.0, 2.05]
    win = (np.abs(x - 4.0) < 0.12) & (z >= 2.0) & (z <= 3.2) & (y >= 1.0) & (y <= 2.05)
    return pts[~(door | win)]


def test_detects_door_and_window_on_gappy_box():
    rng = np.random.default_rng(2)
    pts = _box_with_openings(rng)
    result = fit_planes(pts, voxel_m=0.05, distance_m=0.03, max_planes=8)
    assert result.floor is not None
    assert len(result.walls) >= 3
    kinds = {op.kind for op in result.openings}
    assert "door" in kinds
    assert "window" in kinds
    doors = [op for op in result.openings if op.kind == "door"]
    assert any(0.65 <= op.width_m <= 1.2 for op in doors)
    payload = plan_to_dict(result, capture="gappy-box")
    assert any(o["kind"] == "door" for o in payload["openings"])
    svg_kinds = [o["kind"] for o in payload["openings"]]
    assert "window" in svg_kinds


def test_solid_box_has_no_interior_openings():
    rng = np.random.default_rng(0)
    pts = _box_cloud(rng)
    result = fit_planes(pts, voxel_m=0.05, distance_m=0.03, max_planes=8)
    assert result.floor is not None
    assert result.openings == [] or all(op.width_m < 0.55 for op in result.openings)
    # occupancy heuristic should not invent a door on a solid wall
    assert all(op.kind != "door" for op in result.openings)


def test_path_metrics_on_straight_line():
    Ts = np.stack([np.eye(4) for _ in range(4)])
    for i, T in enumerate(Ts):
        T[0, 3] = float(i)
    assert path_length_m(Ts) == pytest.approx(3.0)
    assert start_end_m(Ts) == pytest.approx(3.0)


def test_pose_roundtrip_translation():
    p = Pose(
        timestamp=1.0,
        frame="000003",
        t=np.array([1.0, 2.0, 3.0]),
        q=np.array([0.0, 0.0, 0.0, 1.0]),
        fx=100.0,
        fy=100.0,
        cx=16.0,
        cy=12.0,
    )
    T = pose_to_T(p)
    back = T_to_pose(p, T)
    assert back.t == pytest.approx(p.t)
    assert back.q == pytest.approx(p.q, abs=1e-6)


def test_drift_cli_writes_json_png(mini_capture: Path, tmp_path: Path):
    ply = tmp_path / "cloud.ply"
    rc = main([str(mini_capture), "--drift", "--out", str(ply), "--frame-stride", "1"])
    assert rc == 0
    data = json.loads((tmp_path / "drift.json").read_text())
    assert "raw_path_m" in data
    assert "opt_path_m" in data
    assert (tmp_path / "drift_path.png").exists()


def test_pose_graph_does_not_explode_on_short_walk(tmp_path: Path):
    cap_dir = write_mini_capture(tmp_path / "walk", n_frames=6)
    from src.ingest import load_capture

    cap = load_capture(cap_dir)
    result = refine_pose_graph(cap, frame_stride=1, pixel_stride=2, loop_min_sep=3)
    assert result.n_frames >= 3
    assert result.mean_pose_delta_m < 1.5
    assert result.opt_path_m < result.raw_path_m * 3.0 + 1.0
