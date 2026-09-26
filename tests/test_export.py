from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.export import plan_to_dict, write_plan_exports, write_plan_svg
from src.planes import fit_planes
from src.run import main
from tests.test_planes import _box_cloud


def test_json_has_height_and_residuals_on_synthetic_box(tmp_path: Path):
    pts = _box_cloud(np.random.default_rng(0))
    result = fit_planes(pts, voxel_m=0.05, distance_m=0.03, max_planes=8)
    payload = plan_to_dict(result, capture="synthetic-box", tier="lidar")
    assert payload["schema"] == "cozmo.room_plan.v1"
    assert payload["tier"] == "lidar"
    assert payload["units"] == "metres"
    assert payload["height_m"] == pytest.approx(2.5, abs=0.12)
    assert payload["height_blocked"] is False
    assert payload["floor"] is not None
    assert payload["ceiling"] is not None
    assert payload["floor"]["rmse_m"] >= 0.0
    assert "tape-measure" in payload["disclaimer"]
    assert len(payload["walls"]) >= 3
    assert len(payload["polygon_xz_m"]) >= 4
    assert "openings" in payload
    jpath, spath = write_plan_exports(
        result, tmp_path, capture="synthetic-box", tier="lidar"
    )
    data = json.loads(jpath.read_text())
    assert data["height_m"] == payload["height_m"]
    svg = spath.read_text()
    assert "<svg" in svg
    assert "1 m" in svg
    assert "synthetic-box" in svg


def test_json_height_null_when_no_ceiling(tmp_path: Path):
    rng = np.random.default_rng(1)
    n = 2000
    floor = np.stack(
        [rng.uniform(0, 3, n), np.zeros(n), rng.uniform(0, 3, n)], axis=1
    )
    y = rng.uniform(0, 1.5, n)
    z = rng.uniform(0, 3, n)
    wall = np.stack([np.zeros(n), y, z], axis=1)
    result = fit_planes(np.concatenate([floor, wall], axis=0), voxel_m=0.05)
    payload = plan_to_dict(result, capture="no-ceil")
    assert payload["height_m"] is None
    assert payload["height_blocked"] is True
    assert payload["ceiling"] is None
    write_plan_exports(result, tmp_path, capture="no-ceil")
    data = json.loads((tmp_path / "plan.json").read_text())
    assert data["height_m"] is None


def test_svg_blocked_polygon(tmp_path: Path):
    payload = {
        "capture": "empty",
        "tier": "lidar",
        "height_blocked": True,
        "polygon_xz_m": [],
        "disclaimer": "test",
    }
    path = write_plan_svg(payload, tmp_path / "plan.svg")
    text = path.read_text()
    assert "BLOCKED" in text


def test_tier_lidar_cli_writes_json_svg(mini_capture: Path, tmp_path: Path):
    ply = tmp_path / "cloud.ply"
    rc = main(
        [
            str(mini_capture),
            "--tier",
            "lidar",
            "--out",
            str(ply),
            "--frame-stride",
            "1",
            "--pixel-stride",
            "1",
        ]
    )
    assert rc == 0
    data = json.loads((tmp_path / "plan.json").read_text())
    assert data["tier"] == "lidar"
    assert data["height_m"] is None
    assert data["height_blocked"] is True
    assert (tmp_path / "plan.svg").exists()
    assert (tmp_path / "preview_plan.png").exists()
