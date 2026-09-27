from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.export import (
    lidar_source_files,
    plan_to_dict,
    write_plan_exports,
    write_plan_svg,
)
from src.planes import fit_planes
from src.run import main
from src.timing import build_run, repeat_key
from tests.test_planes import _box_cloud


def test_json_has_height_and_residuals_on_synthetic_box(tmp_path: Path):
    pts = _box_cloud(np.random.default_rng(0))
    result = fit_planes(pts, voxel_m=0.05, distance_m=0.03, max_planes=8)
    payload = plan_to_dict(result, capture="synthetic-box", tier="lidar")
    assert payload["schema"] == "cozmo.room_plan.v1"
    assert payload["status"] == "baseline"
    assert payload["tier"] == "lidar"
    assert payload["input_tier"] == "lidar"
    assert payload["method"] == "ransac_rgb_d_planes"
    assert payload["measurement_status"] == "estimated"
    assert payload["accuracy_status"] == "not_calibrated"
    assert payload["source_files"] == []
    assert payload["cloud_origin"] == "unspecified"
    assert payload["units"] == "metres"
    assert payload["height_m"] == pytest.approx(2.5, abs=0.12)
    assert payload["height_blocked"] is False
    assert payload["floor"] is not None
    assert payload["ceiling"] is not None
    assert payload["floor"]["rmse_m"] >= 0.0
    assert "tape-measure" in payload["disclaimer"]
    assert len(payload["walls"]) >= 3
    assert len(payload["polygon_xz_m"]) >= 4
    assert payload["footprint_method"] in {
        "wall_lines",
        "wall_inlier_hull",
        "floor_hull",
    }
    assert payload["output_quality"] in {"ok", "warning"}
    if payload["footprint_method"] == "wall_lines":
        assert payload["output_quality"] == "ok"
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


def test_from_ply_lists_only_the_cloud(tmp_path: Path):
    ply = tmp_path / "cloud.ply"
    files, origin = lidar_source_files(tmp_path / "cap", ply=ply, from_ply=True)
    assert files == [str(ply)]
    assert origin == "existing_ply"
    rebuilt, origin = lidar_source_files(tmp_path / "cap", ply=ply, from_ply=False)
    assert origin == "rebuilt_this_run"
    assert all(not name.endswith("imu.csv") for name in rebuilt)
    assert rebuilt[0].endswith("camera_matrix.csv")


def test_run_metadata_records_config_and_excludes_timing_from_repeat_key(tmp_path: Path):
    output = tmp_path / "plan.json"
    run = build_run(
        capture="synthetic-box",
        tier="lidar",
        frame_stride=12,
        pixel_stride=4,
        min_confidence=1,
        invert_extrinsics=False,
        from_ply=True,
        n_points=100,
        n_retained=80,
        n_walls=4,
        n_openings=2,
        phases_s={"planes": 1.25},
        outputs=[output],
    )
    assert run["seeds"]["downsample"] == 0
    assert run["seeds"]["planes"] == 1
    assert run["outputs"] == [str(output.resolve())]
    assert run["determinism_key"] == repeat_key(run)
    assert repeat_key(run)["n_walls"] == 4
    assert "phases_s" not in repeat_key(run)


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
    assert data["input_tier"] == "lidar"
    assert data["status"] == "baseline"
    assert data["accuracy_status"] == "not_calibrated"
    assert data["measurement_status"] == "estimated"
    assert data["cloud_origin"] == "rebuilt_this_run"
    assert data["method"] == "ransac_rgb_d_planes"
    root = mini_capture
    assert data["source_files"] == [
        str(root / "camera_matrix.csv"),
        str(root / "odometry.csv"),
        str(root / "rgb.mp4"),
        str(root / "depth"),
        str(root / "confidence"),
    ]
    assert "imu.csv" not in " ".join(data["source_files"])
    assert "not_calibrated" in (tmp_path / "plan.svg").read_text()
    assert data["height_m"] is None
    assert data["height_blocked"] is True
    rc = main(
        [
            str(mini_capture),
            "--tier",
            "lidar",
            "--from-ply",
            "--out",
            str(ply),
            "--frame-stride",
            "1",
            "--pixel-stride",
            "1",
        ]
    )
    assert rc == 0
    reused = json.loads((tmp_path / "plan.json").read_text())
    assert reused["cloud_origin"] == "existing_ply"
    assert reused["source_files"] == [str(ply)]
    assert (tmp_path / "plan.svg").exists()
    assert (tmp_path / "preview_plan.png").exists()
