from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.adapters import (
    AdapterError,
    identify_tier,
    open_lidar,
    open_photo,
    open_video,
    write_blocked_plan,
)
from src.run import main
from tests.conftest import write_mini_capture


def _write_still(path: Path, bgr: tuple[int, int, int]) -> None:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[:] = bgr
    assert cv2.imwrite(str(path), img)


def test_identify_three_tiers(tmp_path: Path, mini_capture: Path):
    photos = tmp_path / "photo" / "room_01"
    photos.mkdir(parents=True)
    _write_still(photos / "000000.png", (0, 0, 180))
    video = tmp_path / "video" / "room_01.mp4"
    video.parent.mkdir()
    video.write_bytes((mini_capture / "rgb.mp4").read_bytes())
    assert identify_tier(photos) == "photo"
    assert identify_tier(video) == "video"
    assert identify_tier(mini_capture) == "lidar"


def test_photo_consumes_stills_and_refuses_poses(tmp_path: Path):
    photos = tmp_path / "room_01"
    photos.mkdir()
    _write_still(photos / "000000.png", (0, 0, 180))
    _write_still(photos / "000001.png", (0, 180, 0))
    (photos / "odometry.csv").write_text("not used\n")
    (photos / "depth").mkdir()
    adapted = open_photo(photos)
    assert adapted.tier == "photo"
    assert adapted.n_stills == 2
    assert adapted.metric_reconstruction == "blocked"
    assert all(p.endswith(".png") for p in adapted.consumed)
    assert not any("odometry.csv" in p for p in adapted.consumed)
    assert not any(p.endswith("depth") or "/depth/" in p for p in adapted.consumed)
    assert any(p.endswith("odometry.csv") for p in adapted.refused)
    assert any(p.endswith("depth") for p in adapted.refused)


def test_photo_rejects_depth_folder_as_stills(mini_capture: Path):
    with pytest.raises(AdapterError, match="no top-level stills"):
        open_photo(mini_capture)


def test_video_on_record3d_reads_only_rgb(mini_capture: Path):
    adapted = open_video(mini_capture)
    assert adapted.consumed == [str((mini_capture / "rgb.mp4").resolve())]
    assert adapted.metric_reconstruction == "blocked"
    joined = " ".join(adapted.refused)
    assert "odometry.csv" in joined
    assert "camera_matrix.csv" in joined
    assert "depth" in joined
    assert "imu.csv" not in " ".join(adapted.consumed)


def test_lidar_allows_depth_and_poses(mini_capture: Path):
    adapted = open_lidar(mini_capture)
    joined = " ".join(adapted.consumed)
    assert "depth" in joined
    assert "odometry.csv" in joined
    assert "camera_matrix.csv" in joined
    assert adapted.metric_reconstruction == "available"
    assert all("imu.csv" not in p for p in adapted.consumed)


def test_lidar_rejects_photo_folder(tmp_path: Path):
    photos = tmp_path / "room_01"
    photos.mkdir()
    _write_still(photos / "000000.png", (0, 0, 180))
    with pytest.raises(FileNotFoundError, match="not a Record3D folder"):
        open_lidar(photos)


def test_blocked_plan_has_no_invented_geometry(tmp_path: Path):
    photos = tmp_path / "room_01"
    photos.mkdir()
    _write_still(photos / "a.png", (0, 0, 180))
    adapted = open_photo(photos)
    jpath, spath = write_blocked_plan(adapted, tmp_path / "out")
    payload = json.loads(jpath.read_text())
    assert payload["status"] == "blocked"
    assert payload["input_tier"] == "photo"
    assert payload["measurement_status"] == "not_implemented"
    assert payload["method"] == "rgb_keyframes_non_metric"
    assert payload["confidence_status"] == "low"
    assert payload["height_m"] is None
    assert payload["walls"] == []
    assert payload["polygon_xz_m"] == []
    assert payload["output_quality"] == "blocked"
    assert payload["footprint_method"] == ""
    assert payload["source_files"] == adapted.consumed
    assert "BLOCKED" in spath.read_text()


def test_cli_photo_and_video_tiers(tmp_path: Path, mini_capture: Path):
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_still(photos / "000000.png", (0, 0, 180))
    photo_out = tmp_path / "photo_out"
    rc = main([str(photos), "--tier", "photo", "--out", str(photo_out)])
    assert rc == 0
    data = json.loads((photo_out / "plan.json").read_text())
    assert data["tier"] == "photo"
    assert data["status"] == "blocked"
    assert data["source_files"] == [str((photos / "000000.png").resolve())]
    assert data["visual_preview"] == "keyframes_preview.png"
    assert (photo_out / "keyframes_preview.png").exists()
    assert len(data["keyframes"]) == 1

    video_out = tmp_path / "video_out"
    rc = main(
        [str(mini_capture / "rgb.mp4"), "--tier", "video", "--out", str(video_out)]
    )
    assert rc == 0
    video = json.loads((video_out / "plan.json").read_text())
    assert video["tier"] == "video"
    assert video["status"] == "blocked"
    assert video["source_files"] == [str((mini_capture / "rgb.mp4").resolve())]
    assert video["n_video_frames"] == 2
    assert video["visual_preview"] == "keyframes_preview.png"
    assert (video_out / "keyframes_preview.png").exists()
    assert len(video["keyframes"]) == 2


def test_cli_photo_cannot_use_from_ply(tmp_path: Path):
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_still(photos / "000000.png", (0, 0, 180))
    with pytest.raises(SystemExit):
        main([str(photos), "--tier", "photo", "--from-ply"])


def test_cli_inspect_identifies_photo(tmp_path: Path, capsys):
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_still(photos / "000000.png", (0, 0, 180))
    assert main([str(photos), "--inspect"]) == 0
    out = capsys.readouterr().out
    assert "tier: photo" in out
    assert "metric_reconstruction: blocked" in out
    assert not (tmp_path / "plan.json").exists()
