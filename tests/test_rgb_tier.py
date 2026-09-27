from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.adapters import open_photo, open_video, write_blocked_plan
from src.benchmark import photo_frame_indices
from src.rgb_tier import (
    MAX_KEYFRAMES,
    RGB_METHOD,
    even_indices,
    select_photo_keyframes,
    select_video_keyframes,
    write_contact_sheet,
    write_rgb_tier_outputs,
)
from src.run import main


def _write_still(path: Path, bgr: tuple[int, int, int]) -> None:
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:] = bgr
    assert cv2.imwrite(str(path), img)


def test_even_indices_matches_benchmark_helper():
    assert even_indices(5, 12) == photo_frame_indices(5, 12)
    assert even_indices(20, 12) == photo_frame_indices(20, 12)
    assert len(even_indices(100, MAX_KEYFRAMES)) == MAX_KEYFRAMES


def test_select_photo_keyframes_even_spacing(tmp_path: Path):
    stills = []
    for i in range(5):
        path = tmp_path / f"{i:06d}.png"
        _write_still(path, (i * 40, 0, 180))
        stills.append(path)
    keyframes = select_photo_keyframes(stills, max_n=12)
    assert [kf.index for kf in keyframes] == [0, 1, 2, 3, 4]
    assert len(keyframes) == 5


def test_write_contact_sheet(tmp_path: Path):
    frames = [np.full((20, 30, 3), i * 40, dtype=np.uint8) for i in range(4)]
    out = write_contact_sheet(frames, tmp_path / "sheet.png")
    assert out.exists()
    img = cv2.imread(str(out))
    assert img is not None
    assert img.shape[0] > 20 and img.shape[1] > 30


def test_photo_rgb_tier_writes_keyframes_and_preview(tmp_path: Path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(5):
        _write_still(photos / f"{i:06d}.png", (0, i * 40, 180))
    (photos / "odometry.csv").write_text("not used\n")
    adapted = open_photo(photos)
    out = tmp_path / "out"
    paths = write_rgb_tier_outputs(adapted, out)
    assert paths["preview"].exists()
    assert paths["plan_json"].exists()
    kf_pngs = sorted((paths["keyframes_dir"]).glob("*.png"))
    assert len(kf_pngs) == 5
    payload = json.loads(paths["plan_json"].read_text())
    assert payload["status"] == "blocked"
    assert payload["measurement_status"] == "not_implemented"
    assert payload["method"] == RGB_METHOD
    assert payload["confidence_status"] == "low"
    assert payload["uncertainty"]
    assert payload["visual_preview"] == "keyframes_preview.png"
    assert len(payload["keyframes"]) == 5
    assert payload["walls"] == []
    assert payload["polygon_xz_m"] == []
    assert payload["height_m"] is None
    assert not any("odometry.csv" in p for p in payload["source_files"])
    assert any(p.endswith("odometry.csv") for p in payload["refused"])


def test_video_rgb_tier_decodes_without_poses(tmp_path: Path, mini_capture: Path):
    adapted = open_video(mini_capture)
    assert adapted.n_video_frames == 2
    out = tmp_path / "video_out"
    paths = write_rgb_tier_outputs(adapted, out)
    payload = json.loads(paths["plan_json"].read_text())
    assert payload["tier"] == "video"
    assert payload["status"] == "blocked"
    assert payload["confidence_status"] == "low"
    assert len(payload["keyframes"]) == 2
    assert all(k["source"].startswith("video_frame:") for k in payload["keyframes"])
    joined_consumed = " ".join(payload["source_files"])
    assert "rgb.mp4" in joined_consumed
    assert "odometry.csv" not in joined_consumed
    assert "camera_matrix.csv" not in joined_consumed
    assert paths["preview"].exists()


def test_video_stride_thins_then_caps(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    # Fake path only used as source label; indices do not open the file here.
    keyframes = select_video_keyframes(video, n_frames=100, max_n=12, frame_stride=5)
    assert keyframes[0].index == 0
    assert all(kf.index % 5 == 0 for kf in keyframes)
    assert len(keyframes) <= 12


def test_cli_photo_writes_preview(tmp_path: Path):
    photos = tmp_path / "photos"
    photos.mkdir()
    for i in range(3):
        _write_still(photos / f"{i:06d}.png", (0, 0, 180))
    out = tmp_path / "photo_out"
    rc = main([str(photos), "--tier", "photo", "--out", str(out)])
    assert rc == 0
    data = json.loads((out / "plan.json").read_text())
    assert data["visual_preview"] == "keyframes_preview.png"
    assert (out / "keyframes_preview.png").exists()
    assert len(data["keyframes"]) == 3
    assert data["method"] == RGB_METHOD


def test_cli_video_explicit_stride(tmp_path: Path, mini_capture: Path):
    out = tmp_path / "video_out"
    rc = main(
        [
            str(mini_capture / "rgb.mp4"),
            "--tier",
            "video",
            "--out",
            str(out),
            "--frame-stride",
            "1",
        ]
    )
    assert rc == 0
    data = json.loads((out / "plan.json").read_text())
    assert data["status"] == "blocked"
    assert len(data["keyframes"]) == 2


def test_write_blocked_plan_still_empty_geometry(tmp_path: Path):
    photos = tmp_path / "room_01"
    photos.mkdir()
    _write_still(photos / "a.png", (0, 0, 180))
    adapted = open_photo(photos)
    jpath, spath = write_blocked_plan(adapted, tmp_path / "out")
    payload = json.loads(jpath.read_text())
    assert payload["method"] == RGB_METHOD
    assert payload["confidence_status"] == "low"
    assert payload["keyframes"] == []
    assert payload["walls"] == []
    assert "BLOCKED" in spath.read_text()
