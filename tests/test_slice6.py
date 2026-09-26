from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.ingest import load_capture
from src.media import export_stills, export_video, read_rgb_frames
from src.run import main
from tests.conftest import write_mini_capture


def _dominant_channel(bgr: np.ndarray) -> int:
    return int(np.argmax(bgr[0, 0]))


def test_read_rgb_frames_matches_written_colors(mini_capture: Path):
    frames = read_rgb_frames(mini_capture / "rgb.mp4", {0, 1})
    assert set(frames) == {0, 1}
    # mp4 is lossy; frame 0 is red (channel 2), frame 1 is green (channel 1)
    assert _dominant_channel(frames[0]) == 2
    assert _dominant_channel(frames[1]) == 1


def test_export_stills_stride_skips_frames(tmp_path: Path):
    root = write_mini_capture(tmp_path / "cap", n_frames=4)
    cap = load_capture(root)
    result = export_stills(cap, tmp_path / "out", frame_stride=2)
    names = [s.path.name for s in result.stills]
    assert names == ["000000.png", "000002.png"]
    img = cv2.imread(str(result.stills[0].path), cv2.IMREAD_COLOR)
    assert _dominant_channel(img) == 2
    payload = json.loads(result.manifest.read_text())
    assert payload["schema"] == "cozmo.rgb_stills.v1"
    assert payload["n_stills"] == 2
    assert payload["stills"][1]["file"] == "stills/000002.png"
    assert payload["stills"][1]["xyz_m"][0] == 2.0


def test_export_video_roundtrip_frame_count(mini_capture: Path, tmp_path: Path):
    cap = load_capture(mini_capture)
    result = export_video(cap, tmp_path, frame_stride=1)
    assert result.n_frames == 2
    assert result.size == (64, 48)
    clip = cv2.VideoCapture(str(result.path))
    assert clip.isOpened()
    ok0, f0 = clip.read()
    ok1, f1 = clip.read()
    ok2, _ = clip.read()
    clip.release()
    assert ok0 and ok1 and not ok2
    assert _dominant_channel(f0) == 2
    assert _dominant_channel(f1) == 1
    meta = json.loads(result.manifest.read_text())
    assert meta["n_frames"] == 2
    assert meta["frame_stride"] == 1


def test_cli_stills_and_video(mini_capture: Path, tmp_path: Path, capsys):
    ply = tmp_path / "cloud.ply"
    rc = main(
        [
            str(mini_capture),
            "--stills",
            "--video",
            "--out",
            str(ply),
            "--frame-stride",
            "1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "stills: 2 frames" in out
    assert "video: 2 frames" in out
    assert (tmp_path / "stills" / "000000.png").exists()
    assert (tmp_path / "stills.json").exists()
    assert (tmp_path / "video.mp4").stat().st_size > 0
    stills = json.loads((tmp_path / "stills.json").read_text())
    assert stills["stills"][0]["frame"] == "000000"
    assert np.allclose(stills["stills"][1]["xyz_m"][0], 1.0)
