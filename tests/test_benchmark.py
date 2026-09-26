from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.benchmark import (
    SCHEMA,
    discover_rooms,
    main,
    photo_frame_indices,
    prepare_benchmark,
)
from src.ingest import load_capture
from tests.conftest import write_mini_capture


def _dominant_channel(bgr: np.ndarray) -> int:
    return int(np.argmax(bgr[0, 0]))


def test_photo_frame_indices_even_and_reproducible():
    assert photo_frame_indices(5, 3) == [0, 2, 4]
    assert photo_frame_indices(5, 3) == photo_frame_indices(5, 3)
    assert photo_frame_indices(2, 12) == [0, 1]
    assert photo_frame_indices(0, 12) == []
    dropped_last = photo_frame_indices(1714, 12)
    assert len(dropped_last) == len(set(dropped_last)) == 12
    assert dropped_last[-1] == 1713


def test_prepare_benchmark_layout_and_derived_photos(tmp_path: Path):
    store = tmp_path / "store"
    write_mini_capture(store / "c00a170fe1", n_frames=4)
    out = tmp_path / "benchmark"
    payload = prepare_benchmark(store, out, max_stills=2)

    assert payload["schema"] == SCHEMA
    assert payload["rooms"][0]["id"] == "room_01"
    assert payload["rooms"][0]["store_capture"] == "c00a170fe1"
    assert payload["rooms"][0]["photo"]["independent_native_photographs"] is False
    assert payload["rooms"][0]["photo"]["screen_captured"] is False
    assert payload["rooms"][0]["photo"]["source"] == "rgb.mp4"
    assert payload["rooms"][0]["video"]["source"] == "rgb.mp4"
    assert payload["rooms"][0]["lidar"]["files"] == [
        "camera_matrix.csv",
        "odometry.csv",
        "imu.csv",
        "rgb.mp4",
        "depth",
        "confidence",
    ]

    photo_dir = out / "photo" / "room_01"
    pngs = sorted(photo_dir.glob("*.png"))
    assert [p.name for p in pngs] == ["000000.png", "000003.png"]
    first = cv2.imread(str(pngs[0]), cv2.IMREAD_COLOR)
    last = cv2.imread(str(pngs[1]), cv2.IMREAD_COLOR)
    assert _dominant_channel(first) == 2
    assert _dominant_channel(last) == 1

    video = out / "video" / "room_01.mp4"
    assert video.stat().st_size > 0
    src_rgb = (store / "c00a170fe1" / "rgb.mp4").read_bytes()
    assert video.read_bytes() == src_rgb

    lidar = out / "lidar" / "room_01"
    cap = load_capture(lidar)
    assert len(cap.poses) == 4
    assert (lidar / "depth" / "000000.png").exists()
    assert (lidar / "confidence" / "000000.png").exists()

    manifest = json.loads((out / "manifest.json").read_text())
    assert "derived RGB" in manifest["note"]
    assert manifest["rooms"][0]["photo"]["n_stills"] == 2


def test_discover_rooms_maps_known_ids(tmp_path: Path):
    store = tmp_path / "store"
    write_mini_capture(store / "c7d28f72c6", n_frames=1)
    write_mini_capture(store / "c00a170fe1", n_frames=1)
    rooms = discover_rooms(store)
    assert [r.room_id for r in rooms] == ["room_01", "room_02"]
    assert [r.capture_id for r in rooms] == ["c00a170fe1", "c7d28f72c6"]


def test_cli_writes_manifest(tmp_path: Path, capsys):
    store = tmp_path / "store"
    write_mini_capture(store / "c00a170fe1", n_frames=2)
    out = tmp_path / "bench"
    assert main(["--store", str(store), "--out", str(out), "--max-stills", "1"]) == 0
    text = capsys.readouterr().out
    assert "room_01 <- c00a170fe1" in text
    assert "not independent native photographs" in text.lower() or "derived RGB" in text
    assert (out / "manifest.json").exists()
    assert (out / "photo" / "room_01" / "000000.png").exists()
