from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cloud import build_cloud, write_ply
from src.ingest import load_capture
from tests.conftest import SINGLE_ROOM


def test_mini_cloud_is_metric_and_follows_poses(mini_capture: Path, tmp_path: Path):
    cap = load_capture(mini_capture)
    result = build_cloud(
        cap, frame_stride=1, pixel_stride=4, min_confidence=1, z_min=0.1, z_max=8.0
    )
    assert result.n_frames == 2
    assert result.points.shape[0] > 10
    # identity then +1 m in X: two 1 m planes in front of the camera (look −Z)
    zs = result.points[:, 2]
    assert np.median(zs) == pytest.approx(-1.0, abs=0.15)
    xs = result.points[:, 0]
    # mixed of x≈0 (frame 0) and x≈1 (frame 1), plus unprojection spread
    assert xs.min() < 0.5
    assert xs.max() > 0.5
    ply = tmp_path / "mini.ply"
    write_ply(ply, result.points, result.colors)
    assert ply.stat().st_size > 0


@pytest.mark.skipif(not SINGLE_ROOM.exists(), reason="founder dump not on disk")
def test_single_room_cloud_is_room_scale_not_exploded():
    cap = load_capture(SINGLE_ROOM)
    result = build_cloud(
        cap, frame_stride=40, pixel_stride=8, min_confidence=1
    )
    assert result.points.shape[0] > 1000
    span = result.cloud_span_m
    pose_span = result.pose_span_m
    assert pose_span[1] < 0.5
    # room-scale, not millimetres-left-as-metres and not inverted explosion
    assert 2.0 < span[0] < 20.0
    assert 2.0 < span[2] < 20.0
    y = result.points[:, 1]
    y50 = float(np.percentile(y, 50))
    y95 = float(np.percentile(y, 95))
    assert -0.5 < y50 < 1.5
    assert 0.5 < y95 < 4.0

    inverted = build_cloud(
        cap, frame_stride=40, pixel_stride=8, invert_extrinsics=True
    )
    assert float(inverted.cloud_span_m[1]) > float(span[1])
