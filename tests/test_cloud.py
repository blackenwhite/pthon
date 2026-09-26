from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.cloud import (
    camera_to_world,
    read_ply,
    scale_intrinsics,
    unproject_frame,
    write_ply,
    write_preview_pngs,
)
from tests.conftest import make_pose


def test_scale_intrinsics_rgb_to_depth():
    fx, fy, cx, cy = scale_intrinsics(
        1600.0, 1600.0, 960.0, 720.0, (1920, 1440), (256, 192)
    )
    assert fx == pytest.approx(1600.0 * 256 / 1920)
    assert fy == pytest.approx(1600.0 * 192 / 1440)
    assert cx == pytest.approx(960.0 * 256 / 1920)
    assert cy == pytest.approx(720.0 * 192 / 1440)


def test_optical_center_is_along_negative_z():
    depth = np.zeros((3, 3), dtype=np.uint16)
    depth[1, 1] = 2000  # 2.0 m
    pts, _ = unproject_frame(
        depth, fx=2.0, fy=2.0, cx=1.0, cy=1.0,
        R=np.eye(3), t=np.zeros(3), pixel_stride=1, z_min=0.1, z_max=8.0,
    )
    assert pts.shape[0] == 1
    np.testing.assert_allclose(pts[0], [0.0, 0.0, -2.0], atol=1e-5)


def test_image_right_is_plus_x_and_down_is_minus_y():
    depth = np.zeros((3, 3), dtype=np.uint16)
    depth[1, 2] = 2000  # u+1 at center row
    depth[2, 1] = 2000  # v+1 at center col
    pts, _ = unproject_frame(
        depth, fx=2.0, fy=2.0, cx=1.0, cy=1.0,
        R=np.eye(3), t=np.zeros(3), pixel_stride=1, z_min=0.1,
    )
    assert pts.shape[0] == 2
    right = pts[np.argmax(pts[:, 0])]
    down = pts[np.argmin(pts[:, 1])]
    np.testing.assert_allclose(right, [1.0, 0.0, -2.0], atol=1e-5)
    np.testing.assert_allclose(down, [0.0, -1.0, -2.0], atol=1e-5)


def test_depth_millimetres_become_metres():
    depth = np.array([[1000]], dtype=np.uint16)
    pts, _ = unproject_frame(
        depth, fx=1.0, fy=1.0, cx=0.0, cy=0.0,
        R=np.eye(3), t=np.zeros(3), pixel_stride=1, z_min=0.1,
    )
    assert pts[0, 2] == pytest.approx(-1.0)


def test_world_translation_applied():
    depth = np.array([[1000]], dtype=np.uint16)
    t = np.array([1.0, 2.0, 3.0])
    pts, _ = unproject_frame(
        depth, fx=1.0, fy=1.0, cx=0.0, cy=0.0,
        R=np.eye(3), t=t, pixel_stride=1, z_min=0.1,
    )
    np.testing.assert_allclose(pts[0], [1.0, 2.0, 2.0], atol=1e-5)


def test_confidence_mask_drops_low_pixels():
    depth = np.full((2, 2), 1000, dtype=np.uint16)
    conf = np.array([[0, 2], [1, 2]], dtype=np.uint8)
    pts, _ = unproject_frame(
        depth, fx=10.0, fy=10.0, cx=0.5, cy=0.5,
        R=np.eye(3), t=np.zeros(3),
        confidence=conf, min_confidence=2, pixel_stride=1, z_min=0.1,
    )
    assert pts.shape[0] == 2


def test_range_filter_drops_near_and_far():
    depth = np.array([[50, 1000, 20000]], dtype=np.uint16)
    pts, _ = unproject_frame(
        depth, fx=10.0, fy=10.0, cx=1.0, cy=0.0,
        R=np.eye(3), t=np.zeros(3), pixel_stride=1, z_min=0.2, z_max=8.0,
    )
    assert pts.shape[0] == 1
    assert pts[0, 2] == pytest.approx(-1.0)


def test_invert_extrinsics_is_inverse_of_twc():
    pose = make_pose(t=(1.0, 0.0, 0.0), q=(0.0, 0.0, 0.0, 1.0))
    R, t = camera_to_world(pose, invert=False)
    Ri, ti = camera_to_world(pose, invert=True)
    p_cam = np.array([0.0, 0.0, -1.0])
    p_world = R @ p_cam + t
    p_back = Ri @ p_world + ti
    np.testing.assert_allclose(p_back, p_cam, atol=1e-6)


def test_ply_roundtrip(tmp_path: Path):
    points = np.array([[0.1, 0.2, -1.0], [3.0, -4.0, 5.0]], dtype=np.float32)
    colors = np.array([[10, 20, 30], [255, 0, 128]], dtype=np.uint8)
    path = tmp_path / "c.ply"
    write_ply(path, points, colors)
    got_p, got_c = read_ply(path)
    np.testing.assert_allclose(got_p, points, atol=1e-5)
    np.testing.assert_array_equal(got_c, colors)


def test_preview_pngs_are_not_empty(tmp_path: Path):
    rng = np.random.default_rng(0)
    points = rng.normal(size=(500, 3))
    points[:, 1] = np.linspace(0, 2, 500)
    colors = np.full((500, 3), 200, dtype=np.uint8)
    paths = write_preview_pngs(points, colors, tmp_path)
    assert len(paths) == 3
    for p in paths:
        assert p.stat().st_size > 1000
