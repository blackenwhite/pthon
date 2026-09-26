from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.ingest import Pose

ROOT = Path(__file__).resolve().parents[1]
SINGLE_ROOM = ROOT / "store" / "c00a170fe1"


def make_pose(
    *,
    frame: str = "000000",
    t: tuple[float, float, float] = (0.0, 0.0, 0.0),
    q: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    fx: float = 100.0,
    fy: float = 100.0,
    cx: float = 16.0,
    cy: float = 12.0,
) -> Pose:
    return Pose(
        timestamp=0.0,
        frame=frame,
        t=np.array(t, dtype=np.float64),
        q=np.array(q, dtype=np.float64),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )


def write_mini_capture(root: Path, n_frames: int = 2) -> Path:
    """Tiny Record3D-shaped folder for ingest + cloud tests (no founder dump)."""
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    k = "100.0, 0.0, 32.0\n0.0, 100.0, 24.0\n0.0, 0.0, 1.0\n"
    (root / "camera_matrix.csv").write_text(k)
    (root / "imu.csv").write_text("timestamp,ax,ay,az\n0,0,0,0\n")
    odo_lines = [
        "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"
    ]
    rgb_w, rgb_h = 64, 48
    depth_w, depth_h = 32, 24
    writer = cv2.VideoWriter(
        str(root / "rgb.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (rgb_w, rgb_h),
    )
    if not writer.isOpened():
        raise RuntimeError("could not write mini rgb.mp4")
    for i in range(n_frames):
        frame = f"{i:06d}"
        depth = np.full((depth_h, depth_w), 1000, dtype=np.uint16)  # 1.0 m
        conf = np.full((depth_h, depth_w), 2, dtype=np.uint8)
        cv2.imwrite(str(root / "depth" / f"{frame}.png"), depth)
        cv2.imwrite(str(root / "confidence" / f"{frame}.png"), conf)
        bgr = np.zeros((rgb_h, rgb_w, 3), dtype=np.uint8)
        bgr[:] = (0, 0, 180) if i == 0 else (0, 180, 0)
        writer.write(bgr)
        odo_lines.append(
            f"{float(i)}, {frame}, {float(i)}, 0.0, 0.0, 0, 0, 0, 1, 100, 100, 32, 24, , "
        )
    writer.release()
    (root / "odometry.csv").write_text("\n".join(odo_lines) + "\n")
    return root


@pytest.fixture
def mini_capture(tmp_path: Path) -> Path:
    return write_mini_capture(tmp_path / "cap")
