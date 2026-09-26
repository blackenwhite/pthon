"""Parse a Record3D-style capture folder (no reconstruction)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REQUIRED = (
    "camera_matrix.csv",
    "odometry.csv",
    "imu.csv",
    "rgb.mp4",
    "depth",
    "confidence",
)


@dataclass
class Pose:
    timestamp: float
    frame: str
    t: np.ndarray  # (3,) meters
    q: np.ndarray  # (4,) qx,qy,qz,qw
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Capture:
    root: Path
    K: np.ndarray
    poses: list[Pose]
    n_depth: int
    n_confidence: int
    rgb_size: tuple[int, int] | None
    rgb_fps: float | None
    rgb_frame_count: int | None


def _read_K(path: Path) -> np.ndarray:
    rows = []
    with path.open(newline="") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(",")])
    K = np.array(rows, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera_matrix.csv expected 3x3, got {K.shape} in {path}")
    return K


def _read_odometry(path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty odometry: {path}")
        for raw in reader:
            row = {k.strip(): (v.strip() if v is not None else "") for k, v in raw.items()}
            poses.append(
                Pose(
                    timestamp=float(row["timestamp"]),
                    frame=row["frame"].zfill(6),
                    t=np.array(
                        [float(row["x"]), float(row["y"]), float(row["z"])],
                        dtype=np.float64,
                    ),
                    q=np.array(
                        [
                            float(row["qx"]),
                            float(row["qy"]),
                            float(row["qz"]),
                            float(row["qw"]),
                        ],
                        dtype=np.float64,
                    ),
                    fx=float(row["fx"]),
                    fy=float(row["fy"]),
                    cx=float(row["cx"]),
                    cy=float(row["cy"]),
                )
            )
    if not poses:
        raise ValueError(f"no pose rows in {path}")
    return poses


def _count_pngs(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.suffix.lower() == ".png")


def _video_info(path: Path) -> tuple[tuple[int, int] | None, float | None, int | None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, None, None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return (w, h), fps, n


def load_capture(root: Path) -> Capture:
    root = root.resolve()
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root} is not a Record3D folder, missing: {missing}")
    rgb_size, rgb_fps, rgb_frame_count = _video_info(root / "rgb.mp4")
    return Capture(
        root=root,
        K=_read_K(root / "camera_matrix.csv"),
        poses=_read_odometry(root / "odometry.csv"),
        n_depth=_count_pngs(root / "depth"),
        n_confidence=_count_pngs(root / "confidence"),
        rgb_size=rgb_size,
        rgb_fps=rgb_fps,
        rgb_frame_count=rgb_frame_count,
    )


def depth_sample_stats(root: Path, poses: list[Pose], n_samples: int = 5) -> dict:
    """Read a few 16-bit depth frames; Record3D typically stores millimetres."""
    depth_dir = root / "depth"
    indices = np.linspace(0, len(poses) - 1, num=min(n_samples, len(poses)), dtype=int)
    stats = []
    for i in indices:
        name = f"{poses[i].frame}.png"
        path = depth_dir / name
        if not path.exists():
            path = depth_dir / f"{int(poses[i].frame):06d}.png"
        if not path.exists():
            continue
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        valid = img[img > 0]
        stats.append(
            {
                "frame": path.name,
                "shape": list(img.shape),
                "dtype": str(img.dtype),
                "min": int(img.min()),
                "max": int(img.max()),
                "p50_valid": float(np.median(valid)) if valid.size else None,
                "valid_frac": float(valid.size / img.size),
            }
        )
    return {"n_read": len(stats), "frames": stats}


def inspect_text(cap: Capture) -> str:
    ts = np.array([p.timestamp for p in cap.poses])
    xyz = np.stack([p.t for p in cap.poses])
    duration = float(ts[-1] - ts[0])
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    path_len = float(step.sum()) if step.size else 0.0
    bbox = xyz.max(axis=0) - xyz.min(axis=0)
    depth_stats = depth_sample_stats(cap.root, cap.poses)
    lines = [
        f"root: {cap.root}",
        f"K:\n{cap.K}",
        f"poses: {len(cap.poses)}  duration_s: {duration:.2f}  path_m: {path_len:.2f}",
        f"xyz_min: {xyz.min(axis=0)}",
        f"xyz_max: {xyz.max(axis=0)}",
        f"xyz_span_m: {bbox}",
        f"depth pngs: {cap.n_depth}  confidence pngs: {cap.n_confidence}",
        f"rgb: size={cap.rgb_size} fps={cap.rgb_fps} frames={cap.rgb_frame_count}",
        f"per-frame fx (first/last): {cap.poses[0].fx:.2f} / {cap.poses[-1].fx:.2f}",
        f"depth samples: {depth_stats}",
    ]
    return "\n".join(lines)
