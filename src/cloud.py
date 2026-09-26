"""Slice 2: subsample RGB-D + poses into a metric point cloud (PLY)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.ingest import Capture, Pose

# ARKit / Record3D stores millimetres in uint16 depth PNGs.
DEPTH_MM_TO_M = 0.001


@dataclass
class CloudResult:
    points: np.ndarray  # (N, 3) metres, world
    colors: np.ndarray  # (N, 3) uint8 RGB
    n_frames: int
    n_skipped: int
    pose_span_m: np.ndarray
    cloud_span_m: np.ndarray


def _quat_to_R(q: np.ndarray) -> np.ndarray:
    """Hamilton quaternion (qx, qy, qz, qw) → 3×3 rotation."""
    qx, qy, qz, qw = q
    n = float(np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw))
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def camera_to_world(pose: Pose, invert: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Record3D poses are T_wc (camera→world). invert=True treats them as T_cw."""
    R = _quat_to_R(pose.q)
    t = pose.t.astype(np.float64)
    if invert:
        R = R.T
        t = -R @ t
    return R, t


def scale_intrinsics(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rgb_wh: tuple[int, int],
    depth_wh: tuple[int, int],
) -> tuple[float, float, float, float]:
    """K is authored for RGB; depth maps are much smaller."""
    rw, rh = rgb_wh
    dw, dh = depth_wh
    sx = dw / rw
    sy = dh / rh
    return fx * sx, fy * sy, cx * sx, cy * sy


def _read_png(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img


def _depth_path(root: Path, frame: str) -> Path:
    p = root / "depth" / f"{frame}.png"
    if p.exists():
        return p
    return root / "depth" / f"{int(frame):06d}.png"


def _conf_path(root: Path, frame: str) -> Path:
    p = root / "confidence" / f"{frame}.png"
    if p.exists():
        return p
    return root / "confidence" / f"{int(frame):06d}.png"


def unproject_frame(
    depth_mm: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R: np.ndarray,
    t: np.ndarray,
    *,
    confidence: np.ndarray | None = None,
    rgb_bgr: np.ndarray | None = None,
    pixel_stride: int = 4,
    min_confidence: int = 1,
    z_min: float = 0.2,
    z_max: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth image → world points. Camera is OpenGL/ARKit: +Y up, looks along −Z."""
    h, w = depth_mm.shape[:2]
    depth_m = depth_mm.astype(np.float32) * DEPTH_MM_TO_M
    ys = np.arange(0, h, pixel_stride)
    xs = np.arange(0, w, pixel_stride)
    vv, uu = np.meshgrid(ys, xs, indexing="ij")
    z = depth_m[vv, uu]
    mask = (z > z_min) & (z < z_max)
    if confidence is not None:
        mask &= confidence[vv, uu] >= min_confidence
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    u = uu[mask].astype(np.float64)
    v = vv[mask].astype(np.float64)
    z = z[mask].astype(np.float64)
    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    z_cam = -z
    cam = np.stack([x, y, z_cam], axis=1)
    world = (cam @ R.T) + t

    if rgb_bgr is None:
        colors = np.full((world.shape[0], 3), 200, dtype=np.uint8)
    else:
        rgb_small = cv2.resize(rgb_bgr, (w, h), interpolation=cv2.INTER_AREA)
        bgr = rgb_small[vv, uu][mask]
        colors = bgr[:, ::-1].copy()
    return world.astype(np.float32), colors.astype(np.uint8)


def build_cloud(
    cap: Capture,
    *,
    frame_stride: int = 12,
    pixel_stride: int = 4,
    min_confidence: int = 1,
    invert_extrinsics: bool = False,
    z_min: float = 0.2,
    z_max: float = 8.0,
    poses: list[Pose] | None = None,
) -> CloudResult:
    if cap.rgb_size is None:
        raise ValueError("RGB size unknown; cannot scale K to depth")
    rgb_wh = cap.rgb_size
    src_poses = cap.poses if poses is None else poses
    poses = src_poses[:: max(1, frame_stride)]
    wanted = {int(p.frame) for p in poses}

    video = cv2.VideoCapture(str(cap.root / "rgb.mp4"))
    rgb_by_idx: dict[int, np.ndarray] = {}
    if video.isOpened():
        idx = 0
        while True:
            ok, frame = video.read()
            if not ok:
                break
            if idx in wanted:
                rgb_by_idx[idx] = frame
            idx += 1
        video.release()

    chunks_xyz: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []
    skipped = 0
    depth_wh: tuple[int, int] | None = None

    for pose in poses:
        dpath = _depth_path(cap.root, pose.frame)
        depth = _read_png(dpath)
        if depth is None:
            skipped += 1
            continue
        if depth.ndim != 2:
            skipped += 1
            continue
        if depth_wh is None:
            depth_wh = (int(depth.shape[1]), int(depth.shape[0]))
        cpath = _conf_path(cap.root, pose.frame)
        conf = _read_png(cpath) if cpath.exists() else None
        fx, fy, cx, cy = scale_intrinsics(
            pose.fx, pose.fy, pose.cx, pose.cy, rgb_wh, depth_wh
        )
        R, t = camera_to_world(pose, invert=invert_extrinsics)
        frame_i = int(pose.frame)
        pts, cols = unproject_frame(
            depth,
            fx,
            fy,
            cx,
            cy,
            R,
            t,
            confidence=conf,
            rgb_bgr=rgb_by_idx.get(frame_i),
            pixel_stride=pixel_stride,
            min_confidence=min_confidence,
            z_min=z_min,
            z_max=z_max,
        )
        if pts.shape[0] == 0:
            skipped += 1
            continue
        chunks_xyz.append(pts)
        chunks_rgb.append(cols)

    if not chunks_xyz:
        raise RuntimeError("no points reconstructed; check depth paths and masks")

    points = np.concatenate(chunks_xyz, axis=0)
    colors = np.concatenate(chunks_rgb, axis=0)
    xyz_poses = np.stack([p.t for p in cap.poses])
    return CloudResult(
        points=points,
        colors=colors,
        n_frames=len(chunks_xyz),
        n_skipped=skipped,
        pose_span_m=xyz_poses.max(axis=0) - xyz_poses.min(axis=0),
        cloud_span_m=points.max(axis=0) - points.min(axis=0),
    )


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(points.shape[0])
    rec = np.empty(
        n,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    rec["x"] = points[:, 0]
    rec["y"] = points[:, 1]
    rec["z"] = points[:, 2]
    rec["red"] = colors[:, 0]
    rec["green"] = colors[:, 1]
    rec["blue"] = colors[:, 2]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        rec.tofile(f)


def cloud_text(result: CloudResult, out: Path) -> str:
    pmin = result.points.min(axis=0)
    pmax = result.points.max(axis=0)
    return "\n".join(
        [
            f"wrote: {out.resolve()}",
            f"frames_used: {result.n_frames}  frames_skipped: {result.n_skipped}",
            f"points: {result.points.shape[0]}",
            f"cloud_min_m: {pmin}",
            f"cloud_max_m: {pmax}",
            f"cloud_span_m: {result.cloud_span_m}",
            f"pose_span_m: {result.pose_span_m}",
            "convention: T_wc, ARKit cam (+Y up, look −Z); invert with --invert-extrinsics if inside-out",
        ]
    )


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the binary PLY written by write_ply."""
    raw = path.read_bytes()
    marker = b"end_header\n"
    i = raw.find(marker)
    if i < 0:
        raise ValueError(f"not a PLY we wrote: {path}")
    payload = raw[i + len(marker) :]
    rec = np.frombuffer(
        payload,
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    points = np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
    colors = np.stack([rec["red"], rec["green"], rec["blue"]], axis=1)
    return points, colors


def _axis_limits(vals: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    a, b = np.percentile(vals, [lo, hi])
    if b - a < 1e-3:
        a, b = float(vals.min()), float(vals.max())
        if b - a < 1e-3:
            b = a + 1.0
    return float(a), float(b)


def _raster(
    u: np.ndarray,
    v: np.ndarray,
    paint: np.ndarray,
    *,
    width: int = 1280,
    bg: tuple[int, int, int] = (36, 36, 36),
) -> np.ndarray:
    """Orthographic scatter. u,v in metres; paint is (N,3) RGB. v increases upward."""
    u0, u1 = _axis_limits(u)
    v0, v1 = _axis_limits(v)
    span_u = u1 - u0
    span_v = v1 - v0
    aspect = span_v / span_u
    w = width
    h = max(64, int(round(width * aspect)))
    h = min(h, width * 2)
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    col = ((u - u0) / span_u * (w - 1)).astype(np.int32)
    row = ((v1 - v) / span_v * (h - 1)).astype(np.int32)
    m = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    img[row[m], col[m]] = paint[m]
    return img


def _height_colors(y: np.ndarray) -> np.ndarray:
    y0, y1 = _axis_limits(y)
    t = np.clip((y - y0) / (y1 - y0), 0.0, 1.0)
    # floor blue → mid green → ceiling red
    r = np.clip(2.0 * t - 0.5, 0, 1)
    g = np.clip(1.0 - np.abs(2.0 * t - 1.0), 0, 1)
    b = np.clip(1.5 - 2.0 * t, 0, 1)
    return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)


def write_preview_pngs(points: np.ndarray, colors: np.ndarray, out_dir: Path) -> list[Path]:
    """2D views you can open in Preview. macOS Preview cannot show a points-only PLY."""
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    views = [
        ("preview_top_rgb.png", x, z, colors, "top-down XZ, texture colors"),
        ("preview_top_height.png", x, z, _height_colors(y), "top-down XZ, color = height Y"),
        ("preview_side_rgb.png", x, y, colors, "side XY (Y is up)"),
    ]
    paths: list[Path] = []
    for name, u, v, paint, _label in views:
        path = out_dir / name
        bgr = cv2.cvtColor(_raster(u, v, paint), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)
        paths.append(path)
    return paths
