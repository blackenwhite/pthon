"""Slice 6: stills and a shorter video taken from an existing rgb.mp4.

These are adapters over the Record3D color video. They do not measure the room.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.ingest import Capture, Pose

STILLS_SCHEMA = "cozmo.rgb_stills.v1"
VIDEO_SCHEMA = "cozmo.rgb_video.v1"


@dataclass
class StillFrame:
    index: int
    frame: str
    timestamp_s: float
    xyz_m: tuple[float, float, float]
    path: Path


@dataclass
class StillsResult:
    stills: list[StillFrame]
    n_source_frames: int
    fps: float | None
    size: tuple[int, int] | None
    frame_stride: int
    manifest: Path


@dataclass
class VideoResult:
    path: Path
    manifest: Path
    n_frames: int
    fps: float
    size: tuple[int, int]
    frame_stride: int
    n_source_frames: int


def read_rgb_frames(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """BGR frames from rgb.mp4, keyed by 0-based index. Missing video → empty."""
    if not wanted:
        return {}
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        return {}
    found: dict[int, np.ndarray] = {}
    last = max(wanted)
    idx = 0
    try:
        while idx <= last:
            ok, frame = video.read()
            if not ok:
                break
            if idx in wanted:
                found[idx] = frame
            idx += 1
    finally:
        video.release()
    return found


def selected_poses(cap: Capture, frame_stride: int) -> list[Pose]:
    return cap.poses[:: max(1, frame_stride)]


def _xyz(pose: Pose) -> tuple[float, float, float]:
    return (float(pose.t[0]), float(pose.t[1]), float(pose.t[2]))


def export_stills(cap: Capture, out_dir: Path, *, frame_stride: int = 12) -> StillsResult:
    stride = max(1, frame_stride)
    poses = selected_poses(cap, stride)
    wanted = {int(p.frame) for p in poses}
    frames = read_rgb_frames(cap.root / "rgb.mp4", wanted)
    stills_dir = out_dir / "stills"
    stills_dir.mkdir(parents=True, exist_ok=True)
    stills: list[StillFrame] = []
    for pose in poses:
        idx = int(pose.frame)
        bgr = frames.get(idx)
        if bgr is None:
            continue
        path = stills_dir / f"{pose.frame}.png"
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"could not write {path}")
        stills.append(
            StillFrame(
                index=idx,
                frame=pose.frame,
                timestamp_s=float(pose.timestamp),
                xyz_m=_xyz(pose),
                path=path,
            )
        )
    manifest = out_dir / "stills.json"
    payload = {
        "schema": STILLS_SCHEMA,
        "capture": str(cap.root),
        "source": "rgb.mp4",
        "note": "Stills sampled from the existing color video. Not a photo measurement.",
        "fps": cap.rgb_fps,
        "size": list(cap.rgb_size) if cap.rgb_size else None,
        "frame_stride": stride,
        "n_source_frames": cap.rgb_frame_count,
        "n_stills": len(stills),
        "stills": [
            {
                "index": s.index,
                "frame": s.frame,
                "timestamp_s": round(s.timestamp_s, 6),
                "file": f"stills/{s.path.name}",
                "xyz_m": [round(v, 6) for v in s.xyz_m],
            }
            for s in stills
        ],
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return StillsResult(
        stills=stills,
        n_source_frames=cap.rgb_frame_count or 0,
        fps=cap.rgb_fps,
        size=cap.rgb_size,
        frame_stride=stride,
        manifest=manifest,
    )


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for fourcc in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            size,
        )
        if writer.isOpened():
            return writer
    raise RuntimeError(f"could not open a video writer for {path}")


def export_video(cap: Capture, out_dir: Path, *, frame_stride: int = 12) -> VideoResult:
    stride = max(1, frame_stride)
    poses = selected_poses(cap, stride)
    wanted = {int(p.frame) for p in poses}
    frames = read_rgb_frames(cap.root / "rgb.mp4", wanted)
    ordered = [frames[int(p.frame)] for p in poses if int(p.frame) in frames]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "video.mp4"
    manifest = out_dir / "video.json"
    if not ordered:
        payload = {
            "schema": VIDEO_SCHEMA,
            "capture": str(cap.root),
            "source": "rgb.mp4",
            "note": "Shorter clip of the existing color video (same stride as stills). Not a new capture.",
            "file": path.name,
            "fps": None,
            "source_fps": cap.rgb_fps,
            "size": None,
            "frame_stride": stride,
            "n_source_frames": cap.rgb_frame_count,
            "n_frames": 0,
        }
        manifest.write_text(json.dumps(payload, indent=2) + "\n")
        return VideoResult(
            path=path,
            manifest=manifest,
            n_frames=0,
            fps=0.0,
            size=(0, 0),
            frame_stride=stride,
            n_source_frames=cap.rgb_frame_count or 0,
        )
    h, w = ordered[0].shape[:2]
    src_fps = float(cap.rgb_fps) if cap.rgb_fps and cap.rgb_fps > 1e-3 else 30.0
    out_fps = max(src_fps / stride, 1.0)
    writer = _open_writer(path, out_fps, (w, h))
    try:
        for frame in ordered:
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    payload = {
        "schema": VIDEO_SCHEMA,
        "capture": str(cap.root),
        "source": "rgb.mp4",
        "note": "Shorter clip of the existing color video (same stride as stills). Not a new capture.",
        "file": path.name,
        "fps": round(out_fps, 6),
        "source_fps": cap.rgb_fps,
        "size": [w, h],
        "frame_stride": stride,
        "n_source_frames": cap.rgb_frame_count,
        "n_frames": len(ordered),
    }
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    return VideoResult(
        path=path,
        manifest=manifest,
        n_frames=len(ordered),
        fps=out_fps,
        size=(w, h),
        frame_stride=stride,
        n_source_frames=cap.rgb_frame_count or 0,
    )


def stills_text(result: StillsResult) -> str:
    return (
        f"stills: {len(result.stills)} frames from rgb.mp4 "
        f"(stride {result.frame_stride}, source_frames={result.n_source_frames}) "
        f"size={result.size} fps={result.fps}\n"
        f"stills JSON: {result.manifest.resolve()}"
    )


def video_text(result: VideoResult) -> str:
    return (
        f"video: {result.n_frames} frames, {result.fps:.2f} fps, size={result.size} "
        f"(stride {result.frame_stride}, source_frames={result.n_source_frames})\n"
        f"video JSON: {result.manifest.resolve()}"
    )
