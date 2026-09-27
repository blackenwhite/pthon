"""Photo, video, and LiDAR input adapters.

A common interface so later reconstruction can swap tiers without duplicating
the pipeline. File boundaries are enforced here: photo may read stills only,
video may read RGB video only, LiDAR may read depth, confidence, intrinsics,
and poses. Photo and video metric reconstruction is blocked, not faked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.export import (
    ACCURACY_STATUS,
    DISCLAIMER,
    SCHEMA,
    write_plan_json,
    write_plan_svg,
)
from src.ingest import REQUIRED, _video_info, load_capture

TIERS = ("photo", "video", "lidar")
STILL_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v"}
PHOTO_FORBIDDEN = (
    "depth",
    "confidence",
    "odometry.csv",
    "imu.csv",
    "camera_matrix.csv",
    "rgb.mp4",
)
VIDEO_FORBIDDEN = (
    "depth",
    "confidence",
    "odometry.csv",
    "imu.csv",
    "camera_matrix.csv",
)
LIDAR_CONSUMED = (
    "camera_matrix.csv",
    "odometry.csv",
    "rgb.mp4",
    "depth",
    "confidence",
)
PHOTO_ALLOWED = ("still_images",)
VIDEO_ALLOWED = ("rgb_video",)
LIDAR_ALLOWED = ("depth", "confidence", "camera_matrix", "odometry", "rgb_video")

PHOTO_NOTE = (
    "Photo-tier metric reconstruction is not implemented. This adapter consumed "
    "only still images. Depth, poses, and intrinsics were not read. Derived "
    "PNGs from rgb.mp4 are not independent native photographs."
)
VIDEO_NOTE = (
    "Video-tier metric reconstruction is not implemented. This adapter consumed "
    "only RGB video. Depth, poses, and intrinsics were not read. A copied "
    "rgb.mp4 is not a separate native recording."
)


class AdapterError(ValueError):
    """Input does not match the requested tier, or the adapter would have to cheat."""


@dataclass
class AdaptedInput:
    tier: str
    path: Path
    allowed: tuple[str, ...]
    consumed: list[str]
    refused: list[str]
    metric_reconstruction: str
    method: str
    measurement_status: str
    notes: list[str] = field(default_factory=list)
    n_stills: int | None = None
    n_video_frames: int | None = None
    video_size: tuple[int, int] | None = None
    video_fps: float | None = None

    def inspect_text(self) -> str:
        consumed_lines = [f"  {p}" for p in self.consumed] or ["  (none)"]
        refused_lines = [f"  {p}" for p in self.refused] or ["  (none)"]
        lines = [
            f"tier: {self.tier}",
            f"path: {self.path}",
            f"allowed: {', '.join(self.allowed)}",
            f"consumed ({len(self.consumed)}):",
            *consumed_lines,
            f"refused / not read ({len(self.refused)}):",
            *refused_lines,
            f"metric_reconstruction: {self.metric_reconstruction}",
            f"measurement_status: {self.measurement_status}",
            f"method: {self.method}",
        ]
        if self.n_stills is not None:
            lines.append(f"n_stills: {self.n_stills}")
        if self.n_video_frames is not None:
            lines.append(
                f"video: size={self.video_size} fps={self.video_fps} "
                f"frames={self.n_video_frames}"
            )
        lines.extend(self.notes)
        return "\n".join(lines)


def _existing(root: Path, names: tuple[str, ...]) -> list[str]:
    found: list[str] = []
    for name in names:
        item = root / name
        if item.exists():
            found.append(str(item.resolve()))
    return found


def _top_stills(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in STILL_SUFFIXES
    )


def _top_videos(folder: Path) -> list[Path]:
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )


def _is_record3d(folder: Path) -> bool:
    return folder.is_dir() and all((folder / name).exists() for name in REQUIRED)


def identify_tier(path: Path) -> str:
    """Classify a path as photo, video, or lidar without reconstructing."""
    path = path.resolve()
    if not path.exists():
        raise AdapterError(f"input not found: {path}")
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in VIDEO_SUFFIXES:
            return "video"
        if suffix in STILL_SUFFIXES:
            return "photo"
        raise AdapterError(f"cannot identify tier for file {path}")
    if _is_record3d(path):
        return "lidar"
    stills = _top_stills(path)
    videos = _top_videos(path)
    if stills and not videos:
        return "photo"
    if len(videos) == 1 and not stills:
        return "video"
    raise AdapterError(
        f"cannot identify tier for {path}: "
        f"{len(stills)} top-level stills, {len(videos)} top-level videos, "
        "not a Record3D folder"
    )


def open_photo(path: Path) -> AdaptedInput:
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() not in STILL_SUFFIXES:
            raise AdapterError(f"photo adapter expected a still image, got {path}")
        stills = [path]
        root = path.parent
    elif path.is_dir():
        stills = _top_stills(path)
        root = path
    else:
        raise AdapterError(f"photo input not found: {path}")
    if not stills:
        raise AdapterError(
            f"photo adapter found no top-level stills in {path}. "
            "PNG files under depth/ or confidence/ are not allowed."
        )
    consumed = [str(p) for p in stills]
    refused = _existing(root, PHOTO_FORBIDDEN)
    notes = [PHOTO_NOTE]
    if refused:
        notes.append(
            "Nearby depth, pose, or intrinsic files were present and were not read."
        )
    return AdaptedInput(
        tier="photo",
        path=path,
        allowed=PHOTO_ALLOWED,
        consumed=consumed,
        refused=refused,
        metric_reconstruction="blocked",
        method="none",
        measurement_status="not_implemented",
        notes=notes,
        n_stills=len(stills),
    )


def open_video(path: Path) -> AdaptedInput:
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() not in VIDEO_SUFFIXES:
            raise AdapterError(f"video adapter expected an RGB video, got {path}")
        video = path
        root = path.parent
    elif path.is_dir():
        named = path / "rgb.mp4"
        videos = _top_videos(path)
        if named.exists():
            video = named
        elif len(videos) == 1:
            video = videos[0]
        else:
            raise AdapterError(
                f"video adapter needs rgb.mp4 or exactly one top-level video in {path}"
            )
        root = path
    else:
        raise AdapterError(f"video input not found: {path}")
    size, fps, n_frames = _video_info(video)
    if size is None:
        raise AdapterError(f"could not open RGB video {video}")
    refused = _existing(root, VIDEO_FORBIDDEN)
    notes = [VIDEO_NOTE]
    if refused:
        notes.append(
            "Nearby depth, pose, or intrinsic files were present and were not read."
        )
    return AdaptedInput(
        tier="video",
        path=path,
        allowed=VIDEO_ALLOWED,
        consumed=[str(video.resolve())],
        refused=refused,
        metric_reconstruction="blocked",
        method="none",
        measurement_status="not_implemented",
        notes=notes,
        n_video_frames=n_frames,
        video_size=size,
        video_fps=fps,
    )


def open_lidar(path: Path) -> AdaptedInput:
    path = path.resolve()
    cap = load_capture(path)
    consumed = [str((cap.root / name).resolve()) for name in LIDAR_CONSUMED]
    imu = cap.root / "imu.csv"
    refused = [str(imu.resolve())] if imu.exists() else []
    return AdaptedInput(
        tier="lidar",
        path=cap.root,
        allowed=LIDAR_ALLOWED,
        consumed=consumed,
        refused=refused,
        metric_reconstruction="available",
        method="ransac_rgb_d_planes",
        measurement_status="estimated",
        notes=[
            "LiDAR adapter may read depth, confidence, camera_matrix, odometry, "
            "and rgb.mp4. imu.csv is required to recognise the folder and is not "
            "used for plane fitting."
        ],
        n_video_frames=cap.rgb_frame_count,
        video_size=cap.rgb_size,
        video_fps=cap.rgb_fps,
    )


def open_input(path: Path, tier: str) -> AdaptedInput:
    if tier == "photo":
        return open_photo(path)
    if tier == "video":
        return open_video(path)
    if tier == "lidar":
        return open_lidar(path)
    raise AdapterError(f"unknown tier {tier!r}; expected one of {TIERS}")


def blocked_plan_payload(adapted: AdaptedInput) -> dict:
    disclaimer = " ".join(adapted.notes) if adapted.notes else DISCLAIMER
    return {
        "schema": SCHEMA,
        "status": "blocked",
        "tier": adapted.tier,
        "input_tier": adapted.tier,
        "source_files": list(adapted.consumed),
        "allowed": list(adapted.allowed),
        "refused": list(adapted.refused),
        "cloud_origin": "none",
        "method": adapted.method,
        "measurement_status": adapted.measurement_status,
        "accuracy_status": ACCURACY_STATUS,
        "capture": str(adapted.path),
        "units": "metres",
        "disclaimer": disclaimer,
        "n_points": 0,
        "n_downsampled": 0,
        "floor": None,
        "ceiling": None,
        "walls": [],
        "height_m": None,
        "height_blocked": True,
        "height_p05_p95_m": None,
        "polygon_xz_m": [],
        "footprint_method": "",
        "output_quality": "blocked",
        "openings": [],
        "notes": list(adapted.notes),
        "n_stills": adapted.n_stills,
        "n_video_frames": adapted.n_video_frames,
    }


def write_blocked_plan(adapted: AdaptedInput, out_dir: Path) -> tuple[Path, Path]:
    payload = blocked_plan_payload(adapted)
    json_path = write_plan_json(payload, out_dir / "plan.json")
    svg_path = write_plan_svg(payload, out_dir / "plan.svg")
    return json_path, svg_path


def adapter_out_dir(path: Path, tier: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.parent if explicit.suffix else explicit
    stem = path.stem if path.is_file() else path.name
    return Path("out") / f"{tier}_{stem}"
