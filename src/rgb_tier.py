"""Non-metric RGB photo/video outputs: keyframes and a contact-sheet preview.

Metric reconstruction stays blocked. Photo reads stills only. Video decodes
RGB frames only — no odometry, intrinsics, or depth.
"""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.adapters import (
    AdaptedInput,
    STILL_SUFFIXES,
    blocked_plan_payload,
    _top_stills,
)
from src.benchmark import photo_frame_indices
from src.export import write_plan_json, write_plan_svg
from src.media import read_rgb_frames

MAX_KEYFRAMES = 12
RGB_METHOD = "rgb_keyframes_non_metric"
CONFIDENCE_STATUS = "low"
UNCERTAINTY = (
    "No metric scale, poses, or depth were used. Geometry fields stay empty. "
    "Keyframes and the contact sheet are visual evidence only. Derived RGB "
    "from Record3D rgb.mp4 is not an independent native capture."
)
PREVIEW_NAME = "keyframes_preview.png"
KEYFRAMES_DIR = "keyframes"
THUMB_MAX = 320
TILE_PAD = 4


@dataclass
class Keyframe:
    index: int
    source: str
    path: Path | None = None


def even_indices(n: int, max_n: int = MAX_KEYFRAMES) -> list[int]:
    """Evenly spaced indices in ``[0, n)``, at most ``max_n``."""
    return photo_frame_indices(n, max_n)


def select_photo_keyframes(
    stills: list[Path],
    max_n: int = MAX_KEYFRAMES,
) -> list[Keyframe]:
    stills = sorted(stills)
    indices = even_indices(len(stills), max_n)
    return [
        Keyframe(index=i, source=str(stills[i].resolve()), path=stills[i])
        for i in indices
    ]


def select_video_keyframes(
    video: Path,
    n_frames: int,
    max_n: int = MAX_KEYFRAMES,
    frame_stride: int | None = None,
) -> list[Keyframe]:
    """Pick frame indices. Default: even spacing. Optional stride thins first."""
    video = video.resolve()
    if n_frames <= 0:
        return []
    if frame_stride is not None and frame_stride > 0:
        stride_idx = list(range(0, n_frames, frame_stride))
        if not stride_idx:
            stride_idx = [0]
        if len(stride_idx) > max_n:
            pick = even_indices(len(stride_idx), max_n)
            indices = [stride_idx[i] for i in pick]
        else:
            indices = stride_idx
    else:
        indices = even_indices(n_frames, max_n)
    return [
        Keyframe(index=i, source=f"video_frame:{i}", path=video) for i in indices
    ]


def _cell_size(frames_bgr: list[np.ndarray]) -> tuple[int, int]:
    if not frames_bgr:
        return THUMB_MAX, THUMB_MAX
    h0, w0 = frames_bgr[0].shape[:2]
    scale = min(THUMB_MAX / max(h0, 1), THUMB_MAX / max(w0, 1), 1.0)
    return max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))


def write_contact_sheet(
    frames_bgr: list[np.ndarray],
    out_path: Path,
) -> Path:
    """Write a grid mosaic. Empty input yields a labeled placeholder."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames_bgr:
        img = np.full((120, 320, 3), 240, dtype=np.uint8)
        cv2.putText(
            img,
            "no keyframes",
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 40, 40),
            2,
        )
        if not cv2.imwrite(str(out_path), img):
            raise OSError(f"failed to write {out_path}")
        return out_path

    tw, th = _cell_size(frames_bgr)
    n = len(frames_bgr)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = int(math.ceil(n / cols))
    sheet_w = cols * tw + (cols + 1) * TILE_PAD
    sheet_h = rows * th + (rows + 1) * TILE_PAD
    sheet = np.full((sheet_h, sheet_w, 3), 30, dtype=np.uint8)
    for i, frame in enumerate(frames_bgr):
        r, c = divmod(i, cols)
        thumb = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        y0 = TILE_PAD + r * (th + TILE_PAD)
        x0 = TILE_PAD + c * (tw + TILE_PAD)
        sheet[y0 : y0 + th, x0 : x0 + tw] = thumb
        label = str(i)
        cv2.putText(
            sheet,
            label,
            (x0 + 4, y0 + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(out_path), sheet):
        raise OSError(f"failed to write {out_path}")
    return out_path


def _photo_stills(adapted: AdaptedInput) -> list[Path]:
    path = adapted.path
    if path.is_file() and path.suffix.lower() in STILL_SUFFIXES:
        return [path]
    return _top_stills(path)


def _video_path(adapted: AdaptedInput) -> Path:
    if adapted.consumed:
        return Path(adapted.consumed[0])
    path = adapted.path
    if path.is_file():
        return path
    named = path / "rgb.mp4"
    if named.exists():
        return named
    raise FileNotFoundError(f"no RGB video for {path}")


def _load_photo_frames(keyframes: list[Keyframe]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for kf in keyframes:
        assert kf.path is not None
        img = cv2.imread(str(kf.path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"could not read still {kf.path}")
        frames.append(img)
    return frames


def _last_readable_frame(video: Path, claimed_n: int) -> int:
    """Highest 0-based index OpenCV can actually decode.

    ``CAP_PROP_FRAME_COUNT`` often overestimates by one on these Record3D
    copies, so the even-spaced last index can be unreadable.
    """
    if claimed_n <= 0:
        return -1
    # Probe a short window at the reported end; fall back toward zero.
    start = max(0, claimed_n - 32)
    probe = read_rgb_frames(video, set(range(start, claimed_n)))
    if probe:
        return max(probe)
    if start == 0:
        return -1
    earlier = read_rgb_frames(video, {0, start // 2, max(0, start - 1)})
    return max(earlier) if earlier else -1


def _load_video_frames(
    video: Path,
    keyframes: list[Keyframe],
    *,
    claimed_n: int,
    max_n: int,
    frame_stride: int | None,
) -> tuple[list[Keyframe], list[np.ndarray]]:
    wanted = {kf.index for kf in keyframes}
    found = read_rgb_frames(video, wanted)
    if set(found) == wanted:
        return keyframes, [found[kf.index] for kf in keyframes]

    last_ok = _last_readable_frame(video, claimed_n)
    if last_ok < 0:
        raise OSError(f"could not decode any RGB frames from {video}")
    keyframes = select_video_keyframes(
        video,
        last_ok + 1,
        max_n=max_n,
        frame_stride=frame_stride,
    )
    found = read_rgb_frames(video, {kf.index for kf in keyframes})
    missing = {kf.index for kf in keyframes} - set(found)
    if missing:
        raise OSError(f"could not decode video frames {sorted(missing)} from {video}")
    return keyframes, [found[kf.index] for kf in keyframes]


def enrich_blocked_payload(
    adapted: AdaptedInput,
    *,
    keyframes: list[Keyframe],
    preview_name: str = PREVIEW_NAME,
) -> dict:
    payload = blocked_plan_payload(adapted)
    payload["method"] = RGB_METHOD
    payload["confidence_status"] = CONFIDENCE_STATUS
    payload["uncertainty"] = UNCERTAINTY
    payload["visual_preview"] = preview_name
    payload["keyframes"] = [
        {
            "index": kf.index,
            "source": kf.source,
            "path": str(kf.path) if kf.path is not None else None,
        }
        for kf in keyframes
    ]
    notes = list(payload.get("notes") or [])
    notes.append(
        f"Selected {len(keyframes)} non-metric keyframes and wrote {preview_name}."
    )
    payload["notes"] = notes
    payload["disclaimer"] = " ".join(notes) if notes else payload.get("disclaimer")
    return payload


def write_rgb_tier_outputs(
    adapted: AdaptedInput,
    out_dir: Path,
    frame_stride: int | None = None,
    max_keyframes: int = MAX_KEYFRAMES,
) -> dict[str, Path]:
    """Write keyframes, contact sheet, and an enriched blocked plan."""
    if adapted.tier not in ("photo", "video"):
        raise ValueError(f"write_rgb_tier_outputs expects photo|video, got {adapted.tier}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    kf_dir = out_dir / KEYFRAMES_DIR
    if kf_dir.exists():
        shutil.rmtree(kf_dir)
    kf_dir.mkdir(parents=True)

    if adapted.tier == "photo":
        stills = _photo_stills(adapted)
        keyframes = select_photo_keyframes(stills, max_n=max_keyframes)
        frames_bgr = _load_photo_frames(keyframes)
        written: list[Keyframe] = []
        for i, (kf, frame) in enumerate(zip(keyframes, frames_bgr)):
            dest = kf_dir / f"{i:04d}_{kf.path.stem}.png"
            if not cv2.imwrite(str(dest), frame):
                raise OSError(f"failed to write {dest}")
            written.append(
                Keyframe(index=kf.index, source=kf.source, path=dest.resolve())
            )
    else:
        video = _video_path(adapted)
        n_frames = adapted.n_video_frames or 0
        keyframes = select_video_keyframes(
            video,
            n_frames,
            max_n=max_keyframes,
            frame_stride=frame_stride,
        )
        keyframes, frames_bgr = _load_video_frames(
            video,
            keyframes,
            claimed_n=n_frames,
            max_n=max_keyframes,
            frame_stride=frame_stride,
        )
        written = []
        for i, (kf, frame) in enumerate(zip(keyframes, frames_bgr)):
            dest = kf_dir / f"{i:04d}_frame_{kf.index:06d}.png"
            if not cv2.imwrite(str(dest), frame):
                raise OSError(f"failed to write {dest}")
            written.append(
                Keyframe(
                    index=kf.index,
                    source=kf.source,
                    path=dest.resolve(),
                )
            )

    preview = write_contact_sheet(frames_bgr, out_dir / PREVIEW_NAME)
    payload = enrich_blocked_payload(adapted, keyframes=written)
    json_path = write_plan_json(payload, out_dir / "plan.json")
    svg_path = write_plan_svg(payload, out_dir / "plan.svg")
    return {
        "plan_json": json_path,
        "plan_svg": svg_path,
        "preview": preview,
        "keyframes_dir": kf_dir,
    }
