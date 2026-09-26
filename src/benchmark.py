"""Derive photo, video, and LiDAR benchmark inputs from supplied Record3D dumps.

Stills are decoded from rgb.mp4. They are not independent native photographs.
No private-room data is used.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2

from src.ingest import REQUIRED, Capture, load_capture
from src.media import read_rgb_frames

SCHEMA = "cozmo.benchmark.v1"
MANIFEST_NOTE = (
    "Photo stills are PNG frames decoded from the capture rgb.mp4. "
    "They are derived RGB, not independent native photographs and not "
    "screen captures. The video file is a copy of that same rgb.mp4. "
    "The LiDAR folder is a copy of the Record3D dump. All three logical "
    "tiers come from store/; no iPhone 15+ or private-room capture was used."
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORE = REPO_ROOT / "store"
DEFAULT_OUT = REPO_ROOT / "benchmark"

# Frozen baseline first, then the look-up capture, then the report dump.
KNOWN_ROOMS: tuple[tuple[str, str, str], ...] = (
    ("c00a170fe1", "room_01", "single_room"),
    ("c7d28f72c6", "room_02", "look_up_ceiling"),
    ("1a8384c3f6", "room_03", "report_capture"),
)

LIDAR_COPY_FILES = REQUIRED


@dataclass(frozen=True)
class RoomSpec:
    capture_id: str
    room_id: str
    role: str
    source: Path


def _is_record3d(root: Path) -> bool:
    return root.is_dir() and all((root / name).exists() for name in REQUIRED)


def discover_rooms(store: Path) -> list[RoomSpec]:
    store = store.resolve()
    if not store.is_dir():
        raise FileNotFoundError(f"store folder not found: {store}")
    by_id = {p.name: p for p in store.iterdir() if _is_record3d(p)}
    rooms: list[RoomSpec] = []
    used: set[str] = set()
    for capture_id, room_id, role in KNOWN_ROOMS:
        src = by_id.get(capture_id)
        if src is None:
            continue
        rooms.append(RoomSpec(capture_id, room_id, role, src))
        used.add(capture_id)
    extras = sorted(cid for cid in by_id if cid not in used)
    next_n = len(rooms) + 1
    for capture_id in extras:
        rooms.append(
            RoomSpec(
                capture_id,
                f"room_{next_n:02d}",
                "unlisted_store_capture",
                by_id[capture_id],
            )
        )
        next_n += 1
    if not rooms:
        raise FileNotFoundError(f"no Record3D capture folders in {store}")
    return rooms


def photo_frame_indices(n_poses: int, max_stills: int) -> list[int]:
    """Evenly spaced 0-based pose/frame indices. Reproducible for a given capture."""
    if n_poses <= 0 or max_stills <= 0:
        return []
    k = min(int(max_stills), n_poses)
    if k == 1:
        return [0]
    return [int(round(i * (n_poses - 1) / (k - 1))) for i in range(k)]


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _copy_lidar(src: Path, dest: Path) -> list[str]:
    _reset_dir(dest)
    copied: list[str] = []
    for name in LIDAR_COPY_FILES:
        item = src / name
        target = dest / name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        copied.append(name)
    return copied


def _decode_photo_frames(video: Path, indices: list[int]):
    """Decode requested frames; if the last mp4 packet is missing, resample below it.

    OpenCV often reports one more frame than it can read from these Record3D files.
    """
    frames = read_rgb_frames(video, set(indices))
    if indices and indices[-1] not in frames and indices[-1] > 0:
        indices = photo_frame_indices(indices[-1], len(indices))
        frames = read_rgb_frames(video, set(indices))
    return frames, indices


def _extract_photos(cap: Capture, dest: Path, indices: list[int]) -> list[dict]:
    _reset_dir(dest)
    frames, indices = _decode_photo_frames(cap.root / "rgb.mp4", indices)
    stills: list[dict] = []
    for idx in indices:
        bgr = frames.get(idx)
        if bgr is None:
            continue
        name = f"{idx:06d}.png"
        path = dest / name
        if not cv2.imwrite(str(path), bgr):
            raise RuntimeError(f"could not write {path}")
        stills.append(
            {
                "index": idx,
                "file": f"photo/{dest.name}/{name}",
            }
        )
    return stills


def prepare_benchmark(
    store: Path,
    out: Path,
    *,
    max_stills: int = 12,
) -> dict:
    rooms = discover_rooms(store)
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    room_payloads: list[dict] = []
    for spec in rooms:
        cap = load_capture(spec.source)
        n_poses = len(cap.poses)
        indices = photo_frame_indices(n_poses, max_stills)
        photo_dir = out / "photo" / spec.room_id
        stills = _extract_photos(cap, photo_dir, indices)
        video_dir = out / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_path = video_dir / f"{spec.room_id}.mp4"
        shutil.copy2(spec.source / "rgb.mp4", video_path)
        lidar_dir = out / "lidar" / spec.room_id
        lidar_files = _copy_lidar(spec.source, lidar_dir)
        rel_photo = f"photo/{spec.room_id}"
        rel_video = f"video/{spec.room_id}.mp4"
        rel_lidar = f"lidar/{spec.room_id}"
        room_payloads.append(
            {
                "id": spec.room_id,
                "store_capture": spec.capture_id,
                "role": spec.role,
                "source_dir": str(spec.source),
                "photo": {
                    "path": rel_photo,
                    "source": "rgb.mp4",
                    "kind": "derived_rgb_frames",
                    "independent_native_photographs": False,
                    "screen_captured": False,
                    "max_stills": max_stills,
                    "n_stills": len(stills),
                    "stills": stills,
                },
                "video": {
                    "path": rel_video,
                    "source": "rgb.mp4",
                    "kind": "copied_rgb_video",
                    "independent_native_video": False,
                },
                "lidar": {
                    "path": rel_lidar,
                    "kind": "copied_record3d_capture",
                    "files": lidar_files,
                },
            }
        )
    payload = {
        "schema": SCHEMA,
        "store": str(store.resolve()),
        "output": str(out),
        "note": MANIFEST_NOTE,
        "rooms": room_payloads,
    }
    (out / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate benchmark/ photo, video, and LiDAR inputs from store/. "
            "Photo stills are derived from rgb.mp4, not native photographs."
        )
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE,
        help="Record3D dump folder (default: ./store)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Benchmark output folder (default: ./benchmark)",
    )
    parser.add_argument(
        "--max-stills",
        type=int,
        default=12,
        help="Evenly spaced PNG frames per room decoded from rgb.mp4",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = prepare_benchmark(args.store, args.out, max_stills=args.max_stills)
    n = len(payload["rooms"])
    print(f"benchmark: {n} room(s) from {args.store.resolve()}")
    print(payload["note"])
    for room in payload["rooms"]:
        print(
            f"  {room['id']} <- {room['store_capture']} ({room['role']}) "
            f"photos={room['photo']['n_stills']} "
            f"video={room['video']['path']} "
            f"lidar={room['lidar']['path']}"
        )
    print(f"manifest: {(args.out.resolve() / 'manifest.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
