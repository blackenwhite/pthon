"""Run configuration and phase timings written onto plan.json and report.json."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.planes import DOWNSAMPLE_SEED, PLANE_SEED


class PhaseTimer:
    def __init__(self) -> None:
        self.phases_s: dict[str, float] = {}
        self._name: str | None = None
        self._t0: float | None = None

    def start(self, name: str) -> None:
        self.stop()
        self._name = name
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        if self._name is None or self._t0 is None:
            return
        self.phases_s[self._name] = round(time.perf_counter() - self._t0, 3)
        self._name = None
        self._t0 = None


def build_run(
    *,
    capture: str,
    tier: str,
    frame_stride: int,
    pixel_stride: int,
    min_confidence: int,
    invert_extrinsics: bool,
    from_ply: bool,
    n_points: int,
    n_retained: int,
    n_walls: int,
    n_openings: int,
    phases_s: dict[str, float],
    outputs: list[Path],
) -> dict:
    run = {
        "capture": capture,
        "tier": tier,
        "frame_stride": int(frame_stride),
        "pixel_stride": int(pixel_stride),
        "min_confidence": int(min_confidence),
        "invert_extrinsics": bool(invert_extrinsics),
        "from_ply": bool(from_ply),
        "n_points": int(n_points),
        "n_retained": int(n_retained),
        "n_walls": int(n_walls),
        "n_openings": int(n_openings),
        "seeds": {
            "downsample": DOWNSAMPLE_SEED,
            "planes": PLANE_SEED,
            "open3d_segment_plane": "unseeded_fallback",
        },
        "phases_s": {name: float(secs) for name, secs in phases_s.items()},
        "outputs": [str(Path(path).resolve()) for path in outputs],
    }
    run["determinism_key"] = repeat_key(run)
    return run


def repeat_key(run: dict) -> dict:
    """Geometry and configuration a second run of the same cloud must match.

    Phase timings are left out. They move with the machine.
    """
    return {
        "capture": run["capture"],
        "tier": run["tier"],
        "frame_stride": run["frame_stride"],
        "pixel_stride": run["pixel_stride"],
        "min_confidence": run["min_confidence"],
        "invert_extrinsics": run["invert_extrinsics"],
        "from_ply": run["from_ply"],
        "n_points": run["n_points"],
        "n_retained": run["n_retained"],
        "n_walls": run["n_walls"],
        "n_openings": run["n_openings"],
        "seeds": run["seeds"],
    }


def attach_run(path: Path, run: dict) -> None:
    data = json.loads(path.read_text())
    data["run"] = run
    path.write_text(json.dumps(data, indent=2) + "\n")


def run_text(run: dict) -> str:
    phases = " ".join(f"{name}={secs:.3f}s" for name, secs in run["phases_s"].items())
    seeds = run["seeds"]
    return (
        f"run: capture={run['capture']} tier={run['tier']} "
        f"stride={run['frame_stride']}/{run['pixel_stride']} "
        f"from_ply={run['from_ply']} "
        f"points={run['n_points']} retained={run['n_retained']} "
        f"walls={run['n_walls']} openings={run['n_openings']} "
        f"seeds=downsample:{seeds['downsample']},planes:{seeds['planes']} "
        f"{phases}"
    )
