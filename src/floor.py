"""Pick the real floor: the lowest wide horizontal sheet, not the densest slab.

A look-up LiDAR walk puts most points on walls at chest height. Sequential
RANSAC then reports that slab as the floor, so every later height is measured
from the wrong plane. The floor is the lowest candidate that still spans the
room. Search the lower part of the cloud first so wall-height slabs never
exhaust the candidate budget.
"""

from __future__ import annotations

import numpy as np

from src.ceiling import MIN_FILL, MIN_SPAN_M, _coverage
from src.planes import (
    PLANE_SEED,
    FittedPlane,
    RejectedPlane,
    _residuals,
    _segment_one,
    _to_fitted,
)

MAX_CANDIDATES = 8
MIN_INLIERS = 80
# Keep a lower plane even when the wall slab has more inliers.
MIN_INLIER_FRACTION = 0.25


def select_floor(
    xyz: np.ndarray,
    *,
    distance_m: float,
) -> tuple[FittedPlane | None, list[RejectedPlane], list[str]]:
    """Return the lowest wide horizontal plane, every skipped candidate, and notes."""
    if xyz.shape[0] < MIN_INLIERS:
        return None, [], ["floor blocked: not enough points"]

    rng = np.random.default_rng(PLANE_SEED)
    y_cut = float(np.percentile(xyz[:, 1], 35.0))
    low = xyz[xyz[:, 1] <= y_cut]
    scored, rejected, notes = _peel_sheets(
        low if low.shape[0] >= MIN_INLIERS else xyz,
        xyz,
        distance_m=distance_m,
        rng=rng,
    )
    if not scored:
        extra_scored, extra_rej, extra_notes = _peel_sheets(
            xyz, xyz, distance_m=distance_m, rng=rng
        )
        scored.extend(extra_scored)
        rejected.extend(extra_rej)
        notes.extend(extra_notes)

    if not scored:
        return None, rejected, notes + ["floor blocked: no wide horizontal sheet"]

    max_inliers = max(c.n_inliers for c, *_ in scored)
    eligible = [
        item
        for item in scored
        if item[0].n_inliers >= MIN_INLIER_FRACTION * max_inliers
    ]
    chosen, sx, sz, fill = min(eligible, key=lambda item: float(item[0].mean_xyz[1]))
    notes.insert(
        0,
        (
            f"floor: y {chosen.mean_xyz[1]:.2f} m (lowest wide sheet, "
            f"inliers={chosen.n_inliers}, span {sx:.2f} x {sz:.2f} m, fill {fill:.2f})"
        ),
    )
    for cand, *_ in scored:
        if cand is chosen:
            continue
        dy = float(cand.mean_xyz[1] - chosen.mean_xyz[1])
        rejected.append(
            RejectedPlane(
                cand,
                "higher_than_selected_floor",
                f"y {cand.mean_xyz[1]:.2f} m is {dy:.2f} m above the selected floor",
            )
        )
    return chosen, rejected, notes


def _peel_sheets(
    pool: np.ndarray,
    xyz: np.ndarray,
    *,
    distance_m: float,
    rng: np.random.Generator,
) -> tuple[
    list[tuple[FittedPlane, float, float, float]],
    list[RejectedPlane],
    list[str],
]:
    scored: list[tuple[FittedPlane, float, float, float]] = []
    rejected: list[RejectedPlane] = []
    notes: list[str] = []
    remaining = pool
    for _ in range(MAX_CANDIDATES):
        if remaining.shape[0] < MIN_INLIERS:
            break
        got = _segment_one(
            remaining,
            distance_m=distance_m,
            n_iter=500,
            rng=rng,
            require="horizontal",
        )
        if got is None:
            break
        model, _ = got
        idx = np.flatnonzero(_residuals(xyz, model) < distance_m)
        if idx.size < 50:
            break
        cand = _to_fitted(model, xyz[idx], "floor")
        sx, sz, fill = _coverage(cand.points)
        summary = (
            f"y {cand.mean_xyz[1]:.2f} m, inliers {cand.n_inliers}, "
            f"span {sx:.2f} x {sz:.2f} m, fill {fill:.2f}"
        )
        wide = (
            cand.n_inliers >= MIN_INLIERS
            and sx >= MIN_SPAN_M
            and sz >= MIN_SPAN_M
            and fill >= MIN_FILL
        )
        if wide:
            scored.append((cand, sx, sz, fill))
            notes.append(f"floor candidate: {summary}")
        else:
            rejected.append(RejectedPlane(cand, "not_a_room_sheet", summary))
            notes.append(f"rejected floor: not_a_room_sheet {summary}")
        remaining = remaining[_residuals(remaining, model) >= distance_m]
    return scored, rejected, notes
