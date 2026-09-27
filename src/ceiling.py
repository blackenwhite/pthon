"""Same-capture ceiling height.

A horizontal plane is not a ceiling just because it sits above the floor.
Tabletops and a horizontal slice through the walls both look like planes.
A ceiling is a wide sheet with empty space above it and more points on the
sheet than in the volume just below. If nothing passes, height stays null.
"""

from __future__ import annotations

import numpy as np

from src.planes import (
    PLANE_SEED,
    FittedPlane,
    RejectedPlane,
    _residuals,
    _segment_one,
    _to_fitted,
)

# Same gate the reconstruction report already uses for a plausible room height.
HEIGHT_MIN_M = 1.6
HEIGHT_MAX_M = 4.5
MIN_INLIERS = 80
# p10–p90 of the inliers. A patch smaller than this is not a ceiling.
MIN_SPAN_M = 1.2
# Occupied cells inside that span. A wall ring plus a small patch does not fill it.
MIN_FILL = 0.45
FILL_CELL_M = 0.40
# Band just above the candidate. A ceiling has almost no points there.
ABOVE_INNER_M = 0.12
ABOVE_OUTER_M = 0.45
MIN_ABOVE_RATIO = 2.0
# The sheet must be denser than the thicker band underneath it.
MIN_BELOW_RATIO = 1.5
MAX_CANDIDATES = 6
# Pool starts under the minimum so a plane near 1.6 m can still be fit.
POOL_MARGIN_M = 0.15
# Look-up scans often have no empty-above sheet: LiDAR hits walls, RGB shows plaster.
# Interior columns that are empty in the mid-band still give a room height.
COLUMN_CELL_M = 0.40
MIN_BIMODAL_COLUMNS = 8
COLUMN_MIN_POINTS = 12
# A light fixture or second sheet above a real ceiling trips occupied_above.
# Recover the highest wide sheet whose only failure was that band test.
OCCUPIED_SHEET_MIN_FILL = 0.82


def select_ceiling(
    xyz: np.ndarray,
    floor: FittedPlane | None,
    *,
    distance_m: float,
) -> tuple[FittedPlane | None, list[RejectedPlane], list[str]]:
    """Search the upper band. Return the ceiling, every rejected candidate, and notes."""
    if floor is None:
        return None, [], ["ceiling blocked: no floor plane, so height was not searched"]

    floor_y = float(floor.mean_xyz[1])
    pool_y = floor_y + HEIGHT_MIN_M - POOL_MARGIN_M
    pool = xyz[xyz[:, 1] >= pool_y]
    if pool.shape[0] < MIN_INLIERS:
        return (
            None,
            [],
            [
                "ceiling blocked: "
                f"{int(pool.shape[0])} points at least "
                f"{HEIGHT_MIN_M - POOL_MARGIN_M:.2f} m above the floor "
                f"(need {MIN_INLIERS}); height left null"
            ],
        )

    rng = np.random.default_rng(PLANE_SEED)
    rejected: list[RejectedPlane] = []
    notes: list[str] = []
    for _ in range(MAX_CANDIDATES):
        if pool.shape[0] < MIN_INLIERS:
            break
        got = _segment_one(
            pool,
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
        cand = _to_fitted(model, xyz[idx], "ceiling")
        code, detail = _reject_reason(xyz, cand, floor_y, distance_m)
        if code is None:
            dy = float(cand.mean_xyz[1] - floor_y)
            sx, sz, _fill = _coverage(cand.points)
            notes.insert(
                0,
                f"ceiling: {dy:.2f} m above the floor, inliers={cand.n_inliers}, "
                f"span {sx:.2f} x {sz:.2f} m; space above the plane is empty",
            )
            return cand, rejected, notes
        rejected.append(RejectedPlane(cand, code, detail))
        notes.append(f"rejected ceiling: {code} {detail}")
        pool = pool[_residuals(pool, model) >= distance_m]

    recovered = _recover_occupied_sheet(xyz, rejected, floor_y, distance_m)
    if recovered is not None:
        dy = float(recovered.mean_xyz[1] - floor_y)
        notes.insert(
            0,
            (
                f"ceiling: {dy:.2f} m above the floor, inliers={recovered.n_inliers}; "
                "accepted as the highest wide sheet (occupied-above was fixtures/clutter)"
            ),
        )
        return recovered, rejected, notes

    envelope, env_notes = _column_envelope_ceiling(xyz, floor)
    notes.extend(env_notes)
    if envelope is not None:
        return envelope, rejected, notes

    notes.append(
        "ceiling blocked: no empty-above sheet and no bimodal column height; "
        "height left null"
    )
    return None, rejected, notes


def _recover_occupied_sheet(
    xyz: np.ndarray,
    rejected: list[RejectedPlane],
    floor_y: float,
    distance_m: float,
) -> FittedPlane | None:
    """Keep a full ceiling sheet that only failed because something sits above it."""
    scored: list[FittedPlane] = []
    for item in rejected:
        if item.code != "occupied_above":
            continue
        cand = item.plane
        dy = float(cand.mean_xyz[1] - floor_y)
        sx, sz, fill = _coverage(cand.points)
        core, above, _below = _band_counts(xyz, cand, distance_m)
        above_ratio = core / above if above else float(core)
        if dy < HEIGHT_MIN_M or dy > HEIGHT_MAX_M:
            continue
        if cand.n_inliers < MIN_INLIERS:
            continue
        if sx < MIN_SPAN_M or sz < MIN_SPAN_M:
            continue
        if fill < OCCUPIED_SHEET_MIN_FILL:
            continue
        # Wall-height slices have more volume above them than on the sheet.
        if above_ratio < 0.80:
            continue
        scored.append(cand)
    if not scored:
        return None
    return max(scored, key=lambda p: float(p.mean_xyz[1]))


def _column_envelope_ceiling(
    xyz: np.ndarray, floor: FittedPlane
) -> tuple[FittedPlane | None, list[str]]:
    """Height from interior XZ columns that show floor and ceiling clusters."""
    floor_y = float(floor.mean_xyz[1])
    x0, x1 = np.percentile(xyz[:, 0], [15.0, 85.0])
    z0, z1 = np.percentile(xyz[:, 2], [15.0, 85.0])
    interior = (
        (xyz[:, 0] >= x0)
        & (xyz[:, 0] <= x1)
        & (xyz[:, 2] >= z0)
        & (xyz[:, 2] <= z1)
    )
    cloud = xyz[interior]
    if cloud.shape[0] < COLUMN_MIN_POINTS * MIN_BIMODAL_COLUMNS:
        return None, [
            "column envelope skipped: not enough interior points for a height vote"
        ]

    ix = np.floor((cloud[:, 0] - x0) / COLUMN_CELL_M).astype(int)
    iz = np.floor((cloud[:, 2] - z0) / COLUMN_CELL_M).astype(int)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, key in enumerate(zip(ix.tolist(), iz.tolist())):
        buckets.setdefault(key, []).append(i)

    highs: list[np.ndarray] = []
    heights: list[float] = []
    for idxs in buckets.values():
        if len(idxs) < COLUMN_MIN_POINTS:
            continue
        ys = cloud[np.asarray(idxs, dtype=np.int64), 1]
        lo, hi = np.percentile(ys, [8.0, 92.0])
        dy = float(hi - lo)
        if dy < HEIGHT_MIN_M or dy > HEIGHT_MAX_M:
            continue
        if not _bimodal_column(ys, float(lo), float(hi)):
            continue
        high_mask = ys >= (hi - 0.12)
        if int(np.sum(high_mask)) < 3:
            continue
        highs.append(cloud[np.asarray(idxs, dtype=np.int64)][high_mask])
        heights.append(dy)

    if len(heights) < MIN_BIMODAL_COLUMNS:
        return None, [
            "column envelope skipped: "
            f"{len(heights)} bimodal columns (need {MIN_BIMODAL_COLUMNS})"
        ]

    height = float(np.median(np.asarray(heights, dtype=np.float64)))
    high_pts = np.concatenate(highs, axis=0)
    ceiling_y = floor_y + height
    model = np.array([0.0, 1.0, 0.0, -ceiling_y], dtype=np.float64)
    near = high_pts[np.abs(high_pts[:, 1] - ceiling_y) <= 0.20]
    if near.shape[0] < 30:
        near = high_pts
    plane = _to_fitted(model, near, "ceiling")
    plane.confidence = "low"
    return plane, [
        (
            f"ceiling: {height:.2f} m from {len(heights)} bimodal interior columns "
            "(no empty-above LiDAR sheet; RGB may still show plaster)"
        )
    ]


def _bimodal_column(ys: np.ndarray, lo: float, hi: float) -> bool:
    """Floor cluster and a high cluster with a thinner mid-band (not a filled wall)."""
    span = hi - lo
    if span < HEIGHT_MIN_M:
        return False
    n_bins = max(8, int(span / 0.12))
    hist, _edges = np.histogram(ys, bins=n_bins, range=(lo, hi))
    if hist.size < 6:
        return False
    n = int(hist.size)
    low_peak = int(hist[: n // 3].max())
    high_peak = int(hist[2 * n // 3 :].max())
    mid = float(hist[n // 3 : 2 * n // 3].mean())
    if low_peak < 5 or high_peak < 4:
        return False
    return mid < 0.55 * float(min(low_peak, high_peak))


def _reject_reason(
    xyz: np.ndarray,
    cand: FittedPlane,
    floor_y: float,
    distance_m: float,
) -> tuple[str | None, str]:
    dy = float(cand.mean_xyz[1] - floor_y)
    sx, sz, fill = _coverage(cand.points)
    core, above, below = _band_counts(xyz, cand, distance_m)
    above_ratio = core / above if above else float(core)
    below_ratio = core / below if below else float(core)
    summary = (
        f"height {dy:.2f} m, inliers {cand.n_inliers}, "
        f"span {sx:.2f} x {sz:.2f} m, fill {fill:.2f}, "
        f"on_plane {core} above {above} (ratio {above_ratio:.2f}) "
        f"below {below} (ratio {below_ratio:.2f})"
    )
    if dy < HEIGHT_MIN_M:
        return "too_low", summary
    if dy > HEIGHT_MAX_M:
        return "too_high", summary
    if cand.n_inliers < MIN_INLIERS:
        return "sparse_support", summary
    if sx < MIN_SPAN_M or sz < MIN_SPAN_M:
        return "small_extent", summary
    if fill < MIN_FILL:
        return "sparse_coverage", summary
    if above_ratio < MIN_ABOVE_RATIO:
        return "occupied_above", summary
    if below_ratio < MIN_BELOW_RATIO:
        return "not_denser_than_below", summary
    return None, summary


def _coverage(points: np.ndarray) -> tuple[float, float, float]:
    """p10–p90 span in X and Z, and the fraction of that box the inliers occupy."""
    if points.shape[0] < 3:
        return 0.0, 0.0, 0.0
    x0, x1 = np.percentile(points[:, 0], [10.0, 90.0])
    z0, z1 = np.percentile(points[:, 2], [10.0, 90.0])
    sx, sz = float(x1 - x0), float(z1 - z0)
    if sx < 1e-6 or sz < 1e-6:
        return sx, sz, 0.0
    core = points[
        (points[:, 0] >= x0)
        & (points[:, 0] <= x1)
        & (points[:, 2] >= z0)
        & (points[:, 2] <= z1)
    ]
    gx = np.floor((core[:, 0] - x0) / FILL_CELL_M).astype(int)
    gz = np.floor((core[:, 2] - z0) / FILL_CELL_M).astype(int)
    occupied = len(set(zip(gx.tolist(), gz.tolist())))
    nx = max(int(np.floor(sx / FILL_CELL_M)) + 1, 1)
    nz = max(int(np.floor(sz / FILL_CELL_M)) + 1, 1)
    return sx, sz, occupied / (nx * nz)


def _band_counts(
    xyz: np.ndarray, cand: FittedPlane, distance_m: float
) -> tuple[int, int, int]:
    """Points on the plane, just above it, and just below it, inside the inlier span."""
    if cand.points.shape[0] < 3:
        return 0, 0, 0
    x0, x1 = np.percentile(cand.points[:, 0], [10.0, 90.0])
    z0, z1 = np.percentile(cand.points[:, 2], [10.0, 90.0])
    in_xz = (
        (xyz[:, 0] >= x0)
        & (xyz[:, 0] <= x1)
        & (xyz[:, 2] >= z0)
        & (xyz[:, 2] <= z1)
    )
    y = xyz[:, 1]
    plane_y = float(cand.mean_xyz[1])
    band = max(float(distance_m), 0.04)
    core = int(np.sum(in_xz & (np.abs(y - plane_y) <= band)))
    above = int(
        np.sum(in_xz & (y > plane_y + ABOVE_INNER_M) & (y < plane_y + ABOVE_OUTER_M))
    )
    below = int(
        np.sum(in_xz & (y < plane_y - ABOVE_INNER_M) & (y > plane_y - ABOVE_OUTER_M))
    )
    return core, above, below
