"""Keep upright, room-scale wall planes and record why the others were dropped.

RANSAC will fit a cabinet face, a sofa back, or a second copy of the same wall.
Those planes are vertical enough to pass the sampler and dense enough to beat
the inlier cutoff. This filter drops the obvious non-walls and writes a reason
for each one. A plane that is only slightly off is kept with confidence ``low``.
"""

from __future__ import annotations

import numpy as np

from src.planes import FittedPlane, RejectedPlane, _convex_hull_xz, wall_alignment

# |n · floor normal| above this is a tilted slab, not an upright wall.
# Between TILT_LOW (in _mark_low_confidence) and this value the plane is kept
# and marked low: it might be a real wall with a noisy normal.
TILT_MAX = 0.28
# p90−p10 of height above the floor. A sofa back is shorter than this.
MIN_VERTICAL_EXTENT_M = 0.85
# p90−p10 along the wall. A narrow furniture face is shorter than this.
MIN_ALONG_M = 0.70
# Longest occupied run in the body of the plane.
MIN_RUN_M = 1.00
MIN_OCCUPANCY = 0.35
# Near-parallel copy: same direction, separated by about a cabinet depth.
PARALLEL_HDOT = 0.97
PARALLEL_OFFSET_M = 0.55
# Off the room's dominant axes and sitting inside the floor, not on its edge.
OFF_AXIS_HDOT = 0.927  # about 22 degrees
AXIS_CLUSTER_HDOT = 0.951  # about 18 degrees


def select_walls(
    walls: list[FittedPlane],
    floor: FittedPlane | None,
) -> tuple[list[FittedPlane], list[RejectedPlane]]:
    """Return walls to keep, and every dropped plane with a reason code."""
    if floor is None or not walls:
        return list(walls), []

    hull = _convex_hull_xz(floor.points[:, [0, 2]])
    floor_xz = floor.points[:, [0, 2]]
    room_m = _room_span_m(floor_xz)
    floor_edge_p50 = _edge_distance_p50(hull, floor_xz)
    floor_y = float(floor.mean_xyz[1])

    kept: list[FittedPlane] = []
    rejected: list[RejectedPlane] = []
    meta: dict[int, dict] = {}
    for wall in walls:
        stats = _wall_stats(wall, floor, hull, floor_y)
        meta[id(wall)] = stats
        code, detail = _hard_reject(stats, room_m, floor_edge_p50)
        if code is None:
            kept.append(wall)
        else:
            rejected.append(RejectedPlane(wall, code, detail))

    kept, dropped = _drop_near_parallel(kept, meta)
    rejected.extend(dropped)
    kept, dropped = _drop_off_axis(kept, meta, room_m, floor_edge_p50)
    rejected.extend(dropped)
    _mark_low_confidence(kept, meta, floor_edge_p50)
    return kept, rejected


def _room_span_m(floor_xz: np.ndarray) -> float:
    if floor_xz.shape[0] < 2:
        return 1.0
    span_x = float(floor_xz[:, 0].max() - floor_xz[:, 0].min())
    span_z = float(floor_xz[:, 1].max() - floor_xz[:, 1].min())
    return max(min(span_x, span_z), 1.0)


def _hard_reject(stats: dict, room_m: float, floor_edge_p50: float) -> tuple[str | None, str]:
    if stats["tilt"] > TILT_MAX:
        return "tilted_plane", f"|n·floor|={stats['tilt']:.2f} inliers={stats['inliers']}"
    if stats["vext"] < MIN_VERTICAL_EXTENT_M:
        return (
            "short_vertical_extent",
            f"height_p90_p10_m={stats['vext']:.2f} inliers={stats['inliers']}",
        )
    if stats["along"] < MIN_ALONG_M:
        return "short_span", f"along_p90_p10_m={stats['along']:.2f} inliers={stats['inliers']}"
    if stats["run"] < MIN_RUN_M:
        return "fragmented_support", f"run_m={stats['run']:.2f} inliers={stats['inliers']}"
    if stats["occ"] < MIN_OCCUPANCY:
        return "sparse_support", f"occupancy={stats['occ']:.2f} inliers={stats['inliers']}"
    inset_limit = max(0.45, 0.55 * floor_edge_p50)
    along_limit = max(1.6, 0.35 * room_m)
    if stats["inset10"] > inset_limit and stats["along"] < along_limit:
        return (
            "interior_patch",
            (
                f"inset_p10_m={stats['inset10']:.2f} along_p90_p10_m={stats['along']:.2f} "
                f"inliers={stats['inliers']}"
            ),
        )
    return None, ""


def _drop_near_parallel(
    walls: list[FittedPlane],
    meta: dict[int, dict],
) -> tuple[list[FittedPlane], list[RejectedPlane]]:
    """Keep the copy closer to the floor boundary. The inner face is the furniture."""

    def sort_key(wall: FittedPlane) -> tuple:
        stats = meta[id(wall)]
        # Upright planes win. Within a small tilt band, the one nearer the
        # floor boundary wins, so a cabinet face in front of a wall loses.
        tilt_band = int(stats["tilt"] / 0.06)
        return (tilt_band, stats["inset10"], -wall.n_inliers, -stats["run"])

    chosen: list[FittedPlane] = []
    rejected: list[RejectedPlane] = []
    for wall in sorted(walls, key=sort_key):
        stats = meta[id(wall)]
        loser_to: FittedPlane | None = None
        offset_m = 0.0
        hdot = 0.0
        for other in chosen:
            other_stats = meta[id(other)]
            hdot = abs(float(np.dot(stats["nh"], other_stats["nh"])))
            _aligned, offset_m = wall_alignment(wall, other)
            if hdot > PARALLEL_HDOT and offset_m < PARALLEL_OFFSET_M:
                loser_to = other
                break
        if loser_to is None:
            chosen.append(wall)
            continue
        rejected.append(
            RejectedPlane(
                wall,
                "near_parallel_duplicate",
                (
                    f"offset_m={offset_m:.2f} hdot={hdot:.2f} "
                    f"kept_inliers={loser_to.n_inliers} inliers={wall.n_inliers}"
                ),
            )
        )
    return chosen, rejected


def _drop_off_axis(
    walls: list[FittedPlane],
    meta: dict[int, dict],
    room_m: float,
    floor_edge_p50: float,
) -> tuple[list[FittedPlane], list[RejectedPlane]]:
    axes = _dominant_axes(walls, meta)
    if not axes:
        return walls, []
    inset_limit = max(0.45, 0.55 * floor_edge_p50)
    along_limit = max(1.6, 0.50 * room_m)
    chosen: list[FittedPlane] = []
    rejected: list[RejectedPlane] = []
    for wall in walls:
        stats = meta[id(wall)]
        hdot = max(abs(float(np.dot(stats["nh"], axis))) for axis in axes)
        interior = stats["inset10"] > inset_limit and stats["along"] < along_limit
        if hdot < OFF_AXIS_HDOT and interior:
            rejected.append(
                RejectedPlane(
                    wall,
                    "off_axis_interior",
                    f"hdot={hdot:.2f} inset_p10_m={stats['inset10']:.2f} inliers={wall.n_inliers}",
                )
            )
            continue
        chosen.append(wall)
    return chosen, rejected


def _dominant_axes(walls: list[FittedPlane], meta: dict[int, dict]) -> list[np.ndarray]:
    """One or two horizontal normals, clustered mod 180 degrees and weighted by inliers."""
    if not walls:
        return []
    order = sorted(walls, key=lambda w: -w.n_inliers)
    clusters: list[tuple[np.ndarray, int]] = []
    for wall in order:
        nh = meta[id(wall)]["nh"]
        placed = False
        for i, (axis, weight) in enumerate(clusters):
            if abs(float(np.dot(nh, axis))) >= AXIS_CLUSTER_HDOT:
                clusters[i] = (axis, weight + wall.n_inliers)
                placed = True
                break
        if not placed:
            clusters.append((nh, wall.n_inliers))
    clusters.sort(key=lambda item: -item[1])
    primary, primary_n = clusters[0]
    axes = [primary]
    for axis, weight in clusters[1:]:
        hdot = abs(float(np.dot(axis, primary)))
        # A second room direction is nearer 90 degrees than a duplicate of the first.
        if hdot < 0.50 and weight >= 0.20 * primary_n:
            axes.append(axis)
            break
    return axes


def _mark_low_confidence(
    walls: list[FittedPlane],
    meta: dict[int, dict],
    floor_edge_p50: float,
) -> None:
    inset_limit = max(0.45, 0.55 * floor_edge_p50)
    for wall in walls:
        stats = meta[id(wall)]
        if stats["tilt"] > 0.12 or stats["inset10"] > inset_limit:
            wall.confidence = "low"


def _wall_stats(
    wall: FittedPlane,
    floor: FittedPlane,
    hull: np.ndarray,
    floor_y: float,
) -> dict:
    points = wall.points
    nh = _horizontal_normal(wall)
    along = np.array([-nh[2], 0.0, nh[0]], dtype=np.float64)
    signed = points @ along
    p10, p90 = np.percentile(signed, [10.0, 90.0])
    height = points[:, 1] - floor_y
    h10, h90 = np.percentile(height, [10.0, 90.0])
    run_m, occ = _support(signed, height)
    return {
        "tilt": abs(float(np.dot(wall.normal, floor.normal))),
        "vext": float(h90 - h10),
        "along": float(p90 - p10),
        "inset10": _inset_p10(hull, points[:, [0, 2]]),
        "run": run_m,
        "occ": occ,
        "nh": nh,
        "inliers": wall.n_inliers,
    }


def _horizontal_normal(wall: FittedPlane) -> np.ndarray:
    normal = wall.normal.astype(np.float64).copy()
    normal[1] = 0.0
    length = float(np.linalg.norm(normal))
    if length < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return normal / length


def _support(signed: np.ndarray, height: np.ndarray) -> tuple[float, float]:
    body = (height > 0.15) & (height < 2.4)
    samples = signed[body]
    if samples.size < 30:
        return 0.0, 0.0
    bins = np.floor((samples - float(samples.min())) / 0.25).astype(int)
    counts = np.bincount(bins)
    need = max(6, int(0.004 * samples.size))
    occupied = counts >= need
    if occupied.size == 0:
        return 0.0, 0.0
    best = current = 0
    for flag in occupied:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best * 0.25, float(occupied.mean())


def _edge_distance_p50(hull: np.ndarray, xz: np.ndarray) -> float:
    if hull.shape[0] < 4 or xz.shape[0] == 0:
        return 0.0
    return float(np.median(_edge_distance(hull, xz)))


def _inset_p10(hull: np.ndarray, xz: np.ndarray) -> float:
    """How far the inner 10th percentile sits inside the floor hull. Outside is 0."""
    if hull.shape[0] < 4 or xz.shape[0] == 0:
        return 0.0
    edge = _edge_distance(hull, xz)
    inset = np.where(_inside_hull(hull, xz), edge, 0.0)
    return float(np.percentile(inset, 10.0))


def _edge_distance(hull: np.ndarray, xz: np.ndarray) -> np.ndarray:
    start = hull[:-1]
    end = hull[1:]
    edge = end - start
    length2 = np.sum(edge * edge, axis=1)
    delta = xz[:, None, :] - start[None, :, :]
    scale = np.clip(
        np.sum(delta * edge[None, :, :], axis=2) / np.maximum(length2, 1e-12),
        0.0,
        1.0,
    )
    proj = start[None, :, :] + scale[:, :, None] * edge[None, :, :]
    return np.linalg.norm(xz[:, None, :] - proj, axis=2).min(axis=1)


def _inside_hull(hull: np.ndarray, xz: np.ndarray) -> np.ndarray:
    start = hull[:-1]
    end = hull[1:]
    edge = end - start
    shoelace = float(np.sum(start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1]))
    cross = edge[:, 0] * (xz[:, None, 1] - start[None, :, 1]) - edge[:, 1] * (
        xz[:, None, 0] - start[None, :, 0]
    )
    if shoelace < 0.0:
        cross = -cross
    return np.all(cross >= -1e-6, axis=1)
