"""Slice 3: RANSAC floor / walls / ceiling and an XZ footprint polygon."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from src.cloud import _axis_limits, _height_colors, _raster
from src.openings import Opening, detect_openings

HORIZONTAL_DOT = 0.85  # |n · Y|
VERTICAL_DOT = 0.30  # |n · Y| below this → wall
# Voxel overflow subsample. The voxel grid itself is deterministic.
DOWNSAMPLE_SEED = 0
# Floor, wall, and ceiling numpy RANSAC. Open3D segment_plane is unseeded
# and runs only when this sampler finds no horizontal plane.
PLANE_SEED = 1


@dataclass
class FittedPlane:
    abc_d: np.ndarray  # (4,) ax + by + cz + d = 0
    points: np.ndarray  # (M, 3) inliers
    rmse_m: float
    median_residual_m: float
    kind: str  # floor | ceiling | wall | other
    confidence: str = "ok"  # ok | low — low means the furniture filter kept it with a caveat

    @property
    def normal(self) -> np.ndarray:
        n = self.abc_d[:3].astype(np.float64)
        n /= max(np.linalg.norm(n), 1e-12)
        return n

    @property
    def mean_xyz(self) -> np.ndarray:
        return self.points.mean(axis=0)

    @property
    def n_inliers(self) -> int:
        return int(self.points.shape[0])


@dataclass
class RejectedPlane:
    """A vertical plane that was not used as a wall, with the reason it lost."""

    plane: FittedPlane
    code: str
    detail: str


@dataclass
class PlaneResult:
    floor: FittedPlane | None
    ceiling: FittedPlane | None
    walls: list[FittedPlane]
    others: list[FittedPlane]
    polygon_xz: np.ndarray  # (K, 2) closed if K>=3
    height_m: float | None
    height_p05_p95_m: float | None
    n_points: int
    n_downsampled: int
    openings: list[Opening] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rejected_walls: list[RejectedPlane] = field(default_factory=list)
    rejected_ceilings: list[RejectedPlane] = field(default_factory=list)
    footprint_method: str = ""  # wall_lines | wall_inlier_hull | floor_hull | cloud_hull
    output_quality: str = "blocked"  # ok | warning | blocked


def downsample_points(
    points: np.ndarray,
    *,
    voxel_m: float = 0.04,
    max_points: int = 250_000,
) -> np.ndarray:
    if points.shape[0] == 0:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(points, dtype=np.float64)
    )
    pcd = pcd.voxel_down_sample(max(float(voxel_m), 1e-4))
    xyz = np.asarray(pcd.points)
    if xyz.shape[0] > max_points:
        rng = np.random.default_rng(DOWNSAMPLE_SEED)
        pick = rng.choice(xyz.shape[0], size=max_points, replace=False)
        xyz = xyz[pick]
    return xyz.astype(np.float64)


def _fit_plane_svd(pts: np.ndarray) -> np.ndarray | None:
    if pts.shape[0] < 3:
        return None
    c = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - c, full_matrices=False)
    n = vh[-1]
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        return None
    n = n / ln
    d = -float(np.dot(n, c))
    return np.array([n[0], n[1], n[2], d], dtype=np.float64)


def _normal_y(model: np.ndarray) -> float:
    n = model[:3]
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        return 0.0
    return float(n[1] / ln)


def _matches_require(model: np.ndarray, require: str | None) -> bool:
    if require is None:
        return True
    ny = abs(_normal_y(model))
    if require == "vertical":
        return ny <= VERTICAL_DOT
    if require == "horizontal":
        return ny >= HORIZONTAL_DOT
    return True


def _o3d_segment(
    xyz: np.ndarray, distance_m: float, n_iter: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if xyz.shape[0] < 50:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz, dtype=np.float64))
    try:
        model, inliers = pcd.segment_plane(
            distance_threshold=distance_m,
            ransac_n=3,
            num_iterations=max(int(n_iter), 50),
        )
    except Exception:
        return None
    if len(inliers) < 50:
        return None
    return np.asarray(model, dtype=np.float64), np.asarray(inliers, dtype=np.int64)


def _segment_one_numpy(
    xyz: np.ndarray,
    *,
    distance_m: float,
    n_iter: int,
    rng: np.random.Generator,
    require: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Orientation-constrained RANSAC. Open3D always returns the densest plane (floors/tables)."""
    n = int(xyz.shape[0])
    if n < 50:
        return None
    best_count = 0
    best_mask: np.ndarray | None = None
    for _ in range(n_iter):
        i0, i1, i2 = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = xyz[i0], xyz[i1], xyz[i2]
        nvec = np.cross(p1 - p0, p2 - p0)
        ln = float(np.linalg.norm(nvec))
        if ln < 1e-9:
            continue
        nvec = nvec / ln
        if require == "vertical" and abs(nvec[1]) > VERTICAL_DOT:
            continue
        if require == "horizontal" and abs(nvec[1]) < HORIZONTAL_DOT:
            continue
        d = -float(np.dot(nvec, p0))
        dist = np.abs(xyz @ nvec + d)
        mask = dist < distance_m
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
    if best_mask is None or best_count < 50:
        return None
    model = _fit_plane_svd(xyz[best_mask])
    if model is None:
        return None
    if not _matches_require(model, require):
        return None
    idx = np.flatnonzero(_residuals(xyz, model) < distance_m)
    if idx.size < 50:
        return None
    return model, idx


def _segment_one(
    xyz: np.ndarray,
    *,
    distance_m: float,
    n_iter: int,
    rng: np.random.Generator | None = None,
    require: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if rng is None:
        rng = np.random.default_rng(PLANE_SEED)
    # Seeded sampler. Open3D's segment_plane has no seed, so a floor refit changed
    # which points were left for the walls and the wall count jumped between runs.
    if require in ("vertical", "horizontal"):
        got = _segment_one_numpy(
            xyz, distance_m=distance_m, n_iter=n_iter, rng=rng, require=require
        )
        if got is not None or require == "vertical":
            return got
    # Horizontal fallback only: Open3D peels non-horizontal planes until the densest floor remains.
    work = xyz
    for _ in range(12):
        got = _o3d_segment(work, distance_m, n_iter)
        if got is None:
            return None
        model, idx = got
        if _matches_require(model, require):
            idx_full = np.flatnonzero(_residuals(xyz, model) < distance_m)
            if idx_full.size < 50:
                return None
            refined = _fit_plane_svd(xyz[idx_full])
            if refined is None:
                refined = model
            if not _matches_require(refined, require):
                keep = np.ones(work.shape[0], dtype=bool)
                keep[idx] = False
                work = work[keep]
                continue
            idx_full = np.flatnonzero(_residuals(xyz, refined) < distance_m)
            if idx_full.size < 50:
                return None
            return refined, idx_full
        keep = np.ones(work.shape[0], dtype=bool)
        keep[idx] = False
        work = work[keep]
        if work.shape[0] < 50:
            return None
    return None


def extract_planes(
    xyz: np.ndarray,
    *,
    distance_m: float = 0.04,
    max_planes: int = 12,
    min_inliers: int = 80,
    n_iter: int = 400,
    require: str | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    remaining = xyz.copy()
    found: list[tuple[np.ndarray, np.ndarray]] = []
    rng = np.random.default_rng(PLANE_SEED)
    for _ in range(max_planes):
        got = _segment_one(
            remaining,
            distance_m=distance_m,
            n_iter=n_iter,
            rng=rng,
            require=require,
        )
        if got is None:
            break
        model, idx = got
        if idx.size < min_inliers:
            break
        inlier_pts = remaining[idx]
        found.append((model, inlier_pts))
        mask = np.ones(remaining.shape[0], dtype=bool)
        mask[idx] = False
        remaining = remaining[mask]
        if remaining.shape[0] < min_inliers:
            break
    return found


def _residuals(points: np.ndarray, model: np.ndarray) -> np.ndarray:
    n = model[:3]
    ln = float(np.linalg.norm(n))
    if ln < 1e-12:
        return np.full(points.shape[0], np.inf)
    return np.abs(points @ n + model[3]) / ln


def _stats(points: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    r = _residuals(points, model)
    return float(np.sqrt(np.mean(r * r))), float(np.median(r))


def _orient_normal_up(model: np.ndarray) -> np.ndarray:
    m = model.copy()
    if m[1] < 0:
        m = -m
    return m


def _line_xz(wall: FittedPlane, floor: FittedPlane) -> np.ndarray | None:
    """Wall ∩ floor as Ax + Cz + D = 0 in XZ (Y eliminated)."""
    f = floor.abc_d
    w = wall.abc_d
    if abs(f[1]) < 1e-6:
        return None
    # y = -(a_f x + c_f z + d_f) / b_f
    # plug into wall: a_w x + b_w y + c_w z + d_w = 0
    scale = w[1] / f[1]
    A = w[0] - scale * f[0]
    C = w[2] - scale * f[2]
    D = w[3] - scale * f[3]
    nrm = float(np.hypot(A, C))
    if nrm < 1e-8:
        return None
    return np.array([A / nrm, C / nrm, D / nrm], dtype=np.float64)


def _intersect_xz(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    mat = np.array([[l1[0], l1[1]], [l2[0], l2[1]]], dtype=np.float64)
    det = float(np.linalg.det(mat))
    if abs(det) < 1e-8:
        return None
    rhs = np.array([-l1[2], -l2[2]], dtype=np.float64)
    xz = np.linalg.solve(mat, rhs)
    return xz


def _convex_hull_xz(pts_xz: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. Returns closed ring (first = last) if ≥3 verts."""
    pts = np.unique(np.round(pts_xz, 5), axis=0)
    if pts.shape[0] < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.stack(lower[:-1] + upper[:-1], axis=0)
    return np.vstack([hull, hull[0:1]])


def _aabb_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 2:
        return 0.0
    return float((poly[:, 0].max() - poly[:, 0].min()) * (poly[:, 1].max() - poly[:, 1].min()))


# A corner may sit this far outside the floor points. A 20 m spike may not.
FOOTPRINT_MARGIN_M = 0.5
# Drop intersection vertices that land far outside the floor before the ring is judged.
_FAR_INTERSECTION_MARGIN = 1.0


def _floor_margin_box(hull: np.ndarray) -> tuple[float, float, float, float]:
    """Wide XZ box used only to discard intersection spikes before the ring is tested."""
    span_x = float(hull[:, 0].max() - hull[:, 0].min())
    span_z = float(hull[:, 1].max() - hull[:, 1].min())
    margin = max(_FAR_INTERSECTION_MARGIN, 0.25 * max(span_x, span_z))
    return (
        float(hull[:, 0].min()) - margin,
        float(hull[:, 0].max()) + margin,
        float(hull[:, 1].min()) - margin,
        float(hull[:, 1].max()) + margin,
    )


def _inside_box(xz: np.ndarray, box: tuple[float, float, float, float]) -> bool:
    x0, x1, z0, z1 = box
    return x0 <= float(xz[0]) <= x1 and z0 <= float(xz[1]) <= z1


def _ring_vertices(poly: np.ndarray) -> np.ndarray:
    if poly.shape[0] >= 2 and float(np.linalg.norm(poly[0] - poly[-1])) <= 1e-8:
        return poly[:-1]
    return poly


def _close_ring(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if float(np.linalg.norm(pts[0] - pts[-1])) <= 1e-8:
        return pts
    return np.vstack([pts, pts[0:1]])


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """True when open segments ab and cd cross. Shared endpoints do not count."""

    def orient(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    eps = 1e-9
    return o1 * o2 < -eps and o3 * o4 < -eps


def _polygon_self_intersects(poly: np.ndarray) -> bool:
    ring = _ring_vertices(poly)
    n = int(ring.shape[0])
    if n < 4:
        return False
    for i in range(n):
        a = ring[i]
        b = ring[(i + 1) % n]
        if float(np.linalg.norm(b - a)) < 1e-9:
            continue
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c = ring[j]
            d = ring[(j + 1) % n]
            if float(np.linalg.norm(d - c)) < 1e-9:
                continue
            if _segments_cross(a, b, c, d):
                return True
    return False


def _within_floor_margin(poly: np.ndarray, hull: np.ndarray) -> bool:
    """Simple ring, at least a quarter of the floor box, and within 0.5 m of it."""
    ring = _ring_vertices(poly)
    if ring.shape[0] < 3 or hull.shape[0] < 3:
        return False
    if _polygon_self_intersects(poly):
        return False
    margin = FOOTPRINT_MARGIN_M
    x0 = float(hull[:, 0].min()) - margin
    x1 = float(hull[:, 0].max()) + margin
    z0 = float(hull[:, 1].min()) - margin
    z1 = float(hull[:, 1].max()) + margin
    eps = 1e-3
    for x, z in ring:
        if not (x0 - eps <= float(x) <= x1 + eps and z0 - eps <= float(z) <= z1 + eps):
            return False
    hull_area = max(_aabb_area(hull), 1e-6)
    if _aabb_area(poly) < 0.25 * hull_area:
        return False
    return True


def footprint_quality(method: str, polygon: np.ndarray) -> str:
    """ok for a simple wall-line outline, warning for a hull fallback, blocked if undrawable."""
    if polygon.shape[0] < 4:
        return "blocked"
    if method == "wall_lines":
        return "ok"
    if method in ("wall_inlier_hull", "floor_hull", "cloud_hull"):
        return "warning"
    return "blocked"


def polygon_from_walls(
    floor: FittedPlane, walls: list[FittedPlane]
) -> tuple[np.ndarray, str, str]:
    """Pick a closed XZ outline. Returns polygon, method code, and a note.

    Order: wall-line ring, then the wall-inlier hull, then the floor hull.
    A candidate is kept only when it is simple and stays within
    ``FOOTPRINT_MARGIN_M`` of the floor points.
    """
    hull = _convex_hull_xz(floor.points[:, [0, 2]])
    if not walls:
        return hull, "floor_hull", "polygon from floor convex hull (no walls)"
    lines: list[tuple[float, np.ndarray]] = []
    for w in walls:
        line = _line_xz(w, floor)
        if line is None:
            continue
        n = w.normal
        ang = float(np.arctan2(n[2], n[0]))
        lines.append((ang, line))
    box = _floor_margin_box(hull) if hull.shape[0] >= 3 else None
    inter = np.zeros((0, 2), dtype=np.float64)
    if len(lines) >= 2 and box is not None:
        lines.sort(key=lambda t: t[0])
        verts: list[np.ndarray] = []
        n_lines = len(lines)
        for i in range(n_lines):
            l1 = lines[i][1]
            l2 = lines[(i + 1) % n_lines][1]
            # Nearly parallel lines meet tens of metres away and blow up the outline.
            if abs(float(np.dot(l1[:2], l2[:2]))) > 0.985:
                continue
            hit = _intersect_xz(l1, l2)
            if hit is None or not _inside_box(hit, box):
                continue
            verts.append(hit)
        if len(verts) >= 3:
            inter = _close_ring(np.stack(verts, axis=0))
    if _within_floor_margin(inter, hull):
        return inter, "wall_lines", "polygon from wall ∩ floor lines"
    if inter.shape[0] >= 4 and _polygon_self_intersects(inter):
        why = "wall lines self-intersect"
    elif inter.shape[0] >= 4:
        why = "wall lines outside floor margin"
    else:
        why = "line intersections degenerate"
    wall_pts = np.concatenate([w.points[:, [0, 2]] for w in walls], axis=0)
    wall_hull = _convex_hull_xz(wall_pts)
    if _within_floor_margin(wall_hull, hull):
        return (
            wall_hull,
            "wall_inlier_hull",
            f"polygon from wall-inlier convex hull ({why})",
        )
    hull_area = max(_aabb_area(hull), 1e-6)
    if wall_hull.shape[0] < 4:
        wall_why = "wall hull has too few vertices"
    elif _aabb_area(wall_hull) < 0.25 * hull_area:
        wall_why = "wall hull covers too little of the floor"
    else:
        wall_why = "wall hull outside floor margin"
    return (
        hull,
        "floor_hull",
        f"polygon from floor convex hull ({why}; {wall_why})",
    )


def _to_fitted(model: np.ndarray, pts: np.ndarray, kind: str) -> FittedPlane:
    rmse, med = _stats(pts, model)
    if kind in ("floor", "ceiling", "horizontal"):
        model = _orient_normal_up(model)
    return FittedPlane(model, pts, rmse, med, kind)


def wall_alignment(a: FittedPlane, b: FittedPlane) -> tuple[float, float]:
    """Absolute normal dot, and the smaller plane-to-plane offset (metres)."""
    delta = b.mean_xyz - a.mean_xyz
    offset = min(
        abs(float(np.dot(delta, a.normal))),
        abs(float(np.dot(delta, b.normal))),
    )
    return abs(float(np.dot(a.normal, b.normal))), offset


def _dedupe_walls(
    walls: list[FittedPlane],
    *,
    normal_dot: float = 0.95,
    offset_m: float = 0.30,
    max_keep: int = 8,
) -> list[FittedPlane]:
    walls = sorted(walls, key=lambda w: -w.n_inliers)
    unique_walls: list[FittedPlane] = []
    for w in walls:
        dup = False
        for u in unique_walls:
            aligned, offset = wall_alignment(w, u)
            if aligned > normal_dot and offset < offset_m:
                dup = True
                break
        if not dup:
            unique_walls.append(w)
    return unique_walls[:max_keep]


def fit_planes(
    points: np.ndarray,
    *,
    voxel_m: float = 0.04,
    distance_m: float = 0.04,
    max_planes: int = 12,
) -> PlaneResult:
    notes: list[str] = []
    n_all = int(points.shape[0])
    xyz = downsample_points(points, voxel_m=voxel_m)
    n_ds = int(xyz.shape[0])
    min_h = max(80, n_ds // 80)
    min_w = max(60, n_ds // 120)
    rng = np.random.default_rng(PLANE_SEED)

    # Floor: densest horizontal among the lower points (Y up).
    y_cut = float(np.percentile(xyz[:, 1], 40.0))
    low = xyz[xyz[:, 1] <= y_cut]
    floor = None
    got = _segment_one(
        low if low.shape[0] >= 80 else xyz,
        distance_m=distance_m,
        n_iter=500,
        rng=rng,
        require="horizontal",
    )
    if got is None:
        got = _segment_one(
            xyz, distance_m=distance_m, n_iter=500, rng=rng, require="horizontal"
        )
    remaining = xyz
    if got is not None and got[1].size >= min_h:
        model, idx_low = got
        # re-evaluate inliers on the full downsampled cloud
        idx = np.flatnonzero(_residuals(xyz, model) < distance_m)
        if idx.size >= 50:
            floor = _to_fitted(model, xyz[idx], "floor")
            remaining = xyz[_residuals(xyz, model) >= distance_m]

    walls: list[FittedPlane] = []
    others: list[FittedPlane] = []
    raw_walls = extract_planes(
        remaining,
        distance_m=distance_m,
        max_planes=max(4, max_planes - 2),
        min_inliers=min_w,
        n_iter=600,
        require="vertical",
    )
    for model, pts in raw_walls:
        walls.append(_to_fitted(model, pts, "wall"))
        remaining = remaining[_residuals(remaining, model) >= distance_m]
    # Local import: src.walls imports the plane types defined above.
    from src.walls import select_walls

    walls, rejected_walls = select_walls(walls, floor)
    for item in rejected_walls:
        notes.append(f"rejected wall: {item.code} {item.detail}")
    n_low = sum(1 for w in walls if w.confidence == "low")
    notes.append(
        f"wall filter: kept {len(walls)} (low={n_low}), rejected {len(rejected_walls)}"
    )

    # Ceiling is searched on the full downsampled cloud, not on whatever
    # the wall sampler left behind. A table in the leftovers used to win
    # the single RANSAC shot and hide anything higher.
    from src.ceiling import select_ceiling

    ceiling, rejected_ceilings, ceiling_notes = select_ceiling(
        xyz, floor, distance_m=distance_m
    )
    notes.extend(ceiling_notes)

    polygon = np.zeros((0, 2), dtype=np.float64)
    footprint_method = ""
    if floor is not None:
        polygon, footprint_method, how = polygon_from_walls(floor, walls)
        notes.append(how)
    else:
        notes.append("no horizontal floor plane found")
        polygon = _convex_hull_xz(xyz[:, [0, 2]])
        if polygon.shape[0] >= 4:
            footprint_method = "cloud_hull"
            notes.append("polygon from full-cloud convex hull (no floor)")
    output_quality = footprint_quality(footprint_method, polygon)

    height = None
    if floor is not None and ceiling is not None:
        height = float(ceiling.mean_xyz[1] - floor.mean_xyz[1])

    y = points[:, 1]
    p05, p95 = np.percentile(y, [5.0, 95.0])
    height_pct = float(p95 - p05)

    openings = detect_openings(xyz, floor, walls, distance_m=max(distance_m * 2.0, 0.08))
    if openings:
        notes.append(
            f"openings heuristic: {len(openings)} gap(s); width from occupancy bins, not a door detector"
        )
    else:
        notes.append("openings heuristic: none (interior occupancy gaps only)")

    return PlaneResult(
        floor=floor,
        ceiling=ceiling,
        walls=walls,
        others=others,
        polygon_xz=polygon,
        height_m=height,
        height_p05_p95_m=height_pct,
        n_points=n_all,
        n_downsampled=n_ds,
        openings=openings,
        notes=notes,
        rejected_walls=rejected_walls,
        rejected_ceilings=rejected_ceilings,
        footprint_method=footprint_method,
        output_quality=output_quality,
    )


def _plane_line(name: str, p: FittedPlane) -> str:
    n = p.normal
    m = p.mean_xyz
    return (
        f"{name}: inliers={p.n_inliers}  rmse_m={p.rmse_m:.4f}  "
        f"median_resid_m={p.median_residual_m:.4f}  "
        f"normal=({n[0]:.3f},{n[1]:.3f},{n[2]:.3f})  "
        f"mean_xyz_m=({m[0]:.3f},{m[1]:.3f},{m[2]:.3f})"
    )


def planes_text(result: PlaneResult, *, label: str) -> str:
    lines = [
        f"=== planes: {label} ===",
        f"points: {result.n_points}  downsampled: {result.n_downsampled}",
    ]
    if result.floor:
        lines.append(_plane_line("floor", result.floor))
        lines.append(f"floor_y_m (mean inliers): {result.floor.mean_xyz[1]:.4f}")
    else:
        lines.append("floor: BLOCKED (no horizontal plane)")
    lines.append(f"walls: {len(result.walls)}  rejected: {len(result.rejected_walls)}")
    for i, w in enumerate(result.walls):
        extra = "" if w.confidence == "ok" else f"  confidence={w.confidence}"
        lines.append("  " + _plane_line(f"wall{i}", w) + extra)
    for item in result.rejected_walls:
        lines.append(
            f"  rejected {item.code}: inliers={item.plane.n_inliers}  {item.detail}"
        )
    if result.ceiling:
        lines.append(_plane_line("ceiling", result.ceiling))
        lines.append(f"ceiling_y_m (mean inliers): {result.ceiling.mean_xyz[1]:.4f}")
    else:
        lines.append("ceiling: BLOCKED (no ceiling plane on this capture)")
    for item in result.rejected_ceilings:
        lines.append(
            f"  rejected ceiling {item.code}: inliers={item.plane.n_inliers}  {item.detail}"
        )
    if result.height_m is not None:
        lines.append(f"height_m (ceiling_y - floor_y, same scan): {result.height_m:.4f}")
    else:
        lines.append("height_m: BLOCKED")
    if result.height_p05_p95_m is not None:
        lines.append(
            f"height_p95_minus_p05_m (percentile check, not a plane): {result.height_p05_p95_m:.4f}"
        )
    if result.floor is not None and result.floor.points.shape[0] >= 2:
        fx = result.floor.points[:, [0, 2]]
        lines.append(
            "floor_span_xz_m: "
            f"{float(fx[:, 0].max() - fx[:, 0].min()):.2f} x "
            f"{float(fx[:, 1].max() - fx[:, 1].min()):.2f}"
        )
    if result.polygon_xz.shape[0] >= 3:
        p = result.polygon_xz
        lines.append(
            "polygon_span_xz_m: "
            f"{float(p[:, 0].max() - p[:, 0].min()):.2f} x "
            f"{float(p[:, 1].max() - p[:, 1].min()):.2f}"
        )
        lines.append(f"polygon_xz vertices (closed): {result.polygon_xz.shape[0] - 1}")
        for x, z in result.polygon_xz[:-1]:
            lines.append(f"  {x:.3f} {z:.3f}")
    else:
        lines.append("polygon_xz: BLOCKED (too few vertices)")
    lines.append(f"footprint_method: {result.footprint_method or 'none'}")
    lines.append(f"output_quality: {result.output_quality}")
    lines.append(f"openings: {len(result.openings)}")
    for op in result.openings:
        htxt = "null" if op.height_m is None else f"{op.height_m:.3f}"
        lines.append(
            f"  {op.kind} wall{op.wall_index} width_m={op.width_m:.3f} "
            f"height_m={htxt} xz=({op.start_xz[0]:.2f},{op.start_xz[1]:.2f})-"
            f"({op.end_xz[0]:.2f},{op.end_xz[1]:.2f})"
        )
    for n in result.notes:
        lines.append(f"note: {n}")
    lines.append("residuals are fit error, not tape-measure accuracy")
    return "\n".join(lines)


def write_plan_preview(
    points: np.ndarray,
    polygon_xz: np.ndarray,
    out_path: Path,
    openings: list[Opening] | None = None,
) -> Path:
    """Top-down XZ with height colours and the 2D polygon overlay."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    paint = _height_colors(y)
    u0, u1 = _axis_limits(x)
    v0, v1 = _axis_limits(z)
    img = _raster(x, z, paint)
    h, w = img.shape[:2]
    span_u = u1 - u0
    span_v = v1 - v0

    def xz_px(xz: np.ndarray) -> tuple[int, int]:
        col = int((float(xz[0]) - u0) / span_u * (w - 1))
        row = int((v1 - float(xz[1])) / span_v * (h - 1))
        return col, row

    if polygon_xz.shape[0] >= 3:
        cols = ((polygon_xz[:, 0] - u0) / span_u * (w - 1)).astype(np.int32)
        rows = ((v1 - polygon_xz[:, 1]) / span_v * (h - 1)).astype(np.int32)
        pts = np.stack([cols, rows], axis=1).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 0), thickness=2)
        for px, py in pts.reshape(-1, 2):
            cv2.circle(img, (int(px), int(py)), 4, (255, 80, 80), -1)
    for op in openings or []:
        a = xz_px(op.start_xz)
        b = xz_px(op.end_xz)
        color = (40, 220, 90) if op.kind == "door" else (255, 160, 40)
        if op.kind == "unknown":
            color = (200, 200, 200)
        cv2.line(img, a, b, color, 5)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), bgr)
    return out_path
