"""Slice 5: door/window gaps from wall occupancy (heuristic, not a detector)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Opening:
    kind: str  # door | window | unknown
    wall_index: int
    width_m: float
    height_m: float | None
    sill_m: float | None
    start_xz: np.ndarray  # (2,)
    end_xz: np.ndarray  # (2,)
    occupancy: float  # mean body occupancy in the gap vs wall median
    note: str = ""


def _wall_axes(wall) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = wall.normal.astype(np.float64).copy()
    n[1] = 0.0
    ln = float(np.linalg.norm(n))
    if ln < 1e-8:
        n = np.array([1.0, 0.0, 0.0])
    else:
        n = n / ln
    along = np.array([-n[2], 0.0, n[0]], dtype=np.float64)
    al = float(np.linalg.norm(along))
    if al < 1e-8:
        along = np.array([0.0, 0.0, 1.0])
    else:
        along = along / al
    return wall.mean_xyz.astype(np.float64), along, n


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    i = 0
    n = int(mask.size)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i + 1
        while j < n and mask[j]:
            j += 1
        out.append((i, j))
        i = j
    return out


def detect_openings(
    points: np.ndarray,
    floor,
    walls,
    *,
    distance_m: float = 0.10,
    bin_m: float = 0.08,
) -> list[Opening]:
    """Gaps in along-wall occupancy. Interior only; width is the measured signal."""
    if floor is None or not walls or points.shape[0] == 0:
        return []
    floor_y = float(floor.mean_xyz[1])
    found: list[Opening] = []
    for wi, wall in enumerate(walls):
        origin, along, nrm = _wall_axes(wall)
        dist = np.abs((points - origin) @ nrm)
        near = points[dist < distance_m]
        if near.shape[0] < 80:
            continue
        s = (near - origin) @ along
        h = near[:, 1] - floor_y
        keep = (h > -0.15) & (h < 2.9)
        s, h = s[keep], h[keep]
        if s.size < 80:
            continue
        s0, s1 = np.percentile(s, [3.0, 97.0])
        length = float(s1 - s0)
        if length < 1.2:
            continue
        edges = np.arange(s0, s1 + bin_m, bin_m)
        if edges.size < 8:
            continue
        body = (h >= 0.95) & (h <= 1.85)
        low = (h >= 0.0) & (h < 0.50)
        c_body, _ = np.histogram(s[body], bins=edges)
        c_low, _ = np.histogram(s[low], bins=edges)
        med_body = float(np.median(c_body[c_body > 0])) if np.any(c_body > 0) else 0.0
        med_low = float(np.median(c_low[c_low > 0])) if np.any(c_low > 0) else 0.0
        if med_body < 2.0:
            continue
        occ = c_body.astype(np.float64) / med_body
        gap = occ < 0.28
        pad = max(2, int(round(0.20 / bin_m)))
        gap[:pad] = False
        gap[-pad:] = False
        centers = 0.5 * (edges[:-1] + edges[1:])
        for a, b in _runs(gap):
            width = float((b - a) * bin_m)
            if width < 0.55 or width > 2.6:
                continue
            sl = c_low[a:b].astype(np.float64)
            low_occ = float(np.mean(sl) / max(med_low, 1.0))
            body_occ = float(np.mean(occ[a:b]))
            if low_occ < 0.35 and 0.55 <= width <= 1.55:
                kind = "door"
            elif low_occ >= 0.40 and body_occ < 0.32 and width >= 0.60:
                kind = "window"
            else:
                kind = "unknown"
            s_lo = float(centers[a])
            s_hi = float(centers[b - 1])
            margin = 0.35
            neigh_h = h[((s >= s_lo - margin) & (s < s_lo)) | ((s > s_hi) & (s <= s_hi + margin))]
            height = float(np.percentile(neigh_h, 90)) if neigh_h.size >= 20 else None
            in_gap = (s >= s_lo) & (s <= s_hi)
            if kind == "door":
                sill = 0.0
            elif kind == "window":
                low_h = h[in_gap & (h >= 0.0) & (h < 1.1)]
                sill = float(np.percentile(low_h, 90)) if low_h.size >= 8 else None
            else:
                sill = None
            p0 = origin + along * s_lo
            p1 = origin + along * s_hi
            found.append(
                Opening(
                    kind=kind,
                    wall_index=wi,
                    width_m=width,
                    height_m=height,
                    sill_m=sill,
                    start_xz=np.array([p0[0], p0[2]], dtype=np.float64),
                    end_xz=np.array([p1[0], p1[2]], dtype=np.float64),
                    occupancy=body_occ,
                    note="gap in wall occupancy; width from bins, not a detected frame",
                )
            )
    return _dedupe_openings(found)


def _dedupe_openings(openings: list[Opening]) -> list[Opening]:
    kept: list[Opening] = []
    for op in sorted(openings, key=lambda o: o.width_m, reverse=True):
        mid = 0.5 * (op.start_xz + op.end_xz)
        dup = False
        for u in kept:
            um = 0.5 * (u.start_xz + u.end_xz)
            if float(np.linalg.norm(mid - um)) < 0.70:
                dup = True
                break
        if not dup:
            kept.append(op)
    return kept
