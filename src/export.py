"""Slice 4: JSON + SVG from a fitted PlaneResult (metres, honest residuals)."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

from src.planes import FittedPlane, PlaneResult

DISCLAIMER = (
    "Residuals (rmse_m, median_residual_m) are RANSAC fit error against the "
    "point cloud, not tape-measure or laser accuracy. height_m is null unless "
    "a ceiling plane was found on this same capture; do not mix Y from two folders."
)

SCHEMA = "cozmo.room_plan.v1"


def _f(x: float) -> float:
    return float(round(float(x), 6))


def _xyz(v: np.ndarray) -> list[float]:
    return [_f(v[0]), _f(v[1]), _f(v[2])]


def _plane_dict(p: FittedPlane | None) -> dict | None:
    if p is None:
        return None
    n = p.normal
    m = p.mean_xyz
    return {
        "kind": p.kind,
        "n_inliers": p.n_inliers,
        "rmse_m": _f(p.rmse_m),
        "median_residual_m": _f(p.median_residual_m),
        "normal": _xyz(n),
        "mean_xyz_m": _xyz(m),
        "plane_abcd": [_f(c) for c in p.abc_d],
    }


def _polygon_vertices(polygon_xz: np.ndarray) -> list[list[float]]:
    if polygon_xz.shape[0] < 3:
        return []
    pts = polygon_xz.astype(np.float64)
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    return [[_f(x), _f(z)] for x, z in pts]


def plan_to_dict(
    result: PlaneResult,
    *,
    capture: str,
    tier: str = "lidar",
) -> dict:
    height_blocked = result.height_m is None
    return {
        "schema": SCHEMA,
        "tier": tier,
        "capture": capture,
        "units": "metres",
        "disclaimer": DISCLAIMER,
        "n_points": result.n_points,
        "n_downsampled": result.n_downsampled,
        "floor": _plane_dict(result.floor),
        "ceiling": _plane_dict(result.ceiling),
        "walls": [_plane_dict(w) for w in result.walls],
        "height_m": _f(result.height_m) if result.height_m is not None else None,
        "height_blocked": height_blocked,
        "height_p05_p95_m": (
            _f(result.height_p05_p95_m)
            if result.height_p05_p95_m is not None
            else None
        ),
        "polygon_xz_m": _polygon_vertices(result.polygon_xz),
        "notes": list(result.notes),
    }


def write_plan_json(payload: dict, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def write_plan_svg(payload: dict, out_path: Path, *, px_per_m: float = 40.0) -> Path:
    """Top-down XZ: +X right, +Z up the page. SVG y is flipped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    verts = payload.get("polygon_xz_m") or []
    pad = 48.0
    caption_h = 56.0
    if len(verts) < 2:
        w, h = 320.0, 200.0
        body = (
            '<text x="16" y="40" font-size="14" fill="#444">'
            "polygon_xz: BLOCKED (too few vertices)</text>"
        )
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" '
            f'viewBox="0 0 {w:.0f} {h:.0f}">\n{body}\n</svg>\n'
        )
        out_path.write_text(svg)
        return out_path

    xs = [v[0] for v in verts]
    zs = [v[1] for v in verts]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    span_x = max(max_x - min_x, 0.5)
    span_z = max(max_z - min_z, 0.5)
    inner_w = span_x * px_per_m
    inner_h = span_z * px_per_m
    width = inner_w + 2 * pad
    height = inner_h + 2 * pad + caption_h

    def xy(x: float, z: float) -> tuple[float, float]:
        px = pad + (x - min_x) * px_per_m
        py = pad + (max_z - z) * px_per_m
        return px, py

    pairs = [xy(x, z) for x, z in verts]
    pairs.append(pairs[0])
    d = "M " + " L ".join(f"{p:.2f},{q:.2f}" for p, q in pairs) + " Z"
    wall_lines = []
    for i in range(len(verts)):
        a = xy(*verts[i])
        b = xy(*verts[(i + 1) % len(verts)])
        wall_lines.append(
            f'<line x1="{a[0]:.2f}" y1="{a[1]:.2f}" x2="{b[0]:.2f}" y2="{b[1]:.2f}" '
            f'stroke="#1d4ed8" stroke-width="3" />'
        )
    dots = []
    for p, q in pairs[:-1]:
        dots.append(f'<circle cx="{p:.2f}" cy="{q:.2f}" r="4" fill="#b91c1c" />')

    # 1 m scale bar at bottom-left of the drawing area
    bar_y = pad + inner_h + 18
    bar_x0 = pad
    bar_x1 = pad + px_per_m
    height_txt = (
        "BLOCKED"
        if payload.get("height_blocked")
        else f"{payload.get('height_m')} m (same-scan ceiling − floor)"
    )
    cap = payload.get("capture", "")
    disclaimer = escape(str(payload.get("disclaimer", DISCLAIMER))[:220])
    title = escape(f"{cap}  tier={payload.get('tier')}  height={height_txt}")
    body = "\n".join(
        [
            f'<rect x="0" y="0" width="{width:.0f}" height="{height:.0f}" fill="#fafafa" />',
            f'<path d="{d}" fill="rgba(147,197,253,0.25)" stroke="none" />',
            *wall_lines,
            *dots,
            f'<line x1="{bar_x0:.2f}" y1="{bar_y:.2f}" x2="{bar_x1:.2f}" y2="{bar_y:.2f}" '
            f'stroke="#111" stroke-width="2" />',
            f'<text x="{bar_x0:.2f}" y="{bar_y + 16:.2f}" font-size="12" fill="#111">1 m</text>',
            f'<text x="{pad:.2f}" y="{height - 28:.2f}" font-size="12" fill="#111">{title}</text>',
            f'<text x="{pad:.2f}" y="{height - 12:.2f}" font-size="9" fill="#555">{disclaimer}</text>',
        ]
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">\n{body}\n</svg>\n'
    )
    out_path.write_text(svg)
    return out_path


def write_plan_exports(
    result: PlaneResult,
    out_dir: Path,
    *,
    capture: str,
    tier: str = "lidar",
) -> tuple[Path, Path]:
    payload = plan_to_dict(result, capture=capture, tier=tier)
    json_path = write_plan_json(payload, out_dir / "plan.json")
    svg_path = write_plan_svg(payload, out_dir / "plan.svg")
    return json_path, svg_path
