"""Slice 7: reconstruction-damage checks, a short fix loop, and a JSON report.

This is not a detector of physical room damage. It flags a broken outline
(wall intersections that run away), near-duplicate walls, and a missing or
implausible same-scan height. Heights from two captures are never fused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from src.export import ACCURACY_STATUS, MEASUREMENT_STATUS, STATUS
from src.planes import (
    FittedPlane,
    PlaneResult,
    _convex_hull_xz,
    _dedupe_walls,
    polygon_from_walls,
    wall_alignment,
)

SCHEMA = "cozmo.damage_report.v1"
REPORT_METHOD = "reconstruction_health_check"
DISCLAIMER = (
    "Findings describe reconstruction damage (exploded outline, duplicate walls, "
    "missing or implausible planes), not physical room damage and not tape-measure "
    "accuracy. height_m stays null unless that same capture has a ceiling plane. "
    "Do not mix Y from two folders."
)
FIXABLE = frozenset({"exploded_polygon", "duplicate_walls"})
HEIGHT_MIN_M = 1.6
HEIGHT_MAX_M = 4.5
EXPLODE_RATIO = 2.5
MAX_SPAN_M = 40.0


@dataclass
class Finding:
    code: str
    severity: str  # fail | warn | info
    detail: str

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity, "detail": self.detail}


@dataclass
class ScanReport:
    capture: str
    passes: int
    fixes: list[str]
    before: dict
    after: dict
    result: PlaneResult
    versus: dict | None = None
    notes: list[str] = field(default_factory=list)
    input_tier: str = "lidar"
    source_files: list[str] = field(default_factory=list)
    method: str = REPORT_METHOD
    cloud_origin: str = "unspecified"

    def to_dict(self) -> dict:
        payload = {
            "schema": SCHEMA,
            "status": STATUS,
            "input_tier": self.input_tier,
            "source_files": list(self.source_files),
            "cloud_origin": self.cloud_origin,
            "method": self.method,
            "measurement_status": MEASUREMENT_STATUS,
            "accuracy_status": ACCURACY_STATUS,
            "disclaimer": DISCLAIMER,
            "capture": self.capture,
            "passes": self.passes,
            "fixes": list(self.fixes),
            "before": self.before,
            "after": self.after,
            "notes": list(self.notes),
        }
        if self.versus is not None:
            payload["versus"] = self.versus
        return payload


def _xz_span(pts: np.ndarray) -> list[float] | None:
    if pts.shape[0] < 2:
        return None
    return [
        float(pts[:, 0].max() - pts[:, 0].min()),
        float(pts[:, 1].max() - pts[:, 1].min()),
    ]


def _round_span(span: list[float] | None) -> list[float] | None:
    if span is None:
        return None
    return [round(span[0], 4), round(span[1], 4)]


def _polygon_vertices(polygon_xz: np.ndarray) -> list[list[float]]:
    if polygon_xz.shape[0] < 3:
        return []
    pts = polygon_xz.astype(np.float64)
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    return [[round(float(x), 4), round(float(z), 4)] for x, z in pts]


def floor_span(result: PlaneResult) -> list[float] | None:
    if result.floor is None:
        return None
    return _xz_span(result.floor.points[:, [0, 2]])


def polygon_exploded(result: PlaneResult) -> bool:
    span_f = floor_span(result)
    span_p = _xz_span(result.polygon_xz)
    if span_f is None or span_p is None:
        return False
    fx = max(span_f[0], 0.5)
    fz = max(span_f[1], 0.5)
    return (
        span_p[0] > EXPLODE_RATIO * fx
        or span_p[1] > EXPLODE_RATIO * fz
        or max(span_p) > MAX_SPAN_M
    )


def assess(result: PlaneResult) -> list[Finding]:
    findings: list[Finding] = []
    if result.floor is None:
        findings.append(Finding("missing_floor", "fail", "no horizontal floor plane"))
    elif result.polygon_xz.shape[0] < 3:
        findings.append(Finding("polygon_blocked", "warn", "too few polygon vertices"))
    elif polygon_exploded(result):
        span_p = _xz_span(result.polygon_xz)
        span_f = floor_span(result)
        assert span_p is not None and span_f is not None
        findings.append(
            Finding(
                "exploded_polygon",
                "fail",
                (
                    f"polygon span {span_p[0]:.1f} x {span_p[1]:.1f} m "
                    f"vs floor {span_f[0]:.1f} x {span_f[1]:.1f} m"
                ),
            )
        )
    if len(result.walls) < 3:
        findings.append(
            Finding("few_walls", "warn", f"{len(result.walls)} vertical wall(s); expected at least 3")
        )
    for i, a in enumerate(result.walls):
        for j in range(i + 1, len(result.walls)):
            b = result.walls[j]
            aligned, offset = wall_alignment(a, b)
            if aligned > 0.95 and offset < 0.45:
                findings.append(
                    Finding(
                        "duplicate_walls",
                        "warn",
                        f"wall{i} and wall{j} are {offset:.2f} m apart with aligned normals",
                    )
                )
    if result.height_m is None:
        findings.append(
            Finding("height_blocked", "info", "no same-scan ceiling; height left null")
        )
    elif not (HEIGHT_MIN_M <= result.height_m <= HEIGHT_MAX_M):
        findings.append(
            Finding(
                "height_implausible",
                "fail",
                f"height_m={result.height_m:.2f} outside {HEIGHT_MIN_M:.1f}–{HEIGHT_MAX_M:.1f} m",
            )
        )
    return findings


def _merge_duplicate_walls(walls: list[FittedPlane]) -> list[FittedPlane]:
    """Same gate as assess(), wider than the fitter's 0.30 m, so a near-miss pair collapses."""
    return _dedupe_walls(walls, normal_dot=0.95, offset_m=0.45, max_keep=8)


def repair(result: PlaneResult) -> tuple[PlaneResult, list[str]]:
    fixes: list[str] = []
    notes = list(result.notes)
    walls = list(result.walls)
    merged = _merge_duplicate_walls(walls)
    if len(merged) < len(walls):
        n_drop = len(walls) - len(merged)
        fixes.append(f"merged {n_drop} near-duplicate wall(s)")
        notes.append(f"fix: merged {n_drop} near-duplicate wall(s)")
        walls = merged

    polygon = result.polygon_xz
    if result.floor is not None and polygon_exploded(result):
        rebuilt, how = polygon_from_walls(result.floor, walls)
        candidate = PlaneResult(
            floor=result.floor,
            ceiling=result.ceiling,
            walls=walls,
            others=result.others,
            polygon_xz=rebuilt,
            height_m=result.height_m,
            height_p05_p95_m=result.height_p05_p95_m,
            n_points=result.n_points,
            n_downsampled=result.n_downsampled,
            openings=result.openings,
            notes=notes,
        )
        if polygon_exploded(candidate):
            hull = _convex_hull_xz(result.floor.points[:, [0, 2]])
            rebuilt = hull
            how = "floor convex hull"
        polygon = rebuilt
        fixes.append(f"replaced exploded outline with {how}")
        notes.append(f"fix: replaced exploded outline with {how}")

    updated = replace(result, walls=walls, polygon_xz=polygon, notes=notes)
    return updated, fixes


def _snapshot(result: PlaneResult, findings: list[Finding]) -> dict:
    height = None if result.height_m is None else round(float(result.height_m), 4)
    return {
        "n_walls": len(result.walls),
        "n_openings": len(result.openings),
        "height_m": height,
        "height_blocked": result.height_m is None,
        "height_p05_p95_m": (
            None
            if result.height_p05_p95_m is None
            else round(float(result.height_p05_p95_m), 4)
        ),
        "floor_span_xz_m": _round_span(floor_span(result)),
        "polygon_span_xz_m": _round_span(_xz_span(result.polygon_xz)),
        "polygon_xz_m": _polygon_vertices(result.polygon_xz),
        "findings": [f.to_dict() for f in findings],
    }


def fix_loop(result: PlaneResult, *, capture: str, max_passes: int = 3) -> ScanReport:
    before_findings = assess(result)
    current = result
    fixes: list[str] = []
    passes = 1
    for i in range(max_passes):
        passes = i + 1
        findings = assess(current)
        if not any(f.code in FIXABLE for f in findings):
            break
        signature = (len(current.walls), _xz_span(current.polygon_xz))
        current, applied = repair(current)
        if not applied:
            break
        fixes.extend(applied)
        if (len(current.walls), _xz_span(current.polygon_xz)) == signature:
            break
    after_findings = assess(current)
    return ScanReport(
        capture=capture,
        passes=passes,
        fixes=fixes,
        before=_snapshot(result, before_findings),
        after=_snapshot(current, after_findings),
        result=current,
    )


def _span_notes(a: ScanReport, b: ScanReport) -> list[str]:
    notes = ["heights are from each capture alone; Y is not mixed"]
    sa = a.after.get("floor_span_xz_m")
    sb = b.after.get("floor_span_xz_m")
    if not sa or not sb:
        return notes
    for axis, i in (("x", 0), ("z", 1)):
        lo = min(float(sa[i]), float(sb[i]))
        hi = max(float(sa[i]), float(sb[i]))
        if lo > 0.5 and hi > 2.5 * lo:
            notes.append(
                f"floor span on {axis} disagrees: {sa[i]:.1f} m vs {sb[i]:.1f} m"
            )
    return notes


def attach_versus(primary: ScanReport, other: ScanReport) -> ScanReport:
    primary.versus = {
        "capture": other.capture,
        "status": STATUS,
        "input_tier": other.input_tier,
        "source_files": list(other.source_files),
        "cloud_origin": other.cloud_origin,
        "method": other.method,
        "measurement_status": MEASUREMENT_STATUS,
        "accuracy_status": ACCURACY_STATUS,
        "passes": other.passes,
        "fixes": list(other.fixes),
        "before": other.before,
        "after": other.after,
    }
    primary.notes = _span_notes(primary, other)
    return primary


def write_report(report: ScanReport, out_dir: Path) -> Path:
    path = out_dir / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return path


def _finding_lines(findings: list[dict]) -> list[str]:
    if not findings:
        return ["  (none)"]
    return [f"  {f['severity']}: {f['code']} — {f['detail']}" for f in findings]


def report_text(report: ScanReport) -> str:
    after = report.after
    lines = [
        f"=== damage report: {report.capture} ===",
        "detector: reconstruction outline and planes, not physical room damage",
        f"passes: {report.passes}",
        f"walls: {after['n_walls']}  openings: {after['n_openings']}",
        f"height_m: {'BLOCKED' if after['height_blocked'] else after['height_m']}",
        f"height_p95_minus_p05_m (percentile check, not a plane): {after['height_p05_p95_m']}",
        f"floor_span_xz_m: {after['floor_span_xz_m']}",
        f"polygon_span_xz_m: {after['polygon_span_xz_m']}",
        "findings before:",
        *_finding_lines(report.before["findings"]),
        "findings after:",
        *_finding_lines(after["findings"]),
    ]
    if report.fixes:
        lines.append("fixes:")
        lines.extend(f"  {fix}" for fix in report.fixes)
    else:
        lines.append("fixes: none")
    for note in report.notes:
        lines.append(f"note: {note}")
    if report.versus is not None:
        other = report.versus
        oa = other["after"]
        lines.append(f"versus: {other['capture']}")
        lines.append(
            f"  walls: {oa['n_walls']}  height_m: "
            f"{'BLOCKED' if oa['height_blocked'] else oa['height_m']}  "
            f"height_p95_minus_p05_m: {oa['height_p05_p95_m']}  "
            f"polygon_span_xz_m: {oa['polygon_span_xz_m']}"
        )
        lines.append("  findings after:")
        lines.extend(f"  {row}" for row in _finding_lines(oa["findings"]))
    lines.append(DISCLAIMER)
    return "\n".join(lines)
