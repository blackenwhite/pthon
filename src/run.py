"""CLI. Slices 1–7: inspect, cloud, planes, lidar export, drift, stills/video, damage report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.cloud import (
    build_cloud,
    cloud_text,
    read_ply,
    write_ply,
    write_preview_pngs,
)
from src.export import lidar_source_files, write_plan_exports
from src.ingest import inspect_text, load_capture
from src.media import export_stills, export_video, stills_text, video_text
from src.planes import fit_planes, planes_text, write_plan_preview
from src.posegraph import drift_text, refine_pose_graph, write_drift_preview
from src.report import attach_versus, fix_loop, report_text, write_report


def _cloud_out(capture: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return Path("out") / capture.resolve().name / "cloud.ply"


def _ensure_cloud(
    capture: Path,
    *,
    out: Path,
    frame_stride: int,
    pixel_stride: int,
    min_confidence: int,
    invert_extrinsics: bool,
    from_ply: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if from_ply and out.exists():
        points, colors = read_ply(out)
        return points, colors
    cap = load_capture(capture)
    result = build_cloud(
        cap,
        frame_stride=frame_stride,
        pixel_stride=pixel_stride,
        min_confidence=min_confidence,
        invert_extrinsics=invert_extrinsics,
    )
    write_ply(out, result.points, result.colors)
    print(cloud_text(result, out))
    return result.points, result.colors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cozmo room plan MVP")
    parser.add_argument("capture", type=Path, help="Record3D capture directory")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Load metadata and print stats (no reconstruction)",
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Subsample RGB-D + poses and write a metric PLY",
    )
    parser.add_argument(
        "--planes",
        action="store_true",
        help="RANSAC floor/walls/(ceiling) and write a 2D plan PNG",
    )
    parser.add_argument(
        "--tier",
        choices=["lidar"],
        default=None,
        help="Export plan.json + plan.svg from fitted planes (lidar = RGB-D + poses)",
    )
    parser.add_argument(
        "--ceiling",
        type=Path,
        default=None,
        help="Second capture used only for same-scan ceiling height (e.g. with_ceiling)",
    )
    parser.add_argument(
        "--from-ply",
        action="store_true",
        help="Reuse an existing cloud.ply instead of rebuilding",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="PLY path (default: out/<capture>/cloud.ply)",
    )
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.add_argument("--pixel-stride", type=int, default=4)
    parser.add_argument("--min-confidence", type=int, default=1)
    parser.add_argument(
        "--invert-extrinsics",
        action="store_true",
        help="Treat poses as world-from-camera inverse (debug inside-out clouds)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Write PNG orthographic views next to the PLY (open these in Preview)",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Pose-graph vs raw odometry ablation (drift.json + drift_path.png)",
    )
    parser.add_argument(
        "--stills",
        action="store_true",
        help="Write pose-aligned PNG stills from rgb.mp4 (stills/ + stills.json)",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Write a shorter mp4 from rgb.mp4 at --frame-stride (video.mp4 + video.json)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Damage check + fix loop; write report.json (outline/planes, not physical damage)",
    )
    parser.add_argument(
        "--versus",
        type=Path,
        default=None,
        help="Second capture to report beside --report (heights stay independent)",
    )
    args = parser.parse_args(argv)
    want_planes = args.planes or args.tier is not None
    if args.versus is not None and not args.report:
        parser.error("--versus requires --report")
    if (
        not args.inspect
        and not args.cloud
        and not args.preview
        and not want_planes
        and not args.drift
        and not args.stills
        and not args.video
        and not args.report
    ):
        parser.error(
            "pass --inspect, --cloud, --preview, --planes, --tier lidar, "
            "--drift, --stills, --video, and/or --report"
        )
    out = args.out
    wrote_cloud = False
    if args.inspect:
        cap = load_capture(args.capture)
        print(inspect_text(cap))
    if args.cloud:
        cap = load_capture(args.capture)
        out = out or Path("out") / cap.root.name / "cloud.ply"
        result = build_cloud(
            cap,
            frame_stride=args.frame_stride,
            pixel_stride=args.pixel_stride,
            min_confidence=args.min_confidence,
            invert_extrinsics=args.invert_extrinsics,
        )
        write_ply(out, result.points, result.colors)
        print(cloud_text(result, out))
        args.preview = True
        wrote_cloud = True
    if want_planes or args.report:
        ply = _cloud_out(args.capture, out)
        points, _colors = _ensure_cloud(
            args.capture,
            out=ply,
            frame_stride=args.frame_stride,
            pixel_stride=args.pixel_stride,
            min_confidence=args.min_confidence,
            invert_extrinsics=args.invert_extrinsics,
            from_ply=args.from_ply or wrote_cloud,
        )
        fitted = fit_planes(points)
        sources, origin = lidar_source_files(
            args.capture, ply=ply, from_ply=args.from_ply and not wrote_cloud
        )
        if args.report:
            report = fix_loop(fitted, capture=str(args.capture))
            report.source_files = sources
            report.cloud_origin = origin
            fitted = report.result
            if args.versus is not None:
                other_ply = Path("out") / args.versus.resolve().name / "cloud.ply"
                reuse_other = args.from_ply and other_ply.exists()
                o_stride = max(args.frame_stride, 24)
                o_pix = max(args.pixel_stride, 8)
                o_pts, _o_cols = _ensure_cloud(
                    args.versus,
                    out=other_ply,
                    frame_stride=o_stride,
                    pixel_stride=o_pix,
                    min_confidence=args.min_confidence,
                    invert_extrinsics=args.invert_extrinsics,
                    from_ply=reuse_other,
                )
                other_fit = fit_planes(o_pts)
                other = fix_loop(other_fit, capture=str(args.versus))
                other_sources, other_origin = lidar_source_files(
                    args.versus,
                    ply=other_ply,
                    from_ply=reuse_other,
                )
                other.source_files = other_sources
                other.cloud_origin = other_origin
                attach_versus(report, other)
                other_path = write_report(other, other_ply.parent)
                print(report_text(other))
                print(f"versus report JSON: {other_path.resolve()}")
            report_path = write_report(report, ply.parent)
            print(report_text(report))
            print(f"report JSON: {report_path.resolve()}")
        if want_planes:
            print(planes_text(fitted, label=str(args.capture)))
            plan_png = ply.parent / "preview_plan.png"
            write_plan_preview(points, fitted.polygon_xz, plan_png, openings=fitted.openings)
            print(f"plan PNG: {plan_png.resolve()}")
            if args.tier is not None:
                jpath, spath = write_plan_exports(
                    fitted,
                    ply.parent,
                    capture=str(args.capture),
                    tier=args.tier,
                    source_files=sources,
                    cloud_origin=origin,
                )
                print(f"plan JSON: {jpath.resolve()}")
                print(f"plan SVG: {spath.resolve()}")
        if args.ceiling is not None:
            ceil_ply = Path("out") / args.ceiling.resolve().name / "cloud.ply"
            # heavier subsample: this dump is ~9745 frames
            c_stride = max(args.frame_stride, 24)
            c_pix = max(args.pixel_stride, 8)
            c_pts, _c_cols = _ensure_cloud(
                args.ceiling,
                out=ceil_ply,
                frame_stride=c_stride,
                pixel_stride=c_pix,
                min_confidence=args.min_confidence,
                invert_extrinsics=args.invert_extrinsics,
                from_ply=False,
            )
            c_fit = fit_planes(c_pts)
            print(
                planes_text(
                    c_fit,
                    label=f"{args.ceiling} (height from this scan only; not fused with {args.capture})",
                )
            )
            c_png = ceil_ply.parent / "preview_plan.png"
            write_plan_preview(c_pts, c_fit.polygon_xz, c_png, openings=c_fit.openings)
            print(f"ceiling-scan plan PNG: {c_png.resolve()}")
            if args.tier is not None:
                c_sources, c_origin = lidar_source_files(
                    args.ceiling, ply=ceil_ply, from_ply=False
                )
                cj, cs = write_plan_exports(
                    c_fit,
                    ceil_ply.parent,
                    capture=str(args.ceiling),
                    tier=args.tier,
                    source_files=c_sources,
                    cloud_origin=c_origin,
                )
                print(f"ceiling-scan JSON: {cj.resolve()}")
                print(f"ceiling-scan SVG: {cs.resolve()}")
    if args.preview:
        ply = out or Path("out") / args.capture.resolve().name / "cloud.ply"
        if not ply.exists():
            parser.error(f"no PLY at {ply}; run --cloud first")
        points, colors = read_ply(ply)
        pngs = write_preview_pngs(points, colors, ply.parent)
        print("PNG previews (open these in Preview, not the .ply):")
        for p in pngs:
            print(f"  {p.resolve()}")
    if args.drift:
        cap = load_capture(args.capture)
        ply = _cloud_out(args.capture, out)
        drift_dir = ply.parent
        d_stride = max(args.frame_stride, 24)
        d_pix = max(args.pixel_stride, 8)
        drift = refine_pose_graph(
            cap,
            frame_stride=d_stride,
            pixel_stride=d_pix,
            min_confidence=args.min_confidence,
            invert_extrinsics=args.invert_extrinsics,
        )
        print(drift_text(drift, label=str(args.capture)))
        jpath = drift_dir / "drift.json"
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(drift.to_dict(), indent=2) + "\n")
        png = write_drift_preview(drift, drift_dir / "drift_path.png")
        print(f"drift JSON: {jpath.resolve()}")
        print(f"drift PNG: {png.resolve()}")
    if args.stills or args.video:
        cap = load_capture(args.capture)
        media_dir = _cloud_out(args.capture, out).parent
        if args.stills:
            stills = export_stills(cap, media_dir, frame_stride=args.frame_stride)
            print(stills_text(stills))
            if not stills.stills:
                print(f"no stills written; could not read frames from {cap.root / 'rgb.mp4'}")
                return 1
        if args.video:
            clip = export_video(cap, media_dir, frame_stride=args.frame_stride)
            print(video_text(clip))
            if clip.n_frames == 0:
                print(f"no video written; could not read frames from {cap.root / 'rgb.mp4'}")
                return 1
            print(f"video MP4: {clip.path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
