"""CLI. Slice 2: --inspect and --cloud."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cloud import (
    build_cloud,
    cloud_text,
    read_ply,
    write_ply,
    write_preview_pngs,
)
from src.ingest import inspect_text, load_capture


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
    args = parser.parse_args(argv)
    if not args.inspect and not args.cloud and not args.preview:
        parser.error("pass --inspect, --cloud, and/or --preview")
    out = args.out
    if args.cloud or args.inspect:
        cap = load_capture(args.capture)
        if args.inspect:
            print(inspect_text(cap))
        if args.cloud:
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
    if args.preview:
        ply = out or Path("out") / args.capture.resolve().name / "cloud.ply"
        if not ply.exists():
            parser.error(f"no PLY at {ply}; run --cloud first")
        points, colors = read_ply(ply)
        pngs = write_preview_pngs(points, colors, ply.parent)
        print("PNG previews (open these in Preview, not the .ply):")
        for p in pngs:
            print(f"  {p.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
