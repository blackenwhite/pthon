"""CLI. Slice 1: --inspect only."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.ingest import inspect_text, load_capture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Cozmo room plan MVP")
    parser.add_argument("capture", type=Path, help="Record3D capture directory")
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Load metadata and print stats (no reconstruction)",
    )
    args = parser.parse_args(argv)
    if not args.inspect:
        parser.error("Slice 1 only supports --inspect")
    cap = load_capture(args.capture)
    print(inspect_text(cap))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
