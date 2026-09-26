"""Thin wrapper so `python main.py ...` matches `python -m src ...`."""

from src.run import main

if __name__ == "__main__":
    raise SystemExit(main())
