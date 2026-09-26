# Derived benchmark inputs

Logical photo, video, and LiDAR folders are generated from `store/`. Nothing here is a private room scan or a native iPhone photograph.

Regenerate:

```bash
python -m src.benchmark
```

or:

```bash
python scripts/prepare_benchmark.py
```

That writes `photo/`, `video/`, `lidar/`, and `manifest.json`. Those generated files are gitignored; this README is the committed layout note. The manifest states that PNG stills are decoded from `rgb.mp4`.
