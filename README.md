# Cozmo room plan MVP

Founder Record3D dumps live in `store/`. No new capture.

## Slice 1 — inspect

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src store/c00a170fe1 --inspect
```

## Slice 2 — metric cloud

Subsample RGB-D + poses, write a colored PLY (depth mm→m, K scaled to 256×192):

```bash
python -m src store/c00a170fe1 --cloud
```

Default output: `out/c00a170fe1/cloud.ply`. macOS Preview cannot display a points-only PLY (it looks black). Open the PNGs instead:

```bash
python -m src store/c00a170fe1 --preview
open out/c00a170fe1/preview_top_rgb.png out/c00a170fe1/preview_top_height.png out/c00a170fe1/preview_side_rgb.png
```

`--cloud` also writes those PNGs. For a 3D tumble, use [MeshLab](https://www.meshlab.net/) or [CloudCompare](https://www.danielgm.net/cc/) — not Preview. If the cloud looks inside-out, rerun with `--invert-extrinsics`.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

Fast tests use a tiny synthetic capture. Tests marked with the founder dump skip if `store/c00a170fe1` is missing.
