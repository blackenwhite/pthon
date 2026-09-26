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

## Slice 3 — floor, walls, height

RANSAC planes: Open3D voxel downsample + `segment_plane` for floor/ceiling; orientation-constrained numpy RANSAC for walls (Open3D always returns the densest plane, which in this scan is furniture). Floor and walls from `single_room`. **Ceiling height only from the same scan that looked up** (`single_scan_with_ceiling`) — do not mix Y from two folders.

```bash
python -m src store/c00a170fe1 --planes
python -m src store/c00a170fe1 --planes --ceiling store/c7d28f72c6
python -m src store/c7d28f72c6 --planes --frame-stride 24 --pixel-stride 8
```

Open `out/<id>/preview_plan.png`. Stdout reports RMSE / median residual; that is fit error, not tape-measure accuracy. If there is no ceiling plane the CLI prints `BLOCKED` instead of inventing 2.4 m.

Reuse a PLY you already built:

```bash
python -m src store/c00a170fe1 --planes --from-ply
```

## Slice 4 — JSON + SVG

One command fits the same planes and writes `plan.json` + `plan.svg` next to the PLY. Height is `null` unless a ceiling plane exists **on that capture**. Residuals in JSON are fit error, not tape.

```bash
python -m src store/c00a170fe1 --tier lidar
python -m src store/c00a170fe1 --tier lidar --from-ply
python -m src store/c7d28f72c6 --tier lidar --frame-stride 24 --pixel-stride 8
```

Open `out/<id>/plan.svg` in a browser or Preview. `--ceiling` still reports the second scan separately (not fused into the first JSON).

## Slice 5 — openings + drift ablation

Openings are interior occupancy gaps on fitted walls (heuristic). They show up on `--planes` / `--tier lidar` in stdout, `plan.json`, `plan.svg`, and `preview_plan.png` (green door / orange window).

```bash
python -m src store/c00a170fe1 --tier lidar --from-ply
```

Pose-graph vs raw ARKit odometry (sequential ICP + nearby loop closures). Writes `drift.json` and `drift_path.png`:

```bash
python -m src store/c00a170fe1 --drift --from-ply
python -m src store/c00a170fe1 --tier lidar --drift --from-ply
```

This is an ablation, not a claim that drift is solved. Openings are not a trained door detector.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

Fast tests use a tiny synthetic capture. Tests marked with the founder dump skip if `store/c00a170fe1` is missing.
