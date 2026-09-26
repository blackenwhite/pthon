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

## Slice 6 — stills and video from `rgb.mp4`

Same color video the cloud already reads. `--stills` writes pose-aligned PNGs. `--video` writes a shorter mp4 at `--frame-stride` (playback fps is source fps divided by the stride, so duration stays about the same). Neither measures the room.

```bash
python -m src store/c00a170fe1 --stills --video --frame-stride 12
```

Outputs under `out/c00a170fe1/`: `stills/*.png`, `stills.json`, `video.mp4`, `video.json`.

## Slice 7 — damage report and fix loop

Checks the reconstruction, not the room: an outline that runs away from the floor, near-duplicate walls, a missing floor, or a height outside 1.6–4.5 m. A short loop replaces an exploded outline (floor hull if the wall lines are still wild) and merges walls closer than 0.45 m with aligned normals. `height_m` stays null when that capture has no ceiling. Two folders are never fused.

```bash
python -m src store/1a8384c3f6 --report --versus store/c7d28f72c6 --frame-stride 24 --pixel-stride 8
```

Writes `out/1a8384c3f6/report.json` and `out/c7d28f72c6/report.json`. `height_m` is `BLOCKED` unless that capture’s own cloud has a ceiling plane. The p95−p05 figure in the report is a percentile check, not a ceiling measurement.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

Fast tests use a tiny synthetic capture. Tests marked with the founder dump skip if `store/c00a170fe1` is missing.
