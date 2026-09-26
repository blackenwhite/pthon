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

## Current baseline

Frozen at commit `405fee95` (`iteration 1 procceding plan`), before any algorithm change from `PROCEEDING_PLAN.md`. The write-up is [BASELINE_REPORT.md](BASELINE_REPORT.md).

Both captures produce a floor and a plan export in about two seconds from an existing PLY. `height_m` stays blocked, including on the look-up capture. Wall and opening counts change between repeats. Both footprints fell back to a wall-inlier hull. Fit residuals are not tape accuracy.

`rmse_m` and `median_residual_m` are RANSAC fit error against the point cloud, not tape-measure or laser accuracy. `height_m` is null on both captures below. The `height_p95_minus_p05_m` figure is a percentile check, not a ceiling measurement.

This freeze reused the clouds already in `out/` (that directory is gitignored):

| Cloud | Points | File |
|---|---:|---|
| `out/c00a170fe1/cloud.ply` | 428,817 | 6.1 MB, built earlier with the default strides |
| `out/c7d28f72c6/cloud.ply` | 297,183 | 4.3 MB, built earlier with `--frame-stride 24 --pixel-stride 8` |

Rebuild those clouds from `store/`, then export the plan:

```bash
python -m src store/c00a170fe1 --cloud
python -m src store/c7d28f72c6 --cloud --frame-stride 24 --pixel-stride 8
python -m src store/c00a170fe1 --inspect
python -m src store/c00a170fe1 --tier lidar --from-ply
python -m src store/c7d28f72c6 --inspect
python -m src store/c7d28f72c6 --tier lidar --from-ply --frame-stride 24 --pixel-stride 8
```

The stride flags only change a cloud rebuild. `--from-ply` fits planes on the existing PLY.

Stdout and wall-clock time are in `baseline/logs/`. One preserved plan JSON, SVG, and PNG per capture is in `baseline/examples/`. A rerun produces the same kind of files. It does not reproduce the same wall count: Open3D `segment_plane` (used for the floor) is unseeded, so two back-to-back exports of `c00a170fe1` on the same PLY returned 7 walls / 3 openings / 11 polygon vertices, then 6 / 2 / 16. Both runs left `height_m` blocked and fell back to a wall-inlier convex hull because wall-line intersections were degenerate.

Preserved sample (the second `c00a170fe1` run, and the single `c7d28f72c6` run):

| | `c00a170fe1` | `c7d28f72c6` |
|---|---:|---:|
| Role | single room | look-up / ceiling capture |
| Poses | 1,715 over 37.17 s, path 14.49 m | 9,745 over 214.93 s, path 99.76 m |
| Points / downsampled | 428,817 / 192,770 | 297,183 / 250,000 |
| Walls | 6 | 6 |
| Openings | 2 (window, unknown) | 2 (door, door) |
| Polygon | 16 vertices, hull fallback, span 5.48 × 7.18 m | 10 vertices, hull fallback, span 11.73 × 14.88 m |
| Height | BLOCKED | BLOCKED |
| Floor fit RMSE | 0.023 m | 0.023 m |
| Inspect runtime | 0.76 s | 0.98 s |
| Plan-export runtime | 1.69 s on the first run; the preserved sample is a second export and was not timed | 1.90 s |

`c7d28f72c6` still has no ceiling plane. The highest leftover horizontal was 0.64 m above the floor. Its footprint span is larger than the camera path (pose span about 8.3 × 9.1 m), which is the unstable polygon this baseline is recording, not a measured room size.

## Provenance

Each `plan.json` and `report.json` now states its own tier and limits. The console is not required to see that this export is an uncalibrated LiDAR estimate. The fields exist so a later photo or video run cannot be mistaken for a surveyed measurement.

| Field | Value on this baseline | Why it is there |
|---|---|---|
| `status` | `baseline` | Marks the frozen RGB-D result, before a later geometry fix |
| `input_tier` | `lidar` | Only the RGB-D path produces a metric plan. `tier` is the same value |
| `source_files` | Capture files, or the PLY when `--from-ply` | Names what this run actually read |
| `cloud_origin` | `rebuilt_this_run` or `existing_ply` | `--from-ply` does not reopen depth, confidence, RGB, or odometry |
| `method` | `ransac_rgb_d_planes` on plans; `reconstruction_health_check` on reports | Names the algorithm that produced the file |
| `measurement_status` | `estimated` | The geometry is a fit to the cloud |
| `accuracy_status` | `not_calibrated` | Nothing here was checked against a tape or a laser |

`imu.csv` sits in every Record3D folder and is required for the folder to load. Plane fitting does not read it, so it is not listed in `source_files`. The existing disclaimer stays: `rmse_m` is fit error, openings are occupancy-gap heuristics, and `height_m` is null unless that same capture has a ceiling plane. The SVG caption repeats `status`, `tier`, and `accuracy`.

```bash
python -m src store/c00a170fe1 --tier lidar --from-ply
```

With `--from-ply`, `source_files` is the existing `cloud.ply` and `cloud_origin` is `existing_ply`. A run without `--from-ply` lists `camera_matrix.csv`, `odometry.csv`, `rgb.mp4`, `depth`, and `confidence` under the capture, and sets `cloud_origin` to `rebuilt_this_run`. `--report` writes the same provenance block on `report.json`. A `--versus` capture gets its own block on its report and inside the primary report’s `versus` object.

Wall and opening counts still change between repeats because Open3D floor RANSAC is unseeded. Photo and video adapters are not wired yet. Derived inputs for those tiers now live under `benchmark/` (see below).

## Derived benchmark inputs

There is no iPhone 15+ capture and no private-room data. One command rebuilds logical photo, video, and LiDAR folders from the supplied Record3D dumps in `store/`:

```bash
python -m src.benchmark
```

or `python scripts/prepare_benchmark.py`. Default output is `benchmark/`. `--store` and `--out` override the paths. `--max-stills` (default 12) sets how many PNG frames are decoded per room.

| Logical tier | Path | What it actually is |
|---|---|---|
| Photo | `benchmark/photo/room_01/*.png` | Frames decoded from that capture’s `rgb.mp4`. Not independent native photographs, not screen captures. |
| Video | `benchmark/video/room_01.mp4` | A byte-for-byte copy of the same `rgb.mp4`. Not a separate camera recording. |
| LiDAR | `benchmark/lidar/room_01/` | A copy of the Record3D folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `imu.csv`, `camera_matrix.csv`). |

Room ids are stable: `c00a170fe1` → `room_01` (single room), `c7d28f72c6` → `room_02` (look-up / ceiling), `1a8384c3f6` → `room_03` (report dump). `benchmark/manifest.json` repeats that the stills are derived RGB frames. Generated `photo/`, `video/`, `lidar/`, and the manifest are gitignored; `benchmark/README.md` is the committed regeneration note.

This step does not reconstruct from photo or video. Metric plans still come from the LiDAR/RGB-D path. The next step is explicit adapters that enforce those file boundaries.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

Fast tests use a tiny synthetic capture. Tests marked with the founder dump skip if `store/c00a170fe1` is missing.
