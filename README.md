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

RANSAC planes: voxel downsample, then a seeded numpy sampler for the floor, ceiling, and walls. Open3D `segment_plane` is only the fallback when that sampler finds no horizontal plane. It has no seed, so a refit used to change which points were left for the walls. Walls stay orientation-constrained because Open3D’s densest plane on these scans is furniture. Floor and walls from `single_room`. **Ceiling height only from the same scan that looked up** (`single_scan_with_ceiling`) — do not mix Y from two folders. After the vertical planes are fit, a furniture filter drops tilted slabs and near-parallel copies and writes the reason on each rejected plane. See [Wall filtering](#wall-filtering).

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

Repeating a plan export on the same PLY now keeps the same wall count. The floor sampler is seeded. The frozen baseline above was not: Open3D `segment_plane` had no seed, so `c00a170fe1` came out as 7 walls and then 6.

## Input adapters

`--tier` names an adapter with hard file boundaries. The LiDAR path is unchanged. Photo and video select non-metric keyframes and a contact-sheet preview; they do not invent metres.

```bash
python -m src store/c00a170fe1 --tier lidar --from-ply
python -m src benchmark/photo/room_01 --tier photo
python -m src benchmark/video/room_01.mp4 --tier video
python -m src benchmark/photo/room_01 --inspect
```

### Final device / tier matrix

| Tier | Allowed inputs | Forbidden (listed as `refused`, not opened) | Metric reconstruction | RGB-tier outputs |
|---|---|---|---|---|
| `lidar` | `depth/`, `confidence/`, `camera_matrix.csv`, `odometry.csv`, `rgb.mp4` | `imu.csv` is required to recognise the folder and is not used for fitting | Available (`ransac_rgb_d_planes`, still uncalibrated) | Plan JSON/SVG/PNG from RGB-D planes |
| `video` | One RGB video (`rgb.mp4` or a `.mp4` path) | `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`, `imu.csv` | Blocked (`status=blocked`, `measurement_status=not_implemented`, `confidence_status=low`) | Up to 12 evenly spaced keyframes, `keyframes_preview.png`, empty geometry plan |
| `photo` | Top-level stills (`.png` / `.jpg`) | `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`, `imu.csv`, `rgb.mp4`. Depth PNGs inside `depth/` are not stills. | Blocked, same honesty contract as video | Same keyframe + contact-sheet contract over sorted stills |

`--inspect` without `--tier` classifies the path (Record3D folder, photo directory, or video file). `--tier photo` or `--tier video` writes under `out/<tier>_<name>/`:

- `keyframes/*.png` — selected frames (stills copied or RGB-only video decode)
- `keyframes_preview.png` — shared non-metric contact sheet
- `plan.json` / `plan.svg` — `method=rgb_keyframes_non_metric`, empty walls/polygon/`height_m`, plus `keyframes`, `visual_preview`, `confidence_status=low`, and an `uncertainty` note

Keyframe rule: at most 12 evenly spaced indices. For video, an explicit `--frame-stride` thins frames first, then the 12-cap applies; without that flag the default LiDAR stride is ignored so spacing stays even. Combining photo/video with `--cloud`, `--planes`, `--from-ply`, `--drift`, or `--report` is rejected so a PLY cannot smuggle LiDAR into an RGB run.

Derived stills under `benchmark/` remain decoded `rgb.mp4` frames, not native photographs. On the prepared rooms: `benchmark/photo/room_01` → 12 keyframes / 12 stills; `benchmark/video/room_01.mp4` → 12 keyframes from a claimed 1715-frame clip (last readable index 1713 after OpenCV frame-count overestimate). Example artifacts: `out/photo_room_01/keyframes_preview.png`, `out/video_room_01/keyframes_preview.png`.

## Wall filtering

Vertical RANSAC was treating furniture and a second copy of the same wall as extra walls. A cabinet face is vertical, dense, and a few tens of centimetres in front of the real wall. A tilted plane through clutter also clears the old “mostly vertical” cutoff. The filter in `src/walls.py` runs after that sampler. It does not delete a plane quietly: every drop is a `rejected_walls` entry on `plan.json` and a `rejected wall:` note, with a reason code. A plane that is only mildly tilted, or that sits inside the floor hull, stays in `walls` with `confidence` `low`.

Rules, in order:

- `tilted_plane` — the normal is more than about 16° off perpendicular to the floor (`|n · floor| > 0.28`).
- `short_vertical_extent` — the inliers’ height core (10th to 90th percentile above the floor) is under 0.85 m.
- `short_span` — the along-wall core is under 0.70 m.
- `fragmented_support` / `sparse_support` — the body of the plane does not form a continuous run of at least 1 m.
- `interior_patch` — a short plane whose points sit inside the floor hull rather than on its edge.
- `near_parallel_duplicate` — same horizontal direction (normal agreement above 0.97) and less than 0.55 m away. The more upright plane is kept; if they are equally upright, the one closer to the floor boundary is kept.
- `off_axis_interior` — a short interior plane that does not lie on one of the two dominant room directions.

On the frozen clouds the rules that actually fired were `tilted_plane` and `near_parallel_duplicate`. The other codes are covered by synthetic tests (a short cabinet, an inner parallel face, a steep plane). Two back-to-back exports of the same PLY now return the same wall inlier counts. The floor fit that feeds this filter is the seeded sampler, so the leftover cloud no longer changes between runs. Open3D is still the fallback if that sampler finds no horizontal plane.

```bash
python -m src store/c00a170fe1 --tier lidar --from-ply
python -m src store/c7d28f72c6 --tier lidar --from-ply --frame-stride 24 --pixel-stride 8
```

| | Frozen baseline | After this filter (same count on a second run) |
|---|---|---|
| `c00a170fe1` walls | 7, then 6 | 4 (2 of them `confidence=low`) |
| `c00a170fe1` rejected | not recorded | 6 (1 `tilted_plane`, 5 `near_parallel_duplicate`) |
| `c00a170fe1` openings | 3, then 2 | 2 |
| `c00a170fe1` polygon | 11 then 16 vertices, span 5.48 × 7.18 m, wall-inlier hull | 7 vertices, span 6.31 × 7.44 m, still the wall-inlier hull. Floor span on this run is 6.49 × 6.79 m |
| `c7d28f72c6` walls | 6 | 5 (3 of them `confidence=low`) |
| `c7d28f72c6` rejected | not recorded | 5, all `near_parallel_duplicate` |
| `c7d28f72c6` openings | 2 | 2 |
| `c7d28f72c6` polygon | 10 vertices, span 11.73 × 14.88 m, wall-inlier hull | 11 vertices, span 11.61 × 15.62 m, still the wall-inlier hull. Floor span on this run is 11.62 × 15.12 m |
| Height | BLOCKED on both | BLOCKED on both |
| Floor fit RMSE | 0.023 m | 0.023 m |

`height_m` was still null after the wall filter, including on the look-up capture. The polygon is still the wall-inlier hull because wall-line intersections are degenerate. `c7d28f72c6`’s footprint is still larger than that capture’s camera path (about 8.3 × 9.1 m). Fit residuals are still not tape accuracy. Ceiling search is the next section.

## Ceiling detection

The old search took one horizontal plane from the points left after the walls, above the 70th percentile of the whole cloud. On a floor-heavy scan that percentile sits near furniture, so the one shot locked onto a low plane and stopped. Before this change that plane was 1.13 m above the floor on `c00a170fe1` and 0.67 m on `c7d28f72c6`, and `height_m` stayed null.

`src/ceiling.py` now searches the downsampled cloud at least 1.45 m above the floor, up to six horizontal candidates, with the same seeded sampler as the floor. A candidate is kept only when all of these hold:

- height is between 1.6 m and 4.5 m (the same gate the reconstruction report already uses);
- at least 80 inliers, and the inliers’ core span (10th to 90th percentile) is at least 1.2 m in both X and Z;
- those inliers fill at least 45% of that span, so a small patch plus a ring of wall points does not count;
- the band 0.12–0.45 m above the plane has fewer than half as many points as the plane itself (`occupied_above` when the ratio is under 2);
- the plane is denser than the band just below it, so a slice through a volume of points does not count.

Every drop is a `rejected_ceilings` entry on `plan.json` and a `rejected ceiling:` note. Nothing fills in 2.4 m. Wall counts are unchanged, because this search does not remove points from the wall fit.

On the frozen clouds the search still blocks height. The best planes that clear 1.6 m still have points continuing above them (above-ratio about 0.3–0.8, need 2). A second run of each cloud repeats the same rejection codes.

| | Before this search | After (same codes on a second run) |
|---|---|---|
| `c00a170fe1` `height_m` | BLOCKED (leftover plane 1.13 m) | BLOCKED. 6 rejected: 2 `too_low`, 2 `occupied_above`, 2 `sparse_coverage`. Highest candidate 2.12 m, above-ratio 0.58, dropped as `sparse_coverage` |
| `c7d28f72c6` `height_m` | BLOCKED (leftover plane 0.67 m) | BLOCKED. 6 rejected: 2 `too_low`, 4 `occupied_above`. Highest plane 2.07 m, above-ratio 0.37 |
| Walls | 4 and 5 | 4 and 5, same low-confidence counts |

The look-up capture does not contain a sheet with empty space above it. The points above 1.6 m keep going; they are not a ceiling plane. The polygon and the fit residuals were unchanged by this search. Footprint bounds are the next section.

## Footprint

Wall-line corners were allowed to form an outline up to four times the floor box, and a vertex only had to land inside a wide margin. On `c7d28f72c6` that kept the wall-inlier hull, whose Z span (15.62 m) stuck out past the floor points (15.12 m). Sorting those corners around their centre also hid a crossed loop by reordering it into a simple shape.

`polygon_from_walls` now tries three outlines, in order, and keeps the first that passes:

1. `wall_lines` — corners where neighbouring wall lines meet the floor, in wall-angle order. `output_quality` is `ok`.
2. `wall_inlier_hull` — convex hull of the wall inliers. `output_quality` is `warning`.
3. `floor_hull` — convex hull of the floor inliers. This is the bounded fallback. `output_quality` is `warning`.

A candidate is kept only when it has at least three corners, its edges do not cross, its box covers at least a quarter of the floor box, and every corner sits within 0.5 m of the floor points’ X and Z range (`FOOTPRINT_MARGIN_M`). A corner tens of metres away is still dropped before that test. With no floor, the outline is the full-cloud hull (`cloud_hull`, `warning`). Too few vertices sets `output_quality` to `blocked`.

`plan.json` records `footprint_method` and `output_quality`. The same choice is in the `polygon from …` note, including why a candidate was skipped. The plan SVG title repeats both fields. The repair loop uses the same three-way choice when it replaces an exploded outline.

```bash
python -m src store/c00a170fe1 --tier lidar --from-ply
python -m src store/c7d28f72c6 --tier lidar --from-ply --frame-stride 24 --pixel-stride 8
```

A second export of each cloud repeated the method, the span, and the note.

| | After ceiling search | After this gate |
|---|---|---|
| `c00a170fe1` polygon | 7 vertices, span 6.31 × 7.44 m, wall-inlier hull. Floor span 6.49 × 6.79 m | Same 7 vertices and span. `footprint_method=wall_inlier_hull`, `output_quality=warning`. Wall lines were outside the 0.5 m margin |
| `c7d28f72c6` polygon | 11 vertices, span 11.61 × 15.62 m, wall-inlier hull. Floor span 11.62 × 15.12 m | 17 vertices, span 11.62 × 15.12 m, matching the floor. `footprint_method=floor_hull`, `output_quality=warning`. Wall lines self-intersect; the wall hull is outside the 0.5 m margin |
| Walls | 4 and 5 | 4 and 5, same low-confidence counts |
| Height | BLOCKED | BLOCKED |

`c00a170fe1`’s Z span is still 0.65 m longer than the floor span, which the 0.5 m-per-side rule allows. `c7d28f72c6`’s floor hull is still larger than that capture’s camera path (about 8.3 × 9.1 m). The outline follows the reconstructed floor points. Fit residuals are still not tape accuracy.

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

This step does not reconstruct from photo or video. Metric plans still come from the LiDAR/RGB-D path. Adapters for all three tiers are in `src/adapters.py`.

## Runtime and determinism

Step 8 records run metadata on every JSON plan or reconstruction report
written by the LiDAR path. The `run` object includes the capture and tier,
frame and pixel strides, confidence threshold, point counts, wall/opening
counts, phase timings, output paths, and the seeds used by the deterministic
numpy samplers. `open3d_segment_plane` is explicitly labelled
`unseeded_fallback`; it is only used when the seeded horizontal sampler cannot
find a plane. Each run also contains a `determinism_key`, which excludes
timings and output paths for direct repeat comparison.

Repeat a plan export from an existing cloud:

```bash
python -m src store/c00a170fe1 --tier lidar --from-ply
python -m src store/c00a170fe1 --tier lidar --from-ply
```

Compare the two `run.determinism_key` objects directly. The configuration and
geometry counts should match; timings are machine-dependent and are recorded
for comparison rather than treated as a quality metric.

The reproducibility run is preserved in `step8/logs/`:

| Capture | Tier | Input | Points | Retained | Walls | Openings | Phases |
|---|---|---|---:|---:|---:|---:|---|
| `c00a170fe1` | LiDAR | existing PLY | 428,817 | 192,770 | 4 | 2 | `planes=3.586–5.385s`, `export=0.084–0.408s` |
| `c7d28f72c6` | LiDAR | existing PLY | 297,183 | 250,000 | 5 | 2 | `planes=4.935–5.056s`, `export=0.201–0.241s` |

Exact timing values vary by machine. The JSON metadata is the source of truth
for each run.

## Step 9 fix-loop report

The reproducible wall-filtering experiment is documented in
[FIX_LOOP_REPORT.md](FIX_LOOP_REPORT.md). The frozen baseline returned 7 then 6
walls for `c00a170fe1`; the post-fix pipeline repeats at 4 walls and records
six rejected candidates with reason codes. `c7d28f72c6` repeats at 5 walls
with five duplicate-plane rejections. Both captures still correctly report
blocked ceiling height. The report separates these wall-filter results from
the later ceiling and footprint changes and links the reproduction commands,
logs, outputs, and remaining limitations.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

Fast tests use a tiny synthetic capture. Tests marked with the founder dump skip if `store/c00a170fe1` is missing.
