# Step 9 — Reproducible geometry fix loop

## Scope and claim boundary

This report documents one reconstruction-geometry fix loop: reducing false and
unstable wall planes in the supplied RGB-D/LiDAR captures. It is not a report
of physical room damage, a trained door/window detector, or tape-measure
accuracy. Plane residuals are fit errors against the point cloud.

The experiment uses the frozen baseline from commit `405fee95` and the current
post-fix pipeline. The wall-count comparison isolates the wall-selection
change; later ceiling and footprint changes are reported separately rather
than attributed to the wall filter.

## Baseline failure

The frozen implementation extracted vertical planes with insufficient
geometric discrimination. Furniture faces, tilted surfaces, and near-parallel
copies of a real wall could remain in the wall list. The floor fit used
Open3D's unseeded `segment_plane` path, so the points left for subsequent wall
fitting could also differ between repeated exports.

The frozen `c00a170fe1` export demonstrates both problems:

- first run: 7 walls, 3 openings, 11 polygon vertices;
- repeat run: 6 walls, 2 openings, 16 polygon vertices;
- both runs reported `height_m: null`;
- neither run recorded why individual vertical planes were rejected.

The baseline `c7d28f72c6` export produced 6 walls and a 10-vertex wall-inlier
hull. Its highest leftover horizontal plane was only 0.64 m above the floor,
so height was correctly blocked, but the wall and outline evidence was not
stable enough to support a clean room boundary.

The preserved baseline commands were:

```bash
python -m src store/c00a170fe1 --tier lidar --from-ply
python -m src store/c7d28f72c6 --tier lidar --from-ply \
  --frame-stride 24 --pixel-stride 8
```

The original outputs and logs are preserved in
[`baseline/`](/Users/nabajyotimajumdar/F/Pthon/baseline).

## Root-cause hypothesis

The failure was not primarily a lack of RANSAC support. The point clouds
contain many dense vertical surfaces, so a density-only vertical-plane
extractor can select:

1. a furniture face in front of a wall;
2. a tilted slab that is only approximately vertical;
3. multiple observations of the same wall separated by furniture depth; or
4. a short interior patch that does not define the room boundary.

The unseeded floor selection made this worse: removing a different floor set
changed the remaining cloud and could change later wall selections. The
working hypothesis was therefore that deterministic sampling plus conservative
wall evidence checks would reduce false positives while preserving
uncertain-but-plausible walls as low-confidence results.

## Shipped fix

The fix is implemented in [`src/walls.py`](/Users/nabajyotimajumdar/F/Pthon/src/walls.py)
and orchestrated by [`src/planes.py`](/Users/nabajyotimajumdar/F/Pthon/src/planes.py).

- Horizontal and vertical numpy RANSAC uses fixed seeds; Open3D is only an
  explicitly labelled fallback.
- Wall candidates are checked for tilt, vertical extent, along-wall span,
  continuous support, occupancy, and interior placement.
- Near-parallel candidates within 0.55 m are compared and the more
  wall-like candidate is retained.
- Off-axis interior patches are rejected when they do not match dominant room
  directions.
- Mildly uncertain candidates are retained with `confidence: "low"` rather
  than silently discarded.
- Every rejected candidate is exported with a reason code and details in
  `plan.json`.

The synthetic regression coverage is in
[`tests/test_walls.py`](/Users/nabajyotimajumdar/F/Pthon/tests/test_walls.py)
and includes short cabinets, inner parallel faces, tilted planes, low
confidence walls, and repeated exports of the same frozen PLY.

## Prediction

Before measuring the post-fix runs, the expected result was:

- fewer wall candidates caused by furniture and duplicate faces;
- identical wall counts and inlier counts on repeated exports of one PLY;
- explicit rejection provenance;
- no invented ceiling height when the cloud lacks a defensible ceiling plane.

The fix was not expected to calibrate metric accuracy, create a ceiling plane,
or solve multi-room stitching.

## Measured post-fix result

The Step 8 repeated exports in
[`step8/logs/`](/Users/nabajyotimajumdar/F/Pthon/step8/logs) produced the
following stable geometry results:

| Capture | Frozen walls | Post-fix walls | Rejected walls | Openings | Height |
|---|---:|---:|---:|---:|---|
| `c00a170fe1` | 7, then 6 | 4 | 6 | 2 | blocked |
| `c7d28f72c6` | 6 | 5 | 5 | 2 | blocked |

For `c00a170fe1`, the six rejected planes were one `tilted_plane` and five
`near_parallel_duplicate` candidates. Two of the four retained walls were
marked low confidence. For `c7d28f72c6`, all five rejected planes were
`near_parallel_duplicate` candidates, and three of the five retained walls
were marked low confidence.

The repeated post-fix runs also kept the retained point counts stable:

- `c00a170fe1`: 192,770 downsampled points, 4 walls, 2 openings;
- `c7d28f72c6`: 250,000 downsampled points, 5 walls, 2 openings.

The `run.determinism_key` in the generated JSON excludes machine-dependent
timings and output paths, so it is the intended comparison object for repeat
runs. The exact phase timings remain recorded for performance context.

## Separating later changes

The current pipeline also contains the Step 6 ceiling search and Step 7
footprint gate. Those changes are not counted as part of the wall-filter
claim:

- both captures still have `height_m: null`, because no same-capture ceiling
  passed the evidence checks;
- `c00a170fe1` uses a bounded `wall_inlier_hull` with warning quality;
- `c7d28f72c6` now falls back to a bounded `floor_hull` because its wall-line
  loop self-intersects and its wall hull exceeds the floor margin.

These results show that the wall fix improves candidate stability and
provenance, but does not make the room reconstruction surveyed or complete.

## Remaining gap

The post-fix wall count is more stable, but several limitations remain:

- low-confidence walls are still present;
- hull fallbacks remain warning-quality outputs;
- the supplied captures do not provide a defensible same-scan ceiling plane;
- opening detection remains an occupancy-gap heuristic;
- fit residuals are not tape or laser accuracy;
- the system still reconstructs one capture at a time and does not stitch a
  property.

The next engineering priority should be evidence and presentation of the
baseline rather than adding unsupported RGB-only metric reconstruction.

## Reproduction and artifacts

Run the following twice from the repository root:

```bash
python -m src store/c00a170fe1 --tier lidar --from-ply
python -m src store/c00a170fe1 --tier lidar --from-ply

python -m src store/c7d28f72c6 --tier lidar --from-ply \
  --frame-stride 24 --pixel-stride 8
python -m src store/c7d28f72c6 --tier lidar --from-ply \
  --frame-stride 24 --pixel-stride 8
```

Compare the resulting `run.determinism_key`, wall/opening counts, footprint
method, and height status in:

- `out/c00a170fe1/plan.json` and `out/c00a170fe1/plan.svg`;
- `out/c7d28f72c6/plan.json` and `out/c7d28f72c6/plan.svg`;
- `out/*/preview_plan.png`;
- the preserved Step 8 logs.

The repository test command is:

```bash
python -m pytest -q
```
