# Technical report (Path A)

Cozmo RGB-D room-plan baseline. This document is the engineering narrative. Numbers come from `BASELINE_REPORT.md`, `FIX_LOOP_REPORT.md`, and `README.md`. Residual fields in JSON are fit error, not tape or laser accuracy.

## 1. Scope

The assignment is a capture product, three metric tiers, a stitched property, damage/scope, calibration, a consumer-app bake-off, and a cold walk-in. This repository implements a **local Record3D LiDAR/RGB-D baseline** with truthful photo and video **adapters**, one **geometry fix loop**, and explicit **deferred** rows (see `COMPLIANCE_MATRIX.md`).

Submission story: we measured a working RGB-D pipeline, shipped a wall-selection fix with regenerable before/after, and did not invent metres on 2–8 stills.

## 2. Architecture

```text
input adapters (photo | video | lidar)
        │
        ├─ photo/video → keyframes + contact sheet + blocked plan.json
        │
        └─ lidar → RGB-D + poses → metric cloud.ply
                      → RANSAC floor / walls / ceiling
                      → occupancy-gap openings
                      → footprint (wall lines → hull fallbacks)
                      → plan.json + plan.svg + preview_plan.png
                      → optional pose-graph drift ablation
                      → optional reconstruction-health report
```

All processing is on-device relative to the evaluator’s machine: numpy, OpenCV, Open3D. No author API. Large foundation models are not in the dependency set.

**LiDAR path.** `src/ingest.py` loads a Record3D folder. `src/cloud.py` back-projects confidence-filtered depth with scaled intrinsics and ARKit poses. `src/planes.py` voxel-downsamples, fits a seeded horizontal floor, vertical walls (`src/walls.py` filters furniture-like planes), and a conservative ceiling search (`src/ceiling.py`). `src/openings.py` finds along-wall occupancy gaps. Export is `src/export.py`.

**RGB path.** `src/adapters.py` enforces file boundaries. Photo reads top-level stills only. Video decodes RGB only. `src/rgb_tier.py` writes up to 12 evenly spaced keyframes and a blocked plan (`method=rgb_keyframes_non_metric`). Combining `--tier photo|video` with `--cloud` / `--from-ply` is rejected so LiDAR cannot leak in.

## 3. Tier design and device matrix

| Tier | Hardware (protocol) | Reconstruction | Honest accuracy |
|---|---|---|---|
| LiDAR | Pro iPhone + Record3D | Metric planes from RGB-D | Uncalibrated estimate. Floor fit RMSE ~2 cm on frozen clouds. Walls/openings/height not tape-validated. |
| Video | Any iPhone 15+ Camera clip | None | `confidence_status=low`. No wall lengths. |
| Photo | Any iPhone 15+ stills, 2–8/room | None | Same blocked contract. Per-room folders do not stitch. |

Author development hardware is an **iPhone 14** (no assignment-grade photo/LiDAR capture). Derived `benchmark/photo` PNGs are decoded from Record3D `rgb.mp4`, not native photographs (`benchmark/README.md`).

Photo-tier gate of ±8% walls and whole-property stitch: **not claimed**. Video ±3%: **not claimed**.

## 4. Drift handling

ARKit odometry is used to build the cloud. `python -m src <capture> --drift` runs sequential ICP plus nearby loop closures and writes `drift.json` / `drift_path.png` (`src/posegraph.py`). That is an **ablation**, not a proof that accumulated drift is removed. The report should not say “poses used as-is” as the only story; it also should not say loop closure solves the multi-room footprint. There is no stitched property graph.

## 5. Error budget (qualitative)

| Source | Effect | Mitigation in this repo |
|---|---|---|
| Depth noise / confidence | Floor/wall thickness | Confidence threshold; voxel downsample |
| Furniture verticals | Extra walls | Wall filter + rejection codes |
| Missing ceiling in cloud | `height_m` null | Do not impute 2.4 m |
| Degenerate wall corners | Exploded polygon | Footprint gate: wall lines → wall hull → floor hull |
| Odometry drift | Inflated footprint vs true room | Drift ablation; still one-capture |
| Occupancy gaps | False / missed openings | Heuristic only; JSON `note` |
| RGB-only geometry | Arbitrary scale | Blocked plan, no fake scale |

There is **no laser/tape ground-truth table** in the reproduction bundle. Calibration analysis: every export sets `accuracy_status=not_calibrated`. Using plane RMSE as a 95% CI would be confident garbage; we do not.

## 6. Fix loop

Worst observed failure on the frozen baseline: **unstable, furniture-contaminated walls** (`c00a170fe1`: 7 then 6 walls; no rejection provenance). Root cause: density-first vertical RANSAC plus unseeded Open3D floor peeling. Fix: seeded samplers + geometric wall filters (`src/walls.py`). Prediction: fewer duplicates, repeatable counts, no invented ceiling. Measured: walls **4** and **5** on the two captures, stable on repeat `--from-ply`; height still blocked. Full write-up: `FIX_LOOP_REPORT.md`. Before: `baseline/`. After: current code + `step8/logs/`.

This moves a reconstruction defect, not the assignment’s opening-width or ceiling-height gates (those remain unmeasured vs laser).

## 7. Known failure modes

- Ceiling plane absent or occupied-above on supplied look-up capture → height blocked.
- Footprint often warning-quality hull; span can exceed camera path.
- Mirrors, glass, wet floors, low light: depth holes and wrong planes; not specially modelled.
- Photo/video walk-in: pipeline runs, **metres fail closed**.
- Multi-room, damage classes, scope lines, magicplan bake-off: not implemented.
- Repeatability vs tape (1 cm / 0.5% per wall): not demonstrated.

## 8. How to run (evaluator)

Clean machine, Python 3, from repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src path/to/record3d_folder --tier lidar
```

First LiDAR run builds `out/<id>/cloud.ply`. Reuse with `--from-ply`. Photo/video: `CAPTURE_PROTOCOL.md`. Tests: `python -m pytest -q`.
