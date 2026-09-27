# Incremental Proceeding Plan

## 1. Working position

The original assignment is much larger than a normal take-home. It asks for a capture product, three sensor tiers, multi-room reconstruction, damage inspection, calibration, benchmarking, consumer-app comparison, reproducibility, and a live defense.

We will not attempt to build the entire product in one pass. We will build a useful baseline first, measure it honestly, and then add only the highest-value improvements that fit the remaining time.

The intended submission story is:

> We built a local RGB-D room-reconstruction baseline from the supplied Record3D captures, measured its failure modes, shipped a focused geometry improvement, and exposed the remaining limitations instead of claiming unsupported capabilities.

This makes the work defensible without pretending that a two-day project is a production scanning platform.

## 2. Current state

The repository already contains seven incremental commits and a working prototype with:

- Record3D-style capture ingestion;
- RGB-D point-cloud generation;
- camera calibration and pose handling;
- depth confidence and range filtering;
- floor and wall plane fitting;
- same-capture ceiling-height handling;
- heuristic opening detection;
- JSON and SVG plan export;
- RGB still/video extraction;
- pose-graph drift analysis;
- reconstruction-health checks and a repair loop;
- synthetic geometry tests and capture-specific tests.

The current system is a LiDAR/RGB-D baseline. It is not yet a true photo-tier or RGB-only video-tier measurement system.

The current results show useful reconstruction behaviour but also clear weaknesses:

- wall fitting can mistake furniture or other vertical structures for walls;
- some polygons are irregular or unstable;
- ceiling detection is still blocked on the supplied ceiling capture;
- opening detection is occupancy-gap heuristics, not a trained door/window detector;
- fit residuals are not tape-measure accuracy;
- the output currently describes one capture, not a stitched property.

## 3. Scope decision

### In scope for the baseline

```text
supplied RGB-D capture
  → metric point cloud
  → floor/wall/ceiling hypotheses
  → rough room footprint
  → heuristic openings
  → JSON/SVG/PNG outputs
  → reconstruction-health report
```

### In scope if time remains

- a reproducible dataset-preparation script;
- RGB-only photo/video adapters using derived inputs;
- common result/provenance fields;
- a focused wall-filtering or ceiling-detection improvement;
- timing and repeatability measurements;
- a concise baseline and fix-loop report.

### Explicitly deferred unless the baseline is stable

- general RGB-only metric reconstruction;
- robust multi-room stitching;
- physical damage segmentation;
- concealed-damage inference;
- scope line items;
- consumer-app head-to-head comparison;
- TestFlight/iOS capture application;
- arbitrary unseen-room generalisation.

The deferred items are not being ignored. They are being labelled as outside the first executable slice so that engineering time is spent on a reliable result rather than scattered feature stubs.

## 4. Rules for execution

### Small step, small commit

**The agent must not run `git commit` or `git push` for this plan. Nabajyoti creates commits manually after reviewing each step.**

Each step below should produce one observable improvement and one small commit (by Nabajyoti, not the agent). Do not combine unrelated refactors with feature work.

Every commit should include:

1. the code or documentation change;
2. a test or reproducible command where practical;
3. a README update explaining what changed and why;
4. the result or limitation discovered during the step.

Suggested commit format:

```text
step N: short description of one completed change
```

### README is part of the implementation

The README must be updated after each completed step. The assignment auditor should be able to read the README and reconstruct the development story without guessing:

- what the step was trying to solve;
- why that approach was selected;
- what command was run;
- what output was produced;
- what failed or remained uncertain;
- what the next step will address.

The commit history and README should tell the same story. The final README should link to the baseline report, benchmark results, fix-loop report, output examples, and known limitations.

## 5. Executable step plan

**DO NOT COMMIT — AGENT: implement the step, run tests, and update the README; leave all changes unstaged or staged only if Nabajyoti asked otherwise. Nabajyoti runs `git commit` (and push) himself.**


### Step 0 — Freeze the current baseline

**Status:** Done. Commit `02d1929` (`step 0: freeze reproducible RGB-D baseline`). Code under test is `405fee95`. Logs are in `baseline/logs/`, counts in `baseline/freeze.json`, and one plan JSON/SVG/PNG per capture in `baseline/examples/`. The README “Current baseline” section records the commands. `rmse_m` is fit error, not tape accuracy. Open3D floor RANSAC is unseeded: two exports of the same `c00a170fe1` cloud returned 7 walls then 6, while `height_m` stayed blocked on both captures.

**Objective:** Record the current repository state before changing the algorithm.

**Why:** A fix-loop is impossible to defend if the before state is not reproducible.

**Actions:**

- record the current commit hash;
- run the main commands on `store/c00a170fe1` and `store/c7d28f72c6`;
- save stdout, runtime, and generated artifact paths;
- record point counts, wall counts, openings, polygon status, and height status;
- preserve representative plan images and JSON outputs.

**Acceptance check:** Someone can rerun the commands and obtain the same type of outputs.

**Commit:**

```text
step 0: freeze reproducible RGB-D baseline
```

**README update:** Add a “Current baseline” section with commands, outputs, and a clear statement that residuals are fit errors rather than tape accuracy.

### Step 1 — Write the baseline report

**Status:** Done. Commit `67e64e7` (`step 1: document baseline results and limitations`). The write-up is `BASELINE_REPORT.md`, linked from the README. Both frozen captures produce a floor and a plan; ceiling height stays blocked, including on `c7d28f72c6`; both footprints used the wall-inlier hull fallback.

**Objective:** Turn the current implementation into a documented baseline rather than a collection of commands.

**Why:** The project needs a coherent engineering narrative before adding more features.

**Actions:** Create `BASELINE_REPORT.md` containing:

- input capture descriptions;
- pipeline diagram;
- runtime and resource notes;
- generated artifacts;
- observed strengths;
- observed failure modes;
- unsupported assignment requirements;
- explicit claims the system does not make.

**Acceptance check:** A reviewer can understand what currently works in under five minutes.

**Commit:**

```text
step 1: document baseline results and limitations
```

**README update:** Link to the baseline report and summarise its main findings.

### Step 2 — Add formal result status and provenance

**Status:** Done. Commit `121d7fb` (`step 2: add result status and provenance metadata`). `plan.json` and `report.json` carry `status`, `input_tier`, `source_files`, `cloud_origin`, `method`, `measurement_status`, and `accuracy_status`. `--from-ply` lists only the existing cloud. `imu.csv` is not listed because plane fitting does not read it. Fit residuals and opening heuristics are unchanged.

**Objective:** Make every output explicit about its input tier, source files, method, and confidence status.

**Why:** The original assignment scores calibration and honesty. Provenance prevents accidental overclaiming.

**Actions:** Extend plan/report JSON with fields such as:

```json
{
  "status": "baseline",
  "input_tier": "lidar",
  "source_files": [],
  "measurement_status": "estimated",
  "accuracy_status": "not_calibrated"
}
```

Keep the existing disclaimers that fit residuals are not laser/tape accuracy and openings are heuristics.

**Acceptance check:** Every generated plan identifies its tier and provenance without relying on console output.

**Commit:**

```text
step 2: add result status and provenance metadata
```

**README update:** Document the JSON fields and why they exist.

### Step 3 — Add a reproducible dataset-preparation script

**Status:** Done. Commit `12af684` (`step 3: add reproducible benchmark file boundaries`). `python -m src.benchmark` regenerates `benchmark/` from `store/`. Photo PNGs are decoded from `rgb.mp4`; video is a copy of that file; LiDAR is a copied Record3D folder. `benchmark/manifest.json` states that stills are derived RGB, not native photographs. Generated blobs are gitignored; `benchmark/README.md` is the committed note.

**Objective:** Derive logical photo, video, and LiDAR inputs from `store/` without using personal-room data.

**Why:** We do not have an iPhone 15+ and will not use private room captures. The supplied data is the reproducible development source.

**Actions:** Generate:

```text
benchmark/
  photo/room_01/*.png
  video/room_01.mp4
  lidar/room_01/
    rgb.mp4
    depth/
    confidence/
    odometry.csv
    imu.csv
    camera_matrix.csv
```

The photo inputs must be extracted directly from `rgb.mp4`, not screen-captured. The script must write a manifest explaining that the stills are derived RGB frames.

**Acceptance check:** One command regenerates the derived benchmark inputs from `store/`.

**Commit:**

```text
step 3: add reproducible benchmark input preparation
```

**README update:** Explain the provenance of every logical tier and clearly state that derived stills are not independent native photographs.

### Step 4 — Add input adapters without changing reconstruction

**Status:** Done. Commit `ca3ca46` (`step 4: add explicit photo video and lidar input adapters`). `--tier photo|video|lidar` uses `src/adapters.py`. LiDAR still reads depth, confidence, intrinsics, and poses. Video reads only RGB video. Photo reads only top-level stills. Photo/video write `status=blocked` plans and cannot be combined with `--from-ply` or the cloud path.

**Objective:** Give photo, video, and LiDAR inputs a common interface while preserving the current working LiDAR path.

**Why:** A common interface lets us compare tiers and evolve the reconstruction code without duplicating the whole pipeline.

**Actions:** Add adapters with explicit rules:

- LiDAR adapter may read depth, confidence, intrinsics, and poses;
- video adapter may read only RGB video;
- photo adapter may read only still images;
- adapters must record which files were allowed and consumed.

At this stage, photo/video outputs may be marked `blocked` or `not_implemented` for metric reconstruction. The important goal is a truthful interface, not fake accuracy.

**Acceptance check:** The CLI can identify all three tiers and enforce their file boundaries.

**Commit:**

```text
step 4: add explicit photo video and lidar input adapters
```

**README update:** Add a tier matrix showing available inputs, allowed metadata, and current capability.

### Step 5 — Improve wall selection

**Status:** Done. Implemented in `src/walls.py`, not committed (Nabajyoti commits). Horizontal planes now use the seeded numpy sampler, with Open3D only as a fallback, so a second export of the same PLY repeats the wall count. The filter records every drop. On `c00a170fe1`, walls went from 7 then 6 in the freeze to a stable 4 (2 low confidence), with 6 rejected (1 tilted, 5 near-parallel). On `c7d28f72c6`, walls went from 6 to a stable 5 (3 low), with 5 near-parallel duplicates rejected. `height_m` stays blocked. Both footprints are still the wall-inlier hull.

**Objective:** Reduce false wall planes caused by furniture and other vertical objects.

**Why:** This is the most visible weakness in the current baseline and the highest-value geometry improvement per unit of time.

**Actions:** Add conservative filters based on:

- vertical extent;
- inlier count and spatial coverage;
- position relative to the floor footprint;
- dominant room directions;
- duplicate and near-parallel plane suppression;
- wall-like support across the room rather than a small furniture patch.

Do not silently discard uncertain planes; record the rejection reason or downgrade the result confidence.

**Acceptance check:** On the frozen captures, the revised output has a more stable wall count or a more plausible footprint, and the before/after numbers are recorded.

**Commit:**

```text
step 5: filter furniture-like vertical planes
```

**README update:** Explain the false-positive cause, the filtering rule, and the measured change.

### Step 6 — Improve ceiling detection

**Status:** Done. Implemented in `src/ceiling.py`, not committed (Nabajyoti commits). The search tries up to six horizontal planes in the upper band and records every drop. A plane is kept only if it is 1.6–4.5 m above the floor, wide, filled in, and has empty space above it. On `c00a170fe1`, height stays blocked (6 rejected: 2 `too_low`, 2 `occupied_above`, 2 `sparse_coverage`; highest candidate 2.12 m, above-ratio 0.58). On `c7d28f72c6`, height stays blocked (6 rejected: 2 `too_low`, 4 `occupied_above`; highest candidate 2.07 m, above-ratio 0.37). A second run repeats those codes. Wall counts stay 4 and 5. No default height is filled in. The README “Ceiling detection” section has the before/after.

**Objective:** Recover ceiling height when a genuine same-capture ceiling plane is present.

**Why:** Ceiling height is a visible part of the output contract, and the supplied ceiling capture currently remains blocked.

**Actions:** Add a targeted high-horizontal-plane search with:

- upper-height candidate selection;
- horizontal-normal filtering;
- minimum support and spatial coverage;
- floor-to-ceiling plausibility checks;
- explicit blocked output when evidence is insufficient.

Never fill in a typical ceiling height merely because detection failed.

**Acceptance check:** Either the ceiling capture produces a defensible height or the report demonstrates why it remains blocked.

**Commit:**

```text
step 6: improve same-capture ceiling plane detection
```

**README update:** Add the before/after ceiling result and explain the blocked case.

### Step 7 — Stabilise the footprint and outputs

**Status:** Complete. The gate is in `polygon_from_walls` (`src/planes.py`). A candidate outline must be a simple loop and every corner must sit within 0.5 m of the floor points. On `c00a170fe1`, the outline stays the wall-inlier hull (7 vertices, 6.31 × 7.44 m, floor 6.49 × 6.79 m) with `output_quality=warning`; wall lines were outside that margin. On `c7d28f72c6`, the wall-line loop self-intersects and the wall hull sticks out, so the outline is now the floor hull: 17 vertices, span 11.62 × 15.12 m, matching the floor (was 11.61 × 15.62 m). `output_quality=warning`. A second export of each cloud repeats the method and the span. `height_m` stays blocked. Wall counts stay 4 and 5. The README “Footprint” section records which method was selected and when the fallback is used. Nabajyoti commits this step.

**Objective:** Ensure the rendered footprint is bounded, readable, and consistent with the floor evidence.

**Why:** A rough but stable plan is more defensible than a geometrically detailed polygon that can explode or self-intersect.

**Actions:**

- reject self-intersecting or implausibly large polygons;
- prefer a bounded floor-supported fallback when wall intersections are degenerate;
- report the fallback method;
- add a simple output-quality status such as `ok`, `warning`, or `blocked`.

**Acceptance check:** The plan remains within a documented relationship to the reconstructed floor footprint.

**Commit:**

```text
step 7: stabilise bounded room footprint export
```

**README update:** Explain which footprint method was selected and when fallback occurs.

### Step 8 — Measure runtime and determinism

**Objective:** Make the baseline and fix runs comparable.

**Why:** Reproducibility is part of the assignment and gives the fix-loop quantitative evidence.

**Actions:** Record:

- capture name and tier;
- frame/pixel strides;
- points generated and retained;
- wall/opening counts;
- runtime by major phase;
- output paths;
- deterministic seed/configuration.

**Acceptance check:** Repeated runs produce equivalent metrics and a timing record.

**Commit:**

```text
step 8: add deterministic timing and run metadata
```

**README update:** Add the reproducibility command and a timing table.

### Step 9 — Add the fix-loop report

**Objective:** Package one before/after experiment with a readable explanation.

**Why:** The assignment explicitly rewards a correct root cause and shipped fix. We can satisfy the spirit of that requirement with one focused geometry fix.

**Actions:** Create `FIX_LOOP_REPORT.md` containing:

1. worst baseline gate or failure;
2. failing value;
3. root-cause hypothesis;
4. evidence;
5. shipped fix;
6. predicted post-fix value;
7. measured post-fix value;
8. explanation of any remaining gap.

**Acceptance check:** Both before and after runs are reproducible from the repository and the difference is visible in JSON/PNG/SVG outputs.

**Commit:**

```text
step 9: add reproducible geometry fix-loop report
```

**README update:** Link to the fix-loop report and summarise the result.

### Step 10 — Add RGB-only photo/video behaviour only if time remains

**Objective:** Provide a truthful extension for the other logical tiers.

**Why:** The assignment requires three tiers, but a credible RGB-only metric solution is significantly harder than the current LiDAR baseline.

**Actions, in descending priority:**

1. photo/video ingestion and validation;
2. keyframe selection;
3. shared non-metric visual output;
4. uncertainty widening and explicit low-confidence status;
5. metric reconstruction only if evidence supports it.

The photo tier must not access depth or pose files. The video tier must not use hidden odometry during its reported run.

**Acceptance check:** The limitation is visible in the output and report rather than hidden behind invented measurements.

**Commit:**

```text
step 10: add truthful RGB-only tier handling
```

**README update:** Add the final device/tier matrix and disclose that derived RGB inputs were used for development.

## 6. Final submission structure

If time allows, the final repository should contain:

```text
README.md
PROCEEDING_PLAN.md
BASELINE_REPORT.md
FIX_LOOP_REPORT.md
schema/
benchmark/
scripts/
src/
tests/
out/
```

The README should tell the complete story:

1. what the original problem asks for;
2. why the scope was reduced;
3. what baseline was built;
4. how the baseline is reproduced;
5. what failure was selected;
6. what fix was shipped;
7. what changed numerically;
8. what remains unsupported;
9. how the supplied data and derived RGB inputs are disclosed.

## 7. Stop conditions

Stop adding features when any of the following is true:

- the baseline cannot be reproduced from a clean setup;
- a new feature weakens the current output without measurable benefit;
- the change requires hidden metadata or undocumented manual intervention;
- the work would require building a separate production-grade subsystem;
- the remaining time is better spent on the README, benchmark evidence, or defense preparation.

A complete, measurable baseline plus one well-supported improvement is the target. The project should demonstrate engineering judgment, not attempt to deliver the entire startup product for free.
