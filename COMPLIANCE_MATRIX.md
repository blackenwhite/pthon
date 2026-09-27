# Compliance matrix

Status meanings: **done** = implemented and regenerable; **partial** = exists with honest limits; **blocked** = adapter or output exists but metric contract is not claimed; **deferred** = not built; **unavailable** = no data or hardware for this row.

| Requirement | File path | Artifact | Status |
|---|---|---|---|
| One command per capture | `src/run.py`, `README.md` | `python -m src <path> --tier lidar\|photo\|video` | **partial** — LiDAR writes a metric plan; photo/video write keyframes + a blocked plan |
| JSON to published schema | `src/export.py` | `schema=cozmo.room_plan.v1` in `plan.json` | **partial** — LiDAR geometry + provenance; photo/video empty geometry |
| Rendered plan | `src/export.py`, `src/planes.py` | `plan.svg`, `preview_plan.png`; RGB: `keyframes_preview.png` | **partial** |
| Dimensioned per-room plan (walls, height, area, openings) | `src/planes.py`, `src/walls.py`, `src/ceiling.py`, `src/openings.py`, `src/floor.py` | `plan.json` walls / polygon / openings / `height_m` | **partial** — LiDAR only; height often `null`; floor area not a surveyed field; openings are occupancy gaps |
| Stitched multi-room plan, correct adjacency | — | — | **deferred** |
| Photo-tier whole-property stitch from per-room folders | `src/adapters.py`, `src/rgb_tier.py` | Per-folder keyframes only | **blocked** — no stitch |
| Per-surface damage regions, class, metric extent | `src/report.py` | `report.json` | **deferred** — report is reconstruction health, not physical damage |
| Concealed-damage flags + firing rule | — | — | **deferred** |
| Scope line items keyed to surfaces | — | — | **deferred** |
| Confidence interval on every measurement | `src/export.py`, `src/rgb_tier.py` | `rmse_m` / `median_residual_m` on planes; RGB `confidence_status=low` | **partial** — fit residual is not a calibrated CI; RGB does not invent metres |
| Opening widths ≤ 2 cm on ≥ 85%; miss/phantom scored | `src/openings.py` | Heuristic gaps in `plan.json` | **unavailable** — no laser GT table in this repo |
| Ceiling height ≤ 1.5 cm; repeat spread ≤ 1 cm | `src/ceiling.py` | `height_m` or `height_blocked` | **partial** — both frozen captures still block height |
| Repeatability: same room, same tier, 1 cm / 0.5% per wall | `src/timing.py`, `step8/logs/` | Stable wall *count* on same PLY | **partial** — determinism of the fitter, not tape wall lengths |
| Drift accountability + on/off ablation | `src/posegraph.py` | `drift.json`, `drift_path.png` | **partial** — ablation exists; “poses as-is” is not the only path, but loop-closure is not claimed solved |
| Photos / video / LiDAR input tiers | `src/adapters.py` | Hard file boundaries; `--tier` | **done** for ingest; metric reconstruction only on LiDAR |
| Device matrix | `CAPTURE_PROTOCOL.md` | Hardware × tier × honest accuracy | **done** (disclosure, not calibrated accuracy) |
| Capture route (stock protocol) | `CAPTURE_PROTOCOL.md` | Route 2, one page | **done** |
| Benchmark set: 3+ rooms, furnished+damage, all tiers, repeat capture, laser GT | `scripts/prepare_benchmark.py`, `benchmark/README.md` | Derived RGB from Record3D `rgb.mp4` | **partial** — logical folders only; stills are not native photos; no tape/laser GT bundle |
| Head-to-head vs consumer app | — | — | **deferred** |
| Fix loop: worst gate, root cause, shipped fix, before/after | `FIX_LOOP_REPORT.md`, `src/walls.py` | Wall-filter before/after | **done** for reconstruction walls; not a Round-1 opening/height gate pass |
| Reproduction bundle | `README.md`, `requirements.txt`, `baseline/`, `step8/logs/` | Local venv; cached PLY allowed; live `--cloud` path exists | **partial** — LiDAR regenerable; photo metric not implemented |
| Weights/binaries fetched by script | `requirements.txt` | pip only | **done** — no large foundation-model weights |
| Runs without calling our infrastructure | `src/` | Local Open3D / OpenCV / numpy | **done** |
| Process evidence (commit history) | git log | Incremental steps 0–10 | **done** (author commits) |
| Walk-in test, all three tiers cold | `CAPTURE_PROTOCOL.md` | LiDAR metric; photo/video blocked | **partial** — all three *run*; only LiDAR is scored as metres |
| Handheld consumer capture | `CAPTURE_PROTOCOL.md` | Record3D + Camera app | **done** as protocol |
| iPhone 15+ photos/video; Pro LiDAR | `CAPTURE_PROTOCOL.md` | Development on iPhone 14 disclosed | **unavailable** as author hardware; protocol written for 15+ / Pro |
| Mirrors, glass, wet-look, low light | `TECHNICAL_REPORT.md` | Known failure modes | **partial** — documented, not specially handled |
| Technical report ≤ 6 pages | `TECHNICAL_REPORT.md` | Architecture, tiers, drift, error budget, fix loop, failures | **done** |
