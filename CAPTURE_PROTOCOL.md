# Capture route (Route 2) and device matrix

This is the stock protocol. At defense, follow this page literally. There is no TestFlight app.

**Metric output:** only a Record3D LiDAR folder. Photo folders and a video file are accepted and produce a blocked plan (keyframes, no metres).

---

## What to install

| Tier | App | Source | Notes |
|---|---|---|---|
| LiDAR | **Record3D** | App Store | Pro-class iPhone with LiDAR. Export the session as the usual folder (`rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `imu.csv`, `camera_matrix.csv`). |
| Photo | **Camera** (built-in) | preinstalled | Stills only. Do not use Portrait, Cinematic, or a LiDAR scanning app. |
| Video | **Camera** (built-in) | preinstalled | One handheld clip. 4K or 1080p is fine. Do not attach depth or pose files. |

Install time: Record3D from the App Store, then open it once so it can request camera/motion permission. Camera app: nothing to install.

---

## Device matrix (honest)

| Hardware | Photo | Video | LiDAR | What this pipeline delivers |
|---|---|---|---|---|
| iPhone 15 / 16 (non-Pro) | yes | yes | no native LiDAR in this protocol | Photo/video: ingest + keyframes, **no metric plan**. |
| iPhone 15 Pro / 16 Pro (or other Pro with LiDAR) | yes | yes | Record3D | LiDAR: uncalibrated RGB-D room estimate (fit residuals, not tape). Photo/video: same as above. |
| iPhone 14 (author development phone) | practice only | practice only | no | **Not** the assignment’s iPhone 15+ floor. Do not use author-phone stills as the scored photo benchmark. |
| Any iPhone without LiDAR | Camera stills/video | Camera video | **refused** | `--tier lidar` requires a Record3D folder. |

Accuracy claim for every tier: **`accuracy_status=not_calibrated`**. LiDAR `rmse_m` is plane-fit error. Photo/video do not emit wall lengths in metres.

---

## Photo (2–8 stills per room)

1. Lights on. Doors you care about fully open.
2. Use the **1×** camera, not 0.5×. Hold the phone **landscape**. Turn off Live Photo.
3. Stand roughly in the middle of the room. Take **6** photos if you can, **8** at most:
   - four shots toward the four corners (each shot should show floor *and* ceiling if possible);
   - two shots of the longest walls.
4. Overlap: each photo should share a large piece of wall with the previous one. Do not shoot isolated close-ups of furniture.
5. Avoid mirrors, glass, and windows filling more than about a third of the frame if you can walk around them.
6. Put the JPEGs in **one folder per room**. No subfolders. No depth files. Name the folder after the room (`kitchen`, `room_01`).

Handoff: zip those folders. Run:

```bash
python -m src path/to/room_folder --tier photo
```

Expected: `out/photo_<name>/keyframes/`, `keyframes_preview.png`, `plan.json` with empty walls and `status=blocked`.

---

## Video (one handheld walkthrough)

1. Lights on. Hold landscape, 1× lens.
2. Start in a doorway. Walk slowly around the room at walking pace (~0.3–0.5 m/s). Keep the camera at chest height. Pan walls, then a brief look at the ceiling, then the floor.
3. Duration: **30–90 seconds** for one room. Do not stop-start; one clip.
4. Avoid spinning in place and pointing at mirrors.

Handoff: one `.mp4`. Run:

```bash
python -m src path/to/clip.mp4 --tier video
```

Expected: same blocked plan contract as photo.

---

## LiDAR (Record3D, Pro device)

1. Open Record3D. Start a new capture. Hold the phone upright, LiDAR facing the room.
2. Walk the room once at walking pace. Cover every wall. Include a slow look **up** at the ceiling if height matters.
3. Duration: **30–60 seconds** for a single room. Longer walks drift.
4. Stop, export/save the session. Copy the whole folder (do not send only `rgb.mp4`).

Handoff: that folder on a USB stick or AirDrop. Run:

```bash
python -m src path/to/record3d_folder --tier lidar
```

Expected: `out/<folder>/plan.json`, `plan.svg`, `preview_plan.png`. `height_m` may be `null` if no ceiling plane is found. Do not treat JSON residuals as laser measurements.

Repeatability (same room twice): export two separate Record3D sessions. Run the same command on each. Compare `plan.json` wall counts and polygon spans; this repo does not yet claim 1 cm / 0.5% wall agreement vs tape.

---

## What to avoid (all tiers)

- Mixing files across tiers (do not drop `odometry.csv` into a photo folder and expect it to be used; photo/video adapters refuse those files).
- Asking the operator to “scan like magicplan” without Record3D for LiDAR.
- Low light, wet floors, floor-to-ceiling mirrors as the main surface.
- Fusing two captures into one height (this pipeline never does that).

---

## Walk-in (defense)

1. Follow this page for the tier chosen that day.
2. Operator copies files onto the evaluation machine.
3. One command above. First LiDAR run rebuilds the cloud (minutes on a long capture). Photo/video are seconds.
4. Score **metres only on LiDAR**. Photo and video are scored as “pipeline ran and disclosed blocked geometry,” not as a stitched property plan.
