# Cozmo room plan MVP

## Required benchmark data

The repository intentionally does not commit the supplied raw capture data
because it is approximately 2.3 GB. To run the RGB-D baseline and reproduce
the reported measurements, obtain the benchmark-data bundle from the submission
email and extract it at the repository root so that this path exists:

```text
store/
  c00a170fe1/
    camera_matrix.csv
    odometry.csv
    imu.csv
    rgb.mp4
    depth/*.png
    confidence/*.png
  c7d28f72c6/
    camera_matrix.csv
    odometry.csv
    imu.csv
    rgb.mp4
    depth/*.png
    confidence/*.png
  1a8384c3f6/
    camera_matrix.csv
    odometry.csv
    imu.csv
    rgb.mp4
    depth/*.png
    confidence/*.png
```

The `store/` directory is ignored by Git deliberately. It is not missing from
the implementation: it is delivered separately as the benchmark-data bundle.
The captures are supplied Record3D data; no private room capture or iPhone
15+ was used.

After extracting the data, verify the setup with:

```bash
python -m src store/c00a170fe1 --inspect
python -m pytest -q
```

To regenerate the logical photo, video, and LiDAR benchmark inputs from the
raw captures:

```bash
python -m src.benchmark
```

The submission email should provide the benchmark-data link alongside this
repository. If the repository is read without that email, the required data
bundle is the only missing external dependency.

This project turns an iPhone room scan into a simple floor-plan estimate.

Think of it like this: the phone records the room, and this program tries to find the floor, walls, doors, and windows. It then draws a plan and saves the measurements it could estimate.

## What works today

- **LiDAR scans:** Creates a rough single-room plan from a Record3D capture.
- **Photos:** Accepts a folder of photos and creates a visual preview, but does not measure the room.
- **Video:** Accepts a walkthrough video and creates a visual preview, but does not measure the room.

The project does not currently join several rooms into one home, inspect physical damage, or promise survey-grade accuracy. When the evidence is not good enough, it reports that the result is blocked instead of guessing.

## Run it

You need Python 3. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then run one command for your input:

```bash
# Record3D LiDAR folder
python -m src path/to/record3d_folder --tier lidar

# Folder of photos
python -m src path/to/photo_folder --tier photo

# Video file
python -m src path/to/video.mp4 --tier video
```

Results are saved inside the `out/` folder. A LiDAR run can produce a JSON result, a drawn plan, and preview images. Photo and video runs produce selected images and a preview.

For instructions on how to capture a room, read [CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md).

## More information

The detailed development history was moved out of this README so this page stays easy to understand:

- [Engineering notes](ENGINEERING_NOTES.md) — full implementation history, commands, and measurements
- [Compliance matrix](COMPLIANCE_MATRIX.md) — what the assignment asked for and what is complete
- [Technical report](TECHNICAL_REPORT.md) — architecture, limitations, and design decisions
- [Baseline report](BASELINE_REPORT.md) — original measured results
- [Fix-loop report](FIX_LOOP_REPORT.md) — the problem found and the fix that was shipped
- [Proceeding plan](PROCEEDING_PLAN.md) — project scope and development steps
