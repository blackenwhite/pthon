# Cozmo room plan MVP

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
