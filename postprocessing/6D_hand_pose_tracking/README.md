# 6D Hand Pose Tracking with a ZED RGB-D Camera

Extract **6D hand poses** (3D position + orientation) and **fingertip positions** from
[Stereolabs ZED](https://www.stereolabs.com/) recordings (`.svo` / `.svo2`) or from a live ZED
camera, and write them to CSV.

The pipeline combines two components:

1. **[MediaPipe Hands](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)**
   detects 21 landmarks per hand in the ZED's left colour image.
2. **The ZED depth / point cloud** turns those 2D landmarks into metric 3D coordinates: the wrist
   landmark is looked up directly in the point cloud, and the remaining landmarks are
   back-projected around that anchor using the camera intrinsics.

From the resulting 21×3 array per hand the tools derive:

| Quantity | Definition |
| --- | --- |
| **Palm centroid** (x, y, z) | Mean of the wrist (landmark 0), index MCP (5) and pinky MCP (17) |
| **Palm orientation** (yaw, pitch, roll) | Euler angles of the normal to the plane through landmarks 0, 5 and 17 |
| **Index fingertip** (x, y, z) | Landmark 8 |
| **Thumb tip** (x, y, z) | Landmark 4 |
| *(optional)* **All 21 landmarks** | Enabled with `--landmarks` |

Everything is configurable from the command line — there are no hard-coded paths, dataset layouts
or camera settings.

---

## Table of contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Output format](#output-format)
- [Scripts](#scripts)
  - [`zed/save_6d_palm.py` — extract poses to CSV](#zedsave_6d_palmpy--extract-poses-to-csv)
  - [`batch_process_svo.py` — process many recordings](#batch_process_svopy--process-many-recordings)
  - [`zed/show3d_matplot.py` / `zed/show3d_open3d.py` — live preview](#zedshow3d_matplotpy--zedshow3d_open3dpy--live-preview)
  - [`zed/svo_tools/` — record, play back, export](#zedsvo_tools--record-play-back-export)
  - [`download_zed_models.py` — fetch the ZED AI models](#download_zed_modelspy--fetch-the-zed-ai-models)
  - [`zed/gui/` — ZED SDK GUI tools](#zedgui--zed-sdk-gui-tools)
- [Using the package as a library](#using-the-package-as-a-library)
- [Tuning and performance](#tuning-and-performance)
- [Troubleshooting](#troubleshooting)
- [Project structure](#project-structure)
- [Limitations](#limitations)
- [License](#license)

---

## Requirements

**Hardware**

- A Stereolabs ZED camera (ZED, ZED Mini, ZED 2, ZED 2i, ZED X, …), or existing `.svo` / `.svo2`
  recordings made with one.
- An NVIDIA GPU with CUDA. The ZED SDK requires it even for replaying recordings — depth is
  recomputed from the stereo pair on playback, it is not stored in the file.

**Software**

- [ZED SDK](https://www.stereolabs.com/developers/release/) 3.x, 4.x or 5.x, with its CUDA
  dependency. The code detects the SDK generation at run time and adapts to the renamed enums and
  relocated calibration data, so the same scripts work across all three.
- The ZED Python API (`pyzed.sl`), which ships with the SDK — **it is not on PyPI**.
- Python 3.8+.

## Installation

1. **Install the ZED SDK** for your platform from
   <https://www.stereolabs.com/developers/release/> and verify it with the bundled `ZED_Explorer`
   or `ZED_Diagnostic` tool.

2. **Install the ZED Python API.** The SDK provides a `get_python_api.py` script:

   ```bash
   python -m pip install cython numpy
   python /usr/local/zed/get_python_api.py
   ```

   On Windows the script lives in the SDK install directory, by default
   `C:\Program Files (x86)\ZED SDK\get_python_api.py`.

   Verify:

   ```bash
   python -c "import pyzed.sl as sl; print(sl.Camera.get_sdk_version())"
   ```

3. **Install the Python dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **(Optional) Download the ZED AI models** if you want to use the neural depth modes:

   ```bash
   python download_zed_models.py
   ```

There is no `pip install` step for this package itself — the scripts are run directly from the
source tree.

## Quick start

Process a single recording. The CSV is written next to the input file as `take1.csv`:

```bash
python zed/save_6d_palm.py /path/to/take1.svo2
```

Watch the tracking while it runs, and choose where the CSV goes:

```bash
python zed/save_6d_palm.py /path/to/take1.svo2 -o results/take1.csv --display
```

Process every recording under a directory tree, collecting the results in one place:

```bash
python batch_process_svo.py /path/to/recordings -o results/
```

Run on the live camera instead of a file:

```bash
python zed/save_6d_palm.py --live --display -o live_session.csv
```

## Output format

One CSV row per processed frame. Coordinates are in the **ZED camera coordinate frame** (the ZED
SDK default: X right, Y down, Z forward, origin at the left camera's optical centre) and in the
unit chosen with `--units` (metres by default). Angles are in **degrees**.

| Column | Meaning |
| --- | --- |
| `frame` | Frame index in the SVO file (or a running counter in live mode) |
| `timestamp_ns` | Camera timestamp of the image, in nanoseconds |
| `left_detected`, `right_detected` | `1` when all 21 landmarks of that hand were resolved in 3D, else `0` |
| `left_palm_x/y/z`, `right_palm_x/y/z` | Palm centroid |
| `left_yaw_deg`, `left_pitch_deg`, `left_roll_deg` | Palm orientation (same for `right_*`) |
| `left_index_tip_x/y/z`, `right_index_tip_x/y/z` | Index fingertip |
| `left_thumb_tip_x/y/z`, `right_thumb_tip_x/y/z` | Thumb tip |

That is 28 columns. With `--landmarks`, 126 more are appended —
`left_lm00_x` … `left_lm20_z` and `right_lm00_x` … `right_lm20_z` — following the
[MediaPipe hand landmark numbering](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).

**Rows are always written, even when no hand is visible.** Undetected hands are reported as zeros;
always filter on `left_detected` / `right_detected` rather than testing for non-zero coordinates:

```python
import pandas as pd

df = pd.read_csv("results/take1.csv")
right = df[df.right_detected == 1]
print(right[["frame", "right_palm_x", "right_palm_y", "right_palm_z"]].describe())
```

A hand counts as detected only when MediaPipe found it **and** the wrist pixel had valid depth. A
hand that is visible but has no usable depth (out of the `--depth-min`/`--depth-max` range,
occluded, or filtered out by `--depth-confidence`) is reported as undetected.

## Scripts

Every script supports `--help`.

### `zed/save_6d_palm.py` — extract poses to CSV

The main entry point.

```
python zed/save_6d_palm.py [INPUT] [options]
```

| Option | Description |
| --- | --- |
| `INPUT` | Path to an `.svo2` / `.svo` recording. Omit it (or pass `--live`) to use the connected camera. |
| `--live` | Capture from the ZED camera instead of a file. |
| `-o`, `--output` | Output CSV path. Default: `<input name>.csv` next to the input, or `hand_pose_live.csv` for live capture. |
| `--output-dir` | Directory for the default output name. Ignored when `--output` is given. |
| **Processing** | |
| `--frame-skip N` | Process every Nth frame (default `1`). |
| `--start-frame` / `--end-frame` | Restrict processing to a range of SVO frames. |
| `--max-frames N` | Stop after N processed frames — useful for a quick test, and to bound live captures. |
| `--landmarks` | Also write all 21 landmarks per hand. |
| **Camera / SVO** | |
| `--resolution` | `AUTO`, `VGA`, `SVGA`, `HD720`, `HD1080`, `HD1200`, `HD2K`, `QHDPLUS`, `HD4K` (default `HD720`). Live capture only — an SVO replays at its recorded resolution. |
| `--fps` | Live capture frame rate (default `30`). |
| `--depth-mode` | `NONE`, `PERFORMANCE`, `QUALITY`, `ULTRA`, `NEURAL_LIGHT`, `NEURAL`, `NEURAL_PLUS` (default `ULTRA`). Values missing from your SDK fall back to the closest available one. |
| `--units` | `MILLIMETER`, `CENTIMETER`, `METER`, `INCH`, `FOOT` (default `METER`). |
| `--depth-min` / `--depth-max` | Depth clipping range, in `--units` (default `0.3` / `40.0`). |
| `--depth-confidence` | Depth confidence threshold 0–100 (default `100`). Lower discards more uncertain depth pixels. |
| `--svo-real-time` | Replay at the recorded frame rate instead of as fast as possible. Drops frames — leave off for batch work. |
| **Hand tracking** | |
| `--max-hands` | Maximum hands per frame (default `2`). |
| `--detection-confidence` | MediaPipe detection threshold 0–1 (default `0.5`). |
| `--tracking-confidence` | MediaPipe tracking threshold 0–1 (default `0.7`). |
| `--model-complexity` | `0` (fast) or `1` (accurate, default). |
| `--swap-handedness` | Swap the reported left and right hands — see the note below. |
| **Display / logging** | |
| `--display` | Show the annotated video while processing (press `q` to stop early). |
| `--no-annotate` | Skip the landmark overlay even with `--display`. |
| `-q`, `--quiet` | Suppress progress output. |

Processing can be interrupted with `Ctrl+C` (or `q` with `--display`); the frames handled up to
that point are still written to the CSV.

> **Handedness.** MediaPipe labels hands assuming a *mirrored* (selfie-view) image. This package
> feeds it the unmirrored ZED left image, so the default mapping is already inverted to compensate
> and produces anatomically correct labels for a camera facing the subject. If your setup mirrors
> the view — a camera looking over the subject's shoulder, or a recording flipped in
> post-processing — the labels come out reversed; add `--swap-handedness` to correct them. Verify
> once with `--display` on a recording where you know which hand is which.

### `batch_process_svo.py` — process many recordings

Runs `save_6d_palm.py` over a whole set of recordings. Inputs can be **files, directories or glob
patterns**, in any combination, so no particular dataset layout is assumed. Each recording runs in
its own subprocess, so one bad file cannot abort the batch.

```
python batch_process_svo.py INPUT [INPUT ...] [options]
```

| Option | Description |
| --- | --- |
| `INPUT` | Recordings, directories to search, or glob patterns. |
| `-o`, `--output-dir` | Where the CSVs go. Default: next to each recording. |
| `--flat` | Put every CSV directly in `--output-dir` instead of mirroring the input folder structure. |
| `--pattern` | Glob applied inside each input directory, e.g. `"*/session*/frames/*.svo2"`. Overrides `--extensions` and `--no-recursive`. |
| `--extensions` | Comma-separated extensions to search for (default `.svo2,.svo`). |
| `--no-recursive` | Only look at the top level of each input directory. |
| `--skip-existing` | Skip recordings whose CSV already exists — makes an interrupted batch resumable. |
| `--stop-on-error` | Abort as soon as one recording fails. |
| `--capture-output` | Hide per-file output, showing it only on failure. |
| `--dry-run` | Print the commands that would run, then exit. |
| `--worker` / `--python` | Use a different processing script or interpreter. |

**Any option it does not recognise is forwarded verbatim to `save_6d_palm.py`**, so every camera
and tracking switch is available here too:

```bash
# whole tree, CSVs next to each recording
python batch_process_svo.py /data/study

# mirror /data/study's folder structure under results/, every 2nd frame
python batch_process_svo.py /data/study -o results/ --frame-skip 2

# only the recordings matching a layout of your own
python batch_process_svo.py /data --pattern "*/session*/frames/*.svo2"

# explicit files with forwarded tracking options
python batch_process_svo.py a.svo2 b.svo2 --max-hands 1 --depth-mode NEURAL

# check what would happen first
python batch_process_svo.py /data/study -o results/ --dry-run
```

The exit code is `0` when every recording succeeded and `1` otherwise, and a summary of failures
is printed at the end.

### `zed/show3d_matplot.py` / `zed/show3d_open3d.py` — live preview

Interactive viewers for checking the camera placement, depth range and tracking thresholds before
committing to a batch. Both show the colour image and the depth view, plus a 3D skeleton —
`show3d_matplot.py` in matplotlib, `show3d_open3d.py` in an Open3D window (faster and easier to
navigate). Both accept all the camera and tracking options listed above. Press `q` or `Esc` to quit.

```bash
python zed/show3d_matplot.py                            # live camera
python zed/show3d_matplot.py take1.svo2                 # replay a recording
python zed/show3d_matplot.py take1.svo2 --frame 250     # inspect one frame
python zed/show3d_matplot.py take1.svo2 --save-dir out/ # save frames and plots
python zed/show3d_open3d.py take1.svo2
```

Extra options: `--scale` (preview window size), `--zoom` and `--no-plot` (matplotlib version),
`--no-video` (Open3D version), `--save-dir` (writes `images/`, `depth/` and `plots/`).

### `zed/svo_tools/` — record, play back, export

Command-line adaptations of the Stereolabs SVO samples.

```bash
# Record from a live camera; Ctrl+C to stop
python zed/svo_tools/record.py out/take1.svo2 --resolution HD1080 --fps 30 --compression H265

# Play back a recording, optionally writing a video ('s' saves a snapshot, 'q' quits)
python zed/svo_tools/playback.py take1.svo2
python zed/svo_tools/playback.py take1.svo2 -o take1.mp4 --no-display

# Export to a side-by-side video or an image sequence
python zed/svo_tools/save_to_avi.py take1.svo2 take1.avi --mode video-left-right
python zed/svo_tools/save_to_avi.py take1.svo2 frames/ --mode images-left-depth16
```

`save_to_avi.py` export modes (the numeric aliases from the original Stereolabs sample still work):

| Mode | Alias | Output |
| --- | --- | --- |
| `video-left-right` | `0` | Side-by-side left + right video |
| `video-left-depth` | `1` | Side-by-side left + depth-view video |
| `images-left-right` | `2` | Left/right PNG sequence |
| `images-left-depth` | `3` | Left/depth-view PNG sequence |
| `images-left-depth16` | `4` | Left PNG + 16-bit depth PNG sequence |

### `download_zed_models.py` — fetch the ZED AI models

The neural depth modes need model files in the SDK resources directory. The SDK downloads them on
first use; this script fetches them up front, which helps on machines that are offline at run time
or behind a proxy.

```bash
python download_zed_models.py                      # auto-detect the installed SDK version
python download_zed_models.py --sdk-version 4
python download_zed_models.py --output-dir ./models/
python download_zed_models.py --print-url-only     # just list the URLs
```

The default destination is the SDK resources directory for your platform
(`/usr/local/zed/resources/` on Linux, `C:/ProgramData/Stereolabs/resources/` on Windows), which
can be overridden with `--output-dir` or the `ZED_RESOURCES_DIR` environment variable. Writing
there usually needs administrator/root rights. Models already present are skipped unless
`--overwrite` is passed.

### `zed/gui/` — ZED SDK GUI tools

Convenience shell launchers for the GUI tools bundled with the SDK (`ZED_Explorer`,
`ZED_Depth_Viewer`, `ZED_Calibration`, `ZED_Diagnostic`). They search the usual install locations;
point `ZED_TOOLS_DIR` at your SDK if it is installed elsewhere:

```bash
ZED_TOOLS_DIR=/opt/zed/tools ./zed/gui/Explorer.sh
```

## Using the package as a library

The building blocks under `zed/HandTrackingModule/` can be used directly. Add `zed/` to
`sys.path`, or run from inside it:

```python
import sys
sys.path.insert(0, "zed")

import pyzed.sl as sl
from HandTrackingModule.Zed import Zed, is_end_of_svo
from HandTrackingModule.HandTracking import HandTracking

detector = HandTracking(maxHands=2, detectionCon=0.5, trackCon=0.7)

with Zed("take1.svo2", depth_mode="NEURAL", coordinate_units="MILLIMETER") as cam:
    cam.print_information()
    while True:
        err = cam.grab()
        if err != sl.ERROR_CODE.SUCCESS:
            if is_end_of_svo(err):
                break
            raise RuntimeError(err)

        cam.get_image()
        detector.findHands(cam.img, draw=False)
        left, right = detector.find_positions(
            cam.depth_img, cam.point_cloud, cam.camera_params, draw=False
        )

        if right.shape == (21, 3):
            print(detector.calculate_centroid(right),
                  detector.calculate_orientation(right))
```

| Class / function | Purpose |
| --- | --- |
| `Zed` | Opens a live camera or an SVO with every setting as a keyword argument; hides SDK 3/4/5 API differences. Supports the `with` statement. |
| `HandTracking` | MediaPipe detection plus the 3D back-projection, centroid and orientation maths. |
| `Vis3D` | Open3D viewer for the landmarks. |
| `is_end_of_svo(err)` | True when an error code means "end of file", whatever the SDK calls it. |
| `resolve_depth_mode` / `resolve_resolution` | Resolve an enum name against the installed SDK, falling back to the closest available value. |

`HandTracking.find_positions()` is also available under its original name `findpostion()` for
compatibility with existing code.

## Tuning and performance

MediaPipe inference dominates the run time. In rough order of effect:

| Change | Effect |
| --- | --- |
| `--frame-skip 2` (or more) | Near-linear speedup; enough for slow hand motion. |
| `--model-complexity 0` | Noticeably faster MediaPipe inference, slightly less accurate landmarks. |
| Drop `--display` | Displaying and annotating costs real time. It is off by default. |
| `--max-hands 1` | Skips the second hand when you only care about one. |
| `--depth-mode PERFORMANCE` | Cheaper depth than `ULTRA`/`NEURAL`, at the cost of noisier 3D positions. |
| `--detection-confidence 0.7` | Fewer false positives and less tracking work, at the risk of dropped frames. |

Accuracy notes:

- Set `--depth-min` / `--depth-max` to the range where the hands actually are. A tight range
  rejects background depth noise.
- Lower `--depth-confidence` (e.g. `50`) to discard uncertain depth pixels. This raises the number
  of frames marked undetected but makes the surviving 3D positions cleaner.
- The neural depth modes give markedly better depth on hands than `PERFORMANCE`, at a GPU cost.
  Run `download_zed_models.py` first.

## Troubleshooting

**`ModuleNotFoundError: No module named 'pyzed'`**
The ZED Python API is not installed. It ships with the SDK, not with pip — run the SDK's
`get_python_api.py` (see [Installation](#installation)).

**`Unable to open the ZED camera: CAMERA_NOT_DETECTED` / `INVALID_SVO_FILE`**
For a live camera: check the USB connection and that no other process (including `ZED_Explorer`)
is holding the camera. For a recording: check the path, and that the file was produced by a
compatible SDK version. `.svo2` files require SDK 4.1 or newer.

**Every row has `left_detected = 0` and `right_detected = 0`**
Either MediaPipe is not finding hands, or the wrist has no valid depth. Run
`python zed/show3d_matplot.py <file>` to see which: if the overlay shows landmarks but the CSV is
empty, it is a depth problem — widen `--depth-min`/`--depth-max`, lower `--depth-confidence`, or
switch to a neural `--depth-mode`.

**Left and right hands are swapped** — add `--swap-handedness`; see the note under
[`save_6d_palm.py`](#zedsave_6d_palmpy--extract-poses-to-csv).

**`AttributeError` on an `sl.` enum**
An SDK version this code has not been mapped to. The `_DEPTH_MODE_FALLBACKS` and
`_RESOLUTION_FALLBACKS` tables at the top of `zed/HandTrackingModule/Zed.py` are the place to add
new names.

**Neural depth mode fails to start**
The AI models are missing — run `python download_zed_models.py`.

**Processing runs at real-time speed instead of as fast as possible**
`--svo-real-time` is set. Remove it for batch work.

**`OpenCV video writer cannot be opened`**
The requested codec is unavailable on your system. Try `--codec mp4v` (playback) or
`--codec MJPG` (export), or a different output extension.

## Project structure

```
6D_hand_pose_tracking/
├── README.md
├── requirements.txt
├── batch_process_svo.py            # process many recordings
├── download_zed_models.py          # fetch the ZED SDK AI models
└── zed/
    ├── save_6d_palm.py             # main entry point: SVO/live -> CSV
    ├── show3d_matplot.py           # preview with a matplotlib 3D plot
    ├── show3d_open3d.py            # preview with an Open3D viewer
    ├── HandTrackingModule/
    │   ├── Zed.py                  # ZED camera/SVO wrapper, SDK 3/4/5 compatible
    │   ├── HandTracking.py         # MediaPipe detection + 3D landmarks, pose maths
    │   ├── Vis3D.py                # Open3D visualisation
    │   └── cli.py                  # shared command line options
    ├── svo_tools/
    │   ├── record.py               # record an SVO from a live camera
    │   ├── playback.py             # replay an SVO, optionally to video
    │   └── save_to_avi.py          # export an SVO to video or images
    └── gui/                        # launchers for the ZED SDK GUI tools
```

## Limitations

- **Wrist-anchored depth.** Only the wrist landmark is sampled from the point cloud; the other 20
  are back-projected from it using MediaPipe's relative landmark geometry. This is robust to noisy
  per-finger depth, but the accuracy of the whole hand depends on one pixel having good depth. If
  the wrist is occluded, the hand is reported as undetected even when the fingers are visible.
- **Single camera, no temporal filtering.** Poses are computed independently per frame; there is no
  smoothing, no Kalman filter and no multi-view fusion. Expect frame-to-frame jitter, and smooth
  downstream if your application needs it.
- **Orientation convention.** Yaw/pitch/roll come from the palm-plane normal (landmarks 0, 5, 17)
  and are not a general rotation matrix decomposition. They are consistent frame to frame and
  suitable for tracking relative changes, but should not be assumed to match another package's
  Euler-angle convention. Enable `--landmarks` and compute your own frame if you need a specific
  convention.
- **Handedness comes from MediaPipe** and depends on whether the view is mirrored — see
  `--swap-handedness`.

## License

The files under `zed/svo_tools/` are adapted from the
[Stereolabs zed-examples](https://github.com/stereolabs/zed-examples) samples and retain their
original copyright header and license.

No license has been chosen for the rest of this package yet. Add a `LICENSE` file before publishing
if you intend others to reuse it — without one, default copyright applies and others have no
permission to use, modify or redistribute the code.

Third-party components carry their own terms: the ZED SDK is proprietary Stereolabs software,
MediaPipe and OpenCV are Apache-2.0, and Open3D is MIT.
