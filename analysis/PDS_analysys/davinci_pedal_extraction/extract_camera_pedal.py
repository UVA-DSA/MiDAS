import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================================
# --------------------- USER CONFIG ---------------------------
# ============================================================

METRIC = "mean_V"                       # one of: mean_V, mean_R, mean_G, mean_B
THRESHOLD = 20.0                        # difference threshold
HYSTERESIS = 5.0                        # margin to turn off change
BASELINE_FRAMES = 30                    # frames used to estimate baseline
USE_EMA = False                         # True = exponential moving average baseline
EMA_ALPHA = 0.1                         # smoothing factor if USE_EMA
DOWNSAMPLE = 1                          # process every Nth frame

# CPU parallelism
MAX_PROCS = 32                          # request up to this many processes

ROOT_PATH = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/Processed/"
ROI_CSV   = "./patterns/region_output.csv"  # ROI CSV from marking tools

TRIALS = [
    "S105_T1","S106_T1","S106_T2","S112_T1","S112_T2","S116_T1","S116_T2","S116_T4","S116_T5",
    "S118_T1","S200_T1","S201_T1","S201_T2","S202_T1","S203_T1","S204_T1","S209_T2","S210_T1",
    "S214_T1","S214_T4","S214_T6","S215_T3","S215_T4","S217_T2","S217_T3","S217_T4","S218_T1","S219_T1"
]
# ============================================================


def read_rois(csv_path: str):
    df = pd.read_csv(csv_path)
    return [
        {"x": int(r["x"]), "y": int(r["y"]), "w": int(r["w"]),
         "h": int(r["h"]), "name": str(r["name"])}
        for _, r in df.iterrows()
    ]


def crop_roi(img, roi):
    h_img, w_img = img.shape[:2]
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    x2, y2 = x + w, y + h
    x, y = max(0, x), max(0, y)
    x2, y2 = min(w_img, x2), min(h_img, y2)
    return img[y:y2, x:x2] if x2 > x and y2 > y else None


def roi_metrics(bgr_crop):
    if bgr_crop is None or bgr_crop.size == 0:
        return {"mean_R": np.nan, "mean_G": np.nan,
                "mean_B": np.nan, "mean_V": np.nan}
    mean_b = float(np.mean(bgr_crop[:, :, 0]))
    mean_g = float(np.mean(bgr_crop[:, :, 1]))
    mean_r = float(np.mean(bgr_crop[:, :, 2]))
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    mean_v = float(np.mean(hsv[:, :, 2]))
    return {"mean_R": mean_r, "mean_G": mean_g,
            "mean_B": mean_b, "mean_V": mean_v}


def make_output_dirs(out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(out_dir, "plots").mkdir(parents=True, exist_ok=True)


def process_video(video_path: str, roi_csv: str, out_dir: str, trial_id: str, pbar_position: int = 0):
    rois = read_rois(roi_csv)
    make_output_dirs(out_dir)
    out_csv = Path(out_dir, f"{trial_id}_camera_pedal_gt_keshara.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps) if fps and fps > 1e-6 else None

    baselines = {r["name"]: None for r in rois}
    states = {r["name"]: False for r in rois}
    baseline_buffers = {r["name"]: [] for r in rois}

    records = []
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    pbar = tqdm(total=total_frames, desc=f"[{Path(video_path).name}]", position=pbar_position, leave=True)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % DOWNSAMPLE != 0:
            frame_idx += 1
            pbar.update(1)
            continue

        time_s = (frame_idx / fps) if fps else np.nan

        for roi in rois:
            name = roi["name"]
            crop = crop_roi(frame, roi)
            m = roi_metrics(crop)
            val = float(m.get(METRIC, np.nan))

            # Update baseline
            if USE_EMA:
                if baselines[name] is None:
                    baselines[name] = val
                else:
                    baselines[name] = EMA_ALPHA * val + (1 - EMA_ALPHA) * baselines[name]
                base = baselines[name]
            else:
                if len(baseline_buffers[name]) < BASELINE_FRAMES:
                    baseline_buffers[name].append(val)
                    baselines[name] = np.nanmean(baseline_buffers[name])
                base = baselines[name]

            delta = float(val - base) if base is not None else np.nan
            adelta = abs(delta) if not np.isnan(delta) else np.nan

            changed = states[name]
            if not np.isnan(adelta):
                if not changed and adelta >= THRESHOLD:
                    changed = True
                elif changed and adelta <= max(0.0, THRESHOLD - HYSTERESIS):
                    changed = False
            states[name] = changed

            records.append({
                "video": video_path,         # (fix) record actual path
                "frame_idx": frame_idx,
                "time_s": time_s,
                "roi_name": name,
                **m,
                f"{METRIC}_baseline": base,
                f"{METRIC}_delta": delta,
                "changed": int(changed),
                "pressed": changed,
            })

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    df = pd.DataFrame.from_records(records)
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv}")
    print("Extract complete.")


# ---------------------- multiprocessing ----------------------

def _trial_paths(root_path: str, trial: str) -> Optional[str]:
    """Return video path for this trial, or None if missing."""
    # Your layout: ROOT_PATH/<trial>/synched_data/<trial>.mp4
    video_path = os.path.join(root_path, trial, "synched_data", f"{trial}.mp4")
    return video_path if os.path.exists(video_path) else None


def _worker(args):
    """Run one trial in a separate process."""
    # Tame thread oversubscription inside each worker
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    trial, video_path, roi_csv, out_dir, position = args

    if video_path is None:
        return (trial, False, "Video file not found")

    try:
        process_video(video_path, roi_csv, out_dir, trial, pbar_position=position)
        return (trial, True, "OK")
    except Exception as e:
        return (trial, False, f"{type(e).__name__}: {e}")


def main():
    jobs = []
    for i, trial in enumerate(TRIALS):
        video_path = _trial_paths(ROOT_PATH, trial)
        out_dir = os.path.join(ROOT_PATH, trial, "synched_data")
        jobs.append((trial, video_path, ROI_CSV, out_dir, i))

    if not jobs:
        print("No trials to run.")
        return

    procs = min(MAX_PROCS, max(1, cpu_count()))
    print(f"Launching pool with {procs} processes for {len(jobs)} trials…")

    # One trial per worker; maxtasksperchild=1 keeps memory stable
    with Pool(processes=procs, maxtasksperchild=1) as pool:
        for trial, ok, msg in pool.imap_unordered(_worker, jobs, chunksize=1):
            print(("✅" if ok else "❌"), trial, "-", msg)

    print("All done.")


if __name__ == "__main__":
    main()
