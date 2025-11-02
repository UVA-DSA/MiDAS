import os
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ====================== USER CONFIG ======================
ROOT_PATH = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/"

# Euclidean distance in mean BGR color space to flag a change (kept for API; not used in label-based swap)
COLOR_CHANGE_THRESHOLD = 15.0

# Debug window (slow; disabled under multiprocessing)
DEBUG_SHOW = False
DEBUG_SCALE = 0.5

# HSV-driven thresholds (OpenCV HSV: H in [0,179], S,V in [0,255])
BLACK_V_THR = 40
BLUE_H_LO = 95
BLUE_H_HI = 135
BLUE_S_THR = 40
BLUE_V_MIN = 45
BLUE_DOMINANCE_RATIO = 1.08

# Frame skipping for speed (1 = process all frames)
DOWNSAMPLE = 1

# How many frames to keep the swap flag high after a detected swap
EXTEND_FRAMES = 15

# Parallelism
MAX_PROCS = 32

# Shared ROI layout (x, y, w, h)
w = h = 11
REGIONS = [
    (356,  999, w, h),  # Instrument 1
    (673,  999, w, h),  # Instrument 2
    (990,  999, w, h),  # Instrument 3
    (1307, 999, w, h),  # Instrument 4
]

# Trials and their GT folders (same idea as your other script)
TRIALS = [    
    "S105_T1","S106_T1","S106_T2","S112_T1","S112_T2","S116_T1","S116_T2","S116_T4","S116_T5",
    "S118_T1","S200_T1","S201_T1","S201_T2","S202_T1","S203_T1","S204_T1","S209_T2","S210_T1",
    "S214_T1","S214_T4","S214_T6","S215_T3","S215_T4","S217_T2","S217_T3","S217_T4","S218_T1","S219_T1",
]
GT_VIDEO_TRIALS = [
    "INGUNIAL_S105_T1_2024-07-17",
    "INGUINAL_S106_T1_2024-07-17",
    "INGUINAL_S106_T2_2024-07-17",
    "InguinalH_S112_T1_2024-07-18",
    "InguinalH_S112_T2_2024-07-18",
    "VentralH_S116_T1_2024-07-19",
    "VentralHR2_S116_T2_2024-07-19",
    "VentralHR2_S116_T4_2024-07-19",
    "VentralHR2_S116_T5_2024-07-19",
    "VENTRAL_S118_T1_2024-07-19",
    "VH4_S200_T1_2024-07-16",
    "VH_S201_T1_2024-07-16",
    "VH4_S201_T2_2024-07-16",
    "VH1_S202_T1_2024-07-16",
    "VH1_S203_T1_2024-07-16",
    "VH1_S204_T1_2024-07-16",
    "InguinalH_S209_T2_2024-07-17",
    "InguinalH_S210_T1_2024-07-17",
    "InguinalH_S214_T1_2024-07-18",
    "InguinalH_S214_T4_2024-07-18",
    "InguinalH_S214_T6_2024-07-18",
    "INGUINAL_S215_T3_2024-07-18",
    "INGUINAL_S215_T4_2024-07-18",
    "VentralHR2_S217_T2_2024-07-19",
    "VentralHR2_S217_T3_2024-07-19",
    "VentralHR2_S217_T4_2024-07-19",
    "VentralH_S218_T1_2024-07-19",
    "VentralH_S219_T1_2024-07-19",
]
# ========================================================


def region_mean_bgr(frame: np.ndarray, region) -> np.ndarray:
    x, y, w, h = region
    H, W = frame.shape[:2]
    x = max(0, min(x, W - 1))
    y = max(0, min(y, H - 1))
    w = max(0, min(w, W - x))
    h = max(0, min(h, H - y))
    if w <= 0 or h <= 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    mean = roi.reshape(-1, 3).mean(axis=0)  # BGR
    return mean.astype(float)


def classify_overall_color(mean_bgr: np.ndarray,
                           black_thr: float = 35.0,
                           blue_ratio_thr: float = 0.60) -> int:
    """Return 0=BLACK, 1=BLUE."""
    b, g, r = mean_bgr.tolist()
    bgr_uint8 = np.array([[np.clip([b, g, r], 0, 255)]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < BLACK_V_THR:
        return 0
    if BLUE_H_LO <= h <= BLUE_H_HI and s >= BLUE_S_THR and v >= BLUE_V_MIN:
        return 1
    max_rg = max(r, g)
    if max_rg > 0 and b >= BLUE_DOMINANCE_RATIO * max_rg and v >= BLUE_V_MIN:
        return 1
    return 0


def detect_arm_swaps_from_labels(labels_per_frame, extend_frames: int):
    """
    labels_per_frame: list[T] of list[R] (0/1 per region)
    Swap when exactly two regions switch label at the same time (then extend).
    """
    T = len(labels_per_frame)
    if T == 0:
        return []
    R = len(labels_per_frame[0]) if labels_per_frame[0] is not None else 0
    changed = [np.zeros(R, dtype=bool) for _ in range(T)]
    for t in range(1, T):
        prev = labels_per_frame[t - 1]
        curr = labels_per_frame[t]
        changed[t] = (np.array(prev) != np.array(curr))

    arm_swap = np.zeros(T, dtype=int)
    for t in range(1, T):
        if changed[t].sum() == 2:
            t_end = min(T, t + extend_frames)
            arm_swap[t:t_end] = 1
    return arm_swap.tolist()


def analyze_regions_and_save(video_path: str,
                             regions,
                             output_csv: str,
                             threshold: float = COLOR_CHANGE_THRESHOLD,  # kept for signature parity
                             extend_frames: int = EXTEND_FRAMES,
                             debug_show: bool = False,
                             downsample: int = DOWNSAMPLE,
                             pbar_position: int = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Error: Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    frame_id = -1

    colors_per_frame = []  # list[T] of list[R,3]
    labels_per_frame = []  # list[T] of list[R]

    pbar = tqdm(total=total_frames, desc=f"[{Path(video_path).name}]", position=pbar_position, leave=True)
    paused = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            if frame_id % downsample != 0:
                pbar.update(1)
                continue

            region_means = []
            region_labels = []
            for i, region in enumerate(regions):
                mean_bgr = region_mean_bgr(frame, region)
                region_means.append(mean_bgr)
                label = classify_overall_color(mean_bgr)
                region_labels.append(label)

                if debug_show:
                    x, y, w, h = region
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    b, g, r = mean_bgr.tolist()
                    label_str = 'BLUE' if label == 1 else 'BLACK'
                    text = f"R{i+1}: {label_str} ({int(r)},{int(g)},{int(b)})"
                    text_y = y - 6 if y - 6 > 12 else y + h + 16
                    cv2.putText(frame, text, (x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

            colors_per_frame.append(region_means)
            labels_per_frame.append(region_labels)

            if debug_show:
                display = frame
                if DEBUG_SCALE != 1.0:
                    display = cv2.resize(display, None, fx=DEBUG_SCALE, fy=DEBUG_SCALE, interpolation=cv2.INTER_AREA)
                cv2.imshow('arm-swap-debug', display)
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('p'):
                    paused = not paused

            pbar.update(1)
    finally:
        cap.release()
        pbar.close()
        if debug_show:
            try:
                cv2.destroyWindow('arm-swap-debug')
            except Exception:
                pass

    # Post-processing: label-based swap detection
    arm_swap_flags = detect_arm_swaps_from_labels(labels_per_frame, extend_frames)

    # Write CSV
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['obs_frame_idx']
        for i in range(len(regions)):
            header += [f'Region_{i+1}_B', f'Region_{i+1}_G', f'Region_{i+1}_R', f'Region_{i+1}_Label']
        header += ['Arm_Swap']
        csvwriter.writerow(header)

        for idx, region_means in enumerate(colors_per_frame):
            row = [idx]
            for j, mean_bgr in enumerate(region_means):
                b, g, r = mean_bgr.tolist()
                label_str = 'BLUE' if labels_per_frame[idx][j] == 1 else 'BLACK'
                row += [f"{b:.2f}", f"{g:.2f}", f"{r:.2f}", label_str]
            row += [int(arm_swap_flags[idx])]
            csvwriter.writerow(row)


# ---------------------- helpers & multiprocessing ----------------------

def _pick_mkv_in_video_dir(video_dir: str) -> Optional[str]:
    """Return path to newest .mkv inside <GT_FOLDER>/video/, else None."""
    if not os.path.isdir(video_dir):
        return None
    cands = [f for f in os.listdir(video_dir) if f.lower().endswith(".mkv")]
    if not cands:
        return None
    cands.sort(key=lambda fn: os.path.getmtime(os.path.join(video_dir, fn)), reverse=True)
    return os.path.join(video_dir, cands[0])


def _worker(args):
    """One process per trial."""
    # Tame thread oversubscription
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    trial, gt_folder, position = args

    # Input video: ROOT_PATH/<gt_folder>/video/<*.mkv>
    video_dir = os.path.join(ROOT_PATH, gt_folder, "video")
    video_path = _pick_mkv_in_video_dir(video_dir)
    if not video_path:
        return (trial, False, f"No .mkv in {video_dir}")

    # Output CSV: ROOT_PATH/Processed/<trial>/synched_data/<trial>_arm_swap_pedal_gt.csv
    out_dir = os.path.join(ROOT_PATH, "Processed", trial, "synched_data")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{trial}_arm_swap_pedal_gt.csv")

    try:
        analyze_regions_and_save(
            video_path=video_path,
            regions=REGIONS,
            output_csv=out_csv,
            threshold=COLOR_CHANGE_THRESHOLD,
            extend_frames=EXTEND_FRAMES,
            debug_show=False,
            downsample=DOWNSAMPLE,
            pbar_position=position
        )
        return (trial, True, f"Wrote {out_csv}")
    except Exception as e:
        return (trial, False, f"{type(e).__name__}: {e}")


def main():
    trial_mapping = dict(zip(TRIALS, GT_VIDEO_TRIALS))
    jobs = []
    for i, (trial, gt_folder) in enumerate(trial_mapping.items()):
        jobs.append((trial, gt_folder, i))

    if not jobs:
        print("No trials to run.")
        return

    procs = min(MAX_PROCS, max(1, cpu_count()))
    print(f"Launching pool with {procs} processes for {len(jobs)} trials…")

    with Pool(processes=procs, maxtasksperchild=1) as pool:
        for trial, ok, msg in pool.imap_unordered(_worker, jobs, chunksize=1):
            print(("✅" if ok else "❌"), trial, "-", msg)

    print("All done.")


if __name__ == "__main__":
    main()
