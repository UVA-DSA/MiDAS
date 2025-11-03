
import os
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

# ===================== USER CONFIG =====================
ROOT_PATH = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/"
PATTERN_PATH = "./patterns/camera_pattern.png"

# concurrency
MAX_PROCS = 32  # request up to 32 cores

# template-match settings
THRESHOLD = 0.80
DOWNSAMPLE = 1   # process every Nth frame for speed (1 = all)

# ROI: (x, y, w, h)
w = h = 11
REGIONS = [
    (365,  999, w, h),   # Instrument 1
    (682,  999, w, h),   # Instrument 2
    (999,  999, w, h),   # Instrument 3
    (1316, 999, w, h),   # Instrument 4
]

# Your trial → GT folder mapping
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
# =======================================================


def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(0, min(int(w), W - x))
    h = max(0, min(int(h), H - y))
    if w <= 0 or h <= 0:
        return None
    return (x, y, w, h)

def ensure_gray_u8(img):
    if img is None:
        return None
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

def detect_pattern_in_region(gray_frame, pattern, region, threshold=THRESHOLD):
    """Safe template match: returns bool."""
    H, W = gray_frame.shape[:2]
    clamped = clamp_roi(*region, W, H)
    if clamped is None:
        return False
    x, y, w, h = clamped
    roi = gray_frame[y:y+h, x:x+w]
    if roi.size == 0:
        return False

    ph, pw = pattern.shape[:2]
    if ph == 0 or pw == 0:
        return False
    if ph > h or pw > w:
        # template larger than ROI → cannot match
        return False

    res = cv2.matchTemplate(roi, pattern, cv2.TM_CCOEFF_NORMED)
    max_val = float(cv2.minMaxLoc(res)[1])
    return max_val >= threshold

def detect_pattern_in_video(video_path, pattern_path, regions, output_csv,
                            threshold=THRESHOLD, downsample=DOWNSAMPLE, pbar_position=0):
    """Process one video; writes CSV; shows a per-frame tqdm progress bar."""
    # Load pattern
    pattern = cv2.imread(pattern_path, cv2.IMREAD_GRAYSCALE)
    if pattern is None:
        raise FileNotFoundError(f"Pattern not found: {pattern_path}")
    pattern = ensure_gray_u8(pattern)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    # Prepare output
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    f = open(output_csv, 'w', newline='')
    writer = csv.writer(f)
    header = ['obs_frame_idx'] + [f"Instrument_{i+1}_Camera" for i in range(len(regions))]
    writer.writerow(header)

    # TQDM progress bar
    pbar = tqdm(total=total_frames, desc=f"[{Path(video_path).name}]", position=pbar_position, leave=True)

    frame_id = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1

            if frame_id % downsample != 0:
                pbar.update(1)
                continue

            gray = ensure_gray_u8(frame)
            row = [frame_id]
            for region in regions:
                detected = detect_pattern_in_region(gray, pattern, region, threshold=threshold)
                row.append(int(detected))
            writer.writerow(row)

            pbar.update(1)
    finally:
        cap.release()
        f.close()
        pbar.close()

def _pick_video_file(video_dir: str) -> Optional[str]:
    """Pick an .mkv (prefer latest by mtime if multiple)."""
    if not os.path.isdir(video_dir):
        return None
    cands = [f for f in os.listdir(video_dir) if f.lower().endswith(".mkv")]
    if not cands:
        return None
    cands = sorted(cands, key=lambda f: os.path.getmtime(os.path.join(video_dir, f)), reverse=True)
    return os.path.join(video_dir, cands[0])

def _make_output_csv(root: str, trial: str) -> str:
    out_dir = os.path.join(root, "Processed", trial, "synched_data")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return os.path.join(out_dir, f"{trial}_camera_pedal_gt_hamid.csv")

def _worker(args):
    """One process per video."""
    # tame thread oversubscription inside each process
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    trial, gt_folder, root, pattern_path, regions, threshold, downsample, position = args

    video_dir = os.path.join(root, gt_folder, "video")
    video_path = _pick_video_file(video_dir)
    if not video_path:
        return (trial, False, f"No .mkv found in {video_dir}")

    out_csv = _make_output_csv(root, trial)

    try:
        print(f"Processing trial {trial} - video: {video_path} → {out_csv}")
        detect_pattern_in_video(
            video_path=video_path,
            pattern_path=pattern_path,
            regions=regions,
            output_csv=out_csv,
            threshold=threshold,
            downsample=downsample,
            pbar_position=position,   # lets tqdm stack bars nicely
        )
        return (trial, True, f"Wrote {out_csv}")
    except Exception as e:
        return (trial, False, f"{type(e).__name__}: {e}")

def main():
    trial_mapping = dict(zip(TRIALS, GT_VIDEO_TRIALS))
    jobs = []
    # assign each job a tqdm "position" so bars stack instead of overwriting
    for i, (trial, gt_video_path) in enumerate(trial_mapping.items()):
        jobs.append((trial, gt_video_path, ROOT_PATH, PATTERN_PATH, REGIONS, THRESHOLD, DOWNSAMPLE, i))

    if not jobs:
        print("No jobs to run.")
        return

    procs = min(MAX_PROCS, max(1, cpu_count()))
    print(f"Launching pool with {procs} processes for {len(jobs)} videos…")

    # maxtasksperchild=1: each worker handles one video → stable memory
    with Pool(processes=procs, maxtasksperchild=1) as pool:
        # chunksize=1: dispatch jobs immediately
        for trial, ok, msg in pool.imap_unordered(_worker, jobs, chunksize=1):
            print(("✅" if ok else "❌"), trial, "-", msg)

    print("All done.")

if __name__ == "__main__":
    main()
