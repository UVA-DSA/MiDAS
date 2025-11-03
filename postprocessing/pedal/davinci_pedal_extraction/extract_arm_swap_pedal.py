#!/usr/bin/env python3
import os
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple, Optional

import cv2
import numpy as np
from tqdm import tqdm

# ===================== USER CONFIG =====================
ROOT_PATH = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/"

# Regions and patches:
REGIONS_CSV = "./patterns/selected_regions.csv"   # CSV with columns: x0,y0,x1,y1
PATCHES_DIR = "./patterns/patches"                         # contains i_on*.png / i_off*.png

# Concurrency
MAX_PROCS = 32
DOWNSAMPLE = 1            # process every Nth frame (1 = all)

# Arm-swap detection knobs
COLOR_CHANGE_THRESHOLD = 15.0   # (kept for compatibility; not used in label-based detector below)
EXTEND_FRAMES = 15              # keep Arm_Swap=1 for N frames after detection
STABILITY_FRAMES = 5            # require labels stable for next N frames after a swap

# Trials and their GT video folders
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


# ---------------- region & template utils ----------------
def _read_regions_csv(csv_path: str) -> List[Tuple[int, int, int, int]]:
    """CSV columns: x0,y0,x1,y1  → returns list of (x, y, w, h)."""
    rects = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x0 = int(row['x0']); y0 = int(row['y0']); x1 = int(row['x1']); y1 = int(row['y1'])
            w = max(0, x1 - x0)
            h = max(0, y1 - y0)
            rects.append((x0, y0, w, h))
    return rects

def _compute_ab_hist(bgr_img: np.ndarray, bins: int = 32) -> np.ndarray:
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1]; b = lab[:, :, 2]
    hist = cv2.calcHist([a, b], [0, 1], None, [bins, bins], [0, 256, 0, 256])
    hist = cv2.normalize(hist, None, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)

def _load_templates(patches_dir: str, num_regions: int):
    """
    For region i (1-based), load lists of on/off templates:
    filenames start with "i_on" / "i_off" (e.g., 1_on.png, 1_off_2.jpg, ...).
    Returns: list of (on_pairs, off_pairs) per region, where each pair is (bgr_img, ab_hist).
    """
    templates = []
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    try:
        all_files = os.listdir(patches_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"Patches directory not found: {patches_dir}")

    for i in range(1, num_regions + 1):
        on_prefix = f"{i}_on"
        off_prefix = f"{i}_off"
        on_files = [fn for fn in all_files if fn.lower().startswith(on_prefix) and fn.lower().endswith(exts)]
        off_files = [fn for fn in all_files if fn.lower().startswith(off_prefix) and fn.lower().endswith(exts)]

        on_pairs = []
        off_pairs = []
        for fn in sorted(on_files):
            img = cv2.imread(os.path.join(patches_dir, fn), cv2.IMREAD_COLOR)
            if img is not None:
                on_pairs.append((img, None))
        for fn in sorted(off_files):
            img = cv2.imread(os.path.join(patches_dir, fn), cv2.IMREAD_COLOR)
            if img is not None:
                off_pairs.append((img, None))

        # Fallback to exact names if needed
        if not on_pairs:
            img = cv2.imread(os.path.join(patches_dir, f"{i}_on.png"), cv2.IMREAD_COLOR)
            if img is not None:
                on_pairs.append((img, None))
        if not off_pairs:
            img = cv2.imread(os.path.join(patches_dir, f"{i}_off.png"), cv2.IMREAD_COLOR)
            if img is not None:
                off_pairs.append((img, None))

        if not on_pairs or not off_pairs:
            raise FileNotFoundError(
                f"Missing template(s) for region {i} in {patches_dir}. Need '{on_prefix}*' and '{off_prefix}*'."
            )

        # precompute histograms
        on_pairs = [(img, _compute_ab_hist(img)) for (img, _) in on_pairs]
        off_pairs = [(img, _compute_ab_hist(img)) for (img, _) in off_pairs]
        templates.append((on_pairs, off_pairs))
    return templates

def _classify_region_by_templates(frame_bgr: np.ndarray,
                                  rect: Tuple[int, int, int, int],
                                  on_templates, off_templates) -> Tuple[int, np.ndarray]:
    """Return (label, mean_bgr) where label ∈ {0,1}; 1 means 'on/BLUE'."""
    x0, y0, w, h = rect
    H, W = frame_bgr.shape[:2]
    if w <= 0 or h <= 0 or x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
        return 0, np.array([np.nan, np.nan, np.nan], float)

    roi = frame_bgr[y0:y0 + h, x0:x0 + w]
    if roi.size == 0:
        return 0, np.array([np.nan, np.nan, np.nan], float)

    roi_hist = _compute_ab_hist(roi)
    # Bhattacharyya distances
    d_on = min(float(cv2.compareHist(roi_hist, h, cv2.HISTCMP_BHATTACHARYYA)) for (_, h) in on_templates)
    d_off = min(float(cv2.compareHist(roi_hist, h, cv2.HISTCMP_BHATTACHARYYA)) for (_, h) in off_templates)
    label = 1 if d_on <= d_off else 0
    mean_bgr = roi.reshape(-1, 3).mean(axis=0).astype(float)
    return label, mean_bgr


# ---------------- arm-swap logic (label-based) ----------------
def detect_arm_swaps_from_labels(labels_per_frame, extend_frames: int, stability_frames: int = 5):
    """
    labels_per_frame: list[T] of list[R] (0/1 per region).
    Detect a swap at frame t if exactly two regions flip at t,
    previous frame was steady, and the next `stability_frames` frames
    keep the new labels. Raise Arm_Swap=1 for `extend_frames` frames.
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
            if t < 2 or changed[t - 1].sum() != 0:
                continue
            new_labels = np.array(labels_per_frame[t])
            if t + stability_frames >= T:
                continue
            stable = True
            for k in range(1, stability_frames + 1):
                if not np.array_equal(labels_per_frame[t + k], new_labels):
                    stable = False
                    break
            if stable:
                t_end = min(T, t + extend_frames)
                arm_swap[t:t_end] = 1
    return arm_swap.tolist()


# ---------------- per-video processing ----------------
def analyze_regions_and_save(video_path: str,
                             regions: List[Tuple[int, int, int, int]],
                             output_csv: str,
                             patches_dir: str,
                             downsample: int = DOWNSAMPLE,
                             extend_frames: int = EXTEND_FRAMES,
                             stability_frames: int = STABILITY_FRAMES,
                             pbar_position: int = 0):
    """Classify each region via on/off templates per frame, then derive Arm_Swap and write CSV."""
    # Load templates once per video
    templates = _load_templates(patches_dir, len(regions))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    f = open(output_csv, 'w', newline='')
    writer = csv.writer(f)
    # numeric labels (0/1) for easier downstream logic
    header = ['obs_frame_idx']
    for i in range(len(regions)):
        header += [f'Region_{i+1}_B', f'Region_{i+1}_G', f'Region_{i+1}_R', f'Region_{i+1}_Label']
    header += ['Arm_Swap']
    writer.writerow(header)

    pbar = tqdm(total=total_frames, desc=f"[{Path(video_path).name}]", position=pbar_position, leave=True)

    colors_per_frame = []   # list[T] of [R,3]
    labels_per_frame = []   # list[T] of [R]

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

            region_means = []
            region_labels = []
            for i, rect in enumerate(regions):
                on_list, off_list = templates[i]
                lbl, mean_bgr = _classify_region_by_templates(frame, rect, on_list, off_list)
                region_means.append(mean_bgr)
                region_labels.append(int(lbl))

            colors_per_frame.append(region_means)
            labels_per_frame.append(region_labels)
            pbar.update(1)
    finally:
        cap.release()
        pbar.close()

    # post-process → Arm_Swap
    arm_swap_flags = detect_arm_swaps_from_labels(labels_per_frame,
                                                  extend_frames=extend_frames,
                                                  stability_frames=stability_frames)

    # write rows
    for t, region_means in enumerate(colors_per_frame):
        row = [t]
        for j, mean_bgr in enumerate(region_means):
            b, g, r = mean_bgr.tolist()
            row += [f"{b:.2f}", f"{g:.2f}", f"{r:.2f}", int(labels_per_frame[t][j])]
        row += [int(arm_swap_flags[t])]
        writer.writerow(row)

    f.close()


# ---------------- helpers for trials ----------------
def _pick_video_file(video_dir: str) -> Optional[str]:
    """Pick a .mkv (prefer newest)."""
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
    return os.path.join(out_dir, f"{trial}_arm_swap_pedal_gt.csv")


# ---------------- multiprocessing worker ----------------
def _worker(args):
    """One process per video."""
    # Tame thread oversubscription
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    trial, gt_folder, root, regions_csv, patches_dir, downsample, extend_frames, stability_frames, position = args

    video_dir = os.path.join(root, gt_folder, "video")
    video_path = _pick_video_file(video_dir)
    if not video_path:
        return (trial, False, f"No .mkv found in {video_dir}")

    # load regions once per worker
    try:
        regions = _read_regions_csv(regions_csv)
        if not regions:
            return (trial, False, f"No regions in {regions_csv}")
    except Exception as e:
        return (trial, False, f"Regions error: {e}")

    out_csv = _make_output_csv(root, trial)
    try:
        print(f"Processing trial {trial} - video: {video_path} → {out_csv}")
        analyze_regions_and_save(
            video_path=video_path,
            regions=regions,
            output_csv=out_csv,
            patches_dir=patches_dir,
            downsample=downsample,
            extend_frames=extend_frames,
            stability_frames=stability_frames,
            pbar_position=position
        )
        return (trial, True, f"Wrote {out_csv}")
    except Exception as e:
        return (trial, False, f"{type(e).__name__}: {e}")


# ---------------- main ----------------
def main():
    trial_mapping = dict(zip(TRIALS, GT_VIDEO_TRIALS))
    jobs = []
    for i, (trial, gt_video_path) in enumerate(trial_mapping.items()):
        jobs.append((trial, gt_video_path, ROOT_PATH, REGIONS_CSV, PATCHES_DIR,
                     DOWNSAMPLE, EXTEND_FRAMES, STABILITY_FRAMES, i))

    if not jobs:
        print("No jobs to run.")
        return

    procs = min(MAX_PROCS, max(1, cpu_count()))
    print(f"Launching pool with {procs} processes for {len(jobs)} videos…")

    with Pool(processes=procs, maxtasksperchild=1) as pool:
        for trial, ok, msg in pool.imap_unordered(_worker, jobs, chunksize=1):
            print(("✅" if ok else "❌"), trial, "-", msg)

    print("All done.")


if __name__ == "__main__":
    main()
