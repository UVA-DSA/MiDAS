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

# Pixel coordinates (x, y)
ONE_PIXEL   = (600, 1020)   # "Upper Left" checks G+R avg
TWO_PIXEL   = (725, 1020)   # (kept for compatibility; not used in original logic)
THREE_PIXEL = (1230, 1020)  # used with FOUR for Right pedals
FOUR_PIXEL  = (1580, 1020)

THRESHOLD = 150             # same as your script
DOWNSAMPLE = 1              # process every Nth frame (>=1)
MAX_PROCS = 32              # pool size (cap by cpu_count)

# Trial → GT folder mapping (same idea as your other scripts)
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


def _clamp_point(x: int, y: int, W: int, H: int) -> Optional[Tuple[int, int]]:
    """Clamp a single (x,y) to frame bounds; return None if outside image entirely."""
    if W <= 0 or H <= 0:
        return None
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    return (x, y)


def _read_bgr_at(frame: np.ndarray, xy: Tuple[int, int]) -> Optional[np.ndarray]:
    """Read a single pixel's BGR as float array, with bounds clamping."""
    H, W = frame.shape[:2]
    pt = _clamp_point(xy[0], xy[1], W, H)
    if pt is None:
        return None
    x, y = pt
    # frame[y, x] is BGR uint8
    val = frame[y, x]
    if val is None or val.size != 3:
        return None
    return val.astype(float)


def _compute_flags_from_pixels(one_bgr: Optional[np.ndarray],
                               two_bgr: Optional[np.ndarray],
                               three_bgr: Optional[np.ndarray],
                               four_bgr: Optional[np.ndarray],
                               thr: float) -> Tuple[int, int, int, int]:
    """
    Preserve original logic:
      UL: (one.G + one.R)/2 > thr
      LL: (one.B) > thr
      UR: ((four.G + four.R)/2 > thr) or ((three.G + three.R)/2 > thr)
      LR: (four.B > thr) or (three.B > thr)
    """
    def ch(bgr, idx, default=0.0):
        if bgr is None or np.isnan(bgr).any():
            return default
        return float(bgr[idx])

    oneG, oneR, oneB = ch(one_bgr, 1), ch(one_bgr, 2), ch(one_bgr, 0)
    threeG, threeR, threeB = ch(three_bgr, 1), ch(three_bgr, 2), ch(three_bgr, 0)
    fourG, fourR, fourB = ch(four_bgr, 1), ch(four_bgr, 2), ch(four_bgr, 0)

    upper_left  = 1 if ((oneG + oneR) / 2.0) > thr else 0
    lower_left  = 1 if (oneB > thr) else 0
    upper_right = 1 if (((fourG + fourR) / 2.0) > thr or ((threeG + threeR) / 2.0) > thr) else 0
    lower_right = 1 if (fourB > thr or threeB > thr) else 0
    return upper_left, lower_left, upper_right, lower_right


def _pick_mkv_in_video_dir(video_dir: str) -> Optional[str]:
    """Return newest .mkv inside <GT_FOLDER>/video/, else None."""
    if not os.path.isdir(video_dir):
        return None
    cands = [f for f in os.listdir(video_dir) if f.lower().endswith(".mkv")]
    if not cands:
        return None
    cands.sort(key=lambda fn: os.path.getmtime(os.path.join(video_dir, fn)), reverse=True)
    return os.path.join(video_dir, cands[0])


def process_one_video(video_path: str, out_csv: str, pbar_position: int = 0):
    """Implements the original per-pixel channel-threshold logic, frame-by-frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame Num", "Upper Left", "Lower Left", "Upper Right", "Lower Right"])

        frame_num = 0
        pbar = tqdm(total=total, desc=f"[{Path(video_path).name}]", position=pbar_position, leave=True)
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_num += 1

                if frame_num % DOWNSAMPLE != 0:
                    pbar.update(1)
                    continue

                oneColor   = _read_bgr_at(frame, ONE_PIXEL)
                twoColor   = _read_bgr_at(frame, TWO_PIXEL)    # not used in flags; kept for parity
                threeColor = _read_bgr_at(frame, THREE_PIXEL)
                fourColor  = _read_bgr_at(frame, FOUR_PIXEL)

                UL, LL, UR, LR = _compute_flags_from_pixels(oneColor, twoColor, threeColor, fourColor, THRESHOLD)
                writer.writerow([frame_num, UL, LL, UR, LR])

                pbar.update(1)
        finally:
            pbar.close()
            cap.release()


# -------------------------- multiprocessing --------------------------

def _worker(args):
    """One process per trial."""
    # avoid oversubscription
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    trial, gt_folder, position = args

    # Input .mkv
    video_dir = os.path.join(ROOT_PATH, gt_folder, "video")
    video_path = _pick_mkv_in_video_dir(video_dir)
    if not video_path:
        return (trial, False, f"No .mkv in {video_dir}")

    # Output CSV (same naming style as your original script)
    out_csv = os.path.join(ROOT_PATH, "Processed", trial, "synched_data", f"{trial}_primary_secondary_pedals_gt.csv")

    try:
        process_one_video(video_path, out_csv, pbar_position=position)
        return (trial, True, f"Wrote {out_csv}")
    except Exception as e:
        return (trial, False, f"{type(e).__name__}: {e}")


def main():
    trial_mapping = dict(zip(TRIALS, GT_VIDEO_TRIALS))
    jobs = [(trial, gt_folder, i) for i, (trial, gt_folder) in enumerate(trial_mapping.items())]
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
