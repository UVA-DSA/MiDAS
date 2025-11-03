import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

# ====================== USER CONFIG ======================
ROOT_PATH = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/SuturingV2/Processed/"
TRIALS = [
    "S105_T1",
    # "S106_T1","S106_T2","S112_T1","S112_T2","S116_T1","S116_T2","S116_T4","S116_T5",
    # "S118_T1","S200_T1","S201_T1","S201_T2","S202_T1","S203_T1","S204_T1","S209_T2","S210_T1",
    # "S214_T1","S214_T4","S214_T6","S215_T3","S215_T4","S217_T2","S217_T3","S217_T4","S218_T1","S219_T1",
]
MAX_PROCS = 32
PRINT_EVERY = 500  # frames

# Horizontal shifts (in pixels). Positive = move right, Negative = move left.
LEFT_X_OFFSET  = 100    # affects Camera (top-left)
RIGHT_X_OFFSET = 100    # affects UL/UR/LL/LR (top-right)

# ====================== DRAW HELPERS ======================

def _draw_boxed_text(img, text, org, font_scale=0.8, thickness=2,
                     fg=(255, 255, 255), bg=(0, 140, 255), alpha=0.6, pad=6, radius=6):
    """Draw semi-transparent rounded rectangle with text at top-left corner `org`."""
    if not text:
        return img
    x, y = org
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = tw + 2 * pad
    box_h = th + 2 * pad

    # clamp origin so the box stays on-screen
    x = max(0, min(W - box_w, x))
    y = max(0, min(H - box_h, y))

    overlay = img.copy()
    x2, y2 = x + box_w, y + box_h

    # main rounded box
    cv2.rectangle(overlay, (x + radius, y), (x2 - radius, y2), bg, -1)
    cv2.rectangle(overlay, (x, y + radius), (x2, y2 - radius), bg, -1)
    for cx, cy in [(x+radius, y+radius), (x2-radius, y+radius), (x+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, bg, -1)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x + pad, y + pad + th - 2), font, font_scale, fg, thickness, cv2.LINE_AA)
    return img

# ====================== CORE LOGIC ======================

def _compute_index_offset(gt_df):
    """Return -1 if CSV frames look 1-based, else 0."""
    try:
        mn = int(gt_df["obs_frame_idx"].min())
        return -1 if mn == 1 else 0
    except Exception:
        return 0

def _row_for_frame(gt_lookup, frame_idx, expected_keys):
    """Return dict of pedal flags for a given frame index from a dict lookup."""
    row = gt_lookup.get(frame_idx, None)
    if row is None:
        return {k: 0 for k in expected_keys}
    vals = {}
    for k in expected_keys:
        try:
            vals[k] = int(row.get(k, 0))
        except Exception:
            vals[k] = 0
    return vals

def visualize_pedals_on_video(video_path, gt_pedals_csv, output_path, position=0):
    """
    Overlay pedal states per frame:
      - Top-left: Camera_Pedal (only when 1)
      - Top-right stacked: Upper Left/Right (yellow), Lower Left/Right (blue) — only when pressed
      - Bottom-left: Arm_Swap (only when 1)
      - Bottom-right: Clutch_Pedal (only when 1), slightly larger, light-green
    """
    # Load GT
    gt = pd.read_csv(gt_pedals_csv)
    if "obs_frame_idx" not in gt.columns:
        raise ValueError("'obs_frame_idx' missing in {}".format(gt_pedals_csv))

    required = ["Camera_Pedal", "Clutch_Pedal", "Upper Left", "Upper Right", "Lower Left", "Lower Right"]
    for col in required:
        if col not in gt.columns:
            gt[col] = 0
        gt[col] = (gt[col].fillna(0).astype(int) > 0).astype(int)

    offset = _compute_index_offset(gt)
    gt_lookup = {int(r["obs_frame_idx"]) + offset: r for _, r in gt.iterrows()}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {}".format(video_path))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Layout
    margin   = 14
    line_gap = 8
    left_x   = margin + LEFT_X_OFFSET
    top_y    = margin
    right_margin_x = (w - margin) + RIGHT_X_OFFSET  # we clamp when drawing

    # Colors
    cam_bg   = (36, 255, 12)       # light green (camera)
    arm_bg   = (0, 165, 255)       # orange (arm swap)
    clutch_bg= (144, 238, 144)     # light green for clutch (BGR)
    yellow   = (0, 255, 255)       # Upper pedals
    blue     = (255, 128, 0)       # Lower pedals
    white    = (255, 255, 255)

    fidx = -1
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            fidx += 1
            if PRINT_EVERY and fidx % PRINT_EVERY == 0:
                print(f"[{Path(video_path).name}] frame {fidx}/{total}")

            ped = _row_for_frame(gt_lookup, fidx, required)

            # ----- Top-left: Camera -----
            cur_y = top_y
            if ped["Camera_Pedal"] == 1:
                frame = _draw_boxed_text(frame, "CAMERA", (left_x, cur_y), fg=white, bg=cam_bg, alpha=0.65)
                (tw, th), _ = cv2.getTextSize("CAMERA", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cur_y += th + 2 * 6 + line_gap

            # ----- Top-right: Energy pedals -----
            labels = [
                ("Upper Left",  "UL", yellow),
                ("Upper Right", "UR", yellow),
                ("Lower Left",  "LL", blue),
                ("Lower Right", "LR", blue),
            ]
            right_y = top_y
            for col_name, short_tag, bg_color in labels:
                if ped[col_name] == 1:
                    (tw, th), _ = cv2.getTextSize(short_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    box_w = tw + 2 * 6
                    right_x = right_margin_x - box_w
                    right_x = max(0, min(w - box_w, right_x))
                    frame = _draw_boxed_text(frame, short_tag, (right_x, right_y),
                                             fg=white, bg=bg_color, alpha=0.65)
                    right_y += th + 2 * 6 + line_gap

            # # ----- Bottom-left: Arm Swap -----
            # if ped["Arm_Swap"] == 1:
            #     tag = "ARM SWAP"
            #     (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            #     box_w = tw + 2 * 6
            #     box_h = th + 2 * 6
            #     bl_x = margin + LEFT_X_OFFSET
            #     bl_y = h - margin - box_h
            #     frame = _draw_boxed_text(frame, tag, (bl_x, bl_y), fg=white, bg=arm_bg, alpha=0.70)

            # ----- Bottom-right: Clutch (bigger, light green) -----
            if ped["Clutch_Pedal"] == 1:
                tag = "CLUTCH"
                clutch_font_scale = 1.1   # slightly larger
                clutch_thickness  = 2
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, clutch_font_scale, clutch_thickness)
                box_w = tw + 2 * 6
                box_h = th + 2 * 6
                br_x = w - margin - box_w
                br_y = h - margin - box_h
                frame = _draw_boxed_text(frame, tag, (br_x, br_y),
                                         font_scale=clutch_font_scale, thickness=clutch_thickness,
                                         fg=white, bg=clutch_bg, alpha=0.70)

            out.write(frame)
    finally:
        cap.release()
        out.release()

# ====================== MULTIPROCESSING ======================

def _worker(args):
    """Process one trial (one video) in a separate process."""
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    trial, position = args
    video_path = f"{ROOT_PATH}/{trial}/synched_data/{trial}.mp4"
    gt_pedals_csv = f"{ROOT_PATH}/{trial}/synched_data/{trial}_pds_gt.csv"
    out_path = f"{ROOT_PATH}/{trial}/synched_data/{trial}_pedal_visualized.mp4"

    if not os.path.isfile(video_path):
        return (trial, False, f"Missing video: {video_path}")
    if not os.path.isfile(gt_pedals_csv):
        return (trial, False, f"Missing CSV: {gt_pedals_csv}")

    try:
        visualize_pedals_on_video(video_path, gt_pedals_csv, out_path, position=position)
        return (trial, True, f"Wrote {out_path}")
    except Exception as e:
        return (trial, False, f"{type(e).__name__}: {e}")

def main():
    jobs = [(t, i) for i, t in enumerate(TRIALS)]
    if not jobs:
        print("No trials to render.")
        return

    procs = min(MAX_PROCS, max(1, cpu_count()))
    print(f"Launching pool with {procs} processes for {len(jobs)} trials…")

    with Pool(processes=procs, maxtasksperchild=1) as pool:
        for trial, ok, msg in pool.imap_unordered(_worker, jobs, chunksize=1):
            print(("✅" if ok else "❌"), trial, "-", msg)

    print("All done.")

if __name__ == "__main__":
    main()
