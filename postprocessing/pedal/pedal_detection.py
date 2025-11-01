import argparse
import csv
import os
from typing import List, Tuple

import cv2
import numpy as np


def read_points(csv_path: str) -> List[Tuple[int, int]]:
    points: List[Tuple[int, int]] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "x" not in reader.fieldnames or "y" not in reader.fieldnames:
            raise ValueError("CSV must have headers 'x' and 'y'")
        for row in reader:
            try:
                x = int(float(row["x"]))
                y = int(float(row["y"]))
            except Exception:
                continue
            points.append((x, y))
    if len(points) != 4:
        raise ValueError(f"Expected 4 points in CSV, found {len(points)}")
    return points


def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def bgr_to_lab_pixel(bgr: Tuple[int, int, int]) -> np.ndarray:
    arr = np.uint8([[list(bgr)]])  # shape (1,1,3)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return lab[0, 0, :]  # L,a,b


def patch_mean_lab(img_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    # Extract and convert patch to Lab, then average in Lab
    patch = img_bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_lab = lab.reshape(-1, 3).mean(axis=0)
    return mean_lab


def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> float:
    diff = lab1.astype(np.float32) - lab2.astype(np.float32)
    return float(np.sqrt(np.sum(diff * diff)))


def classify_patch(mean_lab: np.ndarray, lower_lab: np.ndarray, upper_lab: np.ndarray, threshold: float) -> str:
    d_lower = delta_e_76(mean_lab, lower_lab)
    d_upper = delta_e_76(mean_lab, upper_lab)
    if d_lower <= threshold or d_upper <= threshold:
        return 'Lower' if d_lower <= d_upper else 'Upper'
    return 'None'


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify dominant color around 4 points for every frame in a video")
    parser.add_argument("video", type=str, help="Path to input video file")
    parser.add_argument("--points", type=str, default="selected_points.csv", help="CSV file with 4 points (headers: x,y)")
    parser.add_argument(
        "--lower-color", type=str, default="#0d589e", help="Hex color for 'Lower' classification"
    )
    parser.add_argument(
        "--upper-color", type=str, default="#fdc40e", help="Hex color for 'Upper' classification"
    )
    parser.add_argument("--window-size", type=int, default=6, help="Square side length in pixels for vicinity")
    parser.add_argument(
        "--threshold", type=float, default=25.0, help="DeltaE (Lab) threshold for color closeness"
    )
    parser.add_argument("--out", type=str, default="pedal_detection.csv", help="Output CSV path")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    points = read_points(args.points)

    lower_lab = bgr_to_lab_pixel(hex_to_bgr(args.lower_color))
    upper_lab = bgr_to_lab_pixel(hex_to_bgr(args.upper_color))

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.video}")

    # Prepare writer
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ["frame", "p1", "p2", "p3", "p4"]
        writer.writerow(header)

        frame_idx = 0
        half = args.window_size // 2
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            labels: List[str] = []
            for (x, y) in points:
                # Clamp ROI bounds
                x0 = max(0, x - half)
                y0 = max(0, y - half)
                x1 = min(w, x + half)
                y1 = min(h, y + half)
                mean_lab = patch_mean_lab(frame, x0, y0, x1, y1)
                label = classify_patch(mean_lab, lower_lab, upper_lab, args.threshold)
                labels.append(label)

            writer.writerow([frame_idx] + labels)
            frame_idx += 1

    cap.release()
    print(f"Wrote classifications to {args.out}")


if __name__ == "__main__":
    main()


