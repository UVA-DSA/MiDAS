
import cv2
import numpy as np
import csv


# Threshold for detecting a color change between consecutive frames for a region.
# Euclidean distance in mean BGR color space (0-255 per channel).
COLOR_CHANGE_THRESHOLD = 15.0
DEBUG_SHOW = False  # Disable on-screen debug visualization
DEBUG_SCALE = 0.5  # Downscale factor for on-screen debug display

# Overall color classification thresholds
# "Black" if brightness below this, otherwise "Blue" if blue_ratio above threshold
BLACK_BRIGHTNESS_THR = 35.0  # legacy param (kept for signature compatibility)
BLUE_RATIO_THR = 0.60        # legacy param (kept for signature compatibility)

# HSV-driven classification thresholds (OpenCV HSV: H in [0,179], S,V in [0,255])
BLACK_V_THR = 40            # V below this -> BLACK
BLUE_H_LO = 95              # Inclusive lower bound for blue hue
BLUE_H_HI = 135             # Inclusive upper bound for blue hue
BLUE_S_THR = 40             # Minimum saturation to consider a color BLUE
BLUE_V_MIN = 45             # Minimum value so dark regions aren't considered BLUE
BLUE_DOMINANCE_RATIO = 1.08 # Fallback: B must exceed max(R,G) by this factor


def region_mean_bgr(frame: np.ndarray, region) -> np.ndarray:
    x, y, w, h = region
    roi = frame[y:y + h, x:x + w]
    if roi.size == 0:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    mean = roi.reshape(-1, 3).mean(axis=0)  # BGR order (OpenCV)
    return mean.astype(float)


def classify_overall_color(mean_bgr: np.ndarray, black_thr: float, blue_ratio_thr: float) -> int:
    """
    Classify region color as 0=BLACK or 1=BLUE using HSV primarily, with a BGR fallback.
    The black_thr and blue_ratio_thr args are kept for backward compatibility but not used directly.
    """
    b, g, r = mean_bgr.tolist()
    # Convert mean BGR to HSV (using uint8 1x1 image for consistent conversion)
    bgr_uint8 = np.array([[np.clip([b, g, r], 0, 255)]], dtype=np.uint8)
    hsv = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Black if value is low
    if v < BLACK_V_THR:
        return 0

    # Blue if hue within range, sufficiently saturated and bright
    if BLUE_H_LO <= h <= BLUE_H_HI and s >= BLUE_S_THR and v >= BLUE_V_MIN:
        return 1

    # Fallback: B dominance check in BGR
    max_rg = max(r, g)
    if max_rg > 0 and b >= BLUE_DOMINANCE_RATIO * max_rg and v >= BLUE_V_MIN:
        return 1

    return 0


def detect_arm_swaps_from_colors(colors_per_frame, threshold: float, extend_frames: int):
    """
    colors_per_frame: list of shape [T][R, 3] mean BGR arrays
    threshold: float distance in BGR space to qualify as a change
    extend_frames: number of frames to continue labeling after a detected swap
    Returns: list[int] arm_swap flags per frame (0/1)
    """
    T = len(colors_per_frame)
    if T == 0:
        return []
    R = len(colors_per_frame[0]) if colors_per_frame[0] is not None else 0

    changed = [np.zeros(R, dtype=bool) for _ in range(T)]
    for t in range(1, T):
        prev = colors_per_frame[t - 1]
        curr = colors_per_frame[t]
        diffs = []
        for r in range(R):
            v0 = prev[r]
            v1 = curr[r]
            if np.any(np.isnan(v0)) or np.any(np.isnan(v1)):
                diffs.append(np.inf)  # treat invalid as change
            else:
                diffs.append(float(np.linalg.norm(v1 - v0)))
        diffs = np.array(diffs)
        changed[t] = diffs > threshold

    # Arm swap if at least two regions changed simultaneously
    arm_swap = np.zeros(T, dtype=int)
    for t in range(1, T):
        if changed[t].sum() >= 2:
            t_end = min(T, t + extend_frames)
            arm_swap[t:t_end] = 1
    return arm_swap.tolist()


def detect_arm_swaps_from_labels(labels_per_frame, extend_frames: int):
    """
    labels_per_frame: list of shape [T][R] with integer labels 0/1 per region (BLACK/BLUE)
    Returns: list[int] arm_swap flags per frame (0/1) when at least two regions switch label simultaneously.
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


def analyze_regions_and_save(video_path, regions, output_csv, threshold=COLOR_CHANGE_THRESHOLD, extend_frames=15, debug_show=DEBUG_SHOW):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_id = -1

    # Storage for mean colors and labels per frame per region
    colors_per_frame = []  # list of [R, 3]
    labels_per_frame = []  # list of [R]

    paused = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        print(f"Processing frame:{frame_id} / {total_frames}")

        region_means = []
        region_labels = []
        for i, region in enumerate(regions):
            mean_bgr = region_mean_bgr(frame, region)
            region_means.append(mean_bgr)
            label = classify_overall_color(mean_bgr, BLACK_BRIGHTNESS_THR, BLUE_RATIO_THR)
            region_labels.append(label)

            if debug_show:
                x, y, w, h = region
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                b, g, r = mean_bgr.tolist()
                label_str = 'BLUE' if label == 1 else 'BLACK'
                text = f"R{i+1}: {label_str} ({int(r)},{int(g)},{int(b)})"
                text_y = y - 6 if y - 6 > 12 else y + h + 16
                cv2.putText(frame, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)

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

    cap.release()
    if debug_show:
        try:
            cv2.destroyWindow('arm-swap-debug')
        except Exception:
            pass

    # Post-processing: detect arm swaps based on simultaneous label switches
    arm_swap_flags = detect_arm_swaps_from_labels(labels_per_frame, extend_frames)

    # Write results per frame with region colors and event flag
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        header = ['Frame_ID']
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


# Example usage (keeps the same regions as before)
video_path = './2024-07-16 15-24-01_left.mp4'
h = w = 11
regions = [
    (356, 999, w, h),   # Instrument 1
    (673, 999, w, h),    # Instrument 2
    (990, 999, w, h),   # Instrument 3
    (1307, 999, w, h)   # Instrument 4
]
output_csv = 'arm_swap_events.csv'

analyze_regions_and_save(video_path, regions, output_csv, threshold=COLOR_CHANGE_THRESHOLD, extend_frames=15)
