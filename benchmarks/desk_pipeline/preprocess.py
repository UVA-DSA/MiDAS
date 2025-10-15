import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from .utils.config import load_config
from .utils.logger import setup_logger

VIDEO_RE = re.compile(r"Peg_Transfer_S0(\d+)_T0(\d+)_([Rr]ight|[Ll]eft)\.avi$")
ANNOT_RE = re.compile(r"Peg_Transfer_S0(\d+)_T0(\d+)\.txt$")

GESTURE_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def parse_annotation_file(path: str) -> List[Tuple[int, int, str, bool]]:
	rows: List[Tuple[int, int, str, bool]] = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			parts = line.split()
			if len(parts) != 4:
				raise ValueError(f"Bad annotation line in {path}: {line}")
			start_f, end_f = int(parts[0]), int(parts[1])
			name = parts[2]
			success = parts[3].lower() in {"1", "true", "yes", "y", "t"}
			rows.append((start_f, end_f, name, success))
	return rows


def build_frame_label_map(ann: List[Tuple[int, int, str, bool]]) -> Dict[int, Tuple[str, bool]]:
	label_map: Dict[int, Tuple[str, bool]] = {}
	for start_f, end_f, name, success in ann:
		for f in range(start_f, end_f + 1):
			label_map[f] = (name, success)
	return label_map


def extract_frames_1hz(video_path: str, out_dir: str, target_fps: int, jpeg_quality: int, logger) -> List[Tuple[int, str]]:
	os.makedirs(out_dir, exist_ok=True)
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")
	orig_fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if orig_fps <= 0:
		orig_fps = 30.0
	step = max(int(round(orig_fps / target_fps)), 1)

	frame_indices: List[Tuple[int, str]] = []
	frame_id = 0
	jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
	while True:
		pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
		ret, frame = cap.read()
		if not ret:
			break
		if pos % step == 0:
			fname = f"frame_{pos:06d}.jpg"
			fpath = os.path.join(out_dir, fname)
			cv2.imwrite(fpath, frame, jpeg_params)
			frame_indices.append((pos, fpath))
		frame_id += 1
	cap.release()
	logger.info(f"Extracted {len(frame_indices)} frames from {os.path.basename(video_path)} (orig_fps={orig_fps:.2f}, total_frames={total_frames})")
	return frame_indices


def discover_videos(video_dir: str) -> List[str]:
	videos: List[str] = []
	for name in os.listdir(video_dir):
		if name.lower().endswith('.avi') and VIDEO_RE.search(name):
			videos.append(os.path.join(video_dir, name))
	return sorted(videos)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	args = parser.parse_args()
	cfg = load_config(args.config)
	paths = cfg["paths"]
	pp = cfg["preprocess"]
	logger = setup_logger("preprocess", paths["logs_dir"], filename="preprocess.log")

	video_dir = os.path.join(paths["desk_root"], "video")
	gest_dir = os.path.join(paths["desk_root"], "gestures")
	frames_root = paths["frames_dir"]
	index_csv = paths["index_csv"]
	os.makedirs(frames_root, exist_ok=True)

	videos = discover_videos(video_dir)
	logger.info(f"Found {len(videos)} videos.")

	records: List[Dict[str, object]] = []

	for vpath in videos:
		vname = os.path.basename(vpath)
		m = VIDEO_RE.match(vname)
		if not m:
			logger.warning(f"Skipping non-matching video name: {vname}")
			continue
		subj, trial, side = m.group(1), m.group(2), m.group(3).lower()
		annot_name = f"Peg_Transfer_S0{subj}_T0{trial}.txt"
		annot_path = os.path.join(gest_dir, annot_name)
		if not os.path.exists(annot_path):
			logger.warning(f"Missing annotation for {vname}: {annot_name}")
			continue
		ann = parse_annotation_file(annot_path)
		frame_to_label = build_frame_label_map(ann)

		video_out_dir = os.path.join(frames_root, f"S{subj}_T{trial}_{side}")
		already_done = os.path.isdir(video_out_dir) and not pp["overwrite"] and len(os.listdir(video_out_dir)) > 0
		if already_done:
			# Build listing from existing files
			frame_files = [f for f in os.listdir(video_out_dir) if f.lower().endswith('.jpg')]
			frame_files = sorted(frame_files)
			frame_tuples: List[Tuple[int, str]] = []
			for fname in frame_files:
				mm = re.search(r"frame_(\d+)\.jpg$", fname)
				if not mm:
					continue
				frame_tuples.append((int(mm.group(1)), os.path.join(video_out_dir, fname)))
		else:
			frame_tuples = extract_frames_1hz(
				vpath,
				video_out_dir,
				int(pp["target_fps"]),
				int(pp["jpeg_quality"]),
				logger,
			)

		for frame_idx, fpath in frame_tuples:
			label_name, success = frame_to_label.get(frame_idx, ("unknown", False))
			records.append({
				"subject_id": int(subj),
				"trial_id": int(trial),
				"side": side,
				"video_name": vname,
				"frame_index": int(frame_idx),
				"image_path": fpath,
				"gesture": label_name,
				"success": bool(success),
			})

	if not records:
		logger.error("No records generated; check dataset paths.")
		return

	df = pd.DataFrame.from_records(records)
	# Filter to known gestures only (S1..S7)
	df = df[df["gesture"].isin(GESTURE_NAMES)].copy()
	# Build numeric label
	label_map = {g: i for i, g in enumerate(GESTURE_NAMES)}
	df["label"] = df["gesture"].map(label_map)

	os.makedirs(os.path.dirname(index_csv), exist_ok=True)
	df.to_csv(index_csv, index=False)
	logger.info(f"Wrote index with {len(df)} rows to {index_csv}")


if __name__ == "__main__":
	main()
