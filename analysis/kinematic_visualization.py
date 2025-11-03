import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Configuration
# -----------------------------
PLOT_SAVE_DIR = './analysis_plots/'
DEFAULT_FPS = 30.0

os.makedirs(PLOT_SAVE_DIR, exist_ok=True)


# -----------------------------
# Column extraction helpers
# -----------------------------
def get_raven_data(data: pd.DataFrame) -> np.ndarray:
	# positions (6) + rotations as 3x3 matrices for 2 PSMs (18)
	raven_columns = [
		'raven_field.pos0', 'raven_field.pos1', 'raven_field.pos2',
		'raven_field.pos3', 'raven_field.pos4', 'raven_field.pos5',
		'raven_field.ori0', 'raven_field.ori1', 'raven_field.ori2',
		'raven_field.ori3', 'raven_field.ori4', 'raven_field.ori5',
		'raven_field.ori6', 'raven_field.ori7', 'raven_field.ori8',
		'raven_field.ori9', 'raven_field.ori10', 'raven_field.ori11',
		'raven_field.ori12', 'raven_field.ori13', 'raven_field.ori14',
		'raven_field.ori15', 'raven_field.ori16', 'raven_field.ori17',
	]
	return data[raven_columns].to_numpy()


def get_console_data(data: pd.DataFrame) -> np.ndarray:
	console_columns = [
		'console_pos0', 'console_pos1', 'console_pos2',
		'console_pos3', 'console_pos4', 'console_pos5',
		'console_rot0', 'console_rot1', 'console_rot2',
		'console_rot3', 'console_rot4', 'console_rot5',
		'console_pedal',
	]
	return data[console_columns].to_numpy()


def get_trakstar_data(data: pd.DataFrame) -> np.ndarray:
	trakstar_columns = [
		'trakstar_sensor_1_x', 'trakstar_sensor_1_y', 'trakstar_sensor_1_z',
		'trakstar_sensor_3_x', 'trakstar_sensor_3_y', 'trakstar_sensor_3_z',
		'trakstar_sensor_1_roll', 'trakstar_sensor_1_elevation', 'trakstar_sensor_1_azimuth',
		'trakstar_sensor_2_roll', 'trakstar_sensor_2_elevation', 'trakstar_sensor_2_azimuth',
	]
	return data[trakstar_columns].to_numpy()


def get_smartwatch_data(data: pd.DataFrame) -> np.ndarray:
	smartwatch_columns = [
		'sw_left_x', 'sw_left_y', 'sw_left_z',
		'sw_right_x', 'sw_right_y', 'sw_right_z',
	]
	return data[smartwatch_columns].to_numpy()


def get_pedal_data_from_df(data: pd.DataFrame) -> np.ndarray:
	return data['console_pedal'].to_numpy()


# -----------------------------
# Transforms and processing
# -----------------------------
def transform_trackstar_to_console_left(trackstar_pos: np.ndarray) -> np.ndarray:
	pos_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
	psm_pos = pos_matrix @ (trackstar_pos[:, 0:3] * 1).T
	trackstar_pos[:, 0:3] = psm_pos.T
	return trackstar_pos


def transform_trackstar_to_console_right(trackstar_pos: np.ndarray) -> np.ndarray:
	pos_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
	psm_pos = pos_matrix @ (trackstar_pos[:, 3:6] * 1).T
	trackstar_pos[:, 3:6] = psm_pos.T
	return trackstar_pos


def transform_console_to_raven_left(console_pos: np.ndarray) -> np.ndarray:
	pos_matrix = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
	psm1_pos = pos_matrix @ console_pos[:, 0:3].T
	console_pos[:, 0:3] = psm1_pos.T
	return console_pos


def transform_console_to_raven_right(console_pos: np.ndarray) -> np.ndarray:
	pos_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
	psm2_pos = pos_matrix @ console_pos[:, 3:6].T
	console_pos[:, 3:6] = psm2_pos.T
	return console_pos


def transform_trackstar_to_console_rot_1(trackstar_rot_left: np.ndarray) -> np.ndarray:
	rot_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
	psm1_rot = rot_matrix @ trackstar_rot_left[:, 0:3].T
	trackstar_rot_left[:, 0:3] = psm1_rot.T
	return trackstar_rot_left


def transform_trackstar_to_console_rot_2(trackstar_rot_right: np.ndarray) -> np.ndarray:
	rot_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
	psm2_rot = rot_matrix @ trackstar_rot_right[:, 0:3].T
	trackstar_rot_right[:, 0:3] = psm2_rot.T
	return trackstar_rot_right


def transform_console_to_raven_rot_1(console_rot_left: np.ndarray) -> np.ndarray:
	rot_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
	psm1_rot = rot_matrix @ console_rot_left[:, 0:3].T
	console_rot_left[:, 0:3] = psm1_rot.T
	return console_rot_left


def rotation_matrix_to_euler(rot_matrix_flat: np.ndarray) -> np.ndarray:
	n_samples = rot_matrix_flat.shape[0]
	euler_angles = np.zeros((n_samples, 3))
	for i in range(n_samples):
		Rmat = rot_matrix_flat[i].reshape(3, 3)
		r = R.from_matrix(Rmat)
		euler_angles[i] = r.as_euler('zyx', degrees=False)
	return euler_angles


def process_raven_rot(raven_psm_rot: np.ndarray) -> np.ndarray:
	processed_rot = raven_psm_rot.copy()
	processed_rot[processed_rot > 4] -= 6.28
	processed_rot[processed_rot < -4] += 6.28
	return processed_rot


def process_trakstar_rot(trakstar_psm_rot: np.ndarray) -> np.ndarray:
	processed_rot = trakstar_psm_rot.copy()
	processed_rot[processed_rot > 4] -= 6.28
	processed_rot[processed_rot < -4] += 6.28
	return processed_rot


def process_smartwatch_pos(smartwatch_data: np.ndarray, pedal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	delta_t = 1 / DEFAULT_FPS
	g = [0, 0, -9.80665, 0, 0, -9.80665]
	switch_pos = np.zeros((smartwatch_data.shape[0], 6))

	sw_acc = smartwatch_data - g
	sw_pos = sw_acc * delta_t ** 2
	diff_sw = np.diff(sw_pos, axis=0)
	zero_mask = pedal_data[:-1] == 0
	diff_sw[zero_mask, :] = 0
	switch_pos[1:, :] = np.cumsum(diff_sw, axis=0)

	scale_factors = np.array([20000, 10000, 10000, -50000, 20000, 20000])
	switch_pos = switch_pos * scale_factors
	return switch_pos[:, 0:3], switch_pos[:, 3:6]


def process_smartwatch_rot(smartwatch_data: np.ndarray, pedal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	left_acc = smartwatch_data[:, 0:3]
	right_acc = smartwatch_data[:, 3:6]
	left_sw_rot = np.zeros((smartwatch_data.shape[0], 2))
	right_sw_rot = np.zeros((smartwatch_data.shape[0], 2))
	left_sw_rot[:, 0] = np.arctan2(left_acc[:, 1], left_acc[:, 2])
	left_sw_rot[:, 1] = -np.arctan2(-left_acc[:, 0], np.sqrt(left_acc[:, 1] ** 2 + left_acc[:, 2] ** 2))
	right_sw_rot[:, 0] = np.arctan2(right_acc[:, 1], right_acc[:, 2])
	right_sw_rot[:, 1] = np.arctan2(-right_acc[:, 0], np.sqrt(right_acc[:, 1] ** 2 + right_acc[:, 2] ** 2))
	left_sw_rot = left_sw_rot - left_sw_rot[0, :]
	right_sw_rot = right_sw_rot - right_sw_rot[0, :]
	return left_sw_rot, right_sw_rot


def cumulate_trackstar_console_data_rot(new_console: np.ndarray, new_trackstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	console_rot = np.zeros((new_console.shape[0], 6))
	trackstar_rot = np.zeros((new_trackstar.shape[0], 6))
	diff_console = np.diff(new_console[:, 6:], axis=0)
	diff_trackstar = np.diff(np.radians(new_trackstar[:, 6:]), axis=0)
	zero_mask = new_console[:-1, -1] == 0
	diff_console[zero_mask, :] = 0
	console_rot[1:, :] = np.cumsum(diff_console, axis=0)
	diff_trackstar[zero_mask, :] = 0
	trackstar_rot[1:, :] = np.cumsum(diff_trackstar, axis=0)
	return console_rot, trackstar_rot


def cumulate_trackstar_console_data(console_data: np.ndarray, trackstar_data: np.ndarray, pedal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	console_pos = np.zeros((console_data.shape[0], 6))
	trackstar_pos = np.zeros((trackstar_data.shape[0], 6))
	diff_console = np.diff(console_data, axis=0)
	diff_trackstar = np.diff(trackstar_data, axis=0)
	zero_mask = pedal[:-1] == 0
	diff_console[zero_mask, :] = 0
	console_pos[1:, :] = np.cumsum(diff_console, axis=0)
	diff_trackstar[zero_mask, :] = 0
	trackstar_pos[1:, :] = np.cumsum(diff_trackstar, axis=0)
	return console_pos, trackstar_pos


def align_console_trackstar_data_pos(new_console: np.ndarray, new_trackstar: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	console_pos = new_console[:, [0, 2, 4, 1, 3, 5]]
	trackstar_pos = new_trackstar[:, [0, 1, 2, 3, 4, 5]]
	trackstar_pos = transform_trackstar_to_console_left(trackstar_pos)
	trackstar_pos = transform_trackstar_to_console_right(trackstar_pos)
	console_pos = console_pos - console_pos[0, :]
	trackstar_pos = trackstar_pos - trackstar_pos[0, :]
	return console_pos, trackstar_pos


def align_trackstar_console_raven_rot(new_trackstar: np.ndarray, new_console: np.ndarray, new_raven: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	console_rot, trackstar_rot = cumulate_trackstar_console_data_rot(new_console, new_trackstar)
	trackstar_psm1_rot = trackstar_rot[:, [0, 1, 2]]
	trackstar_psm2_rot = trackstar_rot[:, [3, 4, 5]]
	trackstar_psm1_rot = process_trakstar_rot(trackstar_psm1_rot)
	trackstar_psm2_rot = process_trakstar_rot(trackstar_psm2_rot)
	trackstar_psm1_rot = transform_trackstar_to_console_rot_1(trackstar_psm1_rot)
	trackstar_psm2_rot = transform_trackstar_to_console_rot_2(trackstar_psm2_rot)
	console_psm1_rot = console_rot[:, [0, 2, 4]]
	console_psm2_rot = console_rot[:, [1, 3, 5]]
	raven_psm1_rot = rotation_matrix_to_euler(new_raven[:, 6:15])
	raven_psm2_rot = rotation_matrix_to_euler(new_raven[:, 15:24])
	raven_psm1_rot = raven_psm1_rot - raven_psm1_rot[0, :]
	raven_psm2_rot = raven_psm2_rot - raven_psm2_rot[0, :]
	raven_psm1_rot = process_raven_rot(raven_psm1_rot)
	raven_psm2_rot = process_raven_rot(raven_psm2_rot)
	console_psm1_rot = transform_console_to_raven_rot_1(console_psm1_rot)
	console_psm2_rot = transform_console_to_raven_rot_1(console_psm2_rot)
	return trackstar_psm1_rot, console_psm1_rot, raven_psm1_rot, trackstar_psm2_rot, console_psm2_rot, raven_psm2_rot


def transform_data(new_console: np.ndarray, new_trackstar: np.ndarray, new_raven: np.ndarray, pedal: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	scale_factors = np.array([2, 1.1, 2.2, 2, 1, 2])
	console, trackstar = cumulate_trackstar_console_data(new_console, new_trackstar, pedal)
	raven_pos = (new_raven[:, :] - new_raven[0, :]) * 1e-2
	trackstar_pos = transform_console_to_raven_left(trackstar)
	trackstar_pos = transform_console_to_raven_right(trackstar)
	console_pos = transform_console_to_raven_left(console)
	console_pos = transform_console_to_raven_right(console)
	trackstar_pos = trackstar_pos * scale_factors
	return console_pos, trackstar_pos, raven_pos


# -----------------------------
# Gesture spans utilities
# -----------------------------
def compute_gesture_spans(gesture_names: Iterable[str], fps: float) -> List[Tuple[float, float, str]]:
	"""Return list of (t_start, t_end, name) for contiguous gesture segments."""
	spans: List[Tuple[float, float, str]] = []
	gesture_list = pd.Series(list(gesture_names)).fillna('Unknown').astype(str).tolist()
	if not gesture_list:
		return spans
	start_idx = 0
	current = gesture_list[0]
	for i in range(1, len(gesture_list)):
		if gesture_list[i] != current:
			spans.append((start_idx / fps, i / fps, current))
			start_idx = i
			current = gesture_list[i]
	# last span
	spans.append((start_idx / fps, len(gesture_list) / fps, current))
	return spans


def build_soft_color_map(gesture_names: List[str]) -> Dict[str, Tuple[float, float, float, float]]:
	"""Assign a unique, soft background color per gesture with good contrast.

	Uses the pastel Set3 colormap; falls back to tab20 if >12 unique gestures.
	Background alpha kept low for line readability.
	"""
	unique = list(dict.fromkeys(gesture_names))
	set3 = plt.get_cmap('Set3')
	tab20 = plt.get_cmap('tab20')
	color_map: Dict[str, Tuple[float, float, float, float]] = {}
	for idx, name in enumerate(unique):
		base = set3(idx % 12) if idx < 12 else tab20(idx % 20)
		r, g, b = base[:3]
		# Ensure backgrounds are light: gently shift toward white
		light_r = 0.85 * r + 0.15 * 1.0
		light_g = 0.85 * g + 0.15 * 1.0
		light_b = 0.85 * b + 0.15 * 1.0
		color_map[name] = (light_r, light_g, light_b, 0.26)
	return color_map


def add_gesture_spans_to_axes(axes: List[plt.Axes], spans: List[Tuple[float, float, str]], color_map: Dict[str, Tuple[float, float, float, float]]) -> None:
	for ax in axes:
		for t0, t1, name in spans:
			ax.axvspan(t0, t1, facecolor=color_map[name], edgecolor='none', zorder=0)



def add_gesture_legend(fig: plt.Figure, color_map: Dict[str, Tuple[float, float, float, float]]) -> None:
	# Desired gesture order (typical sequence)
	desired_order_raw = [
		"S1  Approach peg",
		"S2  Align & grasp",
		"S3  Lift peg",
		"S4  Transfer peg -- Get together",
		"S5  Transfer peg -- Exchange",
		"S6  Approach pole",
		"S7  Align & place",
	]
	def _norm(s: str) -> str:
		return " ".join(str(s).split()).strip().lower()
	name_map = {_norm(k): k for k in color_map.keys()}
	# Mapping from plain names -> canonical S-labeled names for legend labels
	label_map = {
		_norm("Approach peg"): "S1: Approach peg",
		_norm("Align & grasp"): "S2: Align & grasp",
		_norm("Lift peg"): "S3: Lift peg",
		_norm("Transfer peg - Get together"): "S4: Transfer peg - Get together",
		_norm("Transfer peg - Exchange"): "S5: Transfer peg - Exchange",
		_norm("Approach pole"): "S6: Approach pole",
		_norm("Align & place"): "S7: Align & place",
	}
	desired_norm = [_norm(x) for x in desired_order_raw]
	ordered_names: List[str] = []
	for n in desired_norm:
		if n in name_map:
			ordered_names.append(name_map[n])
	# Append any remaining gestures not in desired order, preserving insertion order
	for k in color_map.keys():
		if k not in ordered_names:
			ordered_names.append(k)
	# Build handles in the chosen order with more opaque patches for readability
	handles = []
	for name in ordered_names:
		r, g, b, _ = color_map[name]
		# Use mapped label if available, otherwise keep original
		label_key = _norm(name)
		legend_label = label_map.get(label_key, name)
		handles.append(Patch(facecolor=(r, g, b, 0.55), edgecolor='none', label=legend_label))
	if not handles:
		return
	fig.legend(
		handles=handles,
		labels=[h.get_label() for h in handles],
		loc='lower center',
				ncol=max(1, min(5, len(handles))),
		frameon=False,
		bbox_to_anchor=(0.5, -0.12),
	)


# -----------------------------
# Plotting
# -----------------------------
def plot_pos_rot_side_by_side(
    raven_pos: np.ndarray,
    console_pos: np.ndarray,
    trackstar_pos: np.ndarray,
    raven_rot: np.ndarray,
    console_rot: np.ndarray,
    trackstar_rot: np.ndarray,
    left_sw_pos: Optional[np.ndarray] = None,
    smartwatch_rot: Optional[np.ndarray] = None,
    grasper_state: Optional[np.ndarray] = None,
    fps: float = DEFAULT_FPS,
    out_path: str = f'{PLOT_SAVE_DIR}/pos_rot_side_by_side_left.png',
    gesture_names: Optional[Iterable[str]] = None,
    pedal: Optional[np.ndarray] = None,
):
	# High-contrast, colorblind-safe lines
	COL_RAVEN = '#111111'   # near-black for reference
	COL_CONSOLE = '#4477AA' # blue
	COL_TSTAR = '#228833'   # green
	COL_SW = '#AA3377'      # purple

	T_pos = raven_pos.shape[0]
	T_rot = raven_rot.shape[0]
	t_pos = np.arange(T_pos) / fps
	t_rot = np.arange(T_rot) / fps

	if left_sw_pos is not None:
		t_sw_pos = np.arange(left_sw_pos.shape[0]) / fps
	if smartwatch_rot is not None:
		t_sw_rot = np.arange(smartwatch_rot.shape[0]) / fps

	pos_labels = ['X (mm)', 'Y (mm)', 'Z (mm)']
	rot_labels = ['Roll (rad)', 'Pitch (rad)', 'Yaw (rad)']

	fig = plt.figure(figsize=(12, 6.0), constrained_layout=True)
	gs = fig.add_gridspec(
		nrows=5,
		ncols=2,
		width_ratios=[1, 1],
		height_ratios=[0.5, 0.5, 0.5, 0.35, 0.3],
		wspace=0.02,
		hspace=0.04,
	)
	# Left column: positions
	a_left = [fig.add_subplot(gs[i, 0]) for i in range(3)]
	for i, ax in enumerate(a_left):
		ax.plot(t_pos, raven_pos[:, i], label='PSM', color=COL_RAVEN, linewidth=1.6, zorder=3)
		ax.plot(t_pos, console_pos[:, i], label='MTM', color=COL_CONSOLE, linewidth=1.6, zorder=3)
		ax.plot(t_pos, trackstar_pos[:, i], label='EmHT', color=COL_TSTAR, linewidth=1.6, zorder=3)
		if left_sw_pos is not None:
			ax.plot(t_sw_pos, left_sw_pos[:, i], label='Smartwatch', color=COL_SW, linewidth=1.4, zorder=3)
		ax.set_ylabel(pos_labels[i], fontsize=10)
		ax.grid(True, alpha=0.3)
		ax.set_facecolor('white')
		if i < 2:
			ax.tick_params(labelbottom=False)
		else:
			ax.set_xlabel('Time (s)', fontsize=10)

	# Right column: rotations
	a_right = [fig.add_subplot(gs[i, 1]) for i in range(3)]
	for i, ax in enumerate(a_right):
		ax.plot(t_rot, raven_rot[:, i], label='PSM', color=COL_RAVEN, linewidth=1.6, zorder=3)
		ax.plot(t_rot, console_rot[:, i], label='MTM', color=COL_CONSOLE, linewidth=1.6, zorder=3)
		ax.plot(t_rot, trackstar_rot[:, i], label='EmHT', color=COL_TSTAR, linewidth=1.6, zorder=3)
		if smartwatch_rot is not None and i < 2:
			ax.plot(t_sw_rot, smartwatch_rot[:, i], label='Smartwatch', color=COL_SW, linewidth=1.4, zorder=3)
		ax.set_ylabel(rot_labels[i], fontsize=10)
		ax.grid(True, alpha=0.3)
		ax.set_facecolor('white')
		if i < 2:
			ax.tick_params(labelbottom=False)
		else:
			ax.set_xlabel('Time (s)', fontsize=10)

	# Grasper state (above clutch) for both columns
	ax_grasper_left = fig.add_subplot(gs[3, 0])
	ax_grasper_right = fig.add_subplot(gs[3, 1])
	if grasper_state is not None:
		g = np.asarray(grasper_state, dtype=float)
		# Normalize: max -> 0, min -> 1
		g_min = np.nanmin(g)
		g_max = np.nanmax(g)
		if np.isfinite(g_min) and np.isfinite(g_max) and g_max > g_min:
			g_norm = (g_max - g) / (g_max - g_min)
		else:
			g_norm = np.zeros_like(g)
		t_g = np.arange(g_norm.shape[0]) / fps
		for axg in (ax_grasper_left, ax_grasper_right):
			axg.step(t_g, g_norm, where='post', color='#AA5599', linewidth=1.6, zorder=3)
			axg.fill_between(t_g, 0, g_norm, step='post', color='#AA5599', alpha=0.25, zorder=2)
			axg.set_ylim(-0.1, 1.1)
			axg.set_yticks([0, 1])
			axg.set_ylabel('Grasper', fontsize=9)
			axg.grid(True, alpha=0.25)
			axg.set_facecolor('white')
			axg.tick_params(labelbottom=False)

	# Pedal subplots (bottom, one under each column)
	ax_pedal_left = fig.add_subplot(gs[4, 0])
	ax_pedal_right = fig.add_subplot(gs[4, 1])
	if pedal is not None:
		# Reverse logic: visualize 1 when not pressed, 0 when pressed
		ped = 1 - pedal
		# Left pedal under position subfigures
		ax_pedal_left.step(t_pos[: ped.shape[0]], ped, where='post', color='#CCBB44', linewidth=1.6, zorder=3)
		ax_pedal_left.fill_between(t_pos[: ped.shape[0]], 0, ped, step='post', color='#CCBB44', alpha=0.25, zorder=2)
		ax_pedal_left.set_ylim(-0.1, 1.1)
		ax_pedal_left.set_yticks([0, 1])
		ax_pedal_left.set_ylabel('Clutch', fontsize=9)
		ax_pedal_left.set_xlabel('Time (s)', fontsize=10)
		ax_pedal_left.grid(True, alpha=0.25)
		ax_pedal_left.set_facecolor('white')

		# Right pedal under rotation subfigures
		ax_pedal_right.step(t_pos[: ped.shape[0]], ped, where='post', color='#CCBB44', linewidth=1.6, zorder=3)
		ax_pedal_right.fill_between(t_pos[: ped.shape[0]], 0, ped, step='post', color='#CCBB44', alpha=0.25, zorder=2)
		ax_pedal_right.set_ylim(-0.1, 1.1)
		ax_pedal_right.set_yticks([0, 1])
		ax_pedal_right.set_xlabel('Time (s)', fontsize=10)
		ax_pedal_right.grid(True, alpha=0.25)
		ax_pedal_right.set_facecolor('white')
		# Optional: hide right y-label to reduce clutter
		ax_pedal_right.set_ylabel('')

		# Hide top xlabels on row 3 axes since pedals have their own xlabels
		for ax in a_left + a_right:
			ax.tick_params(labelbottom=False)

	# Build line legend (dedup)
	handles, labels = [], []
	for ax in (a_left[0], a_right[0]):
		h, l = ax.get_legend_handles_labels()
		handles += h; labels += l
	seen, handles_dedup, labels_dedup = set(), [], []
	for h, l in zip(handles, labels):
		if l not in seen:
			seen.add(l)
			handles_dedup.append(h)
			labels_dedup.append(l)
	fig.legend(handles_dedup, labels_dedup, loc='upper center', ncol=4, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 1.08))

	# Gesture spans across all subplots (if provided)
	if gesture_names is not None:
		gesture_list = list(pd.Series(list(gesture_names)).fillna('Unknown').astype(str))
		spans = compute_gesture_spans(gesture_list, fps=fps)
		color_map = build_soft_color_map([name for _, _, name in spans])
		add_gesture_spans_to_axes(a_left + a_right + [ax_grasper_left, ax_grasper_right, ax_pedal_left, ax_pedal_right], spans, color_map)
		add_gesture_legend(fig, color_map)

	for ax in a_left + a_right:
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

	Path(out_path).parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(out_path, dpi=300, bbox_inches='tight')
	plt.close(fig)


# -----------------------------
# Metrics (kept as in notebook for parity)
# -----------------------------
def calculate_cosine_similarity_pos(data1: np.ndarray, data2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
	min_length = min(data1.shape[0], data2.shape[0])
	data1 = data1[:min_length, :6]
	data2 = data2[:min_length, :6]
	cosine_per_dim = np.zeros(6)
	for i in range(6):
		vec1 = data1[:, i].reshape(1, -1)
		vec2 = data2[:, i].reshape(1, -1)
		cosine_per_dim[i] = cosine_similarity(vec1, vec2)[0, 0]
	cosine_left_xyz = cosine_per_dim[:3]
	cosine_right_xyz = cosine_per_dim[3:]
	cosine_overall = cosine_similarity(data1.reshape(1, -1), data2.reshape(1, -1))[0, 0]
	return cosine_left_xyz, cosine_right_xyz, cosine_per_dim, cosine_overall


def calculate_nrmse_pos(data1: np.ndarray, data2: np.ndarray, normalization_method: str = 'range') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
	min_length = min(data1.shape[0], data2.shape[0])
	data1 = data1[:min_length, :6]
	data2 = data2[:min_length, :6]
	squared_diff = (data1 - data2) ** 2
	rmse_total = np.sqrt(np.mean(squared_diff))
	rmse_per_dim = np.sqrt(np.mean(squared_diff, axis=0))
	if normalization_method == 'range':
		normalization_factors = np.ptp(data2, axis=0)
		overall_normalization = np.ptp(data2)
	elif normalization_method == 'mean':
		normalization_factors = np.abs(np.mean(data2, axis=0))
		overall_normalization = np.abs(np.mean(data2))
	elif normalization_method == 'std':
		normalization_factors = np.std(data2, axis=0)
		overall_normalization = np.std(data2)
	else:
		raise ValueError("normalization_method must be 'range', 'mean', or 'std'")
	normalization_factors[normalization_factors == 0] = 1
	if overall_normalization == 0:
		overall_normalization = 1
	nrmse_total = (rmse_total / overall_normalization) * 100
	nrmse_per_dim = (rmse_per_dim / normalization_factors) * 100
	nrmse_left_xyz = nrmse_per_dim[:3]
	nrmse_right_xyz = nrmse_per_dim[3:]
	return nrmse_total, nrmse_left_xyz, nrmse_right_xyz, nrmse_per_dim


def calculate_cosine_similarity_rot(rot_data1: np.ndarray, rot_data2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
	min_length = min(rot_data1.shape[0], rot_data2.shape[0])
	rot_data1 = rot_data1[:min_length, :3]
	rot_data2 = rot_data2[:min_length, :3]
	cosine_per_dim = np.zeros(3)
	for i in range(3):
		vec1 = rot_data1[:, i].reshape(1, -1)
		vec2 = rot_data2[:, i].reshape(1, -1)
		cosine_per_dim[i] = cosine_similarity(vec1, vec2)[0, 0]
	cosine_left_xyz = cosine_per_dim
	cosine_right_xyz = cosine_per_dim
	cosine_overall = cosine_similarity(rot_data1.reshape(1, -1), rot_data2.reshape(1, -1))[0, 0]
	return cosine_left_xyz, cosine_right_xyz, cosine_per_dim, cosine_overall


def calculate_nrmse_rot(rot_data1: np.ndarray, rot_data2: np.ndarray, normalization_method: str = 'range') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
	min_length = min(rot_data1.shape[0], rot_data2.shape[0])
	rot_data1 = rot_data1[:min_length, :3]
	rot_data2 = rot_data2[:min_length, :3]
	squared_diff = (rot_data1 - rot_data2) ** 2
	rmse_total = np.sqrt(np.mean(squared_diff))
	rmse_per_dim = np.sqrt(np.mean(squared_diff, axis=0))
	if normalization_method == 'range':
		normalization_factors = np.ptp(rot_data2, axis=0)
		overall_normalization = np.ptp(rot_data2)
	elif normalization_method == 'mean':
		normalization_factors = np.abs(np.mean(rot_data2, axis=0))
		overall_normalization = np.abs(np.mean(rot_data2))
	elif normalization_method == 'std':
		normalization_factors = np.std(rot_data2, axis=0)
		overall_normalization = np.std(rot_data2)
	else:
		raise ValueError("normalization_method must be 'range', 'mean', or 'std'")
	normalization_factors[normalization_factors == 0] = 1
	if overall_normalization == 0:
		overall_normalization = 1
	nrmse_total = (rmse_total / overall_normalization) * 100
	nrmse_per_dim = (rmse_per_dim / normalization_factors) * 100
	nrmse_left_xyz = nrmse_per_dim
	nrmse_right_xyz = nrmse_per_dim
	return nrmse_total, nrmse_left_xyz, nrmse_right_xyz, nrmse_per_dim


# -----------------------------
# IO and main routine
# -----------------------------
def find_default_csv(trial: Optional[int]) -> Path:
	if trial is not None:
		cand = Path(f'final_annotation_t{trial}.csv')
		if cand.exists():
			return cand
	# Try common trials
	for t in [2, 1, 3, 4, 5, 6]:
		cand = Path(f'final_annotation_t{t}.csv')
		if cand.exists():
			return cand
	raise FileNotFoundError('Could not locate final_annotation_t*.csv in current directory')


def load_and_prepare(csv_path: Path, start_trim: int, end_trim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, Optional[np.ndarray]]:
	data = pd.read_csv(csv_path)
	# Extract arrays
	raven_data = get_raven_data(data)
	console_data = get_console_data(data)
	trakstar_data = get_trakstar_data(data)
	smartwatch_data = get_smartwatch_data(data)
	gesture_series = data['gesture_name'].astype(str).fillna('Unknown') if 'gesture_name' in data.columns else pd.Series(['Unknown'] * len(data))

	# Apply trims (like notebook: [2000:-1200])
	end_index = None if end_trim == 0 else -end_trim
	raven_data = raven_data[start_trim:end_index]
	console_data = console_data[start_trim:end_index]
	trakstar_data = trakstar_data[start_trim:end_index]
	smartwatch_data = smartwatch_data[start_trim:end_index]
	gesture_series = gesture_series.iloc[start_trim:end_index].reset_index(drop=True)

	# Optional grasper state from console_aux0
	if 'console_aux0' in data.columns:
		aux0_full = data['console_aux0'].to_numpy()
		grasper_state = aux0_full[start_trim:end_index]
	else:
		grasper_state = None

	# Split pedal
	pedal_data = console_data[:, -1]
	console_core = console_data[:, :-1]

	# Smartwatch derived
	left_sw_pos, right_sw_pos = process_smartwatch_pos(smartwatch_data, pedal_data)
	left_sw_rot, right_sw_rot = process_smartwatch_rot(smartwatch_data, pedal_data)

	# Rotations alignment
	trackstar_psm1_rot, console_psm1_rot, raven_psm1_rot, trackstar_psm2_rot, console_psm2_rot, raven_psm2_rot = align_trackstar_console_raven_rot(
		trakstar_data, console_core, raven_data
	)

	# Positions alignment
	console_pos_aligned, trackstar_pos_aligned = align_console_trackstar_data_pos(console_core, trakstar_data)
	new_console_pos, new_trackstar_pos, new_raven_pos = transform_data(console_pos_aligned, trackstar_pos_aligned, raven_data, pedal_data)
	return (
		new_raven_pos,
		new_console_pos,
		new_trackstar_pos,
		raven_psm1_rot,
		console_psm1_rot,
		trackstar_psm1_rot,
		left_sw_pos,
		left_sw_rot,
		gesture_series.tolist(),
        pedal_data,
        grasper_state,
	)


def main():
	parser = argparse.ArgumentParser(description='Kinematic visualization with gesture spans')
	parser.add_argument('--csv', type=str, default=None, help='Path to final_annotation_tX.csv')
	parser.add_argument('--trial', type=int, default=None, help='Trial number (if --csv not provided)')
	parser.add_argument('--start-trim', type=int, default=2000, help='Trim N samples from start')
	parser.add_argument('--end-trim', type=int, default=1200, help='Trim N samples from end')
	parser.add_argument('--fps', type=float, default=DEFAULT_FPS, help='Sampling rate (Hz)')
	parser.add_argument('--out', type=str, default=f'{PLOT_SAVE_DIR}/pos_rot_side_by_side_left.png', help='Output image path')
	args = parser.parse_args()

	csv_path = Path(args.csv) if args.csv else find_default_csv(args.trial)
	(
		raven_pos,
		console_pos,
		trackstar_pos,
		raven_rot,
		console_rot,
		trackstar_rot,
		left_sw_pos,
		left_sw_rot,
		gesture_names,
        pedal_data,
        grasper_state,
	) = load_and_prepare(csv_path, start_trim=args.start_trim, end_trim=args.end_trim)

	plot_pos_rot_side_by_side(
		raven_pos,
		console_pos,
		trackstar_pos,
		raven_rot,
		console_rot,
		trackstar_rot,
        left_sw_pos=None,  # hide smartwatch position by default (can be noisy)
        smartwatch_rot=None,  # remove smartwatch from plots
        grasper_state=grasper_state,
		fps=args.fps,
		out_path=args.out,
		gesture_names=gesture_names,
		pedal=pedal_data,
	)

	print(f'Wrote figure to {args.out}')


if __name__ == '__main__':
	main()


