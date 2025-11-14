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


def cumulate_trackstar_console_data_rot(console_arr: np.ndarray, trackstar_arr: np.ndarray, pedal: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Cumulate rotations (console and trackstar) with optional pedal mask.
	Accepts either:
	  - console_arr shape (N, 12+) with rotations in cols 6:12, or shape (N, 6) of rotations only
	  - trackstar_arr shape (N, 12+) with rotations in cols 6:12, or shape (N, 6) of rotations only
	Returns (console_rot_cum (N,6), trackstar_rot_cum (N,6)).
	"""
	# Extract rotation slices flexibly
	if console_arr.shape[1] >= 12:
		console_rot_src = console_arr[:, 6:12]
	elif console_arr.shape[1] == 6:
		console_rot_src = console_arr
	else:
		raise ValueError('console_arr must have 6 rotation columns or 12+ columns with rotations at 6:12')
	if trackstar_arr.shape[1] >= 12:
		trackstar_rot_src = trackstar_arr[:, 6:12]
	elif trackstar_arr.shape[1] == 6:
		trackstar_rot_src = trackstar_arr
	else:
		raise ValueError('trackstar_arr must have 6 rotation columns or 12+ columns with rotations at 6:12')
	N = console_rot_src.shape[0]
	console_rot = np.zeros((N, 6))
	trackstar_rot = np.zeros((trackstar_arr.shape[0], 6))
	diff_console = np.diff(console_rot_src, axis=0)
	diff_trackstar = np.diff(np.radians(trackstar_rot_src), axis=0)
	if pedal is not None:
		zero_mask = pedal[:-1] == 0
	else:
		zero_mask = np.zeros(diff_console.shape[0], dtype=bool)
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
    grasper_left_state: Optional[np.ndarray] = None,
    grasper_right_state: Optional[np.ndarray] = None,
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
	# Left grasper from console_aux0
	if grasper_left_state is not None:
		g_left = np.asarray(grasper_left_state, dtype=float)
		g_left_min = np.nanmin(g_left)
		g_left_max = np.nanmax(g_left)
		if np.isfinite(g_left_min) and np.isfinite(g_left_max) and g_left_max > g_left_min:
			g_left_norm = (g_left_max - g_left) / (g_left_max - g_left_min)
		else:
			g_left_norm = np.zeros_like(g_left)
		t_g_left = np.arange(g_left_norm.shape[0]) / fps
		ax_grasper_left.step(t_g_left, g_left_norm, where='post', color='#AA5599', linewidth=1.6, zorder=3)
		ax_grasper_left.fill_between(t_g_left, 0, g_left_norm, step='post', color='#AA5599', alpha=0.25, zorder=2)
	ax_grasper_left.set_ylim(-0.1, 1.1)
	ax_grasper_left.set_yticks([0, 1])
	ax_grasper_left.set_ylabel('Left Grasper', fontsize=9)
	ax_grasper_left.grid(True, alpha=0.25)
	ax_grasper_left.set_facecolor('white')
	ax_grasper_left.tick_params(labelbottom=False)

	# Right grasper from console_aux1
	if grasper_right_state is not None:
		g_right = np.asarray(grasper_right_state, dtype=float)
		g_right_min = np.nanmin(g_right)
		g_right_max = np.nanmax(g_right)
		if np.isfinite(g_right_min) and np.isfinite(g_right_max) and g_right_max > g_right_min:
			g_right_norm = (g_right_max - g_right) / (g_right_max - g_right_min)
		else:
			g_right_norm = np.zeros_like(g_right)
		t_g_right = np.arange(g_right_norm.shape[0]) / fps
		ax_grasper_right.step(t_g_right, g_right_norm, where='post', color='#AA5599', linewidth=1.6, zorder=3)
		ax_grasper_right.fill_between(t_g_right, 0, g_right_norm, step='post', color='#AA5599', alpha=0.25, zorder=2)
	ax_grasper_right.set_ylim(-0.1, 1.1)
	ax_grasper_right.set_yticks([0, 1])
	ax_grasper_right.set_ylabel('Right Grasper', fontsize=9)
	ax_grasper_right.grid(True, alpha=0.25)
	ax_grasper_right.set_facecolor('white')
	ax_grasper_right.tick_params(labelbottom=False)

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
def _detect_dynamic_start_index(df: pd.DataFrame, default_start: int) -> int:
	"""
	Return the index of the first row where 'raven_console_time_ns' is a finite number != -1.
	Fallback to default_start if the column is missing or contains no valid entries.
	"""
	col = 'raven_console_time_ns'
	if col in df.columns:
		series = pd.to_numeric(df[col], errors='coerce')
		valid = series.notna() & (series != -1)
		idxs = np.flatnonzero(valid.to_numpy())
		if idxs.size > 0:
			return int(idxs[0])
	return int(default_start)


def fit_affine_3d(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""Fit Y ≈ A @ X + b via least squares. X, Y shapes (N, 3). Returns (A(3x3), b(3,))."""
	if X.shape[1] != 3 or Y.shape[1] != 3:
		raise ValueError('fit_affine_3d expects X, Y with 3 columns')
	X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
	W, *_ = np.linalg.lstsq(X_aug, Y, rcond=None)
	A = W[:3, :].T
	b = W[3, :].T
	return A, b


def apply_affine_3d(X: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
	return (A @ X.T).T + b


def fit_mlp_3d(
	X: np.ndarray,
	Y: np.ndarray,
	hidden_layers: Tuple[int, ...] = (16, 16),
	activation: str = 'relu',
	max_iter: int = 3000,
	alpha: float = 0.3,
	random_state: int = 0,
):
	try:
		from sklearn.neural_network import MLPRegressor  # type: ignore
		from sklearn.pipeline import make_pipeline  # type: ignore
		from sklearn.preprocessing import StandardScaler  # type: ignore
		from sklearn.compose import TransformedTargetRegressor  # type: ignore
	except Exception as e:
		raise RuntimeError('scikit-learn is required for --model mlp') from e
	# Scale inputs and targets; enable early stopping and stronger L2 to improve generalization
	base_regressor = make_pipeline(
		StandardScaler(),
		MLPRegressor(
			hidden_layer_sizes=hidden_layers,
			activation=activation,
			max_iter=max_iter,
			alpha=alpha,
			random_state=random_state,
			solver='adam',
		    early_stopping=True,
			validation_fraction=0.1,
			n_iter_no_change=20,
			shuffle=True,		
			learning_rate_init=1e-3,
		
		),
	)
	model = TransformedTargetRegressor(regressor=base_regressor, transformer=StandardScaler())
	model.fit(X, Y)
	return model


def apply_mlp_3d(model, X: np.ndarray) -> np.ndarray:
	return model.predict(X)


# -----------------------------
# Feature engineering for sequences
# -----------------------------
def _edge_pad(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return (X_prev, X, X_next) with edge replication padding."""
	if arr.ndim != 2 or arr.shape[1] != 3:
		raise ValueError('Expected (N,3) array')
	n = arr.shape[0]
	if n == 0:
		return arr.copy(), arr.copy(), arr.copy()
	x_prev = np.vstack([arr[0:1, :], arr[:-1, :]])
	x_next = np.vstack([arr[1:, :], arr[-1:, :]])
	return x_prev, arr, x_next


def _finite_diff(arr: np.ndarray) -> np.ndarray:
	"""Simple first-order difference with zero at t=0; same shape as input."""
	if arr.ndim != 2:
		raise ValueError('Expected 2D array')
	if arr.shape[0] == 0:
		return arr.copy()
	diff = np.vstack([np.zeros((1, arr.shape[1])), np.diff(arr, axis=0)])
	return diff


def build_temporal_features_3d(x_series: np.ndarray, include_velocity: bool = True, include_context: bool = True) -> np.ndarray:
	"""
	Build features from a 3D time series (N,3):
	- context stacking: [x_{t-1}, x_t, x_{t+1}] → 9 dims
	- velocity cues: [x_t - x_{t-1}, x_{t+1} - x_t] → 6 dims
	Returns (N, F).
	"""
	x_prev, x_curr, x_next = _edge_pad(x_series)
	features: List[np.ndarray] = []
	if include_context:
		features.extend([x_prev, x_curr, x_next])
	else:
		features.append(x_curr)
	if include_velocity:
		v_prev = x_curr - x_prev
		v_next = x_next - x_curr
		features.extend([v_prev, v_next])
	return np.hstack(features) if features else x_series


# -----------------------------
# Geometry utilities (rigid fit via Kabsch)
# -----------------------------
def _fit_rigid_transform(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Fit rigid transform Y ≈ R @ X + t.
	Inputs X,Y shape (N,3). Returns (R(3x3), t(3,)).
	"""
	if X.shape[1] != 3 or Y.shape[1] != 3:
		raise ValueError('Expected (N,3) inputs')
	mu_X = X.mean(axis=0)
	mu_Y = Y.mean(axis=0)
	Xc = X - mu_X
	Yc = Y - mu_Y
	H = Xc.T @ Yc
	U, _, Vt = np.linalg.svd(H)
	Rm = Vt.T @ U.T
	if np.linalg.det(Rm) < 0:
		Vt[-1, :] *= -1
		Rm = Vt.T @ U.T
	t = mu_Y - (Rm @ mu_X)
	return Rm, t


# -----------------------------
# Normalization helpers (positions)
# -----------------------------
def _compute_position_frame_stats(
	csv_paths: List[Path],
	start_trim: int,
	end_trim: int,
) -> Dict[str, Dict[str, np.ndarray]]:
	"""
	Compute per-axis min and range for each position frame: console, trackstar, raven.
	Returns: {'console': {'min': (3,), 'range': (3,)}, ...}
	"""
	acc: Dict[str, List[np.ndarray]] = {'console': [], 'trackstar': [], 'raven': []}
	for p in csv_paths:
		(
			rpos_L, rpos_R,
			cpos_L, cpos_R,
			tpos_L, tpos_R,
			_, _,
			_, _,
			_, _,
			_, _, _, _,
		) = _extract_and_cumulate_from_csv(p, start_trim, end_trim)
		acc['console'].append(np.vstack([cpos_L, cpos_R]))
		acc['trackstar'].append(np.vstack([tpos_L, tpos_R]))
		acc['raven'].append(np.vstack([rpos_L, rpos_R]))
	stats: Dict[str, Dict[str, np.ndarray]] = {}
	for frame in acc.keys():
		if acc[frame]:
			stacked = np.vstack(acc[frame])
		else:
			stacked = np.zeros((0, 3))
		if stacked.shape[0] == 0:
			min_v = np.zeros(3)
			ptp_v = np.ones(3)
		else:
			min_v = stacked.min(axis=0)
			max_v = stacked.max(axis=0)
			ptp_v = max_v - min_v
			ptp_v[ptp_v == 0] = 1.0
		stats[frame] = {'min': min_v, 'range': ptp_v}
	return stats


def _apply_pos_norm(x: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
	return (x - stats['min']) / stats['range']


def _invert_pos_norm(x_norm: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
	return x_norm * stats['range'] + stats['min']


# -----------------------------
# MLP variants: residual on rigid (positions) and sin/cos heads (rotations)
# -----------------------------
def fit_rigid_residual_mlp_3d(
	X_series: np.ndarray,
	Y_series: np.ndarray,
	alpha: float = 0.3,
	hidden_layers: Tuple[int, ...] = (16, 16),
	activation: str = 'relu',
	max_iter: int = 3000,
	random_state: int = 0,
	use_temporal_features: bool = True,
	in_norm_stats: Optional[Dict[str, np.ndarray]] = None,
	out_norm_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
	"""
	Fit Y ≈ R X + t + fθ(features(X)).
	If normalization stats are provided, fit in normalized space and return them in the model dict.
	Returns dict with keys: 'R','t','mlp','in_norm','out_norm','use_temporal'.
	"""
	try:
		from sklearn.neural_network import MLPRegressor  # type: ignore
		from sklearn.pipeline import make_pipeline  # type: ignore
		from sklearn.preprocessing import StandardScaler  # type: ignore
		from sklearn.compose import TransformedTargetRegressor  # type: ignore
	except Exception as e:
		raise RuntimeError('scikit-learn is required for residual MLP') from e
	Xn = X_series if in_norm_stats is None else _apply_pos_norm(X_series, in_norm_stats)
	Yn = Y_series if out_norm_stats is None else _apply_pos_norm(Y_series, out_norm_stats)
	Rm, t = _fit_rigid_transform(Xn, Yn)
	base_pred = (Xn @ Rm.T) + t  # (N,3)
	residual = Yn - base_pred
	feat = build_temporal_features_3d(Xn, include_velocity=True, include_context=True) if use_temporal_features else Xn
	reg = make_pipeline(
		StandardScaler(),
		MLPRegressor(
			hidden_layer_sizes=hidden_layers,
			activation=activation,
			max_iter=max_iter,
			alpha=alpha,
			random_state=random_state,
			solver='adam',
			early_stopping=True,
			validation_fraction=0.1,
			n_iter_no_change=20,
			shuffle=True,		
			learning_rate_init=1e-3,
		),
	)
	model = TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())
	model.fit(feat, residual)
	return {
		'R': Rm,
		't': t,
		'mlp': model,
		'in_norm': in_norm_stats,
		'out_norm': out_norm_stats,
		'use_temporal': bool(use_temporal_features),
	}


def apply_rigid_residual_mlp_3d(model_dict: Dict[str, object], X_series: np.ndarray) -> np.ndarray:
	"""Apply residual-on-rigid model to a full sequence X_series (N,3)."""
	in_stats = model_dict.get('in_norm')
	out_stats = model_dict.get('out_norm')
	Xn = X_series if in_stats is None else _apply_pos_norm(X_series, in_stats)  # (N,3)
	Rm: np.ndarray = model_dict['R']  # type: ignore
	t: np.ndarray = model_dict['t']  # type: ignore
	base_pred = (Xn @ Rm.T) + t  # (N,3)
	use_temporal = bool(model_dict.get('use_temporal', True))
	feat = build_temporal_features_3d(Xn, include_velocity=True, include_context=True) if use_temporal else Xn
	residual_pred = model_dict['mlp'].predict(feat)  # type: ignore
	Yn_pred = base_pred + residual_pred
	Y = Yn_pred if out_stats is None else _invert_pos_norm(Yn_pred, out_stats)
	return Y


def fit_mlp_rot_sincos_3d(
	X_series: np.ndarray,
	Y_angles: np.ndarray,
	alpha: float = 0.3,
	hidden_layers: Tuple[int, ...] = (16, 16),
	activation: str = 'relu',
	max_iter: int = 3000,
	random_state: int = 0,
	use_temporal_features: bool = True,
) -> object:
	"""
	Fit MLP on inputs (optionally with temporal features) to predict [sin,cos] per angle.
	Output dimension = 6.
	"""
	try:
		from sklearn.neural_network import MLPRegressor  # type: ignore
		from sklearn.pipeline import make_pipeline  # type: ignore
		from sklearn.preprocessing import StandardScaler  # type: ignore
	except Exception as e:
		raise RuntimeError('scikit-learn is required for rotation MLP') from e
	Y_sc = np.hstack([np.sin(Y_angles), np.cos(Y_angles)])  # (N,6)
	X_feat = build_temporal_features_3d(X_series, include_velocity=True, include_context=True) if use_temporal_features else X_series
	reg = make_pipeline(
		StandardScaler(),
		MLPRegressor(
			hidden_layer_sizes=hidden_layers,
			activation=activation,
			max_iter=max_iter,
			alpha=alpha,
			random_state=random_state,
			solver='adam',
			early_stopping=True,
			validation_fraction=0.1,
			n_iter_no_change=20,
			shuffle=True,		
			learning_rate_init=1e-3,
		),
	)
	reg.fit(X_feat, Y_sc)
	return {'mlp': reg, 'use_temporal': bool(use_temporal_features)}


def apply_mlp_rot_sincos_3d(model_obj: object, X_series: np.ndarray) -> np.ndarray:
	"""Predict angles from sin/cos outputs."""
	if isinstance(model_obj, dict):
		use_temporal = bool(model_obj.get('use_temporal', True))
		reg = model_obj['mlp']
	else:
		use_temporal = True
		reg = model_obj
	X_feat = build_temporal_features_3d(X_series, include_velocity=True, include_context=True) if use_temporal else X_series
	Y_sc = reg.predict(X_feat)  # (N,6)
	sin = Y_sc[:, 0:3]
	cos = Y_sc[:, 3:6]
	angles = np.arctan2(sin, cos)
	return angles

def _extract_and_cumulate_from_csv(
	csv_path: Path,
	start_trim: int,
	end_trim: int,
) -> Tuple[
		np.ndarray, np.ndarray,  # raven_pos_left, raven_pos_right
		np.ndarray, np.ndarray,  # console_pos_left, console_pos_right
		np.ndarray, np.ndarray,  # trackstar_pos_left, trackstar_pos_right
		np.ndarray, np.ndarray,  # raven_rot_left, raven_rot_right
		np.ndarray, np.ndarray,  # console_rot_left, console_rot_right
		np.ndarray, np.ndarray,  # trackstar_rot_left, trackstar_rot_right
		List[str],               # gesture_names
		np.ndarray,              # pedal
		Optional[np.ndarray], Optional[np.ndarray],  # grasper_left_state, grasper_right_state
]:
	# Load
	data = pd.read_csv(csv_path)
	raven_data = get_raven_data(data)
	console_data = get_console_data(data)
	trakstar_data = get_trakstar_data(data)
	gesture_series = data['gesture_name'].astype(str).fillna('Unknown') if 'gesture_name' in data.columns else pd.Series(['Unknown'] * len(data))
	# Trim
	eff_start = _detect_dynamic_start_index(data, start_trim)
	end_index = None if end_trim == 0 else -end_trim
	raven_data = raven_data[eff_start:end_index]
	console_data = console_data[eff_start:end_index]
	trakstar_data = trakstar_data[eff_start:end_index]
	gesture_series = gesture_series.iloc[eff_start:end_index].reset_index(drop=True)
	# Graspers
	if 'console_aux0' in data.columns:
		grasper_left_state = data['console_aux0'].to_numpy()[eff_start:end_index]
	else:
		grasper_left_state = None
	if 'console_aux1' in data.columns:
		grasper_right_state = data['console_aux1'].to_numpy()[eff_start:end_index]
	else:
		grasper_right_state = None
	# Pedal and separate console core
	pedal = console_data[:, -1]
	console_core = console_data[:, :-1]
	# Positions: accumulate with pedal masking
	console_pos_cum, trackstar_pos_cum = cumulate_trackstar_console_data(console_core[:, :6], trakstar_data[:, :6], pedal)
	# Rotations: accumulate with pedal masking
	console_rot_cum, trackstar_rot_cum = cumulate_trackstar_console_data_rot(console_core[:, 6:12], trakstar_data[:, 6:12], pedal)
	# Ground-truth PSM positions (6: left xyz, right xyz) — subtract initial, no fixed scaling
	raven_pos6 = raven_data[:, :6] - raven_data[0, :6]
	raven_pos_left = raven_pos6[:, 0:3]
	raven_pos_right = raven_pos6[:, 3:6]
	# Ground-truth PSM rotations (Euler zyx), subtract initial and wrap
	raven_rot_left = rotation_matrix_to_euler(raven_data[:, 6:15])
	raven_rot_right = rotation_matrix_to_euler(raven_data[:, 15:24])
	raven_rot_left = process_raven_rot(raven_rot_left - raven_rot_left[0, :])
	raven_rot_right = process_raven_rot(raven_rot_right - raven_rot_right[0, :])
	# Feature selections (no fixed frame transforms; let the models learn them)
	console_pos_left = console_pos_cum[:, 0:3]
	console_pos_right = console_pos_cum[:, 3:6]
	trackstar_pos_left = trackstar_pos_cum[:, 0:3]
	trackstar_pos_right = trackstar_pos_cum[:, 3:6]
	# For rotations, reuse established channel groupings but do not apply constant transforms
	console_rot_left = console_rot_cum[:, [0, 2, 4]]
	console_rot_right = console_rot_cum[:, [1, 3, 5]]
	trackstar_rot_left = trackstar_rot_cum[:, 0:3]
	trackstar_rot_right = trackstar_rot_cum[:, 3:6]
	return (
		raven_pos_left, raven_pos_right,
		console_pos_left, console_pos_right,
		trackstar_pos_left, trackstar_pos_right,
		raven_rot_left, raven_rot_right,
		console_rot_left, console_rot_right,
		trackstar_rot_left, trackstar_rot_right,
		gesture_series.tolist(),
		pedal,
		grasper_left_state,
		grasper_right_state,
	)


def _aggregate_training_data(
	csv_paths: List[Path],
	start_trim: int,
	end_trim: int,
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
	"""
	Build training pairs X,Y for each mapping:
	  - positions: console_left/right, trackstar_left/right -> raven_left/right
	  - rotations: console_left/right, trackstar_left/right -> raven_left/right
	  - positions: trackstar_left/right -> console_left/right (for EmHT→MTM metrics)
	  - rotations: trackstar_left/right -> console_left/right (for EmHT→MTM metrics)
	Returns dict: {'pos': {...}, 'rot': {...}} where values are (X, Y).
	"""
	XY: Dict[str, Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]] = {
		'pos': {
			'console_left': ([], []),
			'console_right': ([], []),
			'trackstar_left': ([], []),
			'trackstar_right': ([], []),
			'trackstar_to_console_left': ([], []),
			'trackstar_to_console_right': ([], []),
		},
		'rot': {
			'console_left': ([], []),
			'console_right': ([], []),
			'trackstar_left': ([], []),
			'trackstar_right': ([], []),
			'trackstar_to_console_left': ([], []),
			'trackstar_to_console_right': ([], []),
		},
	}
	for p in csv_paths:
		(
			rpos_L, rpos_R,
			cpos_L, cpos_R,
			tpos_L, tpos_R,
			rrot_L, rrot_R,
			crot_L, crot_R,
			trot_L, trot_R,
			_, _, _, _,
		) = _extract_and_cumulate_from_csv(p, start_trim, end_trim)
		# Positions
		XY['pos']['console_left'][0].append(cpos_L); XY['pos']['console_left'][1].append(rpos_L)
		XY['pos']['console_right'][0].append(cpos_R); XY['pos']['console_right'][1].append(rpos_R)
		XY['pos']['trackstar_left'][0].append(tpos_L); XY['pos']['trackstar_left'][1].append(rpos_L)
		XY['pos']['trackstar_right'][0].append(tpos_R); XY['pos']['trackstar_right'][1].append(rpos_R)
		# EmHT→MTM positions
		XY['pos']['trackstar_to_console_left'][0].append(tpos_L); XY['pos']['trackstar_to_console_left'][1].append(cpos_L)
		XY['pos']['trackstar_to_console_right'][0].append(tpos_R); XY['pos']['trackstar_to_console_right'][1].append(cpos_R)
		# Rotations
		XY['rot']['console_left'][0].append(crot_L); XY['rot']['console_left'][1].append(rrot_L)
		XY['rot']['console_right'][0].append(crot_R); XY['rot']['console_right'][1].append(rrot_R)
		XY['rot']['trackstar_left'][0].append(trot_L); XY['rot']['trackstar_left'][1].append(rrot_L)
		XY['rot']['trackstar_right'][0].append(trot_R); XY['rot']['trackstar_right'][1].append(rrot_R)
		# EmHT→MTM rotations
		XY['rot']['trackstar_to_console_left'][0].append(trot_L); XY['rot']['trackstar_to_console_left'][1].append(crot_L)
		XY['rot']['trackstar_to_console_right'][0].append(trot_R); XY['rot']['trackstar_to_console_right'][1].append(crot_R)
	# Stack lists
	out: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {'pos': {}, 'rot': {}}
	for kind in ['pos', 'rot']:
		for key in XY[kind].keys():
			X_list, Y_list = XY[kind][key]
			X = np.vstack(X_list) if X_list else np.zeros((0, 3))
			Y = np.vstack(Y_list) if Y_list else np.zeros((0, 3))
			out[kind][key] = (X, Y)
	return out


def _fit_models(
	train_pairs: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
	model_type: str = 'affine',
	normalize_frames: bool = False,
	pos_frame_stats: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
	use_temporal_features: bool = True,
) -> Dict[str, Dict[str, object]]:
	"""Fit models for 'pos' and 'rot' across console/trackstar and left/right."""
	models: Dict[str, Dict[str, object]] = {'pos': {}, 'rot': {}}
	for kind in ['pos', 'rot']:
		for key, (X, Y) in train_pairs[kind].items():
			if X.shape[0] == 0:
				continue
			# Determine mapping frames for positions to support normalization
			if kind == 'pos':
				if key.startswith('console_'):
					in_frame = 'console'
					out_frame = 'raven'  # MTM → PSM
				elif key.startswith('trackstar_to_console_'):
					in_frame = 'trackstar'  # EmHT → MTM
					out_frame = 'console'
				elif key.startswith('trackstar_'):
					in_frame = 'trackstar'  # EmHT → PSM
					out_frame = 'raven'
				else:
					in_frame = 'console'
					out_frame = 'raven'
			else:
				in_frame = ''
				out_frame = ''
			# For rotations: always use sin/cos MLP (better wrap handling)
			if kind == 'rot':
				model = fit_mlp_rot_sincos_3d(X, Y)
				models[kind][key] = ('mlp_rot_sincos', model)
				continue
			# Positions
			if model_type == 'affine':
				A, b = fit_affine_3d(X, Y)
				models[kind][key] = ('affine', A, b)
			else:
				in_stats = pos_frame_stats[in_frame] if (normalize_frames and pos_frame_stats is not None) else None
				out_stats = pos_frame_stats[out_frame] if (normalize_frames and pos_frame_stats is not None) else None
				model = fit_rigid_residual_mlp_3d(
					X, Y,
					alpha=0.3,
					hidden_layers=(16, 16),
					activation='relu',
					max_iter=3000,
					random_state=0,
					use_temporal_features=use_temporal_features,
					in_norm_stats=in_stats,
					out_norm_stats=out_stats,
				)
				# package tuple
				models[kind][key] = ('rigid_residual_mlp', model, in_frame, out_frame)
	return models


def _apply_model_3d(model_obj: object, X: np.ndarray) -> np.ndarray:
	"""Dispatch apply for ('affine', A, b) or ('mlp', model)."""
	tag = model_obj[0]
	if tag == 'affine':
		_, A, b = model_obj
		return apply_affine_3d(X, A, b)
	elif tag == 'mlp':
		_, model = model_obj
		return apply_mlp_3d(model, X)
	elif tag == 'rigid_residual_mlp':
		_, model_dict, _, _ = model_obj
		return apply_rigid_residual_mlp_3d(model_dict, X)
	elif tag == 'mlp_rot_sincos':
		_, model = model_obj
		return apply_mlp_rot_sincos_3d(model, X)
	else:
		raise ValueError('Unknown model tuple')


def _save_models(models: Dict[str, Dict[str, object]], outdir: Path) -> None:
	outdir.mkdir(parents=True, exist_ok=True)
	# Save affine params to text and npy; mlp to joblib if available
	affine_lines: List[str] = []
	for kind in ['pos', 'rot']:
		for key, model_obj in models[kind].items():
			if model_obj[0] == 'affine':
				_, A, b = model_obj
				np.save(outdir / f'{kind}__{key}__A.npy', A)
				np.save(outdir / f'{kind}__{key}__b.npy', b)
				affine_lines.append(f'[{kind}:{key}] A=\n{A}\n b={b}\n')
			elif model_obj[0] == 'mlp':
				try:
					import joblib  # type: ignore
					joblib.dump(model_obj[1], outdir / f'{kind}__{key}__mlp.joblib')
				except Exception:
					pass
			elif model_obj[0] == 'rigid_residual_mlp':
				_, model_dict, _, _ = model_obj
				# Save R,t and the residual MLP
				np.save(outdir / f'{kind}__{key}__R.npy', model_dict['R'])
				np.save(outdir / f'{kind}__{key}__t.npy', model_dict['t'])
				try:
					import joblib  # type: ignore
					joblib.dump(model_dict['mlp'], outdir / f'{kind}__{key}__residual_mlp.joblib')
				except Exception:
					pass
			elif model_obj[0] == 'mlp_rot_sincos':
				_, model = model_obj
				try:
					import joblib  # type: ignore
					joblib.dump(model, outdir / f'{kind}__{key}__mlp_rot_sincos.joblib')
				except Exception:
					pass
	if affine_lines:
		with open(outdir / 'learned_affine_parameters.txt', 'w', encoding='utf-8') as f:
			f.write('\n'.join(affine_lines))


def _predict_for_file(
	models: Dict[str, Dict[str, object]],
	csv_path: Path,
	start_trim: int,
	end_trim: int,
) -> Tuple[
		np.ndarray, np.ndarray,  # raven_pos_left, raven_pos_right
		np.ndarray, np.ndarray,  # pred_console_pos_left (MTM→PSM), pred_trackstar_pos_left (EmHT→PSM)
		np.ndarray, np.ndarray,  # pred_console_pos_right (MTM→PSM), pred_trackstar_pos_right (EmHT→PSM)
		np.ndarray, np.ndarray,  # pred_trackstar_to_console_pos_left (EmHT→MTM), pred_trackstar_to_console_pos_right
		np.ndarray, np.ndarray,  # raven_rot_left, raven_rot_right
		np.ndarray, np.ndarray,  # pred_console_rot_left (MTM→PSM), pred_trackstar_rot_left (EmHT→PSM)
		np.ndarray, np.ndarray,  # pred_console_rot_right (MTM→PSM), pred_trackstar_rot_right (EmHT→PSM)
		np.ndarray, np.ndarray,  # pred_trackstar_to_console_rot_left (EmHT→MTM), pred_trackstar_to_console_rot_right
		np.ndarray, np.ndarray,  # console_pos_left_true, console_pos_right_true
		np.ndarray, np.ndarray,  # console_rot_left_true, console_rot_right_true
		List[str], np.ndarray, Optional[np.ndarray], Optional[np.ndarray],
]:
	(
		rpos_L, rpos_R,
		cpos_L, cpos_R,
		tpos_L, tpos_R,
		rrot_L, rrot_R,
		crot_L, crot_R,
		trot_L, trot_R,
		gestures,
		pedal,
		grasper_left_state,
		grasper_right_state,
	) = _extract_and_cumulate_from_csv(csv_path, start_trim, end_trim)
	# Apply models
	pred_cpos_L = _apply_model_3d(models['pos']['console_left'], cpos_L)
	pred_cpos_R = _apply_model_3d(models['pos']['console_right'], cpos_R)
	pred_tpos_L = _apply_model_3d(models['pos']['trackstar_left'], tpos_L)
	pred_tpos_R = _apply_model_3d(models['pos']['trackstar_right'], tpos_R)
	pred_t2c_pos_L = _apply_model_3d(models['pos']['trackstar_to_console_left'], tpos_L)
	pred_t2c_pos_R = _apply_model_3d(models['pos']['trackstar_to_console_right'], tpos_R)
	pred_crot_L = _apply_model_3d(models['rot']['console_left'], crot_L)
	pred_crot_R = _apply_model_3d(models['rot']['console_right'], crot_R)
	pred_trot_L = _apply_model_3d(models['rot']['trackstar_left'], trot_L)
	pred_trot_R = _apply_model_3d(models['rot']['trackstar_right'], trot_R)
	pred_t2c_rot_L = _apply_model_3d(models['rot']['trackstar_to_console_left'], trot_L)
	pred_t2c_rot_R = _apply_model_3d(models['rot']['trackstar_to_console_right'], trot_R)
	return (
		rpos_L, rpos_R,
		pred_cpos_L, pred_tpos_L,
		pred_cpos_R, pred_tpos_R,
		pred_t2c_pos_L, pred_t2c_pos_R,
		rrot_L, rrot_R,
		pred_crot_L, pred_trot_L,
		pred_crot_R, pred_trot_R,
		pred_t2c_rot_L, pred_t2c_rot_R,
		cpos_L, cpos_R,
		crot_L, crot_R,
		gestures, pedal, grasper_left_state, grasper_right_state,
	)


def _compute_file_metrics(
	file_base: str,
	rpos_L: np.ndarray, rpos_R: np.ndarray,
	pred_mtm_pos_L: np.ndarray, pred_mtm_pos_R: np.ndarray,
	pred_emht_pos_L: np.ndarray, pred_emht_pos_R: np.ndarray,
	pred_emht_to_mtm_pos_L: np.ndarray, pred_emht_to_mtm_pos_R: np.ndarray,
	rrot_L: np.ndarray, rrot_R: np.ndarray,
	pred_mtm_rot_L: np.ndarray, pred_mtm_rot_R: np.ndarray,
	pred_emht_rot_L: np.ndarray, pred_emht_rot_R: np.ndarray,
	pred_emht_to_mtm_rot_L: np.ndarray, pred_emht_to_mtm_rot_R: np.ndarray,
	cpos_L_true: Optional[np.ndarray] = None, cpos_R_true: Optional[np.ndarray] = None,
	crot_L_true: Optional[np.ndarray] = None, crot_R_true: Optional[np.ndarray] = None,
) -> Dict[str, float]:
	# Concatenate per-hand for position (6 dims)
	rpos6 = np.hstack([rpos_L, rpos_R])
	mtm_pos6 = np.hstack([pred_mtm_pos_L, pred_mtm_pos_R])
	emht_pos6 = np.hstack([pred_emht_pos_L, pred_emht_pos_R])
	# If console ground-truth provided, build for EmHT→MTM comparisons
	if cpos_L_true is not None and cpos_R_true is not None:
		cpos6_true = np.hstack([cpos_L_true, cpos_R_true])
		emht_to_mtm_pos6 = np.hstack([pred_emht_to_mtm_pos_L, pred_emht_to_mtm_pos_R])
	# Position metrics using existing helpers
	_, _, cos_mtm_pos_per_dim, _ = calculate_cosine_similarity_pos(rpos6, mtm_pos6)
	_, _, cos_emht_pos_per_dim, _ = calculate_cosine_similarity_pos(rpos6, emht_pos6)
	_, _, _, nrmse_mtm_pos_per_dim = calculate_nrmse_pos(rpos6, mtm_pos6, normalization_method='range')
	_, _, _, nrmse_emht_pos_per_dim = calculate_nrmse_pos(rpos6, emht_pos6, normalization_method='range')
	# EmHT→MTM metrics
	if cpos_L_true is not None and cpos_R_true is not None:
		_, _, cos_emht_to_mtm_per_dim, _ = calculate_cosine_similarity_pos(cpos6_true, emht_to_mtm_pos6)
		_, _, _, nrmse_emht_to_mtm_per_dim = calculate_nrmse_pos(cpos6_true, emht_to_mtm_pos6, normalization_method='range')
	# Average across hands for XYZ: dims [0,3], [1,4], [2,5]
	def _avg_axes(vals6: np.ndarray) -> Tuple[float, float, float]:
		return float(np.nanmean([vals6[0], vals6[3]])), float(np.nanmean([vals6[1], vals6[4]])), float(np.nanmean([vals6[2], vals6[5]]))
	mtm_pos_cos_x, mtm_pos_cos_y, mtm_pos_cos_z = _avg_axes(cos_mtm_pos_per_dim)
	emht_pos_cos_x, emht_pos_cos_y, emht_pos_cos_z = _avg_axes(cos_emht_pos_per_dim)
	mtm_pos_nrmse_x, mtm_pos_nrmse_y, mtm_pos_nrmse_z = _avg_axes(nrmse_mtm_pos_per_dim)
	emht_pos_nrmse_x, emht_pos_nrmse_y, emht_pos_nrmse_z = _avg_axes(nrmse_emht_pos_per_dim)
	if cpos_L_true is not None and cpos_R_true is not None:
		emht_to_mtm_pos_cos_x, emht_to_mtm_pos_cos_y, emht_to_mtm_pos_cos_z = _avg_axes(cos_emht_to_mtm_per_dim)
		emht_to_mtm_pos_nrmse_x, emht_to_mtm_pos_nrmse_y, emht_to_mtm_pos_nrmse_z = _avg_axes(nrmse_emht_to_mtm_per_dim)
	# Rotation metrics per hand, then average across hands
	_, _, cos_mtm_rot_L, _ = calculate_cosine_similarity_rot(rrot_L, pred_mtm_rot_L)
	_, _, cos_mtm_rot_R, _ = calculate_cosine_similarity_rot(rrot_R, pred_mtm_rot_R)
	_, _, cos_emht_rot_L, _ = calculate_cosine_similarity_rot(rrot_L, pred_emht_rot_L)
	_, _, cos_emht_rot_R, _ = calculate_cosine_similarity_rot(rrot_R, pred_emht_rot_R)
	_, _, _, nrmse_mtm_rot_L = calculate_nrmse_rot(rrot_L, pred_mtm_rot_L, normalization_method='range')
	_, _, _, nrmse_mtm_rot_R = calculate_nrmse_rot(rrot_R, pred_mtm_rot_R, normalization_method='range')
	_, _, _, nrmse_emht_rot_L = calculate_nrmse_rot(rrot_L, pred_emht_rot_L, normalization_method='range')
	_, _, _, nrmse_emht_rot_R = calculate_nrmse_rot(rrot_R, pred_emht_rot_R, normalization_method='range')
	# EmHT→MTM rotation metrics
	if crot_L_true is not None and crot_R_true is not None:
		_, _, cos_emht_to_mtm_rot_L, _ = calculate_cosine_similarity_rot(crot_L_true, pred_emht_to_mtm_rot_L)
		_, _, cos_emht_to_mtm_rot_R, _ = calculate_cosine_similarity_rot(crot_R_true, pred_emht_to_mtm_rot_R)
		_, _, _, nrmse_emht_to_mtm_rot_L = calculate_nrmse_rot(crot_L_true, pred_emht_to_mtm_rot_L, normalization_method='range')
		_, _, _, nrmse_emht_to_mtm_rot_R = calculate_nrmse_rot(crot_R_true, pred_emht_to_mtm_rot_R, normalization_method='range')
	# Average across hands for roll/pitch/yaw (order from rotation_matrix_to_euler: zyx → we label [yaw, pitch, roll])
	cos_mtm_rot_avg = np.nanmean(np.vstack([cos_mtm_rot_L, cos_mtm_rot_R]), axis=0)
	cos_emht_rot_avg = np.nanmean(np.vstack([cos_emht_rot_L, cos_emht_rot_R]), axis=0)
	nrmse_mtm_rot_avg = np.nanmean(np.vstack([nrmse_mtm_rot_L, nrmse_mtm_rot_R]), axis=0)
	nrmse_emht_rot_avg = np.nanmean(np.vstack([nrmse_emht_rot_L, nrmse_emht_rot_R]), axis=0)
	if crot_L_true is not None and crot_R_true is not None:
		cos_emht_to_mtm_rot_avg = np.nanmean(np.vstack([cos_emht_to_mtm_rot_L, cos_emht_to_mtm_rot_R]), axis=0)
		nrmse_emht_to_mtm_rot_avg = np.nanmean(np.vstack([nrmse_emht_to_mtm_rot_L, nrmse_emht_to_mtm_rot_R]), axis=0)
	# Map to roll, pitch, yaw labels from zyx: [yaw, pitch, roll]
	def _zyx_to_rpy(v: np.ndarray) -> Tuple[float, float, float]:
		return float(v[2]), float(v[1]), float(v[0])
	mtm_rot_cos_roll, mtm_rot_cos_pitch, mtm_rot_cos_yaw = _zyx_to_rpy(cos_mtm_rot_avg)
	emht_rot_cos_roll, emht_rot_cos_pitch, emht_rot_cos_yaw = _zyx_to_rpy(cos_emht_rot_avg)
	mtm_rot_nrmse_roll, mtm_rot_nrmse_pitch, mtm_rot_nrmse_yaw = _zyx_to_rpy(nrmse_mtm_rot_avg)
	emht_rot_nrmse_roll, emht_rot_nrmse_pitch, emht_rot_nrmse_yaw = _zyx_to_rpy(nrmse_emht_rot_avg)
	out = {
		'file': file_base,
		# Position cosine (avg across hands)
		'mtm_pos_cos_x': mtm_pos_cos_x, 'mtm_pos_cos_y': mtm_pos_cos_y, 'mtm_pos_cos_z': mtm_pos_cos_z,  # MTM→PSM vs PSM
		'emht_to_psm_pos_cos_x': emht_pos_cos_x, 'emht_to_psm_pos_cos_y': emht_pos_cos_y, 'emht_to_psm_pos_cos_z': emht_pos_cos_z,
		# Position NRMSE% (avg across hands)
		'mtm_pos_nrmse_x': mtm_pos_nrmse_x, 'mtm_pos_nrmse_y': mtm_pos_nrmse_y, 'mtm_pos_nrmse_z': mtm_pos_nrmse_z,
		'emht_to_psm_pos_nrmse_x': emht_pos_nrmse_x, 'emht_to_psm_pos_nrmse_y': emht_pos_nrmse_y, 'emht_to_psm_pos_nrmse_z': emht_pos_nrmse_z,
		# Rotation cosine (avg across hands; R,P,Y)
		'mtm_rot_cos_roll': mtm_rot_cos_roll, 'mtm_rot_cos_pitch': mtm_rot_cos_pitch, 'mtm_rot_cos_yaw': mtm_rot_cos_yaw,
		'emht_to_psm_rot_cos_roll': emht_rot_cos_roll, 'emht_to_psm_rot_cos_pitch': emht_rot_cos_pitch, 'emht_to_psm_rot_cos_yaw': emht_rot_cos_yaw,
		# Rotation NRMSE% (avg across hands; R,P,Y)
		'mtm_rot_nrmse_roll': mtm_rot_nrmse_roll, 'mtm_rot_nrmse_pitch': mtm_rot_nrmse_pitch, 'mtm_rot_nrmse_yaw': mtm_rot_nrmse_yaw,
		'emht_to_psm_rot_nrmse_roll': emht_rot_nrmse_roll, 'emht_to_psm_rot_nrmse_pitch': emht_rot_nrmse_pitch, 'emht_to_psm_rot_nrmse_yaw': emht_rot_nrmse_yaw,
	}
	# Add EmHT→MTM metrics if available
	if cpos_L_true is not None and cpos_R_true is not None:
		out.update({
			'emht_to_mtm_pos_cos_x': emht_to_mtm_pos_cos_x, 'emht_to_mtm_pos_cos_y': emht_to_mtm_pos_cos_y, 'emht_to_mtm_pos_cos_z': emht_to_mtm_pos_cos_z,
			'emht_to_mtm_pos_nrmse_x': emht_to_mtm_pos_nrmse_x, 'emht_to_mtm_pos_nrmse_y': emht_to_mtm_pos_nrmse_y, 'emht_to_mtm_pos_nrmse_z': emht_to_mtm_pos_nrmse_z,
		})
	if crot_L_true is not None and crot_R_true is not None:
		emht_to_mtm_rot_cos_roll, emht_to_mtm_rot_cos_pitch, emht_to_mtm_rot_cos_yaw = _zyx_to_rpy(cos_emht_to_mtm_rot_avg)
		emht_to_mtm_rot_nrmse_roll, emht_to_mtm_rot_nrmse_pitch, emht_to_mtm_rot_nrmse_yaw = _zyx_to_rpy(nrmse_emht_to_mtm_rot_avg)
		out.update({
			'emht_to_mtm_rot_cos_roll': emht_to_mtm_rot_cos_roll,
			'emht_to_mtm_rot_cos_pitch': emht_to_mtm_rot_cos_pitch,
			'emht_to_mtm_rot_cos_yaw': emht_to_mtm_rot_cos_yaw,
			'emht_to_mtm_rot_nrmse_roll': emht_to_mtm_rot_nrmse_roll,
			'emht_to_mtm_rot_nrmse_pitch': emht_to_mtm_rot_nrmse_pitch,
			'emht_to_mtm_rot_nrmse_yaw': emht_to_mtm_rot_nrmse_yaw,
		})
	return out


def _mean_row(df: pd.DataFrame) -> pd.Series:
	row: Dict[str, float] = {'file': '__MEAN_ALL_FILES__'}
	for c in df.columns:
		if c == 'file':
			continue
		if pd.api.types.is_numeric_dtype(df[c]):
			row[c] = float(df[c].dropna().mean()) if len(df[c].dropna()) > 0 else float('nan')
	return pd.Series(row)

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


def load_and_prepare(csv_path: Path, start_trim: int, end_trim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
	data = pd.read_csv(csv_path)
	# Extract arrays
	raven_data = get_raven_data(data)
	console_data = get_console_data(data)
	trakstar_data = get_trakstar_data(data)
	smartwatch_data = get_smartwatch_data(data)
	gesture_series = data['gesture_name'].astype(str).fillna('Unknown') if 'gesture_name' in data.columns else pd.Series(['Unknown'] * len(data))

	# Apply trims (like notebook: [2000:-1200])
	eff_start = _detect_dynamic_start_index(data, start_trim)
	end_index = None if end_trim == 0 else -end_trim
	raven_data = raven_data[eff_start:end_index]
	console_data = console_data[eff_start:end_index]
	trakstar_data = trakstar_data[eff_start:end_index]
	smartwatch_data = smartwatch_data[eff_start:end_index]
	gesture_series = gesture_series.iloc[eff_start:end_index].reset_index(drop=True)

	# Optional grasper states from console_aux0 (left) and console_aux1 (right)
	if 'console_aux0' in data.columns:
		aux0_full = data['console_aux0'].to_numpy()
		grasper_left_state = aux0_full[eff_start:end_index]
	else:
		grasper_left_state = None
	if 'console_aux1' in data.columns:
		aux1_full = data['console_aux1'].to_numpy()
		grasper_right_state = aux1_full[eff_start:end_index]
	else:
		grasper_right_state = None

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
	        grasper_left_state,
	        grasper_right_state,
	)


def main():
	parser = argparse.ArgumentParser(description='Learn global EmHT/MTM→PSM transforms (pos+rot) and visualize per-file')
	parser.add_argument('--pattern', type=str, default='final_annotation_t*.csv', help='Glob pattern for input CSV files')
	parser.add_argument('--model', type=str, choices=['affine', 'mlp'], default='affine', help='Transformation model type')
	parser.add_argument('--outdir', type=str, default='analysis_outputs', help='Directory to save figures, metrics, and models')
	# Cross-validation now defaults to leave-one-file-out; --train-split is ignored if multiple files
	parser.add_argument('--seed', type=int, default=0, help='Random seed for file split')
	parser.add_argument('--start-trim', type=int, default=2000, help='Trim N samples from start')
	parser.add_argument('--end-trim', type=int, default=1200, help='Trim N samples from end')
	parser.add_argument('--fps', type=float, default=DEFAULT_FPS, help='Sampling rate (Hz)')
	parser.add_argument('--normalize-frames', action='store_true', help='Normalize positions per frame (MTM, PSM, EmHT) by range before learning/applying')
	args = parser.parse_args()

	# Collect files
	import glob as _glob
	csv_paths = sorted([Path(p) for p in _glob.glob(args.pattern)])
	if not csv_paths:
		# Fallback to any default trial if pattern didn't match
		try:
			csv_paths = [find_default_csv(None)]
		except Exception:
			raise FileNotFoundError(f'No files matched pattern {args.pattern}')
	print(f'Found {len(csv_paths)} CSV file(s).')
	# Leave-One-File-Out Cross-Validation (LOFO)
	K = len(csv_paths)
	all_metrics: List[Dict[str, float]] = []
	fig_root = Path(args.outdir) / 'figures' / 'cv'
	fig_root.mkdir(parents=True, exist_ok=True)
	models_root = Path(args.outdir) / 'models' / 'cv'
	models_root.mkdir(parents=True, exist_ok=True)
	if K == 1:
		print('Only one file found; training on it and evaluating on the same (no CV).')
	for i in range(K):
		test_paths = [csv_paths[i]]
		train_paths = [p for j, p in enumerate(csv_paths) if j != i] if K > 1 else [csv_paths[0]]
		print(f'Fold {i+1}/{K}: train={len(train_paths)} file(s), test=1 file.')
		# Compute normalization stats on training set if requested
		pos_stats = _compute_position_frame_stats(train_paths, start_trim=args.start_trim, end_trim=args.end_trim) if args.normalize_frames else None
		# Fit models on training set
		train_pairs = _aggregate_training_data(train_paths, start_trim=args.start_trim, end_trim=args.end_trim)
		models = _fit_models(
			train_pairs,
			model_type=args.model,
			normalize_frames=bool(args.normalize_frames),
			pos_frame_stats=pos_stats,
			use_temporal_features=True,
		)
		# Save models for this fold
		_save_models(models, models_root / f'fold_{i+1}')
		# Evaluate on held-out file
		for csv_path in test_paths:
			base = csv_path.stem
			print(f'  Evaluating on {base} …')
			(
				rpos_L, rpos_R,
				pred_mtm_pos_L, pred_emht_pos_L,
				pred_mtm_pos_R, pred_emht_pos_R,
				pred_emht_to_mtm_pos_L, pred_emht_to_mtm_pos_R,
				rrot_L, rrot_R,
				pred_mtm_rot_L, pred_emht_rot_L,
				pred_mtm_rot_R, pred_emht_rot_R,
				pred_emht_to_mtm_rot_L, pred_emht_to_mtm_rot_R,
				cpos_L_true, cpos_R_true,
				crot_L_true, crot_R_true,
				gestures, pedal, grasper_left_state, grasper_right_state,
			) = _predict_for_file(models, csv_path, start_trim=args.start_trim, end_trim=args.end_trim)
			out_path = str(fig_root / f'fold_{i+1}__{base}__pos_rot_left.png')
			plot_pos_rot_side_by_side(
				rpos_L,
				pred_mtm_pos_L,
				pred_emht_pos_L,
				rrot_L,
				pred_mtm_rot_L,
				pred_emht_rot_L,
				left_sw_pos=None,
				smartwatch_rot=None,
				grasper_left_state=grasper_left_state,
				grasper_right_state=grasper_right_state,
				fps=args.fps,
				out_path=out_path,
				gesture_names=gestures,
				pedal=pedal,
			)
			metrics_row = _compute_file_metrics(
				base,
				rpos_L, rpos_R,
				pred_mtm_pos_L, pred_mtm_pos_R,
				pred_emht_pos_L, pred_emht_pos_R,
				pred_emht_to_mtm_pos_L, pred_emht_to_mtm_pos_R,
				rrot_L, rrot_R,
				pred_mtm_rot_L, pred_mtm_rot_R,
				pred_emht_rot_L, pred_emht_rot_R,
				pred_emht_to_mtm_rot_L, pred_emht_to_mtm_rot_R,
				cpos_L_true=cpos_L_true, cpos_R_true=cpos_R_true,
				crot_L_true=crot_L_true, crot_R_true=crot_R_true,
			)
			metrics_row['fold'] = i + 1
			all_metrics.append(metrics_row)
	# Save CV summary
	if all_metrics:
		df = pd.DataFrame(all_metrics)
		df = pd.concat([df, pd.DataFrame([_mean_row(df)])], ignore_index=True)
		metrics_path = Path(args.outdir) / 'summary_metrics_learned_cv.csv'
		df.to_csv(metrics_path, index=False)
		print(f'Saved CV metrics to {metrics_path}')


if __name__ == '__main__':
	main()


