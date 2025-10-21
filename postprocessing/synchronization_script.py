# %%
import numpy as np
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from typing import List, Optional, Tuple
# import display from IPython.display





trial_id = "t1"


base_path = "/standard/UVA-DSA/MIDAS/Organized/final_data"

trial_path = f"{base_path}/{trial_id}"

trakstar_path = f"{trial_path}/trakstar/trakstar.csv"
pds_path = f"{trial_path}/PDS/PDS.csv"
smartwatch_left_path = f"{trial_path}/smartwatch/sw_left/sw_data.csv"
smartwatch_right_path = f"{trial_path}/smartwatch/sw_right/sw_data.csv"
raven_path = f"{trial_path}/raven_kinematics/raven_{trial_id}.csv"
console_path = f"{trial_path}/console_kinematics/console_{trial_id}.txt"
obs_json_path = f"{trial_path}/video/" # name changes with each recording there is a jsonl file, so we will use glob to



plots_dir = os.path.join(trial_path, "plots")
os.makedirs(plots_dir, exist_ok=True)

# Helper to save figures with a consistent title and filename pattern
def _save_fig(fig, filename: str, suptitle: str):
    try:
        fig.suptitle(suptitle)
    except Exception:
        pass
    # leave room for suptitle
    try:
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    except Exception:
        pass
    out_path = os.path.join(plots_dir, filename)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")



# --- find the OBS jsonl file (pick latest if more than one) ---
obs_json_files = glob.glob(os.path.join(obs_json_path, "*.jsonl"))
if not obs_json_files:
    raise FileNotFoundError(f"No OBS JSON file found in {trial_path} video")

# if you prefer to error on multiple, keep your original logic; otherwise pick latest:
obs_json_path = max(obs_json_files, key=os.path.getmtime)
print(f"Found OBS JSON file: {obs_json_path}")
# --- load JSONL correctly ---
# For huge files, add chunksize=... and concat.
df_obs = pd.read_json(obs_json_path, lines=True)
# Keep only frame rows (optional)
df_obs_frames = df_obs[df_obs["event"] == "frame"].copy()
# Align to your nanoseconds server_time (epoch_ms -> ns)
df_obs_frames["server_time"] = (df_obs_frames["epoch_ms"].astype("int64") * 1_000_000)
# If useful, also keep a datetime for quick plotting:
df_obs_frames["server_time_dt"] = pd.to_datetime(df_obs_frames["server_time"], unit="ns")


# load data
df_trakstar = pd.read_csv(trakstar_path)
df_pds = pd.read_csv(pds_path)
df_sw_left = pd.read_csv(smartwatch_left_path)
df_sw_right = pd.read_csv(smartwatch_right_path)
df_raven = pd.read_csv(raven_path)
df_console = pd.read_csv(console_path, delimiter="\t")


# print some info
print(f"Trakstar data shape: {df_trakstar.shape}")
print(f"PDS data shape: {df_pds.shape}")
print(f"Smartwatch left data shape: {df_sw_left.shape}")
print(f"Smartwatch right data shape: {df_sw_right.shape}")
print(f"Raven data shape: {df_raven.shape}")
print(f"Console data shape: {df_console.shape}")
print("OBS frames only shape :", df_obs_frames.shape)


# %%
# print top 5 of each
print("\nTrakstar data (top 5 rows):")
print(df_trakstar.head())
print("\nPDS data (top 5 rows):")
print(df_pds.head())
print("\nSmartwatch left data (top 5 rows):")
print(df_sw_left.head())
print("\nSmartwatch right data (top 5 rows):")
print(df_sw_right.head())
print("\nRaven data (top 5 rows):")
print(df_raven.head())
print("\nConsole data (top 5 rows):")
print(df_console.head())
print("\nOBS frames only (top 5 rows):")
print(df_obs_frames.head())



# -----------------------------
# I/O
# -----------------------------
def read_data(console_path: str, raven_path: str):
    """Read console (txt) and raven (csv) data."""
    console_data = np.loadtxt(console_path)
    raven_data = pd.read_csv(raven_path)
    return console_data, raven_data


def _cumulate_gated(arr: np.ndarray, pedal: np.ndarray) -> np.ndarray:
    """
    Pedal-gated cumulative integration of an [N, K] array.
    ped==1 -> integrate; ped==0 (or <=0) -> freeze.
    """
    out = np.zeros_like(arr, dtype=float)
    if arr.shape[0] <= 1:
        return out
    diff = np.diff(arr, axis=0)
    ped = np.asarray(pedal).astype(float)
    ped = np.where(np.isnan(ped), 0.0, ped)
    ped = np.where(ped < 0, 0.0, ped)  # treat -1 (no-match) as 0
    zero_mask = ped[:-1] <= 0.0
    diff[zero_mask, :] = 0.0
    out[1:, :] = np.cumsum(diff, axis=0)
    return out

# =========================
# Helpers: math & transforms
# =========================
def cumulate_console_data_from_combined(df: pd.DataFrame) -> np.ndarray:
    """
    Build cumulative console position deltas (gated by pedal) from combined CSV.
    Returns position in order [Xl, Yl, Zl, Xr, Yr, Zr] in CONSOLE frame.
    """
    # Extract pos0..pos5 and pedal from combined console columns
    pos_cols = [f'console_pos{i}' for i in range(6)]
    pedal = df['console_pedal'].to_numpy()
    pos = df[pos_cols].to_numpy()

    console_pos = np.zeros((len(df), 6))
    diff = np.diff(pos, axis=0)
    zero_mask = pedal[:-1] == 0
    diff[zero_mask, :] = 0
    console_pos[1:, :] = np.cumsum(diff, axis=0)

    # Reorder to [0,2,4,1,3,5]
    return console_pos[:, [0, 2, 4, 1, 3, 5]]

def cumulate_console_data(console_data, pedal):
    console_pos = np.zeros((console_data.shape[0], 6))
    
    diff_console = np.diff(console_data, axis=0)

    zero_mask = pedal[:-1] == 0
    diff_console[zero_mask, :] = 0
    console_pos[1:, :] = np.cumsum(diff_console, axis=0)

    return console_pos


def cumulate_trackstar_data( trackstar_data, pedal):
    trackstar_pos = np.zeros((trackstar_data.shape[0], 6))
    
    diff_trackstar = np.diff(trackstar_data, axis=0)

    zero_mask = pedal[:-1] == 0

    diff_trackstar[zero_mask, :] = 0
    trackstar_pos[1:, :] = np.cumsum(diff_trackstar, axis=0)

    return trackstar_pos

def cumulate_console_rot_from_combined(df: pd.DataFrame) -> np.ndarray:
    """
    Build cumulative console rotation deltas (gated by pedal) from combined CSV.
    Returns [Rl, Pl, Yl, Rr, Pr, Yr] in CONSOLE frame.
    """
    rot_cols = [f'console_rot{i}' for i in range(6)]
    pedal = df['console_pedal'].to_numpy()
    rot = df[rot_cols].to_numpy()

    console_rot = np.zeros((len(df), 6))
    diff = np.diff(rot, axis=0)
    zero_mask = pedal[:-1] == 0
    diff[zero_mask, :] = 0
    console_rot[1:, :] = np.cumsum(diff, axis=0)

    # Reorder to [0,2,4,1,3,5]
    return console_rot[:, [0, 2, 4, 1, 3, 5]]


def transform_console_to_raven_left(console_pos: np.ndarray) -> np.ndarray:
    pos_matrix = np.array([[0, 0, 1],
                           [1, 0, 0],
                           [0, 1, 0]])
    psm1_pos = pos_matrix @ console_pos[:, 0:3].T
    console_pos[:, 0:3] = psm1_pos.T
    return console_pos


def transform_console_to_raven_right(console_pos: np.ndarray) -> np.ndarray:
    pos_matrix = np.array([[0, 0, 1],
                           [-1, 0, 0],
                           [0, -1, 0]])
    psm2_pos = pos_matrix @ console_pos[:, 3:6].T
    console_pos[:, 3:6] = psm2_pos.T
    return console_pos


def transform_console_to_raven_rot_1(console_rot_3: np.ndarray) -> np.ndarray:
    """
    Apply rotation transform for console rotations to Raven frame (for one side: Nx3).
    """
    rot_matrix = np.array([[-1,  0, 0],
                           [ 0, -1, 0],
                           [ 0,  0, 1]])
    return (rot_matrix @ console_rot_3.T).T


def rotation_matrix_to_euler(rot_matrix_flat: np.ndarray) -> np.ndarray:
    """Convert Nx9 flattened rotation matrices to Euler (xyz)."""
    n_samples = rot_matrix_flat.shape[0]
    euler_angles = np.zeros((n_samples, 3))
    for i in range(n_samples):
        Rmat = rot_matrix_flat[i].reshape(3, 3)
        r = R.from_matrix(Rmat)
        euler_angles[i] = r.as_euler('xyz')
    return euler_angles


def process_raven_rot(raven_psm_rot: np.ndarray) -> np.ndarray:
    """Wrap raven euler angles to reasonable range."""
    processed = raven_psm_rot.copy()
    processed[processed > 4] -= 6.28
    processed[processed < -4] += 6.28
    return processed


# =========================
# Build arrays FROM combined.csv
# =========================
def build_positions_from_combined(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      console_pos_raven_frame: [Xl,Yl,Zl,Xr,Yr,Zr]
      raven_pos:               [Xl,Yl,Zl,Xr,Yr,Zr] (zeroed & scaled)
    """
    # Console cumulative in its own frame → map to Raven for both arms
    console_pos = cumulate_console_data_from_combined(df)
    console_pos = transform_console_to_raven_left(console_pos)
    console_pos = transform_console_to_raven_right(console_pos)

    # Raven positions (zero-relative & scale like original)
    rpos_cols = [f'raven_field.pos{i}' for i in range(6)]
    rpos = df[rpos_cols].to_numpy().astype(float)
    rpos = (rpos - rpos[0, :]) * 1e-2

    return console_pos, rpos


def build_rotations_from_combined(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      console_psm1_rot_raven: Nx3
      console_psm2_rot_raven: Nx3
      raven_psm1_rot:         Nx3
      raven_psm2_rot:         Nx3
    """
    # Console cumulative rotations (6 -> split per arm, then map to Raven)
    console_rot = cumulate_console_rot_from_combined(df)
    console_psm1 = console_rot[:, [0, 1, 2]]
    console_psm2 = console_rot[:, [3, 4, 5]]
    console_psm1_raven = transform_console_to_raven_rot_1(console_psm1)
    console_psm2_raven = transform_console_to_raven_rot_1(console_psm2)

    # Raven rotations: two 3x3s flattened (ori0..ori8) and (ori9..ori17)
    ori_cols1 = [f'raven_field.ori{i}' for i in range(0, 9)]
    ori_cols2 = [f'raven_field.ori{i}' for i in range(9, 18)]
    r1 = df[ori_cols1].to_numpy().astype(float)
    r2 = df[ori_cols2].to_numpy().astype(float)

    raven_psm1_rot = rotation_matrix_to_euler(r1)
    raven_psm2_rot = rotation_matrix_to_euler(r2)

    # zero-relative & wrap
    raven_psm1_rot = process_raven_rot(raven_psm1_rot - raven_psm1_rot[0, :])
    raven_psm2_rot = process_raven_rot(raven_psm2_rot - raven_psm2_rot[0, :])

    return console_psm1_raven, console_psm2_raven, raven_psm1_rot, raven_psm2_rot


# =========================
# Plotting (from combined.csv only)
# =========================
def plot_xyz_left_from_combined(df: pd.DataFrame):
    console_pos, raven_pos = build_positions_from_combined(df)
    labels = ['X', 'Y', 'Z']
    fig = plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(raven_pos[:, i], label='Raven')
        plt.plot(console_pos[:, i], label='Console')
        plt.title(f'{labels[i]} Coordinate (Left Arm)')
        plt.xlabel('Index'); plt.ylabel(labels[i]); plt.legend()
    _save_fig(fig, f"xyz_left_{trial_id}.png", f"Trial {trial_id} - Left Arm XYZ")


def plot_xyz_right_from_combined(df: pd.DataFrame):
    console_pos, raven_pos = build_positions_from_combined(df)
    labels = ['X', 'Y', 'Z']
    fig = plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(raven_pos[:, i+3], label='Raven')
        plt.plot(console_pos[:, i+3], label='Console')
        plt.title(f'{labels[i]} Coordinate (Right Arm)')
        plt.xlabel('Index'); plt.ylabel(labels[i]); plt.legend()
    _save_fig(fig, f"xyz_right_{trial_id}.png", f"Trial {trial_id} - Right Arm XYZ")


def plot_rotations_from_combined(df: pd.DataFrame):
    c1, c2, r1, r2 = build_rotations_from_combined(df)
    labels = ['Roll', 'Pitch', 'Yaw']

    # PSM1
    fig1 = plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(c1[:, i], label='Console', alpha=0.8)
        plt.plot(r1[:, i], label='Raven', alpha=0.8)
        plt.title(f'PSM1 Left - {labels[i]}')
        plt.xlabel('Index'); plt.ylabel(f'{labels[i]} (rad)'); plt.legend()
    _save_fig(fig1, f"rotations_psm1_{trial_id}.png", f"Trial {trial_id} - PSM1 Rotations")

    # PSM2
    fig2 = plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(c2[:, i], label='Console', alpha=0.8)
        plt.plot(r2[:, i], label='Raven', alpha=0.8)
        plt.title(f'PSM2 Right - {labels[i]}')
        plt.xlabel('Index'); plt.ylabel(f'{labels[i]} (rad)'); plt.legend()
    _save_fig(fig2, f"rotations_psm2_{trial_id}.png", f"Trial {trial_id} - PSM2 Rotations")



# -----------------------------
# Raven / Console column selection
# -----------------------------
def raven_field_names() -> List[str]:
    """Canonical field names in the same order we select them from Raven CSV."""
    return [
        'field.hdr.stamp', 'field.last_seq',
        'field.pos0', 'field.pos1', 'field.pos2', 'field.pos3', 'field.pos4', 'field.pos5',
        'field.ori0', 'field.ori1', 'field.ori2', 'field.ori3', 'field.ori4', 'field.ori5',
        'field.ori6', 'field.ori7', 'field.ori8',
        'field.ori9', 'field.ori10', 'field.ori11', 'field.ori12', 'field.ori13', 'field.ori14',
        'field.ori15', 'field.ori16', 'field.ori17', "field.grasp_d0", "field.grasp_d1"
    ]


def select_columns_from_raven(raven_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """
    Select and format Raven columns.
    - De-dup by 'field.last_seq' (keep first).
    - Convert 'field.hdr.stamp' from ns -> seconds.
    Returns (array, headers).
    """
    fields = raven_field_names()
    rdf = pd.DataFrame(raven_df, columns=fields)
    raven = rdf.drop_duplicates(subset='field.last_seq', keep='first').to_numpy().astype(float)

    print(f"Raven data shape after dedup: {raven.shape}")
    # Optional trim (kept from original; adjust/remove as needed)
    raven = raven[25000:, :]

    print(f"Raven data shape after trim: {raven.shape}")
    
    # Convert ns -> seconds
    raven[:, 0] = raven[:, 0] * 1e-9
    return raven, fields


def console_selected_indices() -> List[int]:
    """
    Indices taken from the raw console matrix (np.loadtxt) we keep.
    Order matters and must match downstream logic.
    """
    return [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21]


def make_console_headers(
    provided_full_headers: Optional[List[str]],
    selected_indices: List[int]
) -> List[str]:
    """
    Build headers for the console subset:
      col0: sequence number
      cols1..6: positions (6 values)
      cols7..12: rotations (6 values)
      cols13..15: auxiliary (we use the last as pedal)
    If 'provided_full_headers' is given and long enough, we map exact names.
    Otherwise we fall back to descriptive defaults.
    """
    if provided_full_headers and max(selected_indices) < len(provided_full_headers):
        return [provided_full_headers[i] for i in selected_indices]

    # Fallback: readable defaults aligned with how the code consumes the data
    return [
        'seq',
        'pos0', 'pos1', 'pos2', 'pos3', 'pos4', 'pos5',
        'rot0', 'rot1', 'rot2', 'rot3', 'rot4', 'rot5',
        'aux0', 'aux1', 'pedal'
    ]


def select_columns_from_console(console_data: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """
    Select console columns and provide headers for them.
    If you know the original console header strings, pass them to
    'select_columns_from_console_with_headers' instead.
    """
    idx = console_selected_indices()
    console = console_data[:, idx]
    headers = make_console_headers(None, idx)  # no provided headers by default
    return console, headers


def select_columns_from_console_with_headers(
    console_data: np.ndarray,
    original_headers: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """
    Same as select_columns_from_console but maps to real header names provided by caller.
    """
    idx = console_selected_indices()
    console = console_data[:, idx]
    headers = make_console_headers(original_headers, idx)
    return console, headers


# -----------------------------
# Synchronization (Raven timeline)
# -----------------------------
def synchronize_data(
    console_data: np.ndarray,
    raven_df: pd.DataFrame,
    console_headers_full: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """
    Synchronize console to Raven samples using Raven's 'field.last_seq' vs console seq.
    Returns:
      new_console (aligned),
      console_headers (for aligned array),
      new_raven    (aligned = selected/truncated),
      raven_headers
    """
    raven, raven_headers = select_columns_from_raven(raven_df)

    if console_headers_full is None:
        console, console_headers = select_columns_from_console(console_data)
    else:
        console, console_headers = select_columns_from_console_with_headers(console_data, console_headers_full)

    new_console = np.zeros((raven.shape[0], console.shape[1]))
    for i in range(raven.shape[0]):
        raven_seq = raven[i, 1]  # 'field.last_seq'
        console_index = np.argmin(np.abs(console[:, 0] - raven_seq))
        new_console[i, :] = console[console_index, :]

    return new_console, console_headers, raven, raven_headers


# -----------------------------
# Console cumulative position / rotation (with pedal gating)
# -----------------------------
def cumulate_console_data(console_data: np.ndarray) -> np.ndarray:
    """
    Build cumulative console position deltas gated by pedal (last col).
    Returns position in order [Xl, Yl, Zl, Xr, Yr, Zr] after index remap.
    """
    console_pos = np.zeros((console_data.shape[0], 6))
    # original console position columns are (1..6) before reindexing
    diff = np.diff(console_data[:, 1:7], axis=0)
    zero_mask = console_data[:-1, -1] == 0  # pedal off => zero the diff
    diff[zero_mask, :] = 0
    console_pos[1:, :] = np.cumsum(diff, axis=0)
    # Reorder to [0,2,4,1,3,5]
    return console_pos[:, [0, 2, 4, 1, 3, 5]]


def cumulate_console_data_rot(console_data: np.ndarray) -> np.ndarray:
    """
    Build cumulative console rotation deltas gated by pedal (last col).
    Returns Euler-like cumulative deltas in order [Rl, Pl, Yl, Rr, Pr, Yr] after index remap.
    """
    console_rot = np.zeros((console_data.shape[0], 6))
    diff = np.diff(console_data[:, 7:13], axis=0)
    zero_mask = console_data[:-1, -1] == 0
    diff[zero_mask, :] = 0
    console_rot[1:, :] = np.cumsum(diff, axis=0)
    return console_rot[:, [0, 2, 4, 1, 3, 5]]


# -----------------------------
# Frame transforms (Console -> Raven)
# -----------------------------
def transform_console_to_raven_left(console_pos: np.ndarray) -> np.ndarray:
    pos_matrix = np.array([[0, 0, 1],
                           [1, 0, 0],
                           [0, 1, 0]])
    psm1_pos = pos_matrix @ console_pos[:, 0:3].T
    console_pos[:, 0:3] = psm1_pos.T
    return console_pos


def transform_console_to_raven_right(console_pos: np.ndarray) -> np.ndarray:
    pos_matrix = np.array([[0, 0, 1],
                           [-1, 0, 0],
                           [0, -1, 0]])
    psm2_pos = pos_matrix @ console_pos[:, 3:6].T
    console_pos[:, 3:6] = psm2_pos.T
    return console_pos


def transform_trakstar_to_raven_left(trakstar_pos: np.ndarray) -> np.ndarray:
    pos_matrix = np.array([
                [ 0,  0, -1],
                [ 1,  0,  0],
                [ 0, -1,  0],
                           ])
    
    psm_pos = pos_matrix @ (trakstar_pos[:, 0:3]*1).T
    trakstar_pos[:, 0:3] = psm_pos.T
    return trakstar_pos 



def transform_trakstar_to_raven_right(trakstar_pos: np.ndarray) -> np.ndarray:
    pos_matrix = np.array([[0,  0, -1], 
                           [-1,  0,  0], 
                           [0,  1, 0]])
    psm_pos = pos_matrix @ (trakstar_pos[:, 3:6]*1).T
    trakstar_pos[:, 3:6] = psm_pos.T
    return trakstar_pos

def align_console_trackstar_data_pos(new_console, new_trackstar):
    console_pos = new_console[:, [0, 2, 4, 1, 3, 5]]
    trackstar_pos = new_trackstar[:, [0, 1, 2, 3, 4, 5]]
    trackstar_pos = transform_trakstar_to_raven_left(trackstar_pos)
    trackstar_pos = transform_trakstar_to_raven_right(trackstar_pos)
    console_pos = console_pos - console_pos[0, :]
    trackstar_pos = trackstar_pos - trackstar_pos[0, :]




def transform_console_to_raven_rot_1(console_rot_left_or_right: np.ndarray) -> np.ndarray:
    """Apply a rotation transform for console rotations to Raven frame (per side)."""
    rot_matrix = np.array([[-1,  0, 0],
                           [ 0, -1, 0],
                           [ 0,  0, 1]])
    psm_rot = rot_matrix @ console_rot_left_or_right[:, 0:3].T
    console_rot_left_or_right[:, 0:3] = psm_rot.T
    return console_rot_left_or_right


# -----------------------------
# Raven rotations utilities
# -----------------------------
def rotation_matrix_to_euler(rot_matrix_flat: np.ndarray) -> np.ndarray:
    """Convert Nx9 flattened rotation matrices to Euler (xyz)."""
    n_samples = rot_matrix_flat.shape[0]
    euler_angles = np.zeros((n_samples, 3))
    for i in range(n_samples):
        Rmat = rot_matrix_flat[i].reshape(3, 3)
        r = R.from_matrix(Rmat)
        euler_angles[i] = r.as_euler('xyz')
    return euler_angles


def process_raven_rot(raven_psm_rot: np.ndarray) -> np.ndarray:
    """Wrap raven euler angles to reasonable range (kept from original)."""
    processed_rot = raven_psm_rot.copy()
    processed_rot[processed_rot > 4] -= 6.28
    processed_rot[processed_rot < -4] += 6.28
    return processed_rot


# -----------------------------
# Alignment (positions and rotations)
# -----------------------------
def build_console_positions_in_raven(new_console: np.ndarray) -> np.ndarray:
    """Cumulative console positions gated by pedal, then map to Raven frame for both arms."""
    console_pos = cumulate_console_data(new_console)  # [Xl,Yl,Zl,Xr,Yr,Zr] in console frame
    console_pos = transform_console_to_raven_left(console_pos)
    console_pos = transform_console_to_raven_right(console_pos)
    return console_pos

def build_trakstar_positions_in_raven(trakstar_df: pd.DataFrame):

# ['obs_frame_idx', 'server_time', 'trakstar_sensor_0_azimuth', 'trakstar_sensor_1_azimuth', 'trakstar_sensor_2_azimuth', 'trakstar_sensor_3_azimuth', 'trakstar_sensor_0_elevation', 'trakstar_sensor_1_elevation', 'trakstar_sensor_2_elevation', 'trakstar_sensor_3_elevation', 'trakstar_sensor_0_roll', 'trakstar_sensor_1_roll', 'trakstar_sensor_2_roll', 'trakstar_sensor_3_roll', 'trakstar_sensor_0_x', 'trakstar_sensor_1_x', 'trakstar_sensor_2_x', 'trakstar_sensor_3_x', 'trakstar_sensor_0_y', 'trakstar_sensor_1_y', 'trakstar_sensor_2_y', 'trakstar_sensor_3_y', 'trakstar_sensor_0_z', 'trakstar_sensor_1_z', 'trakstar_sensor_2_z', 'trakstar_sensor_3_z', 'trakstar_time_ns', 'trakstar_delta_ns', 'trakstar_grasp_d0', 'trakstar_grasp_d1']
    
    print("Building TrakStar positions...")

    # trakstar sensor for left arm: sensor 1
    trakstar_sensor_1_x = trakstar_df['trakstar_sensor_1_x'].to_numpy()
    trakstar_sensor_1_y = trakstar_df['trakstar_sensor_1_y'].to_numpy()
    trakstar_sensor_1_z = trakstar_df['trakstar_sensor_1_z'].to_numpy()
    # trakstar sensor for left arm: sensor 0 - backup
    trakstar_sensor_0_x = trakstar_df['trakstar_sensor_0_x'].to_numpy()
    trakstar_sensor_0_y = trakstar_df['trakstar_sensor_0_y'].to_numpy()
    trakstar_sensor_0_z = trakstar_df['trakstar_sensor_0_z'].to_numpy()

    # trakstar sensor for right arm: sensor 2
    trakstar_sensor_2_x = trakstar_df['trakstar_sensor_2_x'].to_numpy()
    trakstar_sensor_2_y = trakstar_df['trakstar_sensor_2_y'].to_numpy()
    trakstar_sensor_2_z = trakstar_df['trakstar_sensor_2_z'].to_numpy()
    # trakstar sensor for right arm: sensor 3 - backup
    trakstar_sensor_3_x = trakstar_df['trakstar_sensor_3_x'].to_numpy()
    trakstar_sensor_3_y = trakstar_df['trakstar_sensor_3_y'].to_numpy()
    trakstar_sensor_3_z = trakstar_df['trakstar_sensor_3_z'].to_numpy()

    trakstar_pos_s1_s3 = np.stack([trakstar_sensor_1_x, trakstar_sensor_1_y, trakstar_sensor_1_z, trakstar_sensor_3_x, trakstar_sensor_3_y, trakstar_sensor_3_z], axis=1)
    
    trakstar_pos_s1_s3 = transform_trakstar_to_raven_left(trakstar_pos_s1_s3)
    trakstar_pos_s1_s3 = transform_trakstar_to_raven_right(trakstar_pos_s1_s3)


    trakstar_pos_s0_s2 = np.stack([trakstar_sensor_0_x, trakstar_sensor_0_y, trakstar_sensor_0_z, trakstar_sensor_2_x, trakstar_sensor_2_y, trakstar_sensor_2_z], axis=1)

    trakstar_pos_s0_s2 = transform_trakstar_to_raven_left(trakstar_pos_s0_s2)
    trakstar_pos_s0_s2 = transform_trakstar_to_raven_right(trakstar_pos_s0_s2)



    # add transformed trakstar positions to the dataframe
    trakstar_df['trakstar_sensor_1_x_transformed'] = trakstar_pos_s1_s3[:, 0]
    trakstar_df['trakstar_sensor_1_y_transformed'] = trakstar_pos_s1_s3[:, 1]
    trakstar_df['trakstar_sensor_1_z_transformed'] = trakstar_pos_s1_s3[:, 2]
    trakstar_df['trakstar_sensor_2_x_transformed'] = trakstar_pos_s1_s3[:, 3]
    trakstar_df['trakstar_sensor_2_y_transformed'] = trakstar_pos_s1_s3[:, 4]
    trakstar_df['trakstar_sensor_2_z_transformed'] = trakstar_pos_s1_s3[:, 5]

    trakstar_df['trakstar_sensor_0_x_transformed'] = trakstar_pos_s0_s2[:, 0]
    trakstar_df['trakstar_sensor_0_y_transformed'] = trakstar_pos_s0_s2[:, 1]
    trakstar_df['trakstar_sensor_0_z_transformed'] = trakstar_pos_s0_s2[:, 2]
    trakstar_df['trakstar_sensor_3_x_transformed'] = trakstar_pos_s0_s2[:, 3]
    trakstar_df['trakstar_sensor_3_y_transformed'] = trakstar_pos_s0_s2[:, 4]
    trakstar_df['trakstar_sensor_3_z_transformed'] = trakstar_pos_s0_s2[:, 5]

    return trakstar_df



def build_raven_positions(new_raven: np.ndarray) -> np.ndarray:
    """
    Extract Raven positions (subtract initial) and scale (kept *1e-2 from original).
    Assumes pos columns are indices 2..7 (6 values).
    """
    raven_pos = (new_raven[:, 2:8] - new_raven[0, 2:8]) * 1e-2
    return raven_pos


def align_console_raven_rot(new_console: np.ndarray, new_raven: np.ndarray):
    """Compute and plot rotation comparison (Console vs Raven) for both PSMs."""
    console_rot = cumulate_console_data_rot(new_console)
    console_psm1_rot = transform_console_to_raven_rot_1(console_rot[:, [0, 1, 2]].copy())
    console_psm2_rot = transform_console_to_raven_rot_1(console_rot[:, [3, 4, 5]].copy())

    raven_psm1_rot = rotation_matrix_to_euler(new_raven[:, 8:17])
    raven_psm2_rot = rotation_matrix_to_euler(new_raven[:, 17:26])

    # zero-relative
    raven_psm1_rot = process_raven_rot(raven_psm1_rot - raven_psm1_rot[0, :])
    raven_psm2_rot = process_raven_rot(raven_psm2_rot - raven_psm2_rot[0, :])

    plot_rotation_comparison(console_psm1_rot, raven_psm1_rot, "PSM1 Left")
    plot_rotation_comparison(console_psm2_rot, raven_psm2_rot, "PSM2 Right")


# -----------------------------
# Plotting
# -----------------------------
def plot_xyz_raven_console_left(raven_pos: np.ndarray, console_pos: np.ndarray):
    labels = ['X', 'Y', 'Z']
    fig = plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(raven_pos[:, i], label='Raven')
        plt.plot(console_pos[:, i], label='Console')
        plt.title(f'{labels[i]} Coordinate (Left Arm)')
        plt.xlabel('Index')
        plt.ylabel(f'{labels[i]} Value')
        plt.legend()
    _save_fig(fig, f"raven_console_xyz_left_{trial_id}.png", f"Trial {trial_id} - Raven vs Console Left XYZ")


def plot_xyz_raven_console_right(raven_pos: np.ndarray, console_pos: np.ndarray):
    labels = ['X', 'Y', 'Z']
    fig = plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(raven_pos[:, i+3], label='Raven')
        plt.plot(console_pos[:, i+3], label='Console')
        plt.title(f'{labels[i]} Coordinate (Right Arm)')
        plt.xlabel('Index')
        plt.ylabel(f'{labels[i]} Value')
        plt.legend()
    _save_fig(fig, f"raven_console_xyz_right_{trial_id}.png", f"Trial {trial_id} - Raven vs Console Right XYZ")


def plot_x_y_z(data: np.ndarray, x_col: int, y_col: int, z_col: int):
    x = data[:, x_col]
    y = data[:, y_col]
    z = data[:, z_col]
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.plot(x); plt.title('X Coordinate'); plt.xlabel('Index'); plt.ylabel('X Value')
    plt.subplot(1, 3, 2); plt.plot(y); plt.title('Y Coordinate'); plt.xlabel('Index'); plt.ylabel('Y Value')
    plt.subplot(1, 3, 3); plt.plot(z); plt.title('Z Coordinate'); plt.xlabel('Index'); plt.ylabel('Z Value')
    _save_fig(fig, f"xyz_triplet_{trial_id}.png", f"Trial {trial_id} - XYZ Triplet")


def plot_rotation_comparison(console_rot: np.ndarray, raven_rot: np.ndarray, title_prefix: str):
    """Plot console and raven rotation data comparison"""
    labels = ['Roll', 'Pitch', 'Yaw']
    fig = plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.plot(console_rot[:, i], label='Console', alpha=0.8)
        plt.plot(raven_rot[:, i], label='Raven', alpha=0.8)
        plt.title(f'{title_prefix} - {labels[i]}')
        plt.xlabel('Index')
        plt.ylabel(f'{labels[i]} (radians)')
        plt.legend()
    safe_prefix = title_prefix.lower().replace(' ', '_')
    _save_fig(fig, f"rotations_{safe_prefix}_{trial_id}.png", f"Trial {trial_id} - {title_prefix} Rotations")


# -----------------------------
# CSV EXPORT
# -----------------------------
def save_array_with_headers_csv(arr: np.ndarray, headers: List[str]):
    """Save a 2D array with headers to CSV."""
    df = pd.DataFrame(arr, columns=headers)
    return df


def export_aligned_streams_to_csv(
    new_console: np.ndarray,
    console_headers: List[str],
    new_raven: np.ndarray,
    raven_headers: List[str],
):
    """
    Save aligned Console and Raven arrays with their headers to CSV.
    """
    console_df = save_array_with_headers_csv(new_console, console_headers)
    raven_df = save_array_with_headers_csv(new_raven,  raven_headers)
    return console_df, raven_df


# ---- config ----
# optional tolerance: set e.g. 40_000_000 for 40 ms; None means no max tolerance
TOL_NS = None

# output path
synched_pds_output_path = os.path.join(trial_path, "synched_data", f"obs_pds_synced_{trial_id}.csv")
os.makedirs(os.path.dirname(synched_pds_output_path), exist_ok=True)

# 1) Baseline: OBS frames (one row per frame)
obs = df_obs_frames.copy()
if "server_time" not in obs.columns:
    raise ValueError("df_obs_frames must include 'server_time' (epoch ns).")

obs["server_time"] = pd.to_numeric(obs["server_time"], errors="coerce").astype("Int64").fillna(0).astype("int64")
obs = obs.sort_values("server_time").reset_index(drop=True)
obs["obs_frame_idx"] = obs["idx"].astype(int) if "idx" in obs.columns else np.arange(1, len(obs) + 1)
left = obs[["obs_frame_idx", "server_time"]].copy()

# 2) PDS stream
if "Computer Time" not in df_pds.columns:
    raise ValueError("df_pds must include 'Computer Time' (epoch ns).")

pds = df_pds.copy()
pds["Computer Time"] = pd.to_numeric(pds["Computer Time"], errors="coerce").astype("Int64")
pds = pds.dropna(subset=["Computer Time"]).copy()
pds["Computer Time"] = pds["Computer Time"].astype("int64")
pds = pds.sort_values("Computer Time").drop_duplicates(subset=["Computer Time"], keep="last").reset_index(drop=True)

pds_cols = [c for c in pds.columns if c != "Computer Time"]
first_pds = pds["Computer Time"].iloc[0] if len(pds) else np.iinfo(np.int64).max
last_pds  = pds["Computer Time"].iloc[-1] if len(pds) else np.iinfo(np.int64).min

# Handle empty PDS quickly: pure dummies for all OBS frames
if pds.empty:
    out_pds = left.copy()
    press_cols   = [c for c in pds_cols if "Pressure" in c]
    pressed_cols = [c for c in pds_cols if ("Pressed" in c) or ("Pressesd" in c)]
    for c in press_cols:   out_pds[c] = -1
    for c in pressed_cols: out_pds[c] = 0
    out_pds["pds_time_ns"]  = np.nan
    out_pds["pds_delta_ns"] = np.nan
else:
    # 3) Two directional as-of joins (prev/next)
    prev = pd.merge_asof(
        left.sort_values("server_time"),
        pds[["Computer Time"] + pds_cols].sort_values("Computer Time"),
        left_on="server_time", right_on="Computer Time",
        direction="backward", allow_exact_matches=True
    ).rename(columns={c: f"{c}__prev" for c in ["Computer Time"] + pds_cols})

    nxt = pd.merge_asof(
        left.sort_values("server_time"),
        pds[["Computer Time"] + pds_cols].sort_values("Computer Time"),
        left_on="server_time", right_on="Computer Time",
        direction="forward", allow_exact_matches=True
    ).rename(columns={c: f"{c}__next" for c in ["Computer Time"] + pds_cols})

    merged = prev.merge(
        nxt[["obs_frame_idx", "server_time"] + [f"{c}__next" for c in ["Computer Time"] + pds_cols]],
        on=["obs_frame_idx", "server_time"], how="left"
    )

    # 4) Choose nearest of prev/next within the valid PDS range
    merged["delta_prev"] = (merged["Computer Time__prev"] - merged["server_time"]).abs()
    merged["delta_next"] = (merged["Computer Time__next"] - merged["server_time"]).abs()

    before_first = merged["server_time"] < first_pds
    after_last   = merged["server_time"] > last_pds

    pick_prev = (~before_first) & (~after_last) & merged["Computer Time__prev"].notna()
    pick_next = (~before_first) & (~after_last) & merged["Computer Time__next"].notna()

    prefer_prev = (pick_prev & pick_next & (merged["delta_prev"] <= merged["delta_next"]))
    prefer_next = (pick_prev & pick_next & (merged["delta_next"] <  merged["delta_prev"]))
    only_prev   = (pick_prev & ~pick_next)
    only_next   = (pick_next & ~pick_prev)

    use_prev = prefer_prev | only_prev
    use_next = prefer_next | only_next

    # Optional tolerance (ms/ns) — far matches fall back to dummies
    if TOL_NS is not None:
        use_prev = use_prev & (merged["delta_prev"] <= TOL_NS)
        use_next = use_next & (merged["delta_next"] <= TOL_NS)

    # 5) Build output table (chosen PDS values or NaN → dummy later)
    out_pds = merged[["obs_frame_idx", "server_time"]].copy()
    for c in pds_cols:
        out_pds[c] = np.where(use_prev, merged[f"{c}__prev"],
                   np.where(use_next, merged[f"{c}__next"], np.nan))
    out_pds["pds_time_ns"]  = np.where(use_prev, merged["Computer Time__prev"],
                            np.where(use_next, merged["Computer Time__next"], np.nan))
    out_pds["pds_delta_ns"] = out_pds["pds_time_ns"] - out_pds["server_time"]

    # 6) Fill dummies:
    #    - any '*Pressure' -> -1
    #    - any '*Pressed' / '*Pressesd' -> 0
    press_cols   = [c for c in pds_cols if "Pressure" in c]
    pressed_cols = [c for c in pds_cols if ("Pressed" in c) or ("Pressesd" in c)]

    if press_cols:
        out_pds[press_cols] = out_pds[press_cols].astype("float64").fillna(-1)
    if pressed_cols:
        out_pds[pressed_cols] = out_pds[pressed_cols].astype("float64").fillna(0).astype(int)

    # Force dummies anywhere there was no match (covers head, gaps, tail)
    no_match = out_pds["pds_time_ns"].isna()
    if press_cols:
        out_pds.loc[no_match, press_cols] = -1
    if pressed_cols:
        out_pds.loc[no_match, pressed_cols] = 0
        out_pds[pressed_cols] = out_pds[pressed_cols].astype(int)

# 7) Nice column order & save
ordered = ["obs_frame_idx", "server_time", "pds_time_ns", "pds_delta_ns"] + pds_cols
out_pds = out_pds.reindex(columns=ordered)
# force integer (no sci-notation) for time columns
time_cols = ["server_time", "pds_time_ns", "pds_delta_ns"]

# coerce -> numeric, then cast to nullable Int64 so we can keep missing values
for c in time_cols:
    out_pds[c] = pd.to_numeric(out_pds[c], errors="coerce")

# pds_delta_ns might be float-y; round then cast
out_pds["pds_delta_ns"] = out_pds["pds_delta_ns"].round()

out_pds[["server_time", "pds_time_ns", "pds_delta_ns"]] = (
    out_pds[["server_time", "pds_time_ns", "pds_delta_ns"]].astype("Int64")
)

# sanity check
print(out_pds.dtypes[time_cols])
print(out_pds[["obs_frame_idx","server_time","pds_time_ns","pds_delta_ns"]].head(3).to_string(index=False))

# if there pds_time_ns, pds_delta
# fill -1 for time cols where there was no PDS match (head/tail/gaps)
out_pds.loc[no_match, ["pds_time_ns", "pds_delta_ns"]] = -1

# now write
out_pds.to_csv(synched_pds_output_path, index=False)
print(f"Saved OBS+PDS synced data to: {synched_pds_output_path}")

# %% [markdown]
# ## Sync OBS + TrakStar

# %%

# ---- Preconditions: df_obs_frames, df_trakstar are already loaded ----

synched_trakstar_output_path = os.path.join(trial_path, "synched_data", f"obs_trakstar_synced_{trial_id}.csv")
os.makedirs(os.path.dirname(synched_trakstar_output_path), exist_ok=True)

# 0) Optional tolerance for matching (ns). Set to e.g. 40_000_000 (40 ms) or leave None for no limit.
TOL_NS = None

# 1) Baseline OBS frames
obs = df_obs_frames.copy()
if "server_time" not in obs.columns:
    raise ValueError("df_obs_frames must include 'server_time' (epoch ns).")
obs["server_time"] = pd.to_numeric(obs["server_time"], errors="coerce").astype("Int64").fillna(0).astype("int64")
obs = obs.sort_values("server_time").reset_index(drop=True)
obs["obs_frame_idx"] = obs["idx"].astype(int) if "idx" in obs.columns else np.arange(1, len(obs)+1)
left = obs[["obs_frame_idx", "server_time"]].copy()

# 2) TrakStar stream (use 'TrakStar Time' only)
required = ["SensorID","x","y","z","azimuth","elevation","roll","TrakStar Time"]
missing = [c for c in required if c not in df_trakstar.columns]
if missing:
    raise ValueError(f"df_trakstar missing columns: {missing}")

ts = df_trakstar.copy()
ts = ts[required].dropna(subset=["TrakStar Time"]).copy()

# ensure SensorID is int
ts["SensorID"] = pd.to_numeric(ts["SensorID"], errors="coerce").astype("Int64")
ts = ts.dropna(subset=["SensorID"]).copy()
ts["SensorID"] = ts["SensorID"].astype("int64")

# TrakStar Time (seconds) -> ns (int)
ts["TrakStar Time"] = pd.to_numeric(ts["TrakStar Time"], errors="coerce")
ts = ts.dropna(subset=["TrakStar Time"]).copy()
ts["ts_time_ns"] = (ts["TrakStar Time"] * 1_000_000_000).round().astype("int64")

# Pivot to wide: one row per ts_time_ns, columns per sensor & field
value_cols = ["x","y","z","azimuth","elevation","roll"]
ts_wide = ts.pivot_table(index="ts_time_ns", columns="SensorID", values=value_cols, aggfunc="last")
# Flatten MultiIndex columns -> trakstar_sensor_{sid}_{field}
ts_wide.columns = [f"trakstar_sensor_{int(sid)}_{field}" for (field, sid) in ts_wide.columns]
ts_wide = ts_wide.reset_index().sort_values("ts_time_ns").reset_index(drop=True)
ts_cols = [c for c in ts_wide.columns if c != "ts_time_ns"]

# 3) Two as-of joins: previous and next relative to each OBS frame
prev = pd.merge_asof(
    left.sort_values("server_time"),
    ts_wide[["ts_time_ns"] + ts_cols].sort_values("ts_time_ns"),
    left_on="server_time", right_on="ts_time_ns",
    direction="backward", allow_exact_matches=True
).rename(columns={c: c + "__prev" for c in ["ts_time_ns"] + ts_cols})

nxt = pd.merge_asof(
    left.sort_values("server_time"),
    ts_wide[["ts_time_ns"] + ts_cols].sort_values("ts_time_ns"),
    left_on="server_time", right_on="ts_time_ns",
    direction="forward", allow_exact_matches=True
).rename(columns={c: c + "__next" for c in ["ts_time_ns"] + ts_cols})

merged = prev.merge(
    nxt[["obs_frame_idx","server_time"] + [f"{c}__next" for c in ["ts_time_ns"] + ts_cols]],
    on=["obs_frame_idx","server_time"], how="left"
)

# 4) Choose closer of prev/next; enforce dummies outside TrakStar range
first_ts = ts_wide["ts_time_ns"].iloc[0] if len(ts_wide) else np.iinfo(np.int64).max
last_ts  = ts_wide["ts_time_ns"].iloc[-1] if len(ts_wide) else np.iinfo(np.int64).min

merged["delta_prev"] = (merged["ts_time_ns__prev"] - merged["server_time"]).abs()
merged["delta_next"] = (merged["ts_time_ns__next"] - merged["server_time"]).abs()

before_first = merged["server_time"] < first_ts
after_last   = merged["server_time"] > last_ts

pick_prev = (~before_first) & (~after_last) & merged["ts_time_ns__prev"].notna()
pick_next = (~before_first) & (~after_last) & merged["ts_time_ns__next"].notna()

prefer_prev = pick_prev & pick_next & (merged["delta_prev"] <= merged["delta_next"])
prefer_next = pick_prev & pick_next & (merged["delta_next"] <  merged["delta_prev"])
only_prev   = pick_prev & ~pick_next
only_next   = pick_next & ~pick_prev

use_prev = prefer_prev | only_prev
use_next = prefer_next | only_next

if TOL_NS is not None:
    use_prev = use_prev & (merged["delta_prev"] <= TOL_NS)
    use_next = use_next & (merged["delta_next"] <= TOL_NS)

# 5) Build output per OBS frame
out_ts = merged[["obs_frame_idx","server_time"]].copy()
for c in ts_cols:
    out_ts[c] = np.where(use_prev, merged[c + "__prev"],
                 np.where(use_next, merged[c + "__next"], np.nan))

out_ts["trakstar_time_ns"]  = np.where(use_prev, merged["ts_time_ns__prev"],
                                np.where(use_next, merged["ts_time_ns__next"], np.nan))
out_ts["trakstar_delta_ns"] = out_ts["trakstar_time_ns"] - out_ts["server_time"]

# 6) Fill dummies for no match (head/tail/gaps): numeric features -> -1; time cols -> -1
no_match_ts = out_ts["trakstar_time_ns"].isna()

# Fill feature columns first (float)
out_ts[ts_cols] = out_ts[ts_cols].astype("float64").fillna(-1.0)
# Time cols get -1 and cast to int
out_ts.loc[no_match_ts, ["trakstar_time_ns","trakstar_delta_ns"]] = -1
out_ts[["server_time","trakstar_time_ns","trakstar_delta_ns"]] = (
    out_ts[["server_time","trakstar_time_ns","trakstar_delta_ns"]].round().astype("int64")
)

# add two new columns  (trakstar_grasp_d0, trakstar_grasp_d1)
# calculate euclidean distance between sensor 0 and 1 for grasp_d0
out_ts["trakstar_grasp_d0"] = np.sqrt(
    (out_ts["trakstar_sensor_0_x"] - out_ts["trakstar_sensor_1_x"])**2 +
    (out_ts["trakstar_sensor_0_y"] - out_ts["trakstar_sensor_1_y"])**2 +
    (out_ts["trakstar_sensor_0_z"] - out_ts["trakstar_sensor_1_z"])**2
)
# calculate euclidean distance between sensor 2 and 3 for grasp_d1
out_ts["trakstar_grasp_d1"] = np.sqrt(
    (out_ts["trakstar_sensor_2_x"] - out_ts["trakstar_sensor_3_x"])**2 +
    (out_ts["trakstar_sensor_2_y"] - out_ts["trakstar_sensor_3_y"])**2 +
    (out_ts["trakstar_sensor_2_z"] - out_ts["trakstar_sensor_3_z"])**2
)

# normalize grasp_d0 and grasp_d1 to be between 0 and 1
max_d0 = out_ts["trakstar_grasp_d0"].max()
min_d0 = out_ts["trakstar_grasp_d0"].min()
out_ts["trakstar_grasp_d0"] = (out_ts["trakstar_grasp_d0"] - min_d0) / (max_d0 - min_d0) if max_d0 != min_d0 else 0.0

max_d1 = out_ts["trakstar_grasp_d1"].max()
min_d1 = out_ts["trakstar_grasp_d1"].min()
out_ts["trakstar_grasp_d1"] = (out_ts["trakstar_grasp_d1"] - min_d1) / (max_d1 - min_d1) if max_d1 != min_d1 else 0.0


# add the same transformations as done to console positions
out_ts = build_trakstar_positions_in_raven(out_ts)



# Inspect a few rows
print(f"TrakStar columns:", out_ts.columns.tolist())


#


# Save
out_ts.to_csv(synched_trakstar_output_path, index=False)

# %% [markdown]
# ## Sync OBS + Smartwatch

# %%
synched_sw_output_path = os.path.join(trial_path, "synched_data", f"obs_smartwatch_synced_{trial_id}.csv")
os.makedirs(os.path.dirname(synched_sw_output_path), exist_ok=True)

# ---- Preconditions: df_obs_frames, df_sw_left, df_sw_right already loaded ----
# Optional tolerance (ns). Set e.g. 40_000_000 for 40 ms, or None for unlimited.
TOL_NS = None

# 1) Baseline (OBS frames)
obs = df_obs_frames.copy()
if "server_time" not in obs.columns:
    raise ValueError("df_obs_frames must include 'server_time' in epoch ns.")
obs["server_time"] = pd.to_numeric(obs["server_time"], errors="coerce").astype("Int64").fillna(0).astype("int64")
obs = obs.sort_values("server_time").reset_index(drop=True)
obs["obs_frame_idx"] = obs["idx"].astype(int) if "idx" in obs.columns else np.arange(1, len(obs)+1)
left_base = obs[["obs_frame_idx","server_time"]].copy()

def prep_watch(df_watch, side_label):
    """Returns a DataFrame per side with columns:
       time_ns, {side}_x, {side}_y, {side}_z
    """
    if df_watch is None or df_watch.empty:
        return pd.DataFrame(columns=["time_ns", f"sw_{side_label}_x", f"sw_{side_label}_y", f"sw_{side_label}_z"])

    df = df_watch.copy()

    # filter accelerometer rows if sensor_type exists
    if "sensor_type" in df.columns:
        df = df[df["sensor_type"].str.lower() == "accelerometer"].copy()

    # pick time column: prefer pc_epoch_ms, fallback to sw_epoch_ms
    time_col = None
    if "pc_epoch_ms" in df.columns:
        time_col = "pc_epoch_ms"
    elif "sw_epoch_ms" in df.columns:
        time_col = "sw_epoch_ms"
    else:
        raise ValueError(f"Smartwatch ({side_label}) data missing time column (pc_epoch_ms or sw_epoch_ms).")

    # convert to ns
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce").astype("Int64")
    df = df.dropna(subset=[time_col]).copy()
    df["time_ns"] = (df[time_col].astype("int64") * 1_000_000)

    # ensure axes exist
    for c in ("value_X_Axis","value_Y_Axis","value_Z_Axis"):
        if c not in df.columns:
            raise ValueError(f"Smartwatch ({side_label}) missing column: {c}")

    # keep and rename
    df = df[["time_ns","value_X_Axis","value_Y_Axis","value_Z_Axis"]].copy()
    df = df.sort_values("time_ns").drop_duplicates(subset=["time_ns"], keep="last").reset_index(drop=True)
    df.rename(columns={
        "value_X_Axis": f"sw_{side_label}_x",
        "value_Y_Axis": f"sw_{side_label}_y",
        "value_Z_Axis": f"sw_{side_label}_z",
    }, inplace=True)
    return df

# 2) Prep both watches
swL = prep_watch(df_sw_left,  side_label="left")
swR = prep_watch(df_sw_right, side_label="right")

def nearest_join_watch(base, sw_df, side_label):
    """Nearest neighbor (prev/next) per OBS frame for one side."""
    if sw_df.empty:
        out = base.copy()
        out[f"sw_{side_label}_x"] = -1.0
        out[f"sw_{side_label}_y"] = -1.0
        out[f"sw_{side_label}_z"] = -1.0
        out[f"sw_{side_label}_time_ns"]  = -1
        out[f"sw_{side_label}_delta_ns"] = -1
        return out

    first_t = sw_df["time_ns"].iloc[0]
    last_t  = sw_df["time_ns"].iloc[-1]

    prev = pd.merge_asof(
        base.sort_values("server_time"),
        sw_df.sort_values("time_ns"),
        left_on="server_time", right_on="time_ns",
        direction="backward", allow_exact_matches=True
    ).rename(columns={c: c+"__prev" for c in ["time_ns", f"sw_{side_label}_x", f"sw_{side_label}_y", f"sw_{side_label}_z"]})

    nxt = pd.merge_asof(
        base.sort_values("server_time"),
        sw_df.sort_values("time_ns"),
        left_on="server_time", right_on="time_ns",
        direction="forward", allow_exact_matches=True
    ).rename(columns={c: c+"__next" for c in ["time_ns", f"sw_{side_label}_x", f"sw_{side_label}_y", f"sw_{side_label}_z"]})

    m = prev.merge(
        nxt[["obs_frame_idx","server_time"] + [f"{c}__next" for c in ["time_ns", f"sw_{side_label}_x", f"sw_{side_label}_y", f"sw_{side_label}_z"]]],
        on=["obs_frame_idx","server_time"], how="left"
    )

    m["delta_prev"] = (m["time_ns__prev"] - m["server_time"]).abs()
    m["delta_next"] = (m["time_ns__next"] - m["server_time"]).abs()

    before_first = m["server_time"] < first_t
    after_last   = m["server_time"] > last_t

    pick_prev = (~before_first) & (~after_last) & m["time_ns__prev"].notna()
    pick_next = (~before_first) & (~after_last) & m["time_ns__next"].notna()

    prefer_prev = pick_prev & pick_next & (m["delta_prev"] <= m["delta_next"])
    prefer_next = pick_prev & pick_next & (m["delta_next"] <  m["delta_prev"])
    only_prev   = pick_prev & ~pick_next
    only_next   = pick_next & ~pick_prev

    use_prev = prefer_prev | only_prev
    use_next = prefer_next | only_next

    if TOL_NS is not None:
        use_prev = use_prev & (m["delta_prev"] <= TOL_NS)
        use_next = use_next & (m["delta_next"] <= TOL_NS)

    out = m[["obs_frame_idx","server_time"]].copy()
    for axis in ("x","y","z"):
        col = f"sw_{side_label}_{axis}"
        out[col] = np.where(use_prev, m[f"{col}__prev"],
                     np.where(use_next, m[f"{col}__next"], np.nan))

    out[f"sw_{side_label}_time_ns"]  = np.where(use_prev, m["time_ns__prev"],
                                         np.where(use_next, m["time_ns__next"], np.nan))
    out[f"sw_{side_label}_delta_ns"] = out[f"sw_{side_label}_time_ns"] - out["server_time"]

    # Fill dummies for no match
    no_match = out[f"sw_{side_label}_time_ns"].isna()
    for axis in ("x","y","z"):
        out.loc[no_match, f"sw_{side_label}_{axis}"] = -1.0
    out.loc[no_match, [f"sw_{side_label}_time_ns", f"sw_{side_label}_delta_ns"]] = -1

    # Final dtypes: time columns as int64 (no sci-notation)
    out[[f"sw_{side_label}_time_ns", f"sw_{side_label}_delta_ns"]] = (
        out[[f"sw_{side_label}_time_ns", f"sw_{side_label}_delta_ns"]]
        .round().astype("int64")
    )
    return out

# 3) Join both sides
left_with_L = nearest_join_watch(left_base, swL, "left")
left_with_R = nearest_join_watch(left_base, swR, "right")

# Merge both (keep single server_time column from left side)
out_sw = left_with_L.merge(left_with_R.drop(columns=["server_time"]), on="obs_frame_idx", how="left")

# Ensure server_time is int64 and show sample
out_sw["server_time"] = out_sw["server_time"].astype("int64")
(out_sw.head(12))
print(f"Rows: {len(out_sw)}; columns: {list(out_sw.columns)}")

# Save
out_sw.to_csv(synched_sw_output_path, index=False)


# %% [markdown]
# ## Combine Raven & Console Data

# %%

# -----------------------------
# Main
# -----------------------------
# ---- update your paths here ----
console_path = console_path
raven_path   = raven_path


# (Optional) if you know the original console column names for the raw file (all columns),
# provide them here so the CSV uses the *real* names. Example:
# original_console_headers = [...]
original_console_headers = None  # or a list matching the raw console_data column count

# Read
console_data, raven_df = read_data(console_path, raven_path)

# Synchronize (aligned on Raven timeline)
new_console, console_headers, new_raven, raven_headers = synchronize_data(
    console_data,
    raven_df,
    console_headers_full=original_console_headers
)

# ---- CSV output paths ----

# Export aligned streams with headers
console_df_synched, raven_df_synched = export_aligned_streams_to_csv(
    new_console, console_headers,
    new_raven, raven_headers
)


# ---- Plots (optional) ----
# Rotations (Console vs Raven)
align_console_raven_rot(new_console, new_raven)

# Positions (Console vs Raven)
console_pos = build_console_positions_in_raven(new_console)
raven_pos   = build_raven_positions(new_raven)

plot_xyz_raven_console_left(raven_pos, console_pos)
plot_xyz_raven_console_right(raven_pos, console_pos)

# Optional: quick looks
# plot_x_y_z(console_pos, 0, 1, 2)
# plot_x_y_z(raven_pos,   0, 1, 2)


# %% [markdown]
# ## Apply Raven Transformations to Console Data

# %%
# apply the same transformations to console_pos and extend the  console_df_synched DataFrame

print(console_pos.shape) # should be (N, 6)

# Add new columns to console_df_synched
console_df_synched['console_pos0_transformed'] = console_pos[:, 0]
console_df_synched['console_pos1_transformed'] = console_pos[:, 1]
console_df_synched['console_pos2_transformed'] = console_pos[:, 2]
console_df_synched['console_pos3_transformed'] = console_pos[:, 3]
console_df_synched['console_pos4_transformed'] = console_pos[:, 4]
console_df_synched['console_pos5_transformed'] = console_pos[:, 5]

# %%
synched_raven_output_path = os.path.join(trial_path, "synched_data", f"raven_synced_{trial_id}.csv")
os.makedirs(os.path.dirname(synched_raven_output_path), exist_ok=True)

raven_df_synched.head(10)
# save these dataframes
raven_df_synched.to_csv(synched_raven_output_path, index=False)

console_df_synched.head(10)
synched_console_output_path = os.path.join(trial_path, "synched_data", f"console_synced_{trial_id}.csv")
console_df_synched.to_csv(synched_console_output_path, index=False)



def export_combined_prefixed_csv(
    new_console: np.ndarray,
    console_headers: List[str],
    new_raven: np.ndarray,
    raven_headers: List[str],
    transformed_console_pos: np.ndarray,
    out_combined_csv: str
):
    """
    Create a combined CSV of aligned Console + Raven:
      - One shared sequence column named 'seq'
      - All console columns prefixed with 'console_'
      - All raven columns   prefixed with 'raven_'
      - Duplicate sequence from Raven ('field.last_seq') is dropped
    """
    df_console = pd.DataFrame(new_console, columns=console_headers)
    df_raven   = pd.DataFrame(new_raven,   columns=raven_headers)

    # Identify sequence columns
    console_seq_col = console_headers[0]                   # first column of console selection (e.g., 'seq')
    raven_seq_col   = 'field.last_seq'                    # explicit name in Raven headers

    # Build the single shared 'seq' column (prefer console's)
    seq_series = df_console[console_seq_col].copy()

    # Optional sanity check: console seq vs raven seq (comment out if noisy)
    # if np.nanmax(np.abs(seq_series.values - df_raven[raven_seq_col].values)) > 1e-3:
    #     print("Warning: console seq and raven 'field.last_seq' differ beyond tolerance.")

    # Drop duplicate seq from both sides before prefixing
    df_console_noseq = df_console.drop(columns=[console_seq_col])
    df_raven_noseq   = df_raven.drop(columns=[raven_seq_col])

    # Prefix remaining columns
    df_console_pref = df_console_noseq.add_prefix('console_')
    df_raven_pref   = df_raven_noseq.add_prefix('raven_')

    # Concatenate with shared seq first
    df_combined = pd.concat([seq_series.rename('seq'), df_console_pref, df_raven_pref], axis=1)

    # multiply "raven_field.hdr.stamp" column by 1e9 to get rid of decimal point
    if 'raven_field.hdr.stamp' in df_combined.columns:
        df_combined['raven_field.hdr.stamp'] = (df_combined['raven_field.hdr.stamp'] * 1e9).round().astype('int64')

    # add transformed console positions to the combined dataframe
    df_combined['console_pos0_transformed'] = transformed_console_pos[:, 0]
    df_combined['console_pos1_transformed'] = transformed_console_pos[:, 1]
    df_combined['console_pos2_transformed'] = transformed_console_pos[:, 2]
    df_combined['console_pos3_transformed'] = transformed_console_pos[:, 3]
    df_combined['console_pos4_transformed'] = transformed_console_pos[:, 4]
    df_combined['console_pos5_transformed'] = transformed_console_pos[:, 5]

    # Save
    df_combined.to_csv(out_combined_csv, index=False)
    return df_combined


raven_console_combined_output_path = os.path.join(trial_path, "synched_data", f"raven_console_combined_{trial_id}.csv")
out_combined_csv = raven_console_combined_output_path

# transformed console positions
transformed_console_pos = console_pos

# Export combined CSV with prefixes and shared 'seq'
raven_console_combined_df = export_combined_prefixed_csv(
    new_console, console_headers,
    new_raven, raven_headers,
    transformed_console_pos, out_combined_csv
)

# plot selected variables over seq

print(raven_console_combined_df.columns.tolist())


x_axis = "seq"  # x-axis for plots
raven_var = "raven_field.pos0"
console_var = "console_pos0"
transformed_console_var = "console_pos0_transformed"



# columns of interest
columns_of_interest = [x_axis, raven_var, transformed_console_var]

# obs_frame_idx is x-axis
# Plot raven_field.pos0 and console_pos0 over obs_frame_idx

# Extract relevant columns
data = raven_console_combined_df[columns_of_interest].copy()

# bring raven_field.pos0 and console_pos0 to similar scale for better visualization
data[raven_var] = (data[raven_var] - data[raven_var].min()) / (data[raven_var].max() - data[raven_var].min())
# data[console_var] = (data[console_var] - data[console_var].min()) / (data[console_var].max() - data[console_var].min())
data[transformed_console_var] = (data[transformed_console_var] - data[transformed_console_var].min()) / (data[transformed_console_var].max() - data[transformed_console_var].min())

# Plotting
plt.figure(figsize=(15, 7))
for modality in columns_of_interest[1:]:
    plt.plot(data[x_axis], data[modality], label=modality)
plt.xlabel("Frame Number")
plt.ylabel("Position Value")
plt.title(f"Raven {raven_var} and Console {console_var} over Frame Number (Trial {trial_id})")
plt.legend()
plt.grid()
_save_fig(plt.gcf(), f"raven_console_over_seq_{trial_id}.png", f"Trial {trial_id} - Raven vs Console over Seq")

raven_var = "field.pos0"
console_var = "pos0"


print("columns in raven_df_synched:", raven_df_synched.columns.tolist())
print("columns in console_df_synched:", console_df_synched.columns.tolist())
# Extract relevant columns
data_raven = raven_df_synched[["field.last_seq", raven_var]].copy()
data_console = console_df_synched[["seq", transformed_console_var]].copy()

# find the first row where field.last_seq is lower than previous value of field.last_seq
first_decrease_idx = data_raven[data_raven["field.last_seq"].diff() < 0].index
print
if len(first_decrease_idx) > 0:
    first_decrease_idx = first_decrease_idx[0]
    data_raven = data_raven.iloc[first_decrease_idx:]

# find the first row where seq is lower than previous value of seq
first_decrease_idx = data_console[data_console["seq"].diff() < 0].index
if len(first_decrease_idx) > 0:
    first_decrease_idx = first_decrease_idx[0]
    data_console = data_console.iloc[first_decrease_idx:]



# bring raven_field.pos0 and console_pos0 to similar scale for better visualization
data_raven[raven_var] = (data_raven[raven_var] - data_raven[raven_var].min()) / (data_raven[raven_var].max() - data_raven[raven_var].min())
# data_console[console_var] = (data_console[console_var] - data_console[console_var].min()) / (data_console[console_var].max() - data_console[console_var].min())

data_console[transformed_console_var] = (data_console[transformed_console_var] - data_console[transformed_console_var].min()) / (data_console[transformed_console_var].max() - data_console[transformed_console_var].min())


# Plotting
plt.figure(figsize=(15, 7))
plt.plot(data_raven["field.last_seq"], data_raven[raven_var], label=f"Raven {raven_var}")
plt.plot(data_console["seq"], data_console[transformed_console_var], label=f"Console {transformed_console_var}")
plt.xlabel("Frame Number")
plt.ylabel("Position Value")
plt.title(f"Raven {raven_var} and Console {transformed_console_var} over Frame Number (Trial {trial_id})")
plt.legend()
plt.grid()
_save_fig(plt.gcf(), f"raven_console_over_seq_synched_{trial_id}.png", f"Trial {trial_id} - Raven vs Console Synched over Seq")

# Positions (Console vs Raven)
console_pos = build_console_positions_in_raven(console_df_synched.to_numpy())
raven_pos   = build_raven_positions(raven_df_synched.to_numpy())

print(console_pos.shape, raven_pos.shape)

plot_xyz_raven_console_left(raven_pos, console_pos)
plot_xyz_raven_console_right(raven_pos, console_pos)


# %%
from typing import Tuple


# =========================
# Entry
# =========================
# Basic column sanity (raises helpful error early if combined csv doesn't match expected names)
required_cols = (
    ['seq', 'console_pedal'] +
    [f'console_pos{i}' for i in range(6)] +
    [f'console_rot{i}' for i in range(6)] +
    [f'raven_field.pos{i}' for i in range(6)] +
    [f'raven_field.ori{i}' for i in range(18)]
)
missing = [c for c in required_cols if c not in raven_console_combined_df.columns]
if missing:
    raise ValueError(f"Combined CSV missing expected columns: {missing[:8]}{' ...' if len(missing) > 8 else ''}")

# Plots driven ONLY by the combined CSV
plot_xyz_left_from_combined(raven_console_combined_df)
plot_xyz_right_from_combined(raven_console_combined_df)
plot_rotations_from_combined(raven_console_combined_df)


# %% [markdown]
# # SYNC OBS + Raven + Console

# %%
import numpy as np
import pandas as pd

synched_raven_console_obs_output_path = os.path.join(trial_path, "synched_data", f"raven_console_obs_synced_{trial_id}.csv")
os.makedirs(os.path.dirname(synched_raven_console_obs_output_path), exist_ok=True)

# ---- Preconditions: df_obs_frames and df_raven_console exist ----
# Optional tolerance (ns); e.g., 40_000_000 for 40 ms, or None for unlimited.
TOL_NS = None

df_raven_console = raven_console_combined_df

# 1) Baseline (OBS frames)
obs = df_obs_frames.copy()
if "server_time" not in obs.columns:
    raise ValueError("df_obs_frames must include 'server_time' (epoch ns).")
obs["server_time"] = pd.to_numeric(obs["server_time"], errors="coerce").astype("Int64").fillna(0).astype("int64")
obs = obs.sort_values("server_time").reset_index(drop=True)
obs["obs_frame_idx"] = obs["idx"].astype(int) if "idx" in obs.columns else np.arange(1, len(obs)+1)
base = obs[["obs_frame_idx","server_time"]].copy()

# 2) Raven+Console stream
time_col = "raven_field.hdr.stamp"
if time_col not in df_raven_console.columns:
    raise ValueError(f"df_raven_console must include '{time_col}' (epoch ns).")

rc = df_raven_console.copy()
rc[time_col] = pd.to_numeric(rc[time_col], errors="coerce").astype("Int64")
rc = rc.dropna(subset=[time_col]).copy()
rc[time_col] = rc[time_col].astype("int64")

# sort & drop duplicate timestamps (keep latest)
rc = rc.sort_values(time_col).drop_duplicates(subset=[time_col], keep="last").reset_index(drop=True)

rc_cols = [c for c in rc.columns if c != time_col]
first_rc = rc[time_col].iloc[0] if len(rc) else np.iinfo(np.int64).max
last_rc  = rc[time_col].iloc[-1] if len(rc) else np.iinfo(np.int64).min

# 3) Two directional as-of merges (prev & next)
prev = pd.merge_asof(
    base.sort_values("server_time"),
    rc[[time_col] + rc_cols].sort_values(time_col),
    left_on="server_time", right_on=time_col,
    direction="backward", allow_exact_matches=True
).rename(columns={c: f"{c}__prev" for c in [time_col] + rc_cols})

nxt = pd.merge_asof(
    base.sort_values("server_time"),
    rc[[time_col] + rc_cols].sort_values(time_col),
    left_on="server_time", right_on=time_col,
    direction="forward", allow_exact_matches=True
).rename(columns={c: f"{c}__next" for c in [time_col] + rc_cols})

m = prev.merge(
    nxt[["obs_frame_idx","server_time"] + [f"{c}__next" for c in [time_col] + rc_cols]],
    on=["obs_frame_idx","server_time"], how="left"
)

# 4) Choose the closest of prev/next; force dummies outside stream range
m["delta_prev"] = (m[f"{time_col}__prev"] - m["server_time"]).abs()
m["delta_next"] = (m[f"{time_col}__next"] - m["server_time"]).abs()

before_first = m["server_time"] < first_rc
after_last   = m["server_time"] > last_rc

pick_prev = (~before_first) & (~after_last) & m[f"{time_col}__prev"].notna()
pick_next = (~before_first) & (~after_last) & m[f"{time_col}__next"].notna()

prefer_prev = pick_prev & pick_next & (m["delta_prev"] <= m["delta_next"])
prefer_next = pick_prev & pick_next & (m["delta_next"] <  m["delta_prev"])
only_prev   = pick_prev & ~pick_next
only_next   = pick_next & ~pick_prev

use_prev = prefer_prev | only_prev
use_next = prefer_next | only_next

if TOL_NS is not None:
    use_prev = use_prev & (m["delta_prev"] <= TOL_NS)
    use_next = use_next & (m["delta_next"] <= TOL_NS)

# 5) Build per-OBS-frame output
out_rc = m[["obs_frame_idx","server_time"]].copy()

# choose values
for c in rc_cols:
    out_rc[c] = np.where(use_prev, m[f"{c}__prev"],
                  np.where(use_next, m[f"{c}__next"], np.nan))

# chosen sample time & delta (relative to OBS frame)
out_rc["raven_console_time_ns"]  = np.where(use_prev, m[f"{time_col}__prev"],
                                     np.where(use_next, m[f"{time_col}__next"], np.nan))
out_rc["raven_console_delta_ns"] = out_rc["raven_console_time_ns"] - out_rc["server_time"]

# 6) Fill dummies for no-match frames (head/tail/gaps)
no_match = out_rc["raven_console_time_ns"].isna()

# Fill all numeric feature columns with -1 when no match
num_cols = [c for c in rc_cols if pd.api.types.is_numeric_dtype(out_rc[c])]
for c in num_cols:
    out_rc[c] = out_rc[c].astype("float64")  # allow NaN first
out_rc.loc[no_match, num_cols] = -1.0

# If you want console_pedal to be 0 on no-match, uncomment:
# if "console_pedal" in out_rc.columns:
#     out_rc.loc[no_match, "console_pedal"] = 0.0

# Time cols: -1 on no-match, cast to int64 to avoid sci-notation
out_rc.loc[no_match, ["raven_console_time_ns", "raven_console_delta_ns"]] = -1
out_rc[["server_time","raven_console_time_ns","raven_console_delta_ns"]] = (
    out_rc[["server_time","raven_console_time_ns","raven_console_delta_ns"]]
    .round().astype("int64")
)

# Preview
(out_rc.head(10))
print(f"Rows: {len(out_rc)} | rc feature cols: {len(rc_cols)}")

# ---- If you want to merge with previous synced tables (examples):
# synced_all = out.merge(out_ts.drop(columns=["server_time"]), on="obs_frame_idx", how="left") \
#                 .merge(out_sw.drop(columns=["server_time"]), on="obs_frame_idx", how="left") \
#                 .merge(out_rc.drop(columns=["server_time"]), on="obs_frame_idx", how="left")
# display(synced_all.head(10))

# save
out_rc.to_csv(synched_raven_console_obs_output_path, index=False)


# ------------------------------------------------------------------
# Pedal-gated CUMULATION for TrakStar (on transformed-to-Raven signals)
# Requires: out_ts (with transformed columns), out_rc (with console_pedal)
# ------------------------------------------------------------------

# 1) bring pedal into out_ts
if "console_pedal" in out_rc.columns:
    pedal_df = out_rc[["obs_frame_idx", "console_pedal"]].copy()
else:
    # If somehow missing (shouldn't be), create a dummy 0 pedal so code still runs.
    pedal_df = out_rc[["obs_frame_idx"]].copy()
    pedal_df["console_pedal"] = 0.0

out_ts = out_ts.merge(pedal_df, on="obs_frame_idx", how="left")
# treat NaN or negative pedal (e.g., -1 for no-match) as 0
out_ts["console_pedal"] = pd.to_numeric(out_ts["console_pedal"], errors="coerce").fillna(0.0)
out_ts.loc[out_ts["console_pedal"] < 0, "console_pedal"] = 0.0

# 2) choose which transformed TrakStar pair(s) to cumulate.
# We’ll do both (S1/S3) and (S0/S2) if present, and create *_cum columns.

def _maybe_cumulate_triplet(df: pd.DataFrame, cols_xyz: list[str], prefix_out: str):
    missing = [c for c in cols_xyz if c not in df.columns]
    if missing:
        return df  # silently skip if this set isn't present
    arr = df[cols_xyz].to_numpy(dtype=float)
    ped = df["console_pedal"].to_numpy(dtype=float)
    cum = _cumulate_gated(arr, ped)  # [N,3]
    for j, c in enumerate(cols_xyz):
        df[f"{c}_cum"] = cum[:, j]
    # also a convenience 3D-norm cumulative, if useful
    df[f"{prefix_out}_norm_cum"] = np.linalg.norm(cum, axis=1)
    return df

# Left (sensor 1), Right (sensor 2) — primary set
out_ts = _maybe_cumulate_triplet(
    out_ts,
    ["trakstar_sensor_1_x_transformed",
     "trakstar_sensor_1_y_transformed",
     "trakstar_sensor_1_z_transformed"],
    prefix_out="trakstar_s1"
)
out_ts = _maybe_cumulate_triplet(
    out_ts,
    ["trakstar_sensor_2_x_transformed",
     "trakstar_sensor_2_y_transformed",
     "trakstar_sensor_2_z_transformed"],
    prefix_out="trakstar_s2"
)

# Backup set (sensor 0 and 3) — if you want them too:
out_ts = _maybe_cumulate_triplet(
    out_ts,
    ["trakstar_sensor_0_x_transformed",
     "trakstar_sensor_0_y_transformed",
     "trakstar_sensor_0_z_transformed"],
    prefix_out="trakstar_s0"
)
out_ts = _maybe_cumulate_triplet(
    out_ts,
    ["trakstar_sensor_3_x_transformed",
     "trakstar_sensor_3_y_transformed",
     "trakstar_sensor_3_z_transformed"],
    prefix_out="trakstar_s3"
)

# cumulate grasp_d0 and grasp_d1 too
out_ts = _maybe_cumulate_triplet(
    out_ts,
    ["trakstar_grasp_d0",
        "trakstar_grasp_d1"], 
    prefix_out="trakstar_grasp_cumulated"
)

# 3) Save the updated TrakStar sync CSV (overwrite prior file)
out_ts.to_csv(synched_trakstar_output_path, index=False)
print(f"[UPDATED] wrote pedal-gated cumulative TrakStar to: {synched_trakstar_output_path}")


# trakstar synched df
out_ts.head(10)

# pds synched df
out_pds.head(10)

# smartwatch synched df
out_sw.head(10)

# raven+console synched df
out_rc.head(10)

# combine all
synched_all = out_ts.merge(out_pds.drop(columns=["server_time"]), on="obs_frame_idx", how="left") \
                .merge(out_sw.drop(columns=["server_time"]), on="obs_frame_idx", how="left") \
                .merge(out_rc.drop(columns=["server_time"]), on="obs_frame_idx", how="left")

# ---- Build OBS baseline (one row per OBS frame) ----
obs = df_obs_frames.copy()
if "server_time" not in obs.columns:
    raise ValueError("df_obs_frames must include 'server_time' (epoch ns).")

# ensure proper types
obs["server_time"] = pd.to_numeric(obs["server_time"], errors="coerce").astype("Int64").fillna(0).astype("int64")
obs = obs.sort_values("server_time").reset_index(drop=True)
obs["obs_frame_idx"] = obs["idx"].astype(int) if "idx" in obs.columns else np.arange(1, len(obs)+1)

combined = obs[["obs_frame_idx", "server_time"]].copy()

# ---- List your per-modality synced DataFrames here (add/remove as needed) ----
# out     -> PDS
# out_ts  -> TrakStar
# out_sw  -> Smartwatch L/R
# out_rc  -> Raven+Console
to_merge_names = ["out", "out_ts", "out_sw", "out_rc"]

for name in to_merge_names:
    if name in globals() and isinstance(globals()[name], pd.DataFrame) and not globals()[name].empty:
        dfm = globals()[name].copy()
        # Make sure the key exists
        if "obs_frame_idx" not in dfm.columns:
            raise ValueError(f"{name} is missing 'obs_frame_idx'")
        # Drop duplicate server_time from right side; keep the baseline's server_time
        dfm = dfm.drop(columns=["server_time"], errors="ignore")
        # Defensive: remove duplicate obs_frame_idx rows if any
        if dfm["obs_frame_idx"].duplicated().any():
            dfm = dfm.sort_values("obs_frame_idx").drop_duplicates(subset=["obs_frame_idx"], keep="last")
        # Merge
        combined = combined.merge(dfm, on="obs_frame_idx", how="left")
        print(f"[OK] merged {name} (rows={len(dfm)})")
    else:
        print(f"[INFO] Skipping {name}: not found or empty.")

# ---- Clean up time/delta columns so they are integers (no sci-notation) ----
time_ns_cols  = [c for c in combined.columns if c.endswith("_time_ns")]
delta_ns_cols = [c for c in combined.columns if c.endswith("_delta_ns")]

for c in time_ns_cols:
    combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(-1).round().astype("int64")
for c in delta_ns_cols:
    combined[c] = pd.to_numeric(combined[c], errors="coerce").fillna(-1).round().astype("int64")

# ---- Optional: tidy column order ----
ordered_prefix = ["obs_frame_idx", "server_time"]
ordered_times  = [c for c in [
    "pds_time_ns","pds_delta_ns",
    "trakstar_time_ns","trakstar_delta_ns",
    "sw_left_time_ns","sw_left_delta_ns",
    "sw_right_time_ns","sw_right_delta_ns",
    "raven_console_time_ns","raven_console_delta_ns",
] if c in combined.columns]
other_cols = [c for c in combined.columns if c not in (ordered_prefix + ordered_times)]
combined = combined.reindex(columns=ordered_prefix + ordered_times + other_cols)

# ---- Save ----
out_all_path = os.path.join(trial_path, "synched_data", f"test_synced_all_{trial_id}.csv")
os.makedirs(os.path.dirname(out_all_path), exist_ok=True)
combined.to_csv(out_all_path, index=False)

print(f"[DONE] wrote: {out_all_path} | rows={len(combined)} cols={len(combined.columns)}")
(combined.head(10))


# %% [markdown]
# # Final Plot for Verification

# %%
# ---------- Config ----------
# Use your existing combined DataFrame variable. If it's called something else, rename here:
DF = combined  # or: DF = pd.read_csv("path/to/synced_all_*.csv")

USE_DATETIME_X = True   # True = wall-clock time; False = seconds from start
N_COLS = 3              # grid columns for the subplot layout

# Edit this list to choose what to plot (keys are printed below after discovery):
SELECT = [
    # examples:
    "pds_pressed5",
    "ts0_x", 
    # "ts0_y", "ts1_z", "ts2_azimuth", "ts3_elevation", 
    "ts0_azimuth",
    "sw_left_x",
    # "sw_left_y", "sw_left_z", "sw_right_x",
    "console_pos0", 
    # "console_rot0",
     "console_pedal",
    "raven_pos0", 
    "raven_ori2",
    "raven_ori11"
]

# ---------- Build time axis ----------
import math
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

df = DF.copy()
if "server_time" not in df.columns:
    raise ValueError("Combined data must include 'server_time' (epoch ns).")

df["server_time"] = pd.to_numeric(df["server_time"], errors="coerce").astype("Int64").fillna(0).astype("int64")
if USE_DATETIME_X:
    X = pd.to_datetime(df["server_time"], unit="ns", errors="coerce")
    xfmt = DateFormatter("%H:%M:%S.%f")
else:
    X = (df["server_time"] - df["server_time"].iloc[0]) / 1e9  # seconds from start
    xfmt = None

# ---------- Discover available signals -> keys ----------
SIGNALS = {}  # key -> dict(col=<colname>, label=<pretty>, kind= "line" | "step", mask_neg1=<bool>)

def add(col, key, label=None, kind="line", mask_neg1=False):
    if col in df.columns:
        SIGNALS[key] = dict(col=col, label=label or key, kind=kind, mask_neg1=mask_neg1)

# PDS (pressures -> mask -1; pressed -> step 0/1)
for p in (1,2,3,4,6,7):
    add(f"Pedal {p} Pressure", f"pds_p{p}", label=f"PDS Pedal {p} Pressure", mask_neg1=True)
for p in (1,2,3,4,5,6,7):
    pressed_col = f"Pedal {p} Pressed" if f"Pedal {p} Pressed" in df.columns else ("Pedal 5 Pressesd" if p == 5 and "Pedal 5 Pressesd" in df.columns else None)
    if pressed_col:
        add(pressed_col, f"pds_pressed{p}", label=f"PDS Pedal {p} Pressed", kind="step")

# TrakStar sensors (0..3) — x/y/z & azimuth/elevation/roll; treat -1 as missing
for sid in (0,1,2,3):
    for fld in ("x","y","z","azimuth","elevation","roll"):
        col = f"trakstar_sensor_{sid}_{fld}"
        key = f"ts{sid}_{fld}"
        lab = f"TrakStar S{sid} {fld}"
        add(col, key, label=lab, mask_neg1=True)

# Smartwatch L/R — treat -1 as missing (your sync filled -1 for no-match)
for side in ("left","right"):
    for axis in ("x","y","z"):
        col = f"sw_{side}_{axis}"
        key = f"sw_{side}_{axis}"
        lab = f"SW {side} {axis.upper()}"
        add(col, key, label=lab, mask_neg1=True)

# Console (positions, rotations, pedal). Do NOT mask -1 here (valid values can be negative).
for i in range(6):
    add(f"console_pos{i}", f"console_pos{i}", label=f"Console pos{i}")
    add(f"console_rot{i}", f"console_rot{i}", label=f"Console rot{i}")
add("console_pedal", "console_pedal", label="Console pedal")

# Raven fields (positions 0..5, ori 0..17). Do NOT mask -1; real values can be negative.
for i in range(6):
    add(f"raven_field.pos{i}", f"raven_pos{i}", label=f"Raven pos{i}")
for i in range(18):
    add(f"raven_field.ori{i}", f"raven_ori{i}", label=f"Raven ori{i}")

# (Optional) time/delta diagnostics — lines (no masking)
for name in ("pds","trakstar","sw_left","sw_right","raven_console"):
    tcol = f"{name}_time_ns"
    dcol = f"{name}_delta_ns"
    if tcol in df.columns:
        add(tcol, f"{name}_time_ns", label=f"{name} time (ns)")
    if dcol in df.columns:
        add(dcol, f"{name}_delta_ns", label=f"{name} Δt (ns)")

# Show what's available so you can pick
print("Available keys:")
print(", ".join(SIGNALS.keys()) or "(none)")

# If SELECT empty, auto-pick a small demo set
if not SELECT:
    candidates = ["pds_p1","pds_pressed1","ts0_x","ts0_y","sw_left_x","sw_right_x","console_pedal","raven_pos0"]
    SELECT = [k for k in candidates if k in SIGNALS] or list(SIGNALS.keys())[:6]
    print("SELECT was empty; using:", SELECT)

# Keep only valid keys
SELECT = [k for k in SELECT if k in SIGNALS]
if not SELECT:
    raise ValueError("No valid keys in SELECT. Update your selection based on the printed list above.")

# ---------- Plot grid ----------
n = len(SELECT)
ncols = max(1, N_COLS)
nrows = math.ceil(n / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 2.8*nrows), sharex=True)
axes = np.atleast_1d(axes).ravel()

for i, key in enumerate(SELECT):
    meta = SIGNALS[key]
    col = meta["col"]
    y = pd.to_numeric(df[col], errors="coerce")  # numeric for safety

    # Mask sentinels (-1) for streams that used -1 as "no data"
    if meta.get("mask_neg1", False):
        y = y.mask(np.isclose(y, -1))

    ax = axes[i]
    if meta["kind"] == "step":
        ax.step(X, y.fillna(0), where="post")  # pressed flags are binary; fill NaN->0
        ax.set_ylim(-0.1, 1.1)
    else:
        ax.plot(X, y, linewidth=1.0)

    ax.set_title(meta["label"], fontsize=10)
    ax.grid(True, alpha=0.3)

# Hide extra axes if grid > n
for j in range(i+1, len(axes)):
    axes[j].set_visible(False)

# X-axis formatting
if USE_DATETIME_X:
    for ax in axes[:n]:
        ax.xaxis.set_major_formatter(xfmt)
    fig.autofmt_xdate()
else:
    axes[max(0, n-1)].set_xlabel("Time (s from start)")

plt.tight_layout()
_save_fig(fig, f"synced_signals_grid_{trial_id}.png", f"Trial {trial_id} - Synced Signals Grid")




