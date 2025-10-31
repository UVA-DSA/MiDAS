# multimodal_gesture_dataset_selective.py
from __future__ import annotations
import os
from typing import Dict, List, Optional, Sequence, Tuple, Callable, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import fnmatch
import torch.nn.functional as F

import glob
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from torchvision.io import VideoReader

from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np
from typing import Tuple, List

def _labels_for_dataset_windows(ds) -> np.ndarray:
    """Build integer labels per window from ds.samples and ds.class_map (no __getitem__)."""
    return np.asarray([ds.class_map[s["gesture_code"]] for s in ds.samples], dtype=np.int64)

def stratified_gesture_train_test(
    y: np.ndarray,
    test_ratio: float,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Return stratified (train_idx, test_idx)."""
    try:
        X = np.arange(len(y))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        return train_idx.tolist(), test_idx.tolist()
    except Exception:
        # Fallback: manual per-class split
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_te = max(1, int(round(n * test_ratio))) if n > 1 else 0
            te.extend(idx[:n_te].tolist())
            tr.extend(idx[n_te:].tolist())
        return tr, te



# --------------- basic helpers ---------------
def class_coverage(df_train, df_test):
    tr = set(df_train['gesture_code'].unique().tolist())
    te = set(df_test['gesture_code'].unique().tolist())
    missing = sorted(te - tr)
    return missing


def _gather_csvs_from_dir(dir_path: str, glob_pattern: str = "*.csv", recursive: bool = False) -> list[str]:
    pattern = str(Path(dir_path) / ("**/" + glob_pattern if recursive else glob_pattern))
    return sorted(glob.glob(pattern, recursive=recursive))

def _infer_trial_id(path: str) -> str:
    p = Path(path)
    stem = p.stem

    # Prefer parent folder like S106_T1 when inside synched_data/
    parent = p.parent
    if parent.name.lower() in {"synched_data", "synced_data", "synched", "synced"}:
        grand = parent.parent.name
        if re.match(r"^S\d+_T\d+$", grand, flags=re.IGNORECASE):
            return grand

    # Else extract S###_T# from file name if present
    m = re.search(r"(S\d+_T\d+)", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)

    # Last resort: old behavior, but DO NOT strip to just 't1'
    m2 = re.search(r"_(t\d+)\b", stem, flags=re.IGNORECASE)
    if m2:
        # include a prefix to avoid collisions
        return m2.group(1)  # <-- just 't1'

    # Final fallback: parent dir or full stem
    return p.parent.name or stem


def _read_and_tag_csv(
    path: str,
    fillna_value: float | None,
    *,
    downsample_stride: int = 1,
    ignore_clutch: bool = False,
    clutch_column: str = "console_pedal",
    clutch_pressed_value: int | float | bool = 1,
    drop_neg1: bool = True,
    neg1_exclude_cols: Optional[Sequence[str]] = ("obs_frame_idx","seq","server_time", 'frame_number', 'timestamp', 'trakstar_time_ns','trakstar_delta_ns','sw_left_time_ns', 'sw_left_delta_ns', 'sw_right_time_ns', 'sw_right_delta_ns', 'raven_console_time_ns', 'raven_console_delta_ns',
    "Pedal 1 Pressure","Pedal 2 Pressure","Pedal 3 Pressure","Pedal 4 Pressure","Pedal 6 Pressure","Pedal 7 Pressure"),
) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["source_csv"] = str(path)
    df["trial_id"] = _infer_trial_id(path)

    # drop first rows with -1 in any of the columns until first valid row
    if drop_neg1:
        # ---- Drop rows with sentinel (-1) ----
        drop_mode = "leading"  # "leading" -> drop only initial bad run; "all" -> drop every bad row anywhere
        treat_nan_as_missing = False
        exclude_cols = set(neg1_exclude_cols or ())

        check_cols = [c for c in df.columns if c not in exclude_cols]
        if check_cols:
            vals = df[check_cols].apply(pd.to_numeric, errors="coerce")
            bad_cell = vals.eq(-1)
            if treat_nan_as_missing:
                bad_cell = bad_cell | vals.isna()
            row_is_bad = bad_cell.any(axis=1)

            if drop_mode == "leading":
                arr = row_is_bad.to_numpy()
                if arr.size == 0:
                    leading_bad = 0
                elif not arr[0]:
                    leading_bad = 0
                else:
                    idx = np.nonzero(~arr)[0]
                    leading_bad = int(idx[0]) if idx.size > 0 else len(arr)
                if leading_bad > 0:
                    df = df.iloc[leading_bad:].reset_index(drop=True)
            elif drop_mode == "all":
                df = df.loc[~row_is_bad].reset_index(drop=True)
            else:
                raise ValueError("drop_mode must be 'leading' or 'all'")
    else:
        print(f"Not dropping -1 rows for file: {path}")

    # downsample (prefer obs_frame_idx modulo)
    if downsample_stride > 1:
        if "obs_frame_idx" in df.columns and pd.api.types.is_integer_dtype(df["obs_frame_idx"]):
            df = df[df["obs_frame_idx"] % downsample_stride == 0].reset_index(drop=True)
        else:
            df = df.iloc[::downsample_stride].reset_index(drop=True)

    # optional: drop clutch-pressed
    if ignore_clutch and clutch_column in df.columns:
        col = df[clutch_column]
        if pd.api.types.is_bool_dtype(col):
            keep_mask = ~col.astype(bool)
        else:
            if not pd.api.types.is_numeric_dtype(col):
                col = pd.to_numeric(col, errors="coerce")
            keep_mask = ~(col == float(clutch_pressed_value))
        df = df[keep_mask].reset_index(drop=True)

    # drop rows with NaN values
    rows_before = len(df)   
    df = df.dropna()
    # print(df.head(3))
    # print(f"[INFO] Loaded '{path}' before-rows:{rows_before} after-rows={len(df)} stride={downsample_stride} (after dropna).")

    return df

def _format_video_pattern(pattern: str, csv_path: str, trial_id: str) -> str:
    """pattern supports {trial_id}, {stem}, {parent}"""
    p = Path(csv_path)
    return pattern.format(trial_id=trial_id, stem=p.stem, parent=p.parent.name)

# --------------- video utils ---------------

def _to_chw_uint8(frame_hwc: torch.Tensor) -> torch.Tensor:
    if frame_hwc.ndim != 3:
        raise ValueError(f"Expected 3D frame tensor, got shape {tuple(frame_hwc.shape)}")
    if frame_hwc.shape[-1] in (1, 3, 4):
        img = frame_hwc.permute(2, 0, 1).contiguous()
    elif frame_hwc.shape[0] in (1, 3, 4):
        img = frame_hwc.contiguous()
    else:
        img = frame_hwc.permute(2, 0, 1).contiguous()
    return img.to(torch.uint8)

def _resize_chw_uint8(img_chw_u8: torch.Tensor, size_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
    if size_hw is None:
        return img_chw_u8
    imgf = (img_chw_u8.float().unsqueeze(0) / 255.0)  # [1,C,H,W]
    imgf = F.interpolate(imgf, size=size_hw, mode="bilinear", align_corners=False)
    out = (imgf.squeeze(0) * 255.0).clamp(0, 255).to(torch.uint8)
    return out

class _VideoStreamCache:
    """Lightweight VideoReader cache with robust FPS detection."""
    def __init__(self, stream: str = "video", output_size: Optional[Tuple[int,int]] = None, default_fps: float = 30.0):
        self.stream = stream
        self.output_size = output_size
        self.default_fps = float(default_fps)
        self._readers: Dict[str, VideoReader] = {}
        self._fps: Dict[str, float] = {}
        self._duration_s: Dict[str, float] = {}

    def _make_reader(self, path: str) -> VideoReader:
        return VideoReader(path, stream=self.stream)

    def _estimate_fps_from_pts(self, vr: VideoReader, max_frames: int = 60) -> Optional[float]:
        try: vr.seek(0.0)
        except Exception: return None
        pts_list: List[float] = []
        for i, pkt in enumerate(vr):
            pts = pkt.get("pts", None)
            if pts is not None: pts_list.append(float(pts))
            if i + 1 >= max_frames: break
        if len(pts_list) < 2: return None
        deltas = np.diff(np.array(pts_list, dtype=np.float64))
        deltas = deltas[deltas > 0]
        if deltas.size == 0: return None
        median_dt = float(np.median(deltas))
        if median_dt <= 0: return None
        return 1.0 / median_dt

    def get_reader(self, path: str) -> VideoReader:
        if path not in self._readers:
            vr = self._make_reader(path)
            self._readers[path] = vr
            fps = None; duration_s = 0.0
            try:
                md = vr.get_metadata()
                vid_md = md.get("video", None)
                if isinstance(vid_md, dict):
                    f = vid_md.get("fps", None)
                    if isinstance(f, (list, tuple)) and len(f) > 0:
                        fps = float(f[0])
                    elif isinstance(f, (int, float)):
                        fps = float(f)
                    d = vid_md.get("duration", None)
                    if isinstance(d, (list, tuple)) and len(d) > 0 and d[0] is not None:
                        duration_s = float(d[0])
                    elif isinstance(d, (int, float)):
                        duration_s = float(d)
            except Exception:
                fps = None
            if fps is None or not np.isfinite(fps) or fps <= 1e-3:
                est = self._estimate_fps_from_pts(vr, max_frames=60)
                fps = float(est) if est is not None and np.isfinite(est) and est > 1e-3 else self.default_fps
            self._fps[path] = float(fps)
            self._duration_s[path] = float(duration_s)
        return self._readers[path]

    def get_fps(self, path: str) -> float:
        self.get_reader(path)
        return self._fps[path]

    @staticmethod
    def _t_for_frame(idx: int, fps: float) -> float:
        return max(0.0, idx / max(1e-6, fps))

    def read_frames(self, path: str, frame_indices: np.ndarray) -> torch.Tensor:
        if frame_indices.size == 0:
            return torch.empty(0, 3, 0, 0)
        vr = self.get_reader(path)
        fps = self.get_fps(path)

        order = np.argsort(frame_indices)
        sorted_idx = frame_indices[order]
        unique_idx, inv = np.unique(sorted_idx, return_inverse=True)

        try: vr.seek(self._t_for_frame(int(unique_idx[0]), fps))
        except Exception:
            try: vr.seek(0.0)
            except Exception: pass

        frames: Dict[int, torch.Tensor] = {}
        needed = set(int(i) for i in unique_idx)
        last_needed = int(unique_idx[-1])

        for pkt in vr:
            pts = pkt.get("pts", None)
            if pts is None: continue
            curr = int(round(float(pts) * fps))
            if curr in needed and curr not in frames:
                raw = pkt["data"]
                img_u8 = _to_chw_uint8(raw)
                img_u8 = _resize_chw_uint8(img_u8, self.output_size)
                img = (img_u8.float() / 255.0)
                frames[curr] = img
            if curr >= last_needed and len(frames) == len(unique_idx):
                break

        if len(frames) < len(unique_idx):
            for idx in unique_idx:
                ii = int(idx)
                if ii in frames: continue
                try:
                    vr.seek(self._t_for_frame(ii, fps))
                    pkt = next(iter(vr))
                    raw = pkt["data"]
                    img_u8 = _to_chw_uint8(raw)
                    img_u8 = _resize_chw_uint8(img_u8, self.output_size)
                    img = (img_u8.float() / 255.0)
                    frames[ii] = img
                except Exception:
                    if frames:
                        sample = next(iter(frames.values())); C, H, W = sample.shape
                    else:
                        C, H, W = 3, 224, 224
                    frames[ii] = torch.zeros(C, H, W, dtype=torch.float32)

        gathered = [frames[int(i)] for i in unique_idx]
        gathered = torch.stack(gathered, 0)
        gathered = gathered[inv]
        out = torch.empty_like(gathered)
        out[order] = gathered
        return out

# --------------- feature cache for .npy ---------------

class _FeatureCache:
    """Lazy npy loader with simple caching per path."""
    def __init__(self):
        self._arr: Dict[str, np.ndarray] = {}

    def get(self, path: str) -> np.ndarray:
        if path not in self._arr:
            self._arr[path] = np.load(path, mmap_mode="r")
        return self._arr[path]

# --------------- dataset ---------------

class MultimodalGestureDataset(Dataset):

    # CSV-based modality column rules (images handled specially)
    DEFAULT_MODALITY_RULES: Dict[str, Callable[[str], bool]] = {
        "trakstar": lambda c: c.startswith("trakstar_"),
        "sw_left":  lambda c: c.startswith("sw_left_"),
        "sw_right": lambda c: c.startswith("sw_right_"),
        "console":  lambda c: c.startswith("console_"),
        "raven":    lambda c: c.startswith("raven_"),
        "pedals":   lambda c: c.startswith("Pedal "),
        "handkp":   lambda c: c.startswith("left_") or c.startswith("right_"),
    }

    # map selection token -> filename suffix for features
    IMAGE_FEAT_SUFFIX: Dict[str, str] = {
        "resnet": "resnet",
        "resnet_deskpt": "resnet_deskpt",
        "resnet_deskpt_v2": "resnet_deskpt_v2",
        "resnet_raven": "resnet_raven",
        "resnet_v2": "resnet_v2",
        "resnet_v3": "resnet_v3",
        "dinov3": "dinov3",
        "i3d_flow":    "flow",
        "i3d_rgb":     "rgb",
        "timm": "timm",
        # add more here later
    }

    def __init__(
        self,
        # One of:
        base_path: str | None = None,
        csv_path: str | None = None,
        dir_path: str | None = None,
        csv_paths: Optional[Sequence[str]] = None,
        glob_pattern: str = "*.csv",
        recursive: bool = False,

        # Sampling
        source_hz: int = 30,
        sample_rate: int = 30,

        # Optional: clutch filtering
        ignore_clutch: bool = False,
        clutch_column: str = "console_pedal",
        clutch_pressed_value: int | float | bool = 1,

        # Windowing
        clip_len: int = 32,
        step: int = 8,
        include_modalities: Optional[Sequence[str]] = None,
        drop_short_segments: bool = True,

        # Data handling
        fillna_value: float = 0.0,
        normalize: bool = False,
        normalization_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,

        # Video / images
        video_root: Optional[str] = None,
        video_pattern: str = "{trial_id}.mp4",
        video_map: Optional[Dict[str, str]] = None,
        video_stream: str = "video",
        video_output_size: Optional[Tuple[int,int]] = (224, 224),

        # Schema / selection
        class_map: Optional[Dict[str, int]] = None,
        modality_rules: Optional[Dict[str, Callable[[str], bool]]] = None,
        modality_selections: Optional[Dict[str, Sequence[str]]] = None,
        modality_exclude: Optional[Dict[str, Sequence[str]]] = None,

        selected_indices: Optional[Sequence[int]] = None,

        drop_neg1: bool = True,

        classes_to_allow: Optional[Sequence[str]] = [],

        # Misc
        seed: int = 0,
        windowing: str = "fixed",
    ):
        super().__init__()

        print("\n--- Initializing MultimodalGestureDataset ---\n")
        print("Selected classes to allow:", classes_to_allow)
        # sampling validation
        if sample_rate <= 0 or sample_rate > source_hz:
            raise ValueError(f"sample_rate must be in (0, {source_hz}] — got {sample_rate}.")
        if source_hz % sample_rate != 0:
            raise ValueError(f"sample_rate must divide source_hz exactly; got source_hz={source_hz}, sample_rate={sample_rate}.")
        self.source_hz = int(source_hz)
        self.sample_rate = int(sample_rate)
        self.downsample_stride = self.source_hz // self.sample_rate

        self.ignore_clutch = bool(ignore_clutch)
        self.clutch_column = clutch_column
        self.clutch_pressed_value = clutch_pressed_value

        self.samples: List[Dict] = []

        # collect CSVs
        collected_paths: list[str] = []
        if csv_paths: collected_paths = list(csv_paths)
        elif dir_path:
            collected_paths = _gather_csvs_from_dir(dir_path, glob_pattern, recursive)
            if not collected_paths:
                raise FileNotFoundError(f"No CSVs found in '{dir_path}' matching '{glob_pattern}' (recursive={recursive}).")
        elif csv_path: collected_paths = [csv_path]
        else: raise ValueError("Provide one of: csv_path, dir_path, or csv_paths.")

        # read CSVs
        if base_path is not None:
            dfs = [
                _read_and_tag_csv(
                    base_path + p + "/synched_data/final_annotation_" + p + ".csv",
                    fillna_value=None,
                    downsample_stride=self.downsample_stride,
                    ignore_clutch=self.ignore_clutch,
                    clutch_column=self.clutch_column,
                    clutch_pressed_value=self.clutch_pressed_value,
                    drop_neg1=drop_neg1,
                ) for p in collected_paths
            ]
        else:
            dfs = [
                _read_and_tag_csv(
                    p,
                    fillna_value=None,
                    downsample_stride=self.downsample_stride,
                    ignore_clutch=self.ignore_clutch,
                    clutch_column=self.clutch_column,
                    clutch_pressed_value=self.clutch_pressed_value,
                    drop_neg1=drop_neg1,
                ) for p in collected_paths
            ]

        self.df = pd.concat(dfs, axis=0, ignore_index=True)
        sort_keys = [k for k in ["trial_id", "obs_frame_idx", "source_csv"] if k in self.df.columns]
        self.df = self.df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

        if fillna_value is not None:
            self.df = self.df.fillna(fillna_value)

        for col in ["obs_frame_idx", "gesture_code"]:
            if col not in self.df.columns:
                raise ValueError(f"CSV(s) missing required column: {col}")

        # Always drop Idle if needed
        self.df = self.df[self.df["gesture_code"] != "Idle"].reset_index(drop=True)

        # --- Optional class filtering with robust matching ---
        if classes_to_allow:
            keys = list(classes_to_allow.keys())
            vals = list(classes_to_allow.values())

            print(f"Selected classes to allow: {classes_to_allow}")
            print("gesture_code dtype before:", self.df["gesture_code"].dtype)

            gc_str = self.df["gesture_code"].astype(str).str.strip()
            gc_lower = gc_str.str.lower()
            gc_int = pd.to_numeric(self.df["gesture_code"], errors="coerce").astype("Int64")

            allowed_str = {str(k).strip().lower() for k in keys}
            allowed_int = set()
            for k in keys:
                try:
                    allowed_int.add(int(str(k).strip()))
                except ValueError:
                    pass

            allowed_str |= {str(v).strip().lower() for v in vals}

            before = len(self.df)
            mask = gc_int.isin(list(allowed_int)) | gc_lower.isin(list(allowed_str))
            self.df = self.df[mask].reset_index(drop=True)
            after = len(self.df)
            print(f"Filtered rows: {before} -> {after}")
            print("Unique gesture_codes after filtering:",
                  sorted(self.df['gesture_code'].astype(str).unique().tolist()))

            missing = set(allowed_int) - set(gc_int.dropna().unique().tolist())
            if missing:
                print(f"Note: these allowed IDs had no rows present: {sorted(missing)}")
        else:
            print("classes_to_allow is empty — keeping all non-Idle gestures.")

        sort_keys = ["trial_id", "obs_frame_idx"] if "trial_id" in self.df.columns else ["obs_frame_idx"]
        self.df = self.df.sort_values(sort_keys).reset_index(drop=True)

        # === NEW/UPDATED: Normalize gesture_code dtype consistently ===
        # Try to coerce to int; if it fails, keep strings (and we will sort numerically by casting)
        _gesture_is_numeric = True
        try:
            self.df["gesture_code"] = pd.to_numeric(self.df["gesture_code"], errors="raise").astype(int)
        except Exception:
            _gesture_is_numeric = False
            self.df["gesture_code"] = self.df["gesture_code"].astype(str).str.strip()
        self._gesture_is_numeric = _gesture_is_numeric  # keep for later lookups

        # === NEW/UPDATED: Build a stable class_map ===
        if class_map is not None:
            self.class_map = dict(class_map)
        else:
            if classes_to_allow:
                # Respect the order of classes_to_allow, but only include those present
                if _gesture_is_numeric:
                    allowed_ids = []
                    for k in classes_to_allow.keys():
                        try:
                            allowed_ids.append(int(str(k).strip()))
                        except ValueError:
                            # ignore textual keys when using numeric labels
                            pass
                    present = set(int(x) for x in self.df["gesture_code"].unique())
                    order = [k for k in allowed_ids if k in present]
                    self.class_map = {k: i for i, k in enumerate(order)}
                else:
                    # String labels: normalize to lowercase for matching, keep original token as key
                    present = set(self.df["gesture_code"].unique())
                    # Respect provided keys/values order; include whichever is present
                    ordered = []
                    for k in classes_to_allow.keys():
                        ks = str(k).strip()
                        if ks in present:
                            ordered.append(ks)
                    for v in classes_to_allow.values():
                        vs = str(v).strip()
                        if vs in present and vs not in ordered:
                            ordered.append(vs)
                    self.class_map = {k: i for i, k in enumerate(ordered)}
            else:
                # No predefined list: sort naturally
                if _gesture_is_numeric:
                    uniq = sorted(int(u) for u in self.df["gesture_code"].dropna().unique().tolist())
                    self.class_map = {u: i for i, u in enumerate(uniq)}
                else:
                    uniq = sorted(self.df["gesture_code"].dropna().unique().tolist(), key=lambda s: int(s) if s.isdigit() else s)
                    self.class_map = {s: i for i, s in enumerate(uniq)}

        print("\n--- Gesture class mapping ---")
        print(f"Final class_map (gesture_code -> index): {self.class_map}")
        print("--- Gesture class mapping ---\n")

        

        # modality selection setup
        self.modality_rules = dict(self.DEFAULT_MODALITY_RULES if modality_rules is None else modality_rules)
        self.include_modalities = list(include_modalities or self.modality_rules.keys())
        self.modality_selections = modality_selections or {}
        self.modality_exclude = modality_exclude or {}

        # CSV-backed columns per modality (images handled specially)
        self.modality_cols: Dict[str, List[str]] = {}
        all_cols = list(self.df.columns)

        def match_any(patterns: Sequence[str], col: str) -> bool:
            return any(fnmatch.fnmatch(col, p) for p in patterns)

        for m in self.include_modalities:
            if m in ("images", "images_feat"):
                # images: special — no CSV columns to gather
                continue
            rule = self.modality_rules.get(m)
            if rule is None:
                raise ValueError(f"No rule defined for modality '{m}'")
            candidates = [c for c in all_cols if rule(c)]
            allow = self.modality_selections.get(m)
            if allow:
                candidates = sorted({c for c in candidates if match_any(allow, c)})
            deny = self.modality_exclude.get(m)
            if deny:
                candidates = [c for c in candidates if not match_any(deny, c)]
            numeric = [c for c in candidates if pd.api.types.is_numeric_dtype(self.df[c])]
            if numeric:
                self.modality_cols[m] = numeric

        sort_keys = [k for k in ["trial_id", "source_csv", "obs_frame_idx"] if k in self.df.columns]
        self.df = self.df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)

        # gesture runs
        self.df["_run_boundary"] = (
            (self.df["gesture_code"] != self.df["gesture_code"].shift(1)) |
            (self.df["trial_id"]     != self.df["trial_id"].shift(1))     |
            (self.df["source_csv"]   != self.df["source_csv"].shift(1))
        )
        self.df["_run_id"] = self.df["_run_boundary"].cumsum()

        # video & feature caches
        self.video_root = video_root
        self.video_pattern = video_pattern
        self.video_map = dict(video_map) if video_map else {}
        self._video_cache = _VideoStreamCache(stream=video_stream, output_size=video_output_size)
        self._feat_cache = _FeatureCache()
        # precompute video path for each row
        self.df["_video_path"] = self.df.apply(self._resolve_video_for_row, axis=1)

        # images selections normalized
        imgs_sel = self.modality_selections.get("images", []) or []
        self.images_want_rgb = ("rgb" in imgs_sel)
        # which feature sets are requested for images
        self.images_feat_kinds: List[str] = [k for k in imgs_sel if k in self.IMAGE_FEAT_SUFFIX]
        print(f"Dataset will load image features: {self.images_feat_kinds}")
        # windowing
        self.clip_len = int(clip_len)
        self.step = int(step)
        self.full_clip = (self.clip_len == -1)
        self.drop_short_segments = bool(drop_short_segments)
        self.windowing = windowing
        self.samples: List[Dict] = []
        self._make_windows()

        # keep only a chosen subset of windows if provided
        if selected_indices is not None:
            sel = set(int(i) for i in selected_indices)
            self.samples = [self.samples[i] for i in range(len(self.samples)) if i in sel]

        # normalization
        self.normalize = normalize
        self.norm_stats = normalization_stats or {}
        if self.normalize and not self.norm_stats and self.modality_cols:
            self._compute_norm_stats()

        self.rng = np.random.default_rng(seed)

    # ---------- utils ----------

    def _resolve_video_for_row(self, row: pd.Series) -> str:
        trial = str(row.get("trial_id", ""))
        src = str(row.get("source_csv", ""))
        if src in self.video_map:
            return self.video_map[src]
        if trial in self.video_map:
            return self.video_map[trial]
        csv_dir = str(Path(src).parent)
        base_root = self.video_root if self.video_root is not None else csv_dir
        filename = _format_video_pattern(self.video_pattern, src, trial)
        return str(Path(base_root) / filename)

    def _make_windows(self):
        self.samples.clear()
        for _, seg in self.df.groupby(["_run_id"], sort=False):
            label_str = seg["gesture_code"].iloc[0]
            seg_len = len(seg)
            base = int(seg.index[0])

            if self.full_clip:
                self.samples.append({"start": base, "end": base + seg_len, "gesture_code": label_str})
                continue

            if seg_len < self.clip_len:
                if self.drop_short_segments:
                    continue

            max_start = seg_len - self.clip_len
            if max_start < 0:
                continue

            for s in range(0, max_start + 1, self.step):
                self.samples.append({"start": base + s, "end": base + s + self.clip_len, "gesture_code": label_str})

    def _compute_norm_stats(self):
        if not self.samples:
            raise RuntimeError("No samples to compute normalization stats from.")
        idxs = np.concatenate([np.arange(s["start"], s["end"]) for s in self.samples])
        used = self.df.iloc[idxs]
        for m, cols in self.modality_cols.items():
            X = used[cols].to_numpy(dtype=np.float32)
            mean, std = X.mean(axis=0), X.std(axis=0)
            std[std == 0] = 1.0
            self.norm_stats[m] = (mean, std)

    # ---------- dataset API ----------

    def __len__(self) -> int:
        return len(self.samples)

    def _get_modality_tensor(self, start: int, end: int, modality: str) -> torch.Tensor:
        cols = self.modality_cols[modality]
        X = self.df.loc[start:end - 1, cols].to_numpy(dtype=np.float32)
        if self.normalize and modality in self.norm_stats:
            mean, std = self.norm_stats[modality]
            X = (X - mean) / std
        return torch.from_numpy(X)  # [T, F]

    # --- images: frames (rgb) ---

    def _load_clip_frames(self, start: int, end: int) -> Optional[torch.Tensor]:
        if not self.images_want_rgb:
            return None
        slice_df = self.df.iloc[start:end]
        video_path = slice_df["_video_path"].iloc[0]
        if not Path(video_path).exists():
            return None
        obs_idx = slice_df["obs_frame_idx"].to_numpy(dtype=np.int64)
        fps = self._video_cache.get_fps(video_path)
        vid_frames = np.round(obs_idx * (fps / float(self.source_hz))).astype(np.int64)
        vid_frames[vid_frames < 0] = 0
        return self._video_cache.read_frames(video_path, vid_frames)  # [T, C, H, W]

    # --- images: features (resnet/i3d/...) ---

    def _build_feat_path(self, video_path: str, kind: str) -> str:
        """
        Map /.../t1.mp4 + kind='resnet' -> /.../t1_resnet50.npy
        """
        suf = self.IMAGE_FEAT_SUFFIX[kind]  # e.g., 'resnet50'
        p = Path(video_path)
        stem = p.stem  # 't1'
        return str(p.with_name(f"{stem}_{suf}.npy"))

    def _load_clip_image_features(self, start: int, end: int) -> Optional[torch.Tensor]:
        """
        Load and concatenate requested feature sets for this clip:
        returns [T, F_total] or None if no feature kind requested or files missing.
        """
        if not self.images_feat_kinds:
            return None
        slice_df = self.df.iloc[start:end]
        video_path = slice_df["_video_path"].iloc[0]
        if not Path(video_path).exists():
            print(f"Video path does not exist for image features: {video_path}")
            return None

        obs_idx = slice_df["obs_frame_idx"].to_numpy(dtype=np.int64)
        fps = self._video_cache.get_fps(video_path)
        vid_frames = np.round(obs_idx * (fps / float(self.source_hz))).astype(np.int64)
        vid_frames[vid_frames < 0] = 0

        feats_list = []
        for kind in self.images_feat_kinds:
            feat_path = self._build_feat_path(video_path, kind)
            if not Path(feat_path).exists():
                continue
            arr = self._feat_cache.get(feat_path)
            # unify to [N, F]
            if arr.ndim == 2:
                N0, N1 = arr.shape
                if N0 < N1:
                    arr_t = arr.T  # [T, F]
                else:
                    arr_t = arr    # [T, F]
            elif arr.ndim == 1:
                arr_t = arr.reshape(-1, 1)
            else:
                raise ValueError(f"Unexpected feature array shape for {feat_path}: {arr.shape}")

            max_row = arr_t.shape[0] - 1
            idx = np.clip(vid_frames, 0, max_row)
            pick = arr_t[idx]  # [T, F_kind]
            feats_list.append(torch.from_numpy(pick.astype(np.float32)))

        if not feats_list:
            return None

        feats = torch.cat(feats_list, dim=-1)  # [T, sum(F_kinds)]
        return feats

    def __getitem__(self, i: int):
        s = self.samples[i]
        start, end = s["start"], s["end"]
        code_raw = self.df.loc[start, "gesture_code"]

        # === NEW/UPDATED: normalize code to match class_map key type ===
        if len(self.class_map) > 0:
            first_key = next(iter(self.class_map.keys()))
            if isinstance(first_key, int):
                code_key = int(code_raw) if not isinstance(code_raw, int) else code_raw
            else:
                code_key = str(code_raw)
        else:
            code_key = int(code_raw) if self._gesture_is_numeric else str(code_raw)

        if code_key not in self.class_map:
            raise KeyError(f"Label '{code_key}' not found in class_map keys {list(self.class_map.keys())}")

        label = self.class_map[code_key]

        item: Dict[str, torch.Tensor | str | int] = {}

        # CSV-backed modalities -> [T, F]
        for m in self.modality_cols.keys():
            item[m] = self._get_modality_tensor(start, end, m)

        # metadata
        obs_idx = self.df.loc[start:end - 1, "obs_frame_idx"].to_numpy(dtype=np.int64)
        item["obs_frame_idx"] = torch.from_numpy(obs_idx)
        item["label"] = torch.tensor(label, dtype=torch.long)
        item["gesture_code"] = code_key  # store normalized form
        item["trial_id"] = self.df.loc[start, "trial_id"]
        item["source_csv"] = self.df.loc[start, "source_csv"]

        # images: frames
        frames = self._load_clip_frames(start, end)
        if frames is not None:
            item["images"] = frames  # [T, C, H, W]

        # images: features (concatenated over selected types)
        im_feats = self._load_clip_image_features(start, end)
        if im_feats is not None:
            item["images_feat"] = im_feats  # [T, F_total]

        return item

    # ---------- stats helpers ----------

    def _get_class_stats(self) -> Dict[str, int]:
        stats = {c: 0 for c in self.class_map.keys()}
        for s in self.samples:
            stats[s["gesture_code"]] += 1
        return stats

    @property
    def num_classes(self) -> int:
        return len(self.class_map)

    @property
    def classes(self) -> List[str]:
        inv = {v: k for k, v in self.class_map.items()}
        return [inv[i] for i in range(len(inv))]

    # ---------- padding helpers ----------

    @staticmethod
    def _pad_and_stack_2d(seqs: List[torch.Tensor], pad_value: float = 0.0):
        lengths = [s.shape[0] for s in seqs]
        B = len(seqs)
        T_max = max(lengths)
        Fdim = seqs[0].shape[1]
        device = seqs[0].device
        out = seqs[0].new_full((B, T_max, Fdim), pad_value)
        mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            t = s.shape[0]
            out[i, :t] = s
            mask[i, :t] = True
        return out, mask

    @staticmethod
    def _pad_and_stack_1d(seqs: List[torch.Tensor], pad_value: int = 0):
        lengths = [s.shape[0] for s in seqs]
        B = len(seqs)
        T_max = max(lengths)
        device = seqs[0].device
        out = seqs[0].new_full((B, T_max), pad_value)
        mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            t = s.shape[0]
            out[i, :t] = s
            mask[i, :t] = True
        return out, mask

    # ---------- collate ----------

    @staticmethod
    def collate_fn(batch):
        """
        - Numeric modalities: any tensor [T, F] keyed by modality name (including 'images_feat')
        - Special: 'images' [T, C, H, W] are excluded from the concatenation flow (used for viz)
        """
        out: Dict[str, torch.Tensor] = {}
        keys = list(batch[0].keys())

        # include all tensor keys except special ones
        special_exclude = {"label", "obs_frame_idx", "images"}
        tensor_modalities = [k for k in keys if isinstance(batch[0][k], torch.Tensor) and k not in special_exclude]

        # detect variable length across any modality
        def lengths_equal_for_mod(m):
            lens = [b[m].shape[0] for b in batch]
            return len(set(lens)) == 1

        variable_length = any(not lengths_equal_for_mod(m) for m in tensor_modalities) if tensor_modalities else False

        if not variable_length:
            for m in tensor_modalities:
                out[m] = torch.stack([b[m] for b in batch], dim=0)  # [B, T, F]
            out["label"] = torch.stack([b["label"] for b in batch], dim=0)
            out["obs_frame_idx"] = torch.stack([b["obs_frame_idx"] for b in batch], dim=0)
            out["gesture_code"] = [b["gesture_code"] for b in batch]
            out["trial_id"] = [b.get("trial_id", "") for b in batch]
            out["source_csv"] = [b.get("source_csv", "") for b in batch]
            # frames (if present)
            if "images" in batch[0]:
                out["images"] = torch.stack([b["images"] for b in batch], dim=0)  # [B, T, C, H, W]
            return out

        # variable-length padding
        attention_mask: Dict[str, torch.Tensor] = {}
        for m in tensor_modalities:
            seqs = [b[m] for b in batch]  # list of [T, F]
            padded, mask = MultimodalGestureDataset._pad_and_stack_2d(seqs, pad_value=0.0)
            out[m] = padded
            attention_mask[m] = mask

        obs_seqs = [b["obs_frame_idx"] for b in batch]
        obs_padded, obs_mask = MultimodalGestureDataset._pad_and_stack_1d(obs_seqs, pad_value=0)
        out["obs_frame_idx"] = obs_padded
        out["obs_mask"] = obs_mask

        out["label"] = torch.stack([b["label"] for b in batch], dim=0)
        out["gesture_code"] = [b["gesture_code"] for b in batch]
        out["trial_id"] = [b.get("trial_id", "") for b in batch]
        out["source_csv"] = [b.get("source_csv", "") for b in batch]
        out["attention_mask"] = attention_mask

        # pad images if present
        if "images" in batch[0]:
            imgs = [b.get("images", None) for b in batch]
            if all(x is not None for x in imgs):
                T_list = [x.shape[0] for x in imgs]
                Tm = max(T_list)
                B = len(imgs)
                C, H, W = imgs[0].shape[1:]
                pad_imgs = imgs[0].new_zeros((B, Tm, C, H, W))
                mask_img = torch.zeros(B, Tm, dtype=torch.bool)
                for i, x in enumerate(imgs):
                    t = x.shape[0]
                    pad_imgs[i, :t] = x
                    mask_img[i, :t] = True
                out["images"] = pad_imgs
                out["images_mask"] = mask_img

        return out



# ----------------------- (Optional) quick viz utils -----------------------

def print_batch_stats(batch):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: dtype={v.dtype}, shape={tuple(v.shape)}, "
                  f"min={v.min().item():.3f}, max={v.max().item():.3f}")
            # (optional) per-dim stats could go here
        elif isinstance(v, list):
            print(f"{k}: list of {len(v)} items, head: {v[:3]}")
        else:
            print(f"{k}: type={type(v)}, value={v}")


def visualize_dataset_sample(
    batch: dict,
    idx: int = 0,
    save_dir: str = "./viz",
    max_features_per_mod: int = 8,
    max_frames_in_grid: int = 64,
    make_gif: bool = True,
    gif_fps: int = 10,
    batch_idx: int = 0
):
    os.makedirs(save_dir, exist_ok=True)

    gesture = batch["gesture_code"][idx] if isinstance(batch.get("gesture_code"), list) else str(batch.get("gesture_code", ""))
    trial_id = batch.get("trial_id", ["?"])[idx] if isinstance(batch.get("trial_id"), list) else batch.get("trial_id", "?")

    clip_name = f"trial_{trial_id}_gesture_{gesture}_batchidx_{batch_idx}"

    # Identify 2D modalities [B,T,F]
    modality_keys = []
    for k, v in batch.items():
        if isinstance(v, torch.Tensor) and k not in ("label", "obs_frame_idx", "images", "images_mask"):
            if v.dim() == 3:
                modality_keys.append(k)
    obs_mask = batch.get("obs_mask", None)
    attn_masks = batch.get("attention_mask", {})

    def get_valid_T_for_mod(mod_key: str) -> int:
        if isinstance(attn_masks, dict) and mod_key in attn_masks:
            m = attn_masks[mod_key][idx]
            return int(m.sum().item())
        if isinstance(obs_mask, torch.Tensor):
            return int(obs_mask[idx].sum().item())
        return int(batch[mod_key][idx].shape[0])

    # Frames grid
    if "images" in batch:
        imgs = batch["images"][idx]  # [T_max, C, H, W]
        t_img = int(batch["images_mask"][idx].sum().item()) if "images_mask" in batch else imgs.shape[0]
        if t_img > 0:
            imgs = imgs[:t_img]
            T_show = min(t_img, max_frames_in_grid)
            cols = int(np.ceil(np.sqrt(T_show)))
            rows = int(np.ceil(T_show / cols))

            fig = plt.figure(figsize=(cols * 2.2, rows * 2.2))
            for t in range(T_show):
                ax = plt.subplot(rows, cols, t + 1)
                im = imgs[t].detach().cpu().numpy().transpose(1, 2, 0)
                ax.imshow(im)
                ax.set_title(f"t={t}", fontsize=8)
                ax.axis("off")
            fig.suptitle(f"Frames (trial={trial_id}, gesture={gesture})", fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            grid_path = os.path.join(save_dir, f"{clip_name}_grid.png")
            fig.savefig(grid_path, dpi=150)
            plt.close(fig)
            print(f"Saved frames grid to: {grid_path}")

    # Signals
    if len(modality_keys) > 0:
        n_mods = len(modality_keys)
        fig = plt.figure(figsize=(12, 2.5 * n_mods))
        for i, m in enumerate(modality_keys, 1):
            x = batch[m][idx]
            T_valid = get_valid_T_for_mod(m)
            x = x[:T_valid]
            if x.ndim != 2:
                print(f"Skipping modality '{m}' with shape {tuple(x.shape)}")
                continue
            T_cur, Fdim = x.shape
            tt = np.arange(T_cur)
            ax = plt.subplot(n_mods, 1, i)
            F_plot = min(Fdim, max_features_per_mod)
            for f in range(F_plot):
                ax.plot(tt, x[:, f].detach().cpu().numpy(), linewidth=1.0)
            ax.set_title(f"{m} (T={T_cur}, F={Fdim})", fontsize=10)
            ax.set_xlabel("t"); ax.set_ylabel("value")
            ax.grid(True, linewidth=0.5, alpha=0.5)
        fig.suptitle(f"Signals (trial={trial_id}, gesture={gesture})", fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        ts_path = os.path.join(save_dir, f"{clip_name}_signals.png")
        fig.savefig(ts_path, dpi=150)
        plt.close(fig)
        print(f"Saved signals figure to: {ts_path}")

    # Animation (optional)
    if make_gif and "images" in batch:
        try:
            plot_mods = modality_keys[:2]
            n_sig = len(plot_mods)

            img_ratio = 4
            height_ratios = [img_ratio] + [1]*n_sig
            fig = plt.figure(figsize=(8, 2.2 * (img_ratio/2 + n_sig)))
            gs  = fig.add_gridspec(nrows=1+n_sig, ncols=1, height_ratios=height_ratios)

            imgs = batch["images"][idx]
            if "images_mask" in batch:
                T_anim = int(batch["images_mask"][idx].sum().item())
                imgs = imgs[:T_anim]
            else:
                T_anim = imgs.shape[0]

            # make frames bigger ONLY for GIF
            scale = 2.0
            _, H, W = imgs.shape[1:]
            up_h, up_w = int(round(H*scale)), int(round(W*scale))
            imgs_up = torch.nn.functional.interpolate(
                imgs.float(), size=(up_h, up_w), mode="bilinear", align_corners=False
            )

            ax_img   = fig.add_subplot(gs[0, 0])
            frame0   = imgs_up[0].detach().cpu().numpy().transpose(1, 2, 0)
            im_artist = ax_img.imshow(frame0)
            ax_img.set_title(f"Video (trial={trial_id}, gesture={gesture})", fontsize=12)
            ax_img.axis("off")

            lines = []
            data_cache = []
            for i, m in enumerate(plot_mods, start=1):
                ax = fig.add_subplot(gs[i, 0])
                x = batch[m][idx]
                T_valid = get_valid_T_for_mod(m)
                x = x[:T_valid]
                data_cache.append(x)
                mod_lines = []
                Fdim = x.shape[1] if x.ndim == 2 else 0
                F_plot = min(Fdim, max_features_per_mod)
                for _ in range(F_plot):
                    (ln,) = ax.plot([], [], linewidth=1.0)
                    mod_lines.append(ln)
                ax.set_xlim(0, max(T_valid - 1, 1))
                ymin, ymax = (float(x.min()), float(x.max())) if T_valid > 0 else (-1.0, 1.0)
                if ymin == ymax:
                    ymin -= 1.0; ymax += 1.0
                ax.set_ylim(ymin, ymax)
                ax.set_title(f"{m} (F={Fdim})", fontsize=10)
                ax.grid(True, linewidth=0.5, alpha=0.5)
                lines.append(mod_lines)

            def update(t):
                fimg = imgs_up[t].detach().cpu().numpy().transpose(1, 2, 0)
                im_artist.set_data(fimg)
                for mi, x in enumerate(data_cache):
                    T_valid = x.shape[0]
                    tt = np.arange(min(t+1, T_valid))
                    for fi, ln in enumerate(lines[mi]):
                        ln.set_data(tt, x[:tt.size, fi].detach().cpu().numpy())
                return [im_artist] + [ln for group in lines for ln in group]

            ani = FuncAnimation(fig, update, frames=range(T_anim), interval=1000//gif_fps, blit=False)
            gif_path = os.path.join(save_dir, f"{clip_name}_anim.gif")
            print(f"Saving animation GIF to: {gif_path}")
            ani.save(gif_path, writer=PillowWriter(fps=gif_fps), dpi=120)
            plt.close(fig)
            print(f"Saved animation GIF to: {gif_path}")
        except Exception as e:
            print(f"Animation skipped due to error: {e}")


# ----------------------- test block -----------------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # ROOT_DIR = "/standard/UVA-DSA/MIDAS/Organized/final_data"
    ROOT_DIR = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/Suturing/Processed/"

    total_samples = 0

    trials =  [    
    "S105_T1",
    "S106_T1",
    "S106_T2",
    "S112_T1",
    "S112_T2",
    "S116_T1",
    "S116_T2",
    "S116_T4",
    "S116_T5",
    "S118_T1",
    "S200_T1",
    "S201_T1",
    "S201_T2",
    "S202_T1",
    "S203_T1",
    "S204_T1",
    "S209_T2",
    "S210_T1",
    "S214_T1",
    "S214_T4",
    "S214_T6",
    "S215_T3",
    "S215_T4",
    "S217_T2",
    "S217_T3",
    "S217_T4",
    "S218_T1",
    "S219_T1"
    ]
    csvs = []

    for trial_id in trials:
        # gather CSVs
        csvs.append(f"{ROOT_DIR}/{trial_id}/synched_data/final_annotation_{trial_id}.csv")

    dataset = MultimodalGestureDataset(
        csv_paths=csvs,
        clip_len=-1,              # or -1 for full gesture
        step=1,
        sample_rate=30,           # 30Hz
        ignore_clutch=False,
        clutch_pressed_value=0,

        include_modalities=["trakstar"],
        modality_selections={
            "trakstar": [
                "trakstar_sensor_0_*x", "trakstar_sensor_0_*y", "trakstar_sensor_0_azimuth",
                "trakstar_sensor_2_*x", "trakstar_sensor_2_*y", "trakstar_sensor_2_azimuth",
            ],
            # "images": ["resnet"],  # optional
        },
        # modality_exclude={"trakstar": ["*elevation*", "*roll*"]},
        normalize=True,
        classes_to_allow={
            0: "No Gesture",
            1: "Cinch Loop Down (Barbed Suture)",
            2: "Cold Cut",
            3: "Cold Hook",
            4: "Cut suture",
            5: "Grasp Long End of Suture",
            6: "Loop around Free Arm-DoubleLoop",
            7: "Loop around Free Arm-SingleLoop",
            8: "One-hand Spread",
            9: "Orient Needle",
            10: "Pass Needle Through Loop",
            11: "Pedicalize",
            12: "Peel",
            13: "Pull Needle",
            14: "Pull Suture into Slip Knot and Cinch",
            15: "Pull Suture-BothHands",
            16: "Pull Suture-Fulcrum",
            17: "Pull Suture-OneHand",
            18: "Push Needle",
            19: "Reduce",
            20: "Square Knot-1: Grasp and Pull Short End Through Loop",
            21: "Square Knot-2",
            22: "Target Needle",
            23: "Two-hand Spread",
        },

        video_root=None,
        video_pattern="{trial_id}.mp4",
        video_output_size=(224, 224),
    )

    print("Dataset stats:", dataset._get_class_stats())

    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=MultimodalGestureDataset.collate_fn)

    batch = next(iter(loader))
    print("\nBatch stats:")
    print_batch_stats(batch)
    print(f"\nDataset has {len(dataset)} samples, {dataset.num_classes} classes: {dataset.classes}")

    total_samples += len(dataset)
    # visualize_dataset_sample(batch, batch_idx=0, save_dir="./viz_refactor")
