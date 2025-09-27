# multimodal_gesture_dataset_selective.py
from __future__ import annotations
import os
from typing import Dict, List, Optional, Sequence, Tuple, Callable
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import fnmatch

# --- Add these imports at the top (or ensure they exist) ---
import glob
from pathlib import Path

# --- Add these helpers above the class (or make them @staticmethods) ---
def _gather_csvs_from_dir(dir_path: str, glob_pattern: str = "*.csv", recursive: bool = False) -> list[str]:
    pattern = str(Path(dir_path) / ("**/" + glob_pattern if recursive else glob_pattern))
    return sorted(glob.glob(pattern, recursive=recursive))

import re

def _infer_trial_id(path: str) -> str:
    p = Path(path)
    name = p.name  # e.g., final_annotation_t2.csv
    # Prefer explicit t# in filename
    m = re.search(r'_(t\d+)\b', name, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # If the immediate parent is a generic "synched_data" (or similar), use grandparent
    parent = p.parent.name.lower()
    if parent in {"synched_data", "synced_data", "synched", "synced"} and p.parent.parent.name:
        return p.parent.parent.name
    # Else use parent folder; fallback to stem
    return p.parent.name or p.stem

def _read_and_tag_csv(path: str, fillna_value: float | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_csv"] = str(path)
    df["trial_id"] = _infer_trial_id(path)
    if fillna_value is not None:
        df = df.fillna(fillna_value)
    return df


class MultimodalGestureDataset(Dataset):

    DEFAULT_MODALITY_RULES: Dict[str, Callable[[str], bool]] = {
        "trakstar":    lambda c: c.startswith("trakstar_sensor_"),
        "sw_left":     lambda c: c.startswith("sw_left_"),
        "sw_right":    lambda c: c.startswith("sw_right_"),
        "console":     lambda c: c.startswith("console_"),
        "raven_field": lambda c: c.startswith("raven_field."),
        "pedals":      lambda c: c.startswith("Pedal "),
        # "timestamps": lambda c: c.endswith("_time_ns") or c.endswith("_delta_ns"),
    }

    def __init__(
        self,
        # One of these:
        csv_path: str | None = None,              # old behavior (single file)
        dir_path: str | None = None,              # NEW: directory containing many CSVs
        csv_paths: Optional[Sequence[str]] = None,# NEW: explicit list of CSV paths
        glob_pattern: str = "*.csv",              # NEW: which files to pick in dir
        recursive: bool = False,                  # NEW: recurse into subfolders

        clip_len: int = 32,
        step: int = 8,
        include_modalities: Optional[Sequence[str]] = None,
        drop_short_segments: bool = True,
        fillna_value: float = 0.0,
        normalize: bool = False,
        normalization_stats: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
        image_dir: Optional[str] = None,
        image_pattern: str = "{:06d}.jpg",
        image_loader: Optional[callable] = None,
        class_map: Optional[Dict[str, int]] = None,
        modality_rules: Optional[Dict[str, Callable[[str], bool]]] = None,
        modality_selections: Optional[Dict[str, Sequence[str]]] = None,
        modality_exclude: Optional[Dict[str, Sequence[str]]] = None,
        seed: int = 0,
        windowing: str = "fixed",   # keep your earlier extension if you added it
    ):
        super().__init__()

        # ---------- Load one or many CSVs ----------
        collected_paths: list[str] = []
        if csv_paths:
            collected_paths = list(csv_paths)
        elif dir_path:
            collected_paths = _gather_csvs_from_dir(dir_path, glob_pattern, recursive)
            if not collected_paths:
                raise FileNotFoundError(f"No CSVs found in '{dir_path}' matching '{glob_pattern}' (recursive={recursive}).")
        elif csv_path:
            collected_paths = [csv_path]
        else:
            raise ValueError("Provide one of: csv_path, dir_path, or csv_paths.")

        dfs = [_read_and_tag_csv(p, fillna_value=None) for p in collected_paths]
        self.df = pd.concat(dfs, axis=0, ignore_index=True)

        sort_keys = ["trial_id", "obs_frame_idx", "source_csv"]
        self.df = self.df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
        # (stable mergesort helps maintain per-file order when keys tie)


        # Now handle NaNs globally (after union of columns)
        if fillna_value is not None:
            self.df = self.df.fillna(fillna_value)

        # Ensure required columns
        for col in ["obs_frame_idx", "gesture_code"]:
            if col not in self.df.columns:
                raise ValueError(f"CSV(s) missing required column: {col}")

        # Sort by (trial_id, obs_frame_idx) to keep per-trial order
        sort_keys = ["trial_id", "obs_frame_idx"] if "trial_id" in self.df.columns else ["obs_frame_idx"]
        self.df = self.df.sort_values(sort_keys).reset_index(drop=True)

        # ---------- Rest of your existing init follows ----------
        self.class_map = dict(class_map) if class_map else {
            c: i for i, c in enumerate(sorted(self.df["gesture_code"].dropna().unique().tolist()))
        }
        self.modality_rules = dict(self.DEFAULT_MODALITY_RULES if modality_rules is None else modality_rules)
        self.include_modalities = list(include_modalities or self.modality_rules.keys())
        self.modality_selections = modality_selections or {}
        self.modality_exclude = modality_exclude or {}

        self.modality_cols = {}
        all_cols = list(self.df.columns)

        def match_any(patterns: Sequence[str], col: str) -> bool:
            return any(fnmatch.fnmatch(col, p) for p in patterns)

        for m in self.include_modalities:
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

        # Gesture runs per trial (keep segments separated across trials)
        self.df["_run_boundary"] = (
            (self.df["gesture_code"] != self.df["gesture_code"].shift(1)) |
            (self.df["trial_id"]     != self.df["trial_id"].shift(1))     |
            (self.df["source_csv"]   != self.df["source_csv"].shift(1))
        )
        self.df["_run_id"] = self.df["_run_boundary"].cumsum()


        self.clip_len = int(clip_len)
        self.step = int(step)
        self.drop_short_segments = bool(drop_short_segments)
        self.windowing = windowing  # if you kept the full/fixed option
        self.samples = []
        self._make_windows()  # unchanged except it groups by _run_id

        self.normalize = normalize
        self.norm_stats = normalization_stats or {}
        if self.normalize and not self.norm_stats:
            self._compute_norm_stats()

        self.image_dir = image_dir
        self.image_pattern = image_pattern
        self.image_loader = image_loader
        self.rng = np.random.default_rng(seed)


    def _make_windows(self):
        self.samples.clear()
        for _, seg in self.df.groupby("_run_id", sort=False):
            label_str = seg["gesture_code"].iloc[0]
            seg_len = len(seg)
            base = int(seg.index[0])

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

    def __len__(self) -> int:
        return len(self.samples)

    def _get_modality_tensor(self, start: int, end: int, modality: str) -> torch.Tensor:
        cols = self.modality_cols[modality]
        X = self.df.loc[start:end - 1, cols].to_numpy(dtype=np.float32)
        if self.normalize:
            mean, std = self.norm_stats[modality]
            X = (X - mean) / std
        return torch.from_numpy(X)  # [T, F]

    def _maybe_load_images(self, obs_idx_seq: np.ndarray) -> Optional[torch.Tensor]:
        if self.image_dir is None or self.image_loader is None:
            return None
        imgs = []
        for fid in obs_idx_seq:
            path = os.path.join(self.image_dir, self.image_pattern.format(int(fid)))
            imgs.append(self.image_loader(path))
        return torch.stack(imgs, dim=0)

    def __getitem__(self, i: int):
        s = self.samples[i]
        start, end = s["start"], s["end"]
        code = s["gesture_code"]
        label = self.class_map[code]

        item = {}
        for m in self.modality_cols.keys():
            item[m] = self._get_modality_tensor(start, end, m)  # [T, F]

        obs_idx = self.df.loc[start:end - 1, "obs_frame_idx"].to_numpy(dtype=np.int64)
        item["obs_frame_idx"] = torch.from_numpy(obs_idx)
        item["label"] = torch.tensor(label, dtype=torch.long)
        item["gesture_code"] = code

        imgs = self._maybe_load_images(obs_idx)
        if imgs is not None:
            item["images"] = imgs
        return item

    @property
    def num_classes(self) -> int:
        return len(self.class_map)

    @property
    def classes(self) -> List[str]:
        inv = {v: k for k, v in self.class_map.items()}
        return [inv[i] for i in range(len(inv))]

    @staticmethod
    def collate_fn(batch):
        out = {}
        keys = list(batch[0].keys())
        tensor_modalities = [k for k in keys if isinstance(batch[0][k], torch.Tensor)
                             and k not in ("label", "obs_frame_idx", "images")]
        for m in tensor_modalities:
            out[m] = torch.stack([b[m] for b in batch], dim=0)  # [B, T, F]
        out["label"] = torch.stack([b["label"] for b in batch], dim=0)
        out["obs_frame_idx"] = torch.stack([b["obs_frame_idx"] for b in batch], dim=0)
        out["gesture_code"] = [b["gesture_code"] for b in batch]
        if "images" in batch[0]:
            out["images"] = torch.stack([b["images"] for b in batch], dim=0)
        return out


# test block

from torch.utils.data import DataLoader

if __name__ == "__main__":

    train_files = [
    "G:/Research/MIDAS/Organized/09-18-25/hamid/t1/synched_data/final_annotation_t1.csv",
    "G:/Research/MIDAS/Organized/09-18-25/hamid/t2/synched_data/final_annotation_t2.csv",
    "G:/Research/MIDAS/Organized/09-18-25/hamid/t3/synched_data/final_annotation_t3.csv",
    "G:/Research/MIDAS/Organized/09-18-25/hamid/t4/synched_data/final_annotation_t4.csv",
    "G:/Research/MIDAS/Organized/09-18-25/hamid/t5/synched_data/final_annotation_t5.csv"
    ]

    test_files = [
    "G:/Research/MIDAS/Organized/09-18-25/hamid/t6/synched_data/final_annotation_t6.csv",
    "G:/Research/MIDAS/Organized/09-18-25/hamid/t7/synched_data/final_annotation_t7.csv"
    ]
    
    train_dataset = MultimodalGestureDataset(
        csv_paths=train_files,
        clip_len=30,
        step=8,
        include_modalities=["trakstar", "sw_left", "console"],

        # Allowlist patterns (fnmatch)
        modality_selections={
            # Only sensors 0 and 2, and only x/y/azimuth features
            "trakstar": [
                "trakstar_sensor_0_*x", "trakstar_sensor_0_*y", "trakstar_sensor_0_azimuth",
                "trakstar_sensor_2_*x", "trakstar_sensor_2_*y", "trakstar_sensor_2_azimuth",
            ],
            # Only smartwatch-left x and y (drop z)
            "sw_left": ["sw_left_x", "sw_left_y"],
            # Only console position, not rotation
            "console": ["console_pos*"],
        },

        # Denylist patterns if you want to drop something after allowlist
        modality_exclude={
            "trakstar": ["*elevation*", "*roll*"],  # just in case the allowlist was broad
        },

        normalize=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                        collate_fn=MultimodalGestureDataset.collate_fn)
    batch = next(iter(train_loader))

    # print more details about batch content
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: dtype={v.dtype}, shape={tuple(v.shape)}, "
                f"min={v.min().item():.3f}, max={v.max().item():.3f}")
        elif k == "gesture_code":
            # show a few sample codes for sanity
            print(f"{k}: {len(v)} items -> {v[:5]}{' ...' if len(v) > 5 else ''}")
        elif isinstance(v, list):
            print(f"{k}: list of {len(v)} items, first 3: {v[:3]}")
        else:
            print(f"{k}: type={type(v)}, value={v}")

    # iterate through all samples to check for errors
    print(f"\nTrain Dataset has {len(train_dataset)} samples, {train_dataset.num_classes} classes: {train_dataset.classes}")

    # use loader to iterate through all samples to check for errors
    for i, sample in enumerate(train_loader):
        pass

    print(f"Iterated through all {i+1} batches from Train DataLoader successfully.")

    # test dataset
    test_dataset = MultimodalGestureDataset(
        csv_paths=test_files,
        clip_len=30,
        step=8,
        include_modalities=["trakstar", "sw_left", "console"],
        normalize=True,
    )

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                        collate_fn=MultimodalGestureDataset.collate_fn)
    test_batch = next(iter(test_loader))

    # print more details about test batch content
    for k, v in test_batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: dtype={v.dtype}, shape={tuple(v.shape)}, "
                f"min={v.min().item():.3f}, max={v.max().item():.3f}")
        elif k == "gesture_code":
            # show a few sample codes for sanity
            print(f"{k}: {len(v)} items -> {v[:5]}{' ...' if len(v) > 5 else ''}")
        elif isinstance(v, list):
            print(f"{k}: list of {len(v)} items, first 3: {v[:3]}")
        else:
            print(f"{k}: type={type(v)}, value={v}")

    # iterate through all samples to check for errors
    print(f"\nTest Dataset has {len(test_dataset)} samples, {test_dataset.num_classes} classes: {test_dataset.classes}")

    # use loader to iterate through all samples to check for errors
    for i, sample in enumerate(test_loader):
        pass

    print(f"Iterated through all {i+1} batches from Test DataLoader successfully.")
