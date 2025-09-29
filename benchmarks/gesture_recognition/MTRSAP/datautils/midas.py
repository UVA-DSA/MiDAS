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
import re


# --- Add these helpers above the class (or make them @staticmethods) ---
def _gather_csvs_from_dir(dir_path: str, glob_pattern: str = "*.csv", recursive: bool = False) -> list[str]:
    pattern = str(Path(dir_path) / ("**/" + glob_pattern if recursive else glob_pattern))
    return sorted(glob.glob(pattern, recursive=recursive))


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
        "raven":       lambda c: c.startswith("raven_"),
        "pedals":      lambda c: c.startswith("Pedal "),
        # "timestamps": lambda c: c.endswith("_time_ns") or c.endswith("_delta_ns"),
    }

    def __init__(
        self,
        # One of these:
        base_path: str | None = None,             # used for appending csv_paths (keep as-is if you rely on it)
        csv_path: str | None = None,              # single file
        dir_path: str | None = None,              # directory containing many CSVs
        csv_paths: Optional[Sequence[str]] = None,# explicit list of CSV paths
        glob_pattern: str = "*.csv",              # pattern for dir scan
        recursive: bool = False,                  # recurse into subfolders

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
        windowing: str = "fixed",   # kept for back-compat; clip_len takes precedence now
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

        # NOTE: This block preserves your existing behavior; if you pass absolute paths,
        #       prefer: dfs = [_read_and_tag_csv(p, fillna_value=None) for p in collected_paths]
        #       but we keep your current pattern to avoid breaking existing pipelines.
        if base_path is not None:
            dfs = [_read_and_tag_csv(base_path + p + "/synched_data/final_annotation_" + p + ".csv", fillna_value=None)
                   for p in collected_paths]
        else:
            dfs = [_read_and_tag_csv(p, fillna_value=None) for p in collected_paths]

        self.df = pd.concat(dfs, axis=0, ignore_index=True)

        sort_keys = ["trial_id", "obs_frame_idx", "source_csv"]
        self.df = self.df.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
        # (stable mergesort helps maintain per-file order when keys tie)

        # Handle NaNs globally (after union of columns)
        if fillna_value is not None:
            self.df = self.df.fillna(fillna_value)

        # Ensure required columns
        for col in ["obs_frame_idx", "gesture_code"]:
            if col not in self.df.columns:
                raise ValueError(f"CSV(s) missing required column: {col}")

        # Sort by (trial_id, obs_frame_idx) to keep per-trial order
        sort_keys = ["trial_id", "obs_frame_idx"] if "trial_id" in self.df.columns else ["obs_frame_idx"]
        self.df = self.df.sort_values(sort_keys).reset_index(drop=True)

        # ---------- Build class map & modality columns ----------
        self.class_map = dict(class_map) if class_map else {
            c: i for i, c in enumerate(sorted(self.df["gesture_code"].dropna().unique().tolist()))
        }
        self.modality_rules = dict(self.DEFAULT_MODALITY_RULES if modality_rules is None else modality_rules)
        self.include_modalities = list(include_modalities or self.modality_rules.keys())
        self.modality_selections = modality_selections or {}
        self.modality_exclude = modality_exclude or {}

        self.modality_cols: Dict[str, List[str]] = {}
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

        # ---------- Gesture runs per trial (separate trials/files) ----------
        self.df["_run_boundary"] = (
            (self.df["gesture_code"] != self.df["gesture_code"].shift(1)) |
            (self.df["trial_id"]     != self.df["trial_id"].shift(1))     |
            (self.df["source_csv"]   != self.df["source_csv"].shift(1))
        )
        self.df["_run_id"] = self.df["_run_boundary"].cumsum()

        # ---------- Windowing ----------
        self.clip_len = int(clip_len)
        self.step = int(step)
        self.full_clip = (self.clip_len == -1)   # NEW: full-gesture mode flag
        self.drop_short_segments = bool(drop_short_segments)
        self.windowing = windowing  # kept for back-compat; ignored if full_clip is True
        self.samples: List[Dict] = []
        self._make_windows()  # builds samples

        # ---------- Normalization ----------
        self.normalize = normalize
        self.norm_stats = normalization_stats or {}
        if self.normalize and not self.norm_stats:
            self._compute_norm_stats()

        # ---------- Optional images ----------
        self.image_dir = image_dir
        self.image_pattern = image_pattern
        self.image_loader = image_loader

        # ---------- RNG ----------
        self.rng = np.random.default_rng(seed)

    # ---------------- Window builder ----------------
    def _make_windows(self):
        self.samples.clear()
        for _, seg in self.df.groupby("_run_id", sort=False):
            label_str = seg["gesture_code"].iloc[0]
            seg_len = len(seg)
            base = int(seg.index[0])

            if self.full_clip:
                # One sample = entire gesture segment
                self.samples.append({
                    "start": base,
                    "end": base + seg_len,
                    "gesture_code": label_str
                })
                continue

            # ---- fixed-length windows (existing behavior) ----
            if seg_len < self.clip_len:
                if self.drop_short_segments:
                    continue

            max_start = seg_len - self.clip_len
            if max_start < 0:
                continue

            for s in range(0, max_start + 1, self.step):
                self.samples.append({
                    "start": base + s,
                    "end": base + s + self.clip_len,
                    "gesture_code": label_str
                })

    # ---------------- Normalization stats ----------------
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

    # ---------------- Dataset API ----------------
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

        item: Dict[str, torch.Tensor | str] = {}
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

    # ---------------- Stats helpers ----------------
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

    # ---------------- Padding helpers (for variable-length) ----------------
    @staticmethod
    def _pad_and_stack_2d(seqs: List[torch.Tensor], pad_value: float = 0.0):
        """
        Pad a list of [T, F] tensors to [B, T_max, F] and return (padded, mask).
        mask: [B, T_max] with True for valid timesteps.
        """
        lengths = [s.shape[0] for s in seqs]
        B = len(seqs)
        T_max = max(lengths)
        F = seqs[0].shape[1]
        device = seqs[0].device
        out = seqs[0].new_full((B, T_max, F), pad_value)
        mask = torch.zeros(B, T_max, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            t = s.shape[0]
            out[i, :t] = s
            mask[i, :t] = True
        return out, mask

    @staticmethod
    def _pad_and_stack_1d(seqs: List[torch.Tensor], pad_value: int = 0):
        """
        Pad a list of [T] tensors to [B, T_max] and return (padded, mask).
        """
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

    # ---------------- Collate ----------------
    @staticmethod
    def collate_fn(batch):
        """
        Fixed-length clips -> stack as before.
        Full-gesture (variable-length) -> pad per modality and return attention masks.
        """
        out: Dict[str, torch.Tensor] = {}
        keys = list(batch[0].keys())

        # Identify tensor modalities (exclude label/idx/images)
        tensor_modalities = [k for k in keys
                             if isinstance(batch[0][k], torch.Tensor)
                             and k not in ("label", "obs_frame_idx", "images")]

        # Detect variable length (any modality with differing T within the batch)
        def lengths_equal_for_mod(m):
            lens = [b[m].shape[0] for b in batch]
            return len(set(lens)) == 1

        variable_length = False
        if tensor_modalities:
            variable_length = any(not lengths_equal_for_mod(m) for m in tensor_modalities)

        if not variable_length:
            # ---- fixed-length: original behavior ----
            for m in tensor_modalities:
                out[m] = torch.stack([b[m] for b in batch], dim=0)  # [B, T, F]
            out["label"] = torch.stack([b["label"] for b in batch], dim=0)  # [B]
            out["obs_frame_idx"] = torch.stack([b["obs_frame_idx"] for b in batch], dim=0)  # [B, T]
            out["gesture_code"] = [b["gesture_code"] for b in batch]
            if "images" in batch[0]:
                out["images"] = torch.stack([b["images"] for b in batch], dim=0)  # [B, T, C, H, W]
            return out

        # ---- variable-length (full-gesture) ----
        attention_mask: Dict[str, torch.Tensor] = {}
        for m in tensor_modalities:
            seqs = [b[m] for b in batch]                     # list of [T, F]
            padded, mask = MultimodalGestureDataset._pad_and_stack_2d(seqs, pad_value=0.0)
            out[m] = padded                                  # [B, T_max, F]
            attention_mask[m] = mask                         # [B, T_max] (True=valid)

        # Pad obs_frame_idx similarly
        obs_seqs = [b["obs_frame_idx"] for b in batch]       # list of [T]
        obs_padded, obs_mask = MultimodalGestureDataset._pad_and_stack_1d(obs_seqs, pad_value=0)
        out["obs_frame_idx"] = obs_padded                    # [B, T_max]
        out["obs_mask"] = obs_mask                           # [B, T_max]

        out["label"] = torch.stack([b["label"] for b in batch], dim=0)          # [B]
        out["gesture_code"] = [b["gesture_code"] for b in batch]

        # Package per-modality masks (handy for per-modality encoders)
        out["attention_mask"] = attention_mask               # dict: {mod: [B, T_max]}

        # Images: add padding if you need variable-length visuals later
        return out


# ---------------- test block ----------------
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ROOT_DIR = "/standard/UVA-DSA/MIDAS/Organized/09-18-25/hamid"

    train_files = [
        f"{ROOT_DIR}/t1/synched_data/final_annotation_t1.csv",
        f"{ROOT_DIR}/t2/synched_data/final_annotation_t2.csv",
        f"{ROOT_DIR}/t3/synched_data/final_annotation_t3.csv",
        f"{ROOT_DIR}/t4/synched_data/final_annotation_t4.csv",
        f"{ROOT_DIR}/t5/synched_data/final_annotation_t5.csv"
    ]

    test_files = [
        f"{ROOT_DIR}/t6/synched_data/final_annotation_t6.csv",
        f"{ROOT_DIR}/t7/synched_data/final_annotation_t7.csv"
    ]

    # ----- Fixed-length example (unchanged behavior) -----
    train_dataset = MultimodalGestureDataset(
        csv_paths=train_files,
        clip_len=30,  # fixed 1s @ 30Hz
        step=8,
        include_modalities=["trakstar", "sw_left", "console"],
        modality_selections={
            "trakstar": [
                "trakstar_sensor_0_*x", "trakstar_sensor_0_*y", "trakstar_sensor_0_azimuth",
                "trakstar_sensor_2_*x", "trakstar_sensor_2_*y", "trakstar_sensor_2_azimuth",
            ],
            "sw_left": ["sw_left_x", "sw_left_y"],
            "console": ["console_pos*"],
        },
        modality_exclude={
            "trakstar": ["*elevation*", "*roll*"],
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
            print(f"{k}: {len(v)} items -> {v[:5]}{' ...' if len(v) > 5 else ''}")
        elif isinstance(v, list):
            print(f"{k}: list of {len(v)} items, first 3: {v[:3]}")
        else:
            print(f"{k}: type={type(v)}, value={v}")

    print(f"\nTrain Dataset has {len(train_dataset)} samples, {train_dataset.num_classes} classes: {train_dataset.classes}")

    # iterate through all batches once
    for i, _ in enumerate(train_loader):
        pass
    print(f"Iterated through all {i+1} batches from Train DataLoader successfully.")

    # ----- Full-gesture example (clip_len == -1) -----
    full_dataset = MultimodalGestureDataset(
        csv_paths=test_files,
        clip_len=-1,  # <-- entire gesture segments
        include_modalities=["trakstar", "sw_left", "console"],
        normalize=True,
    )

    full_loader = DataLoader(full_dataset, batch_size=8, shuffle=False,
                             collate_fn=MultimodalGestureDataset.collate_fn)
    full_batch = next(iter(full_loader))

    print("\n[Full-gesture mode] Keys:", full_batch.keys())
    for m, x in full_batch.items():
        if isinstance(x, torch.Tensor):
            print(m, x.shape)
    if "attention_mask" in full_batch:
        print("Per-modality masks:", {k: v.shape for k, v in full_batch["attention_mask"].items()})

    print(f"\nFull-gesture Dataset has {len(full_dataset)} samples, {full_dataset.num_classes} classes: {full_dataset.classes}")
    for i, _ in enumerate(full_loader):
        pass
    print(f"Iterated through all {i+1} batches from Full-gesture DataLoader successfully.")
