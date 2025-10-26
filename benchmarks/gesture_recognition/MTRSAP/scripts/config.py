# config.py
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import random
import math

# ------------------ Feature candidate lists ------------------
trakstar_candidates = [
    'trakstar_sensor_0_azimuth','trakstar_sensor_1_azimuth','trakstar_sensor_2_azimuth','trakstar_sensor_3_azimuth',
    'trakstar_sensor_0_elevation','trakstar_sensor_1_elevation','trakstar_sensor_2_elevation','trakstar_sensor_3_elevation',
    'trakstar_sensor_0_roll','trakstar_sensor_1_roll','trakstar_sensor_2_roll','trakstar_sensor_3_roll',
    'trakstar_sensor_0_x','trakstar_sensor_1_x','trakstar_sensor_2_x','trakstar_sensor_3_x',
    'trakstar_sensor_0_y','trakstar_sensor_1_y','trakstar_sensor_2_y','trakstar_sensor_3_y',
    'trakstar_sensor_0_z','trakstar_sensor_1_z','trakstar_sensor_2_z','trakstar_sensor_3_z'
]

console_candidates = [
    'console_pos0','console_pos1','console_pos2','console_pos3','console_pos4','console_pos5',
    'console_rot0','console_rot1','console_rot2','console_rot3','console_rot4','console_rot5',
    'console_aux0','console_aux1','console_pedal'
]

raven_candidates = [
    'raven_field.pos0','raven_field.pos1','raven_field.pos2','raven_field.pos3','raven_field.pos4','raven_field.pos5',
    'raven_field.ori0','raven_field.ori1','raven_field.ori2','raven_field.ori3','raven_field.ori4','raven_field.ori5',
    'raven_field.ori6','raven_field.ori7','raven_field.ori8','raven_field.ori9','raven_field.ori10','raven_field.ori11',
    'raven_field.ori12','raven_field.ori13','raven_field.ori14','raven_field.ori15','raven_field.ori16','raven_field.ori17'
]

sw_left_candidates  = ['sw_left_x','sw_left_y','sw_left_z']
sw_right_candidates = ['sw_right_x','sw_right_y','sw_right_z']

# ------------------ Model-side configs ------------------
@dataclass
class ModalityCfg:
    name: str
    in_dim: int               # derived from dataloader selections (len of selected columns)

@dataclass
class ModelCfg:
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    num_classes: int = 8
    include_modalities: List[str] = field(default_factory=list)       # derived
    fusion: str = "concat_tokens"                                     # "concat_tokens", "late_gated", "cross_attend"
    modality_dropout_p: float = 0.0
    modalities: Dict[str, ModalityCfg] = field(default_factory=dict)  # derived {mod: ModalityCfg}

@dataclass
class MultiMTRSAPModelCfg:
    modalities: List[str] = field(default_factory=list)       # derived
    in_dims: List[int] = field(default_factory=list)          # derived
    num_classes: int = 8
    tcn_channels: List[int] = field(default_factory=lambda: [128, 128, 256])
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    seq_to_one: bool = False
    causal_tcn: bool = True
 
# ------------------ Dataloader (single source of truth) ------------------
dataloader_params: Dict = {
    "experiment_name": "hamid_console_DS_NOCLUTCH_only",
    "base_path": "/standard/UVA-DSA/MIDAS/Organized/final_data/",
    "batch_size": 4,
    "sample_rate": 10,
    "ignore_clutch": True,  # whether to ignore clutching periods in data
    "clutch_pressed_value": 0, # value indicating clutch pressed in console_aux1
    "return_images": False,             # actually load frames

    # List **all** trials here once:
    # "all_trials": ["bt1","bt2","bt3","bt4","bt5"], # bootcamp data
    "all_trials": ["t1","t3","t4","t5", "t6","t7"], # hamid data

    # Cross-validation control:
    #   scheme: "leave_one_out" | "group_k_fold"
    #   k: for group_k_fold (ignored for leave_one_out)
    #   val_ratio: fraction of the non-test trials used for validation (0 < val_ratio < 1)
    #   seed: for deterministic shuffles
    #   shuffle: whether to shuffle trial order before splitting
    "cv": {
        "scheme": "leave_one_out",   # or "group_k_fold"
        "k": 5,                      # used only if scheme == "group_k_fold"
        "val_ratio": 0.25,           # from the remaining (non-test) trials
        "seed": 42,
        "shuffle": True
    },

    # Active modalities for this experiment (order matters for reporting):
    # "modalities": ["console"],
    "modalities": ["console"],
    # "modalities": ["sw_left","sw_right"],
    # "modalities": ["raven","console"],
    # "modalities": ["trakstar"],
    # Column selections per modality
    "selections": {
        # "trakstar": trakstar_candidates,
        # "trakstar": ['trakstar_sensor_0_x','trakstar_sensor_0_y','trakstar_sensor_0_z',
        #                 'trakstar_sensor_1_x','trakstar_sensor_1_y','trakstar_sensor_1_z',
        #                 'trakstar_sensor_2_x','trakstar_sensor_2_y','trakstar_sensor_2_z',
        #              'trakstar_sensor_3_x','trakstar_sensor_3_y','trakstar_sensor_3_z'],
        # "sw_left":  sw_left_candidates,
        # "sw_right": sw_right_candidates,
        # "raven":    raven_candidates,
        "console":  console_candidates,
    },

    # Windowing
    "observation_window": 10,   # 1s at 30 Hz; use -1 for full-gesture mode
    "step": 1,

    # Labels / human-readable map
    "keysteps": {
        "S1": "Approach peg",
        "S2": "Align & grasp",
        "S3": "Lift peg",
        "S4": "Transfer peg - Get together",
        "S5": "Transfer peg - Exchange",
        "S6": "Approach pole",
        "S7": "Align & place",
        "Idle": "Idle"
    },

    # (optional) normalization stats populated after computing on TRAIN
    "train_class_stats": {},
    "val_class_stats": {}
}

# ------------------ Other hyperparams ------------------
learning_params = {
    "lr": 1e-5,
    "epochs": 1,
    "weight_decay": 1e-5,
    "patience": 10,
    "lr_drop": 20,
    "best_chkpoint": "./checkpoints/job_xxx/val_best_model.pt",
}

tcn_model_params = {
    "encoder_params": {
        "in_channels": 128,
        "kernel_size": 13,
        "out_channels": 64,
    },
    "decoder_params": {
        "in_channels": 60,
        "kernel_size": 31,
        "out_channels": 60
    }
}

transformer_params = {
    "d_model": 64,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "batch_first": True,
    "sample_rate": 48000,
    "n_mels": 64,
    "hop_length": 512,
    "n_fft": 1024,
    "resnet_dim": 2048,
}

# ------------------ CV helpers ------------------
def _split_train_val(remaining: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    """Split remaining trials into train/val by a ratio (deterministic)."""
    rnd = random.Random(seed)
    rem = list(remaining)
    rnd.shuffle(rem)
    val_n = max(1, int(round(len(rem) * val_ratio))) if len(rem) > 1 else 0
    val_trials = rem[:val_n] if val_n > 0 else []
    train_trials = rem[val_n:]
    # Ensure at least one train trial
    if not train_trials and len(rem) > 0:
        train_trials = rem[-1:]
        val_trials = rem[:-1]
    return train_trials, val_trials

def _chunks(lst: List[str], k: int) -> List[List[str]]:
    """Split list into k near-equal chunks (deterministic, no external deps)."""
    n = len(lst)
    base = n // k
    extra = n % k
    chunks = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        chunks.append(lst[start:start+size])
        start += size
    return chunks

import random
from typing import Dict, List, Tuple

def _split_train_val(remaining: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    rem = list(remaining)
    rnd.shuffle(rem)
    if len(rem) <= 1:
        return rem, []
    val_n = max(1, int(round(len(rem) * val_ratio)))
    val_trials = rem[:val_n]
    train_trials = rem[val_n:]
    if not train_trials and len(rem) > 0:
        train_trials = rem[-1:]
        val_trials = rem[:-1]
    return train_trials, val_trials

def _chunks(lst: List[str], k: int) -> List[List[str]]:
    n = len(lst)
    base = n // k
    extra = n % k
    out = []
    s = 0
    for i in range(k):
        sz = base + (1 if i < extra else 0)
        out.append(lst[s:s+sz])
        s += sz
    return out
import re
import random
from typing import Dict, List, Tuple

# --- tiny helpers used below ---

def _infer_subject_trial(trial_name: str) -> Tuple[str, str]:
    """
    Parse '..._S01_T06' -> ('S01','T06'). Falls back to ('UNK','TRIAL') if not found.
    """
    m = re.search(r"_S(\d+)_T(\d+)", trial_name)
    if m:
        return f"S{int(m.group(1)):02d}", f"T{int(m.group(2)):02d}"
    # tolerant fallback: try 'S\d+' & 'T\d+' anywhere
    ms = re.search(r"S(\d+)", trial_name)
    mt = re.search(r"T(\d+)", trial_name)
    subj = f"S{int(ms.group(1)):02d}" if ms else "UNK"
    trl  = f"T{int(mt.group(1)):02d}" if mt else "TRIAL"
    return subj, trl

def _split_train_val(items: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    arr = list(items)
    rnd.shuffle(arr)
    n = len(arr)
    n_val = max(1, int(round(n * val_ratio))) if n > 1 else (1 if n == 1 else 0)
    val = arr[:n_val]
    train = arr[n_val:]
    return train, val

def _chunks(lst: List[str], k: int):
    n = len(lst)
    size = max(1, n // k)
    for i in range(0, n, size):
        yield lst[i:i + size]

# --- NEW: LOUO-aware build_cv_splits ---

def build_cv_splits(dataloader_cfg: Dict) -> List[Dict[str, List[str]]]:
    """
    Trial-level CV with options:
      - scheme: "leave_one_out" | "group_k_fold" | "gesture_flat" | "trial_split" | "leave_one_user_out" (NEW)
      - val_same_as_test (bool): if True, validation == test set (useful for LOUO sanity checks)
      - For LOUO when val_same_as_test=False, we split the held-out user's trials into val/test by val_ratio.
    """
    cv = dataloader_cfg.get("cv", {})
    scheme = cv.get("scheme", "leave_one_out")
    k = int(cv.get("k", 5))
    val_ratio = float(cv.get("val_ratio", 0.2))
    seed = int(cv.get("seed", 0))
    shuffle = bool(cv.get("shuffle", True))
    val_same_as_test = bool(cv.get("val_same_as_test", False))  # ignored for gesture_flat
    flat_test_ratio = float(cv.get("test_ratio", 0.2))
    flat_seed = seed

    trials = list(dataloader_cfg.get("all_trials", []))
    if not trials:
        raise ValueError("Please set dataloader_params['all_trials'].")

    # sort/shuffle trials once for reproducibility
    trials_sorted = list(trials)
    if shuffle:
        rnd = random.Random(seed)
        rnd.shuffle(trials_sorted)
    else:
        trials_sorted.sort()

    folds: List[Dict[str, List[str]]] = []

    # ----------------- NEW: Leave-One-User-Out -----------------
    if scheme in ("leave_one_user_out", "louo"):
        by_subject = {}
        for t in trials_sorted:
            subj, _ = _infer_subject_trial(t)
            by_subject.setdefault(subj, []).append(t)

        subjects = sorted(by_subject.keys())

        for i, held_subj in enumerate(subjects, start=1):
            test_trials = list(by_subject[held_subj])  # held user's trials
            # Remaining users
            remaining_subjs = [s for s in subjects if s != held_subj]
            train_pool = [t for s in remaining_subjs for t in by_subject[s]]

            # ✅ strict LOUO: training and test
            train_trials = train_pool
            val_trials = test_trials

            folds.append({
                "name": f"louo_{held_subj}_strict",
                "train_trials": train_trials,
                "val_trials":   val_trials,   # from training users
                "test_trials":  test_trials,  # from held user
            })


    # ----------------- Existing schemes (unchanged) -----------------
    elif scheme == "leave_one_out":
        for i, test_trial in enumerate(trials_sorted, start=1):
            remaining = [t for t in trials_sorted if t != test_trial]
            if val_same_as_test:
                train_trials = remaining
                val_trials = [test_trial]
            else:
                train_trials, val_trials = _split_train_val(remaining, val_ratio, seed + i)
            folds.append({
                "name": f"loo_{test_trial}" + ("_valEqTest" if val_same_as_test else ""),
                "train_trials": train_trials,
                "val_trials": val_trials,
                "test_trials": [test_trial],
            })

    elif scheme == "group_k_fold":
        if k < 2:
            raise ValueError("group_k_fold requires k >= 2")
        for i, test_trials in enumerate(_chunks(trials_sorted, k), start=1):
            remaining = [t for t in trials_sorted if t not in test_trials]
            if val_same_as_test:
                train_trials = remaining
                val_trials = list(test_trials)
            else:
                train_trials, val_trials = _split_train_val(remaining, val_ratio, seed + i)
            folds.append({
                "name": f"kfold_{i:02d}_of_{k}" + ("_valEqTest" if val_same_as_test else ""),
                "train_trials": train_trials,
                "val_trials": val_trials,
                "test_trials": list(test_trials),
            })

    elif scheme == "gesture_flat":
        folds.append({
            "name": "gesture_flat",
            "train_trials": list(trials_sorted),
            "val_trials":   [],
            "test_trials":  list(trials_sorted),
            "use_gesture_flat": True,
            "flat_test_ratio":  flat_test_ratio,
            "flat_seed":        flat_seed,
        })

    elif scheme == "trial_split":
        rnd = random.Random(seed)
        trls = list(trials_sorted)
        rnd.shuffle(trls)
        n = len(trls)
        test_n = max(1, int(round(n * flat_test_ratio)))
        test_trials = trls[:test_n]
        remaining = trls[test_n:]
        train_trials, val_trials = _split_train_val(remaining, val_ratio, seed + 1)
        folds.append({
            "name": f"trial_split_{flat_test_ratio:.2f}",
            "train_trials": train_trials,
            "val_trials":   val_trials,
            "test_trials":  test_trials,
        })
        print("\n----------------- Trial Split Info ----------------")
        print(f"Created trial splits with num trials: total={n}, train={len(train_trials)}, val={len(val_trials)}, test={len(test_trials)}")
        print(f"Trial split fold: train={train_trials}, val={val_trials}, test={test_trials}")
        print("---------------------------------------------------\n")

    else:
        raise ValueError(f"Unknown CV scheme: {scheme}")

    # cleanup (skip degenerate folds unless gesture_flat)
    cleaned = []
    for f in folds:
        if f.get("use_gesture_flat"):
            cleaned.append(f); continue
        if len(f["test_trials"]) == 0 or len(f["train_trials"]) == 0:
            continue
        cleaned.append(f)
    return cleaned


# ------------------ Model config derivation ------------------
def build_model_cfg_from_dataloader(
    dataloader_cfg: Dict,
    *,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    dropout: float = 0.1,
    num_classes: int = 8,
    fusion: str = "concat_tokens",
    modality_dropout_p: float = 0.1,
) -> ModelCfg:
    """Derive ModelCfg.modalities and include list from dataloader selections."""
    mods = dataloader_cfg.get("modalities", [])
    sel  = dataloader_cfg.get("selections", {})

    problems = []
    modalities_dict: Dict[str, ModalityCfg] = {}
    for m in mods:
        cols = sel.get(m, [])
        if m == "images":
            modalities_dict[m] = ModalityCfg(name=m, in_dim=2048)  # set placeholder feature dim (ResNet output etc.)
            continue

        if not isinstance(cols, list) or len(cols) == 0:
            problems.append(m)
            continue

        if m == "console":
            modalities_dict[m] = ModalityCfg(name=m, in_dim=len(cols)-1)
        else:
            modalities_dict[m] = ModalityCfg(name=m, in_dim=len(cols))

    if problems:
        raise ValueError(
            f"Missing or empty selections for modalities: {problems}. "
            f"Add column selections in dataloader_params['selections']."
        )

    # derive num_classes from keysteps map
    num_classes = len(dataloader_cfg.get("keysteps", {})) or num_classes

    return ModelCfg(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        include_modalities=list(modalities_dict.keys()),
        fusion=fusion,
        modality_dropout_p=modality_dropout_p,
        modalities=modalities_dict,
    )

# ------------------ Model config derivation ------------------
def build_multimtrsap_model_cfg_from_dataloader(
    dataloader_cfg: Dict,
    *,
    d_model: int = 256,
    nhead: int = 4,
    num_layers: int = 4,
    dropout: float = 0.1,
    num_classes: int = 8,
    seq_to_one: bool = False,
    dim_feedforward: int = 1024,
    causal_tcn: bool = True,
    tcn_channels: List[int] = [128, 128, 256],
) -> MultiMTRSAPModelCfg:
    """Derive ModelCfg.modalities and include list from dataloader selections."""
    mods = dataloader_cfg.get("modalities", [])
    sel  = dataloader_cfg.get("selections", {})

    problems = []
    modalities_dict: Dict[str, ModalityCfg] = {}
    for m in mods:
        cols = sel.get(m, [])
        if m == "images":
            modalities_dict[m] = ModalityCfg(name=m, in_dim=2048)  # set placeholder feature dim (ResNet output etc.)
            continue

        if not isinstance(cols, list) or len(cols) == 0:
            problems.append(m)
            continue

        if m == "console":
            modalities_dict[m] = ModalityCfg(name=m, in_dim=len(cols)-1)
        else:
            modalities_dict[m] = ModalityCfg(name=m, in_dim=len(cols))

    if problems:
        raise ValueError(
            f"Missing or empty selections for modalities: {problems}. "
            f"Add column selections in dataloader_params['selections']."
        )

    # derive num_classes from keysteps map
    num_classes = len(dataloader_cfg.get("keysteps", {})) or num_classes

    return MultiMTRSAPModelCfg(
        modalities=list(modalities_dict.keys()),
        in_dims=[modalities_dict[m].in_dim for m in modalities_dict],
        num_classes=num_classes,
        tcn_channels=[128, 128, 256],
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=1024,
        dropout=dropout,
        seq_to_one= seq_to_one,
        causal_tcn=True)


# ------------------ Unified args namespace ------------------
RECORD_RESULTS = True

class DefaultArgsNamespace:
    def __init__(self, fold_index: int = 0):
        self.record_results = RECORD_RESULTS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Source config
        self.dataloader_params = dict(dataloader_params)  # shallow copy

        # Build folds and pick one
        folds = build_cv_splits(self.dataloader_params)
        if not folds:
            raise RuntimeError("No valid CV folds were produced.")
        if fold_index < 0 or fold_index >= len(folds):
            raise IndexError(f"fold_index {fold_index} out of range [0, {len(folds)-1}]")
        self.fold = folds[fold_index]

        # Inject chosen fold back into dataloader_params for the rest of the pipeline
        self.dataloader_params["train_trials"] = self.fold["train_trials"]
        self.dataloader_params["val_trials"]   = self.fold["val_trials"]
        self.dataloader_params["test_trials"]  = self.fold["test_trials"]

        # Make experiment name reflect the fold
        base_exp = self.dataloader_params.get("experiment_name", "exp")
        self.dataloader_params["experiment_name"] = f"{base_exp}_{self.fold['name']}"

        # Model config derived from dataloader config
        self.mmtransformercfg = build_model_cfg_from_dataloader(
            self.dataloader_params,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            fusion="concat_tokens",  # or "late_gated","cross_attend"
            modality_dropout_p=0.1,
        )

        # Other blocks (kept if your code references them)
        self.learning_params = learning_params
        self.transformer_params = transformer_params
        self.tcn_model_params = tcn_model_params

# ------------------ Test ------------------
if __name__ == "__main__":
    # Example: iterate over all folds
    args0 = DefaultArgsNamespace(fold_index=0)
    print("Experiment:", args0.dataloader_params["experiment_name"])
    print("Fold:", args0.fold["name"])
    print("Train:", args0.dataloader_params["train_trials"])
    print("Val:",   args0.dataloader_params["val_trials"])
    print("Test:",  args0.dataloader_params["test_trials"])
    print("Active modalities:", args0.mmtransformercfg.include_modalities)
    print("Modalities (name -> in_dim):", {k: v.in_dim for k, v in args0.mmtransformercfg.modalities.items()})

    print("Number of folds available:", len(build_cv_splits(dataloader_params)))
