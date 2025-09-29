# config.py
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# trakstar candidates #
trakstar_candidates = ['trakstar_sensor_0_azimuth', 'trakstar_sensor_1_azimuth', 'trakstar_sensor_2_azimuth', 'trakstar_sensor_3_azimuth', 'trakstar_sensor_0_elevation', 'trakstar_sensor_1_elevation', 'trakstar_sensor_2_elevation', 'trakstar_sensor_3_elevation', 'trakstar_sensor_0_roll', 'trakstar_sensor_1_roll', 'trakstar_sensor_2_roll', 'trakstar_sensor_3_roll', 'trakstar_sensor_0_x', 'trakstar_sensor_1_x', 'trakstar_sensor_2_x', 'trakstar_sensor_3_x', 'trakstar_sensor_0_y', 'trakstar_sensor_1_y', 'trakstar_sensor_2_y', 'trakstar_sensor_3_y', 'trakstar_sensor_0_z', 'trakstar_sensor_1_z', 'trakstar_sensor_2_z', 'trakstar_sensor_3_z']

# console candidates #
console_candidates = ['console_pos0', 'console_pos1', 'console_pos2', 'console_pos3', 'console_pos4', 'console_pos5', 'console_rot0', 'console_rot1', 'console_rot2', 'console_rot3', 'console_rot4', 'console_rot5', 'console_aux0', 'console_aux1', 'console_pedal']


# raven candidates #
raven_candidates = ['raven_field.pos0', 'raven_field.pos1', 'raven_field.pos2', 'raven_field.pos3', 'raven_field.pos4', 'raven_field.pos5', 'raven_field.ori0', 'raven_field.ori1', 'raven_field.ori2', 'raven_field.ori3', 'raven_field.ori4', 'raven_field.ori5', 'raven_field.ori6', 'raven_field.ori7', 'raven_field.ori8', 'raven_field.ori9', 'raven_field.ori10', 'raven_field.ori11', 'raven_field.ori12', 'raven_field.ori13', 'raven_field.ori14', 'raven_field.ori15', 'raven_field.ori16', 'raven_field.ori17']


# sw_left candidates #
sw_left_candidates = ['sw_left_x', 'sw_left_y', 'sw_left_z']

# sw_right candidates #
sw_right_candidates = ['sw_right_x', 'sw_right_y', 'sw_right_z']

# 

# ---------- Model-side configs ----------
@dataclass
class ModalityCfg:
    name: str
    in_dim: int               # derived from dataloader selections (len of selected columns)
    # (optional) downsample, stride, etc. could be added here later

@dataclass
class ModelCfg:
    d_model: int = 256
    nhead: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    num_classes: int = 8
    include_modalities: List[str] = field(default_factory=list)     # derived
    fusion: str = "concat_tokens"                                   # "concat_tokens", "late_gated", "cross_attend"
    modality_dropout_p: float = 0.0
    modalities: Dict[str, ModalityCfg] = field(default_factory=dict) # derived {mod: ModalityCfg}

# ---------- Dataloader (single source of truth) ----------
dataloader_params: Dict = {
    "base_path": "/standard/UVA-DSA/MIDAS/Organized/09-18-25/hamid/",
    "batch_size": 16,
    "fps": 30,
    "train_trials": ["t1", "t2", "t3", "t4","t5"],
    "val_trials":   ["t5", "t6", "t7"],
    "test_trials":  ["t6", "t7"],

    # Active modalities for this experiment (order matters for reporting):
    "modalities": ["trakstar"],
    # "modalities": ["raven"],

    # Column selections (patterns or explicit names) per modality
    "selections": {
        "trakstar": trakstar_candidates,
        # "trakstar": ["trakstar_sensor_0_x", "trakstar_sensor_0_y", "trakstar_sensor_0_z", "trakstar_sensor_1_x", "trakstar_sensor_1_y", "trakstar_sensor_1_z", "trakstar_sensor_2_x", "trakstar_sensor_2_y", "trakstar_sensor_2_z", "trakstar_sensor_3_x", "trakstar_sensor_3_y", "trakstar_sensor_3_z"],
        "sw_left":  ["sw_left_x", "sw_left_y", "sw_left_z"],
        "sw_right": ["sw_right_x", "sw_right_y", "sw_right_z"],
        "raven":    ["raven_field.pos0","raven_field.pos1", "raven_field.pos2", "raven_field.ori0", "raven_field.ori1", "raven_field.ori2",],
        # add others only when you use them:
        # "console":  [...],
        # "imu":      [...],
    },

    # Windowing
    "observation_window": 120,   # 4s at 30 Hz
    # "observation_window": -1,   # full clip for gesture
    "step": 8,

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

# ---------- Other hyperparams ----------
learning_params = {
    "lr": 1e-5,
    "epochs": 50,
    "weight_decay": 1e-5,
    "patience": 3,
    "lr_drop": 20,
    "best_chkpoint": "./checkpoints/job_xxx/val_best_model.pt",
}

tcn_model_params = {
    "encoder_params": { #some of these gets updated during runtime based on the feature dimension of the given data
        "in_channels": 1024,
        "kernel_size": 45,
        "out_channels": 256,
    },
    "decoder_params": {
        "in_channels": 60,
        "kernel_size": 31,
        "out_channels": 60
    }
}


# Keep only what you actually use from the old blocks. If you still need TCN/audio/ResNet, keep them;
# otherwise delete to avoid config drift.
transformer_params = {
    "d_model": 256,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.1,
    "batch_first": True,
    # audio/video params only if you really use them
    "sample_rate": 48000,
    "n_mels": 64,
    "hop_length": 512,
    "n_fft": 1024,
    "resnet_dim": 2048,
}

# ---------- Derivation helpers ----------
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

    # Validate: every listed modality must have a non-empty selection list
    problems = []
    modalities_dict: Dict[str, ModalityCfg] = {}
    for m in mods:
        cols = sel.get(m, [])
        if not isinstance(cols, list) or len(cols) == 0:
            problems.append(m)
        else:
            modalities_dict[m] = ModalityCfg(name=m, in_dim=len(cols))

    if problems:
        raise ValueError(
            f"Missing or empty selections for modalities: {problems}. "
            f"Add column selections in dataloader_params['selections']."
        )
    
    num_classes = len(dataloader_cfg.get("keysteps", {}))

    return ModelCfg(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=num_classes,
        include_modalities=list(modalities_dict.keys()),  # preserve order from 'mods'
        fusion=fusion,
        modality_dropout_p=modality_dropout_p,
        modalities=modalities_dict,
    )

# ---------- Unified args namespace ----------
RECORD_RESULTS = True

class DefaultArgsNamespace:
    def __init__(self):
        self.record_results = RECORD_RESULTS
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Single source of truth:
        self.dataloader_params = dataloader_params

        # Model config is DERIVED from dataloader config:
        self.mmtransformercfg = build_model_cfg_from_dataloader(
            self.dataloader_params,
            d_model=128,
            nhead=4,
            num_layers=2,
            dropout=0.1,
            num_classes=10,
            fusion="cross_attend", # "concat_tokens", "late_gated", "cross_attend"
            modality_dropout_p=0.1,
        )

        # Keep extra blocks only if used by your codebase:
        self.learning_params = learning_params
        self.transformer_params = transformer_params

        self.tcn_model_params = tcn_model_params





# ---------- Test ----------
if __name__ == "__main__":
    args = DefaultArgsNamespace()
    print("Active modalities:", args.mmtransformercfg.include_modalities)
    print("Modalities (name -> in_dim):", {k: v.in_dim for k, v in args.mmtransformercfg.modalities.items()})
