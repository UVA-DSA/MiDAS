# train_kfold_yaml.py
from utils.utils import *  # your training/eval helpers (train_mmt_one_epoch, validate_mmt, test_mmt_model, initialize_mmt_model, etc.)
from scripts.config import build_cv_splits, build_multimtrsap_model_cfg_from_dataloader  # reuse your helpers
import torch
from torch.optim.lr_scheduler import StepLR
import wandb

from datetime import datetime
from typing import List, Dict, Any
import argparse
import warnings
import os
import time
import json
import yaml  # <-- YAML loader

warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")
torch.manual_seed(0)


# --------------- small utilities ---------------

def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def flatten_numeric(d: Dict[str, Any]) -> Dict[str, float]:
    """
    Keep only numeric scalars from a (possibly nested) results dict.
    If your test_mmt_model returns nested dicts, flatten keys as a.b
    """
    out = {}

    def _walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            if isinstance(obj, (int, float)) or (hasattr(obj, "item") and obj.dim() == 0):
                try:
                    out[prefix] = float(obj if not hasattr(obj, "item") else obj.item())
                except Exception:
                    pass

    _walk("", d)
    return out


def write_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_csv(path: str, rows: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # collect union of keys
    headers = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    # write
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            vals = []
            for h in headers:
                vals.append(str(r.get(h, "")))
            f.write(",".join(vals) + "\n")


def load_yaml_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    # basic shape with defaults
    cfg.setdefault("dataloader_params", {})
    cfg.setdefault("learning_params", {
        "lr": 1e-5,
        "epochs": 10,
        "weight_decay": 1e-5,
        "patience": 10,
        "lr_drop": 20,
    })
    cfg.setdefault("transformer_params", {
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "batch_first": True,
    })
    cfg.setdefault("tcn_model_params", {
        "encoder_params": {"in_channels": 128, "kernel_size": 13, "out_channels": 64},
        "decoder_params": {"in_channels": 60, "kernel_size": 31, "out_channels": 60},
    })
    return cfg


# --------------- per-fold args namespace ---------------

class DefaultArgsNamespace:
    """
    A minimal args container built from a config dict that contains at least:
      - dataloader_params
      - learning_params
      - transformer_params (optional)
      - tcn_model_params (optional)
      - model_cfg_overrides (optional)  # to override d_model, nhead, etc.
    """
    def __init__(self, cfg: Dict[str, Any], fold_index: int = 0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1) take dataloader params from file (no global imports)
        self.dataloader_params: Dict[str, Any] = dict(cfg["dataloader_params"])

        # 2) build CV folds from the loaded dataloader params
        folds = build_cv_splits(self.dataloader_params)
        if not folds:
            raise RuntimeError("No valid CV folds were produced from the provided dataloader_params['all_trials'] and ['cv'].")

        if fold_index < 0 or fold_index >= len(folds):
            raise IndexError(f"fold_index {fold_index} out of range [0, {len(folds)-1}]")
        self.fold = folds[fold_index]

        # inject chosen fold back
        self.dataloader_params["train_trials"] = self.fold["train_trials"]
        self.dataloader_params["val_trials"]   = self.fold["val_trials"]
        self.dataloader_params["test_trials"]  = self.fold["test_trials"]

        base_exp = self.dataloader_params.get("experiment_name", "exp")
        self.dataloader_params["experiment_name"] = f"{base_exp}_{self.fold['name']}"

        # 3) model cfg derived from dataloader selections (and optional overrides)
        overrides = cfg.get("model_cfg_overrides", {})
        self.multiMTRSAPcfg = build_multimtrsap_model_cfg_from_dataloader(self.dataloader_params, **overrides)
        print("Derived MultiMTRSAP model cfg:", self.multiMTRSAPcfg)

        # 4) the rest
        self.learning_params     = dict(cfg["learning_params"])
        self.transformer_params  = dict(cfg.get("transformer_params", {}))
        self.tcn_model_params    = dict(cfg.get("tcn_model_params", {}))
        self.record_results      = bool(cfg.get("record_results", True))


# --------------- main ---------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script (YAML-config + CV)")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
    parser.add_argument('--job_id', type=str, default=None, help='SLURM job ID or custom tag')
    parser.add_argument('--fold_index', type=int, default=None, help='Run a single fold (0-based). Omit to run all.')
    parser.add_argument('--wandb', type=str, choices=["on", "off"], default="off")
    cmd_args = parser.parse_args()

    # Load the YAML config
    cfg = load_yaml_config(cmd_args.config)
    dataloader_cfg = cfg["dataloader_params"]

    # base job id (stable across folds)
    if cmd_args.job_id is None:
        cmd_args.job_id = str(int(time.time()))
    base_job_id = "MultiMTRSAP_" + cmd_args.job_id
    print(f"Base Job ID: {base_job_id}")
    print(f"Config file: {os.path.abspath(cmd_args.config)}")

    # Build CV folds from YAML dataloader_params
    folds = build_cv_splits(dataloader_cfg)
    if not folds:
        raise RuntimeError("No valid folds produced by build_cv_splits(). Check dataloader_params['all_trials'] and CV settings.")
    print(f"Discovered {len(folds)} folds:")
    for i, f in enumerate(folds):
        print(f"  [{i}] {f['name']}: train={f['train_trials']}  val={f['val_trials']}  test={f['test_trials']}")

    # choose which folds to run
    fold_indices = list(range(len(folds))) if cmd_args.fold_index is None else [cmd_args.fold_index]

    # top-level run directory (stable across folds)
    run_root = os.path.abspath(f'./results/job_{base_job_id}')
    ckpt_root = os.path.abspath(f'./checkpoints/job_{base_job_id}')
    os.makedirs(run_root, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    # Optionally, copy the YAML into the job root for provenance
    try:
        import shutil
        shutil.copy2(cmd_args.config, os.path.join(run_root, os.path.basename(cmd_args.config)))
    except Exception as e:
        print(f"Warning: could not copy config into results dir: {e}")

    # collect per-fold metrics for summary
    fold_rows_for_csv: List[Dict[str, Any]] = []
    numeric_metrics_per_key: Dict[str, List[float]] = {}

    for fi in fold_indices:
        fold = folds[fi]
        print(f"\n=== Running Fold {fi}: {fold['name']} ===")

        # Build args for this fold from YAML + fold
        args = DefaultArgsNamespace(cfg, fi)
        experiment_name = args.dataloader_params['experiment_name']

        print("Loaded args:")
        for k, v in vars(args).items():
            print(f"  {k}: {v}")

        # per-fold folders (unique)
        fold_results_dir = os.path.join(run_root, experiment_name)
        fold_ckpt_dir    = os.path.join(ckpt_root, experiment_name)
        os.makedirs(fold_results_dir, exist_ok=True)
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        print(f"Fold results dir: {fold_results_dir}")
        print(f"Fold checkpoints dir: {fold_ckpt_dir}")

        keysteps = args.dataloader_params['keysteps']
        out_classes = len(keysteps)
        modality = args.dataloader_params['modalities']
        print(f"Keysteps: {keysteps}")
        print(f"Modalities: {modality}")
        print(f"Trials (train/val/test): {args.dataloader_params['train_trials']} / {args.dataloader_params['val_trials']} / {args.dataloader_params['test_trials']}")

        best_val_path = os.path.join(fold_ckpt_dir, 'val_best_model.pt')

        # W&B
        wandb_mode = "online" if cmd_args.wandb == "on" else "disabled"
        wandb_logger = wandb.init(
            project="MIDAS Gesture Recognition",
            group=f"Gesture Recognition ({base_job_id})",
            mode=wandb_mode,
            name=experiment_name,
            notes=f"job={base_job_id}, fold={fold['name']}",
            config={"args": str(args.dataloader_params)},
        )

        # Data
        train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats = MIDAS_get_dataloaders(args)
        args.dataloader_params['train_class_stats'] = train_class_stats
        args.dataloader_params['val_class_stats']   = val_class_stats

        print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
        print_one_batch(train_loader)

        device = args.device
        print(f"Using device: {device}")

        # Learning params
        lr = float(args.learning_params.get("lr", 1e-5))
        weight_decay = float(args.learning_params.get("weight_decay", 1e-5))
        epochs = int(args.learning_params.get("epochs", 10))
        lr_drop = int(args.learning_params.get("lr_drop", 20))
        print(f"Learning params: lr={lr}, weight_decay={weight_decay}, epochs={epochs}, lr_drop={lr_drop}")

        # Model
        model, optimizer, criterion = initialize_multiMTRSAP_model(args, device)

        scheduler = StepLR(optimizer, step_size=lr_drop, gamma=0.1)


        for epoch in range(1, epochs + 1):
            print(f"\n[Fold {fi}] Epoch {epoch}/{epochs}")
            # Train
            train_metrics = train_multiMTRSAP_one_epoch(model, train_loader,criterion, optimizer, device, wandb_logger, args=args )

            # Validate
            val_metrics = validate_multiMTRSAP(model, val_loader, criterion, device, wandb_logger, args=args )
            if scheduler is not None:
                scheduler.step()


            # (Optional) test each epoch; save under per-epoch subdir
            epoch_dir = os.path.join(fold_results_dir, "epochs")
            os.makedirs(epoch_dir, exist_ok=True)
            _ = test_multiMTRSAP_model(model, test_loader, criterion, device, wandb_logger, epoch, epoch_dir, args)

            print("*"*10, "="*10, "*"*10)


        # ---- Final test with best checkpoint (if you enabled it) ----
        if os.path.exists(best_val_path):
            model.load_state_dict(torch.load(best_val_path, map_location=device))
            print(f"Loaded best val checkpoint from: {best_val_path}")

        final_results = test_multiMTRSAP_model(
            model, test_loader, criterion, device, wandb_logger, epoch="final", results_dir=fold_results_dir, args=args
        )
        print(f"[{experiment_name}] Final Test Results: {final_results}")

        # Save final numeric results to JSON for this fold
        fold_numeric = flatten_numeric(final_results)
        fold_numeric["fold_name"] = fold["name"]
        fold_numeric["experiment_name"] = experiment_name
        write_json(os.path.join(fold_results_dir, "results_final.json"), fold_numeric)

        # Accumulate for summary
        if fold_numeric:
            fold_rows_for_csv.append(fold_numeric)
            for k, v in fold_numeric.items():
                if k in ("fold_name", "experiment_name"):
                    continue
                if is_number(v):
                    numeric_metrics_per_key.setdefault(k, []).append(float(v))

        wandb_logger.finish()

    # ---------------- Summary across folds ----------------
    summary_dir = os.path.join(run_root, "summary_all_folds")
    os.makedirs(summary_dir, exist_ok=True)

    # Save the per-fold table
    write_csv(os.path.join(summary_dir, "fold_results.csv"), fold_rows_for_csv)

    # Compute mean/std for numeric metrics
    summary_stats = {}
    for k, vals in numeric_metrics_per_key.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        std = var ** 0.5
        summary_stats[k] = {"mean": mean, "std": std, "n_folds": len(vals)}

    # Save summary JSON + a README that includes the *resolved* dataloader params used by the last fold's args
    write_json(os.path.join(summary_dir, "summary_mean_std.json"), summary_stats)

    # Dump the *last* args' dataloader params (they’re fold-injected) for provenance
    with open(os.path.join(summary_dir, "README.txt"), "w") as f:
        f.write(f"Job: {base_job_id}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folds run: {fold_indices} of total {len(folds)}\n")
        f.write(f"Root run dir: {run_root}\n")
        f.write("Each fold has its own subfolder named after the experiment_name (which includes the fold name).\n")
        f.write("This folder contains per-epoch outputs (under epochs/) and results_final.json for the final test.\n")
        f.write("This summary folder aggregates all folds into fold_results.csv and summary_mean_std.json.\n")
        f.write("\nResolved dataloader_params for the last fold:\n")
        try:
            f.write(json.dumps(args.dataloader_params, indent=2))  # 'args' from last loop
        except Exception:
            f.write("<unavailable>\n")

    print("\n=== Cross-validation complete ===")
    print(f"Per-fold results under: {run_root}")
    print(f"Summary written to:     {summary_dir}")


