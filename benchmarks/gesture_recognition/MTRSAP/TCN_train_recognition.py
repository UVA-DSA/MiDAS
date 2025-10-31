# train_kfold.py
from utils.utils import *
from scripts.config import build_cv_splits, build_model_cfg_from_dataloader  # <- only import helpers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb
from datetime import datetime

import argparse
import warnings
import os
import time
import json
from typing import Dict, Any, List

try:
    import yaml  # optional, for YAML configs
except Exception:
    yaml = None

warnings.filterwarnings("ignore", message="Accurate seek is not implemented for pyav backend")
torch.manual_seed(0)


def is_number(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def flatten_numeric(d: Dict[str, Any]) -> Dict[str, float]:
    out = {}
    def _walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            if isinstance(obj, (int, float)) or (hasattr(obj, "item") and getattr(obj, "dim", lambda:1)() == 0):
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
    headers = []
    for r in rows:
        for k in r.keys():
            if k not in headers:
                headers.append(k)
    with open(path, "w") as f:
        f.write(",".join(headers) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(h, "")) for h in headers) + "\n")


def load_config_file(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r") as f:
        if ext in (".yml", ".yaml"):
            if yaml is None:
                raise RuntimeError("pyyaml not installed; install it or use JSON.")
            return yaml.safe_load(f)
        elif ext == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config extension: {ext} (use .yaml/.yml or .json)")


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
        self.mmtransformercfg = build_model_cfg_from_dataloader(self.dataloader_params, **overrides)

        # 4) the rest
        self.learning_params     = dict(cfg["learning_params"])
        self.transformer_params  = dict(cfg.get("transformer_params", {}))
        self.tcn_model_params    = dict(cfg.get("tcn_model_params", {}))
        self.record_results      = bool(cfg.get("record_results", True))


if __name__ == "__main__":

    # --- CLI ---
    parser = argparse.ArgumentParser(description="Training script for recognition w/ CV (file-based config)")
    parser.add_argument('--config', required=True, help='Path to YAML/JSON config file for this job')
    parser.add_argument('--job_id', type=str, help='SLURM job ID or custom tag', default=None)
    parser.add_argument('--wandb', type=str, choices=["on", "off"], default="off")
    parser.add_argument('--fold_index', type=int, default=None, help="Run a single fold (0-based). Omit to run all.")
    cmd_args = parser.parse_args()

    # load the per-job config (immutable snapshot)
    cfg = load_config_file(cmd_args.config)
    print("Loaded config:")
    print(cfg)

    # base job id (stable across folds)
    if cmd_args.job_id is None or cmd_args.job_id == "0":
        cmd_args.job_id = str(int(time.time()))
    base_job_id = "TCN_" + cmd_args.job_id

    # build folds *from file config* (for printing)
    folds = build_cv_splits(cfg["dataloader_params"])
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

    # **snapshot** the exact config used into the run root
    snap_cfg_path = os.path.join(run_root, "config_used" + os.path.splitext(cmd_args.config)[1])
    if not os.path.exists(snap_cfg_path):
        # don’t overwrite if resuming
        with open(snap_cfg_path, "w") as f:
            if snap_cfg_path.endswith((".yml", ".yaml")) and yaml is not None:
                yaml.safe_dump(cfg, f, sort_keys=False)
            else:
                json.dump(cfg, f, indent=2)

    # collect per-fold metrics for summary
    fold_rows_for_csv: List[Dict[str, Any]] = []
    numeric_metrics_per_key: Dict[str, List[float]] = {}

    for fi in fold_indices:
        print(f"\n=== Running Fold {fi}: {folds[fi]['name']} ===")
        args = DefaultArgsNamespace(cfg, fold_index=fi)

        modalitys = args.dataloader_params['modalities']
        modality_string = '_'.join(modalitys)
        args.dataloader_params['experiment_name'] = f"{modality_string}_{args.dataloader_params['sample_rate']}hz"
        experiment_name = args.dataloader_params['experiment_name']

        # per-fold folders (unique)
        fold_results_dir = os.path.join(run_root, experiment_name, f"fold_{fi}")
        fold_ckpt_dir = os.path.join(ckpt_root, experiment_name, f"fold_{fi}")
        os.makedirs(fold_results_dir, exist_ok=True)
        os.makedirs(fold_ckpt_dir, exist_ok=True)

        print(f"Fold results dir: {fold_results_dir}")
        print(f"Fold checkpoints dir: {fold_ckpt_dir}")

        # W&B
        wandb_mode = "online" if cmd_args.wandb == "on" else "disabled"
        wandb_logger = wandb.init(
            project="MIDAS Gesture Recognition",
            group=f"Gesture Recognition ({base_job_id})",
            mode=wandb_mode,
            name=experiment_name,
            notes=f"job={base_job_id}, fold={folds[fi]['name']}",
            config={"args": str(args.dataloader_params)},  # keep light
        )

        keysteps = args.dataloader_params['keysteps']
        out_classes = len(keysteps)
        modality = args.dataloader_params['modalities']
        selections = args.dataloader_params['selections']

        print(f"Keysteps: {keysteps}")
        print(f"Modalities: {modality}")
        print(f"Selections: {selections}")
        print(f"Trials (train/val/test): {args.dataloader_params['train_trials']} / {args.dataloader_params['val_trials']} / {args.dataloader_params['test_trials']}")

        # Data
        if args.dataloader_params.get('dataset_name', 'MIDAS') == 'MIDAS':
            print("Using MIDAS dataset...")
            train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats, class_names = MIDAS_get_dataloaders(args)
            
            # invert class_names so we can go from contiguous index → original_id
            inv_class_names = {v: k for k, v in class_names.items()}

            # build mapping contiguous_index → human-readable name
            class_id_to_name = {
                idx: keysteps.get(orig_id, str(orig_id))
                for idx, orig_id in inv_class_names.items()
            }
            print("Class ID → Name mapping:", class_id_to_name)
        elif args.dataloader_params.get('dataset_name') == 'DESK':
            print("Using DESK dataset...")
            train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats = DESK_get_dataloaders(args)
        elif args.dataloader_params.get('dataset_name') == 'JIGSAWS':
            print("Using JIGSAWS dataset...")
            train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats = DESK_get_dataloaders(args) # reuse DESK loader for JIGSAWS

        args.dataloader_params['train_class_stats'] = train_class_stats
        args.dataloader_params['val_class_stats'] = val_class_stats

        # save the class stats for reference
        stats_json = f"{fold_results_dir}/{fi}_class_stats.json"
        write_json(stats_json, {
            "train_class_stats": train_class_stats,
            "val_class_stats": val_class_stats,
            "test_class_stats": test_class_stats,
        })
        print(f"Saved class stats to: {stats_json}")


        print(f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}, Test samples: {len(test_loader.dataset)}")
        print_one_batch(train_loader)

        # Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Feature dims
        feature_dim = get_feature_dim(train_loader, args, None, device)
        print(f"Feature dimension: {feature_dim}")
  
        model,optimizer,criterion = initialize_tcn_model(args,feature_dim,device,out_classes)

        scheduler = StepLR(optimizer, step_size=args.learning_params["lr_drop"], gamma=0.1)

        # Track best val
        best_val_path = os.path.join(fold_ckpt_dir, 'val_best_model.pt')
        min_val_loss = float('inf')

        best_model = None

        # Train loop
        for epoch in range(1, args.learning_params["epochs"] + 1):
            print("*"*10, "="*10, "*"*10)
            print(f"[{experiment_name}] Epoch: {epoch}")

            train_loss = train_TCN_one_epoch(model, train_loader, criterion, optimizer, device, wandb_logger, args)
            wandb_logger.log({"avg_train_loss": train_loss, "epoch": epoch})
            print(f"Epoch: {epoch}, Train Loss: {train_loss}")

            val_loss = validate_TCN(model, val_loader, criterion, device, wandb_logger, args)
            print(f"Epoch: {epoch}, Val Loss: {val_loss}")

            if val_loss < min_val_loss:
                min_val_loss = val_loss
                best_model = model
                # torch.save(model.state_dict(), best_val_path)  # uncomment if you want real best-ckpt saving

            scheduler.step()

            epoch_dir = os.path.join(fold_results_dir, "epochs")
            os.makedirs(epoch_dir, exist_ok=True)
            if best_model is not None:
                print(f"Testing best model at epoch {epoch} on test set...")
                _ = test_TCN_model(best_model, test_loader, criterion, device, wandb_logger, epoch, epoch_dir, args)

            print("*"*10, "="*10, "*"*10)

        # Final test (optionally with best ckpt)
        if os.path.exists(best_val_path):
            model.load_state_dict(torch.load(best_val_path, map_location=device))
            print(f"Loaded best val checkpoint from: {best_val_path}")

        final_results = test_TCN_model(
            best_model, test_loader, criterion, device, wandb_logger, epoch="final", results_dir=fold_results_dir, args=args
        )
        print(f"[{experiment_name}] Final Test Results: {final_results}")

        fold_numeric = flatten_numeric(final_results)
        fold_numeric["fold_name"] = folds[fi]['name']
        fold_numeric["experiment_name"] = experiment_name
        write_json(os.path.join(fold_results_dir, "results_final.json"), fold_numeric)

        # accumulate for summary
        fold_rows_for_csv.append(fold_numeric)
        for k, v in fold_numeric.items():
            if k in ("fold_name", "experiment_name"):
                continue
            if is_number(v):
                numeric_metrics_per_key.setdefault(k, []).append(float(v))

        wandb_logger.finish()

    # Summary across folds
    summary_dir = os.path.join(run_root, "summary_all_folds")
    os.makedirs(summary_dir, exist_ok=True)

    write_csv(os.path.join(summary_dir, "fold_results.csv"), fold_rows_for_csv)

    summary_stats = {}
    for k, vals in numeric_metrics_per_key.items():
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / max(1, len(vals) - 1)
        std = var ** 0.5
        summary_stats[k] = {"mean": mean, "std": std, "n_folds": len(vals)}
    write_json(os.path.join(summary_dir, "summary_mean_std.json"), summary_stats)

    with open(os.path.join(summary_dir, "README.txt"), "w") as f:
        f.write(f"Job: {base_job_id}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Folds: {len(fold_indices)} of total {len(folds)}\n")
        f.write(f"Root run dir: {run_root}\n")
        f.write("Each fold folder is named after experiment_name (includes fold name).\n")
        f.write("This summary aggregates all folds into fold_results.csv and summary_mean_std.json.\n")
        f.write("\nConfig snapshot used (see run_root/config_used.*).\n")

    print("\n=== Cross-validation complete ===")
    print(f"Per-fold results under: {run_root}")
    print(f"Summary written to:     {summary_dir}")
