# train_kfold.py
from utils.utils import *
from scripts.config import build_cv_splits, build_model_cfg_from_dataloader  # <- only import helpers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import wandb
from datetime import datetime

from datautils.mstscn_dataset import BatchGenerator, Trainer

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

        self.dataloader_params["train_trials"] = {}
        self.dataloader_params["val_trials"]   = {}
        self.dataloader_params["test_trials"]  = {}



        self.learning_params     = dict(cfg["learning_params"])
        self.transformer_params  = dict(cfg.get("transformer_params", {}))
        self.tcn_model_params    = dict(cfg.get("tcn_model_params", {}))
        self.record_results      = bool(cfg.get("record_results", True))


if __name__ == "__main__":

    # --- CLI ---
    cmd_args = argparse.Namespace()
    cmd_args.fold_index = 1

    cmd_args.config = "./configs/RAVEN_DATA/exp1.yaml"
    # load the per-job config (immutable snapshot)
    cfg = load_config_file(cmd_args.config)
    print("Loaded config:")
    print(cfg)

    # --- Setup args ---
    args = DefaultArgsNamespace(cfg, fold_index=cmd_args.fold_index)

    # Data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    args.dataloader_params['train_class_stats'] = {}
    args.dataloader_params['val_class_stats'] = {}

    args.dataset = "50salads"
    args.split = "1"
    args.num_epochs = 50
    args.features_dim = 2048
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if args.dataset == "50salads":
        sample_rate = 2

    rivanna_standard_root = "/standard/UVA-DSA/MSTCN/"
    vid_list_file = rivanna_standard_root +args.dataset+"/splits/train.split"+args.split+".bundle"
    vid_list_file_tst = rivanna_standard_root +args.dataset+"/splits/test.split"+args.split+".bundle"
    features_path = rivanna_standard_root +args.dataset+"/features/"
    gt_path = rivanna_standard_root +args.dataset+"/groundTruth/"

    mapping_file = rivanna_standard_root +args.dataset+"/mapping.txt"

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)


    # Initial model (with dummy dims)
    args.transformer_params['input_dim'] = args.features_dim
    args.transformer_params['output_dim'] = num_classes


    model_dir = "./models/" + args.dataset + "/split_" + args.split
    results_dir = "./results/" + args.dataset + "/split_" + args.split
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    num_epochs = 50
    batch_size = 1
    lr = 0.001
    model, optimizer, criterion = init_model(args, device)

    # Dataset
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)

    batch_gen.read_data(vid_list_file)
    batch_input_tensor, batch_target_tensor, mask = batch_gen.next_batch(1)
    # print(batch_input_tensor.size())
    # print(batch_target_tensor.size())
    # print(mask.size())

    # def __init__(self, model, num_classes, dataset, split):

    trainer = Trainer(model, num_classes, args.dataset, args.split)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=batch_size, learning_rate=lr, device=device)

    # trainer.sanity_check(model, batch_gen, device)
