import argparse
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from tqdm import tqdm

from .utils.config import load_config
from .utils.logger import setup_logger
from .dataset import DeskFramesDataset, build_transforms, stratified_split
from .model import ResNet50ForGestures


def set_seed(seed: int):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)


def save_confmat(cm: np.ndarray, class_names: List[str], out_path: str):
	import matplotlib.pyplot as plt
	import seaborn as sns
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()


def train_one_epoch(model, loader, optimizer, scaler, device, step_log_interval, logger):
	model.train()
	all_preds: List[int] = []
	all_tgts: List[int] = []
	accum_loss = 0.0
	for step, (images, targets) in enumerate(loader, 1):
		images = images.to(device, non_blocking=True)
		targets = targets.to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			logits, _ = model(images)
			loss = F.cross_entropy(logits, targets)
		if scaler is not None:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()
		accum_loss += loss.item()
		preds = logits.argmax(dim=1)
		all_preds.extend(preds.detach().cpu().tolist())
		all_tgts.extend(targets.detach().cpu().tolist())
		if step % step_log_interval == 0:
			acc = accuracy_score(all_tgts, all_preds)
			prec, rec, f1, _ = precision_recall_fscore_support(all_tgts, all_preds, average='macro', zero_division=0)
			logger.info(f"Step {step}: train_loss={(accum_loss/step):.4f} acc={acc:.4f} prec={prec:.4f} rec={rec:.4f} f1={f1:.4f}")
	acc = accuracy_score(all_tgts, all_preds)
	prec, rec, f1, _ = precision_recall_fscore_support(all_tgts, all_preds, average='macro', zero_division=0)
	return (accum_loss/len(loader)), acc, prec, rec, f1


def evaluate(model, loader, device):
	model.eval()
	all_preds: List[int] = []
	all_tgts: List[int] = []
	accum_loss = 0.0
	with torch.no_grad():
		for images, targets in loader:
			images = images.to(device, non_blocking=True)
			targets = targets.to(device, non_blocking=True)
			logits, _ = model(images)
			loss = F.cross_entropy(logits, targets)
			accum_loss += loss.item()
			preds = logits.argmax(dim=1)
			all_preds.extend(preds.detach().cpu().tolist())
			all_tgts.extend(targets.detach().cpu().tolist())
	acc = accuracy_score(all_tgts, all_preds)
	prec, rec, f1, _ = precision_recall_fscore_support(all_tgts, all_preds, average='macro', zero_division=0)
	cm = confusion_matrix(all_tgts, all_preds, labels=list(range(7)))
	return (accum_loss/len(loader)), acc, prec, rec, f1, cm


def build_optimizer(model, cfg):
	name = cfg["optim"]["name"].lower()
	lr = float(cfg["optim"]["lr"])
	weight_decay = float(cfg["optim"]["weight_decay"])
	betas = tuple(cfg["optim"]["betas"])
	eps = float(cfg["optim"]["eps"])
	if name == "adamw":
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
	raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
	name = cfg["scheduler"]["name"].lower()
	warmup_epochs = int(cfg["scheduler"]["warmup_epochs"])
	min_lr = float(cfg["scheduler"]["min_lr"])
	max_epochs = int(cfg["training"]["epochs"])
	if name == "cosine":
		# Use ReduceLROnPlateau for validation-based scheduling
		return torch.optim.lr_scheduler.ReduceLROnPlateau(
			optimizer, 
			mode='max', 
			factor=0.5, 
			patience=3, 
			min_lr=min_lr
		)
	raise ValueError(f"Unsupported scheduler: {name}")


def save_checkpoint(state: Dict, is_best: bool, ckpt_dir: str, epoch: int, keep_all: bool, logger):
	os.makedirs(ckpt_dir, exist_ok=True)
	path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth")
	torch.save(state, path)
	logger.info(f"Saved checkpoint: {path}")
	
	# Keep all checkpoints if requested
	if not keep_all:
		# Clean up older checkpoints if not keeping all
		ckpts = sorted([p for p in os.listdir(ckpt_dir) if p.endswith('.pth') and 'best' not in p])
		if len(ckpts) > 3:  # Keep last 3 regular checkpoints
			to_remove = ckpts[:-3]
			for name in to_remove:
				try:
					os.remove(os.path.join(ckpt_dir, name))
				except OSError:
					pass
	
	if is_best:
		best_path = os.path.join(ckpt_dir, "best.pth")
		import shutil
		shutil.copy(path, best_path)
		with open(os.path.join(ckpt_dir, "best_checkpoint.txt"), "w", encoding="utf-8") as f:
			f.write(best_path)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, required=True)
	args = parser.parse_args()
	cfg = load_config(args.config)
	paths = cfg["paths"]
	logger = setup_logger("train", paths["logs_dir"], filename="train.log")

	set_seed(int(cfg["training"]["seed"]))
	device = cfg["training"]["device"]
	batch_size = int(cfg["training"]["batch_size"])
	num_workers = int(cfg["training"]["num_workers"])
	pin_mem = bool(cfg["training"]["pin_memory"])
	amp_enabled = bool(cfg["training"]["amp"]) and torch.cuda.is_available() and device == "cuda"

	# Load index
	index_csv = paths["index_csv"]
	df = pd.read_csv(index_csv)
	train_tf, test_tf = build_transforms(cfg)

	split = stratified_split(
		df, 
		train_size=float(cfg["split"]["train_size"]),
		val_size=float(cfg["split"]["val_size"]),
		test_size=float(cfg["split"]["test_size"]),
		random_state=int(cfg["split"]["random_state"]), 
		stratify_by=str(cfg["split"]["stratify_by"])
	)

	train_ds = DeskFramesDataset(df.iloc[split.train_idx].reset_index(drop=True), transform=train_tf)
	val_ds = DeskFramesDataset(df.iloc[split.val_idx].reset_index(drop=True), transform=test_tf)
	test_ds = DeskFramesDataset(df.iloc[split.test_idx].reset_index(drop=True), transform=test_tf)

	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
	test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

	model = ResNet50ForGestures(num_classes=int(cfg["training"]["num_classes"]), pretrained=True, freeze_early_layers=True)
	model.to(device)
	
	# Log trainable parameters
	trainable_params = model.get_trainable_params()
	total_params = sum(p.numel() for p in model.parameters())
	logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable ({trainable_params/total_params*100:.1f}%)")

	optimizer = build_optimizer(model, cfg)
	scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=max(1, len(train_loader)))
	scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

	best_metric_name = cfg["checkpoint"]["best_metric"]
	maximize = bool(cfg["checkpoint"]["maximize"])
	keep_all = bool(cfg["checkpoint"]["keep_all"])
	ckpt_dir = paths["ckpt_dir"]
	step_log_interval = int(cfg["logging"]["step_log_interval"])
	csv_log = bool(cfg["logging"]["csv_log"])

	csv_rows: List[Dict] = []
	best_value = -float("inf") if maximize else float("inf")
	patience = int(cfg["early_stopping"]["patience"]) if cfg["early_stopping"]["enabled"] else None
	min_delta = float(cfg["early_stopping"]["min_delta"]) if cfg["early_stopping"]["enabled"] else 0.0
	no_improve_epochs = 0

	for epoch in range(1, int(cfg["training"]["epochs"]) + 1):
		logger.info(f"Epoch {epoch}")
		start_t = time.time()
		train_loss, train_acc, train_prec, train_rec, train_f1 = train_one_epoch(model, train_loader, optimizer, scaler, device, step_log_interval, logger)
		val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = evaluate(model, val_loader, device)
		test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = evaluate(model, test_loader, device)
		
		# Step scheduler based on validation metric
		scheduler.step(val_f1)
		elapsed = time.time() - start_t

		logger.info(
			f"Epoch {epoch} done in {elapsed:.1f}s | "
			f"train_loss={train_loss:.4f} acc={train_acc:.4f} prec={train_prec:.4f} rec={train_rec:.4f} f1={train_f1:.4f} | "
			f"val_loss={val_loss:.4f} acc={val_acc:.4f} prec={val_prec:.4f} rec={val_rec:.4f} f1={val_f1:.4f} | "
			f"test_loss={test_loss:.4f} acc={test_acc:.4f} prec={test_prec:.4f} rec={test_rec:.4f} f1={test_f1:.4f}"
		)

		metric_value = {
			"val_acc": val_acc,
			"val_precision_macro": val_prec,
			"val_recall_macro": val_rec,
			"val_f1_macro": val_f1,
		}.get(best_metric_name, val_f1)

		is_better = (metric_value > best_value + min_delta) if maximize else (metric_value < best_value - min_delta)
		if is_better:
			best_value = metric_value
			no_improve_epochs = 0
		else:
			no_improve_epochs += 1

		# Save checkpoints
		state = {
			"epoch": epoch,
			"model_state": model.state_dict(),
			"optimizer_state": optimizer.state_dict(),
			"scheduler_state": scheduler.state_dict(),
			"scaler_state": scaler.state_dict() if scaler is not None else None,
			"best_value": best_value,
		}
		save_checkpoint(state, is_better, ckpt_dir, epoch, keep_all, logger)

		# Save confusion matrices for both val and test
		if cfg["logging"]["save_confusion_matrix"]:
			val_cm_path = os.path.join(paths["logs_dir"], f"confusion_matrix_val_epoch_{epoch:03d}.png")
			test_cm_path = os.path.join(paths["logs_dir"], f"confusion_matrix_test_epoch_{epoch:03d}.png")
			save_confmat(val_cm, [f"S{i}" for i in range(1,8)], val_cm_path)
			save_confmat(test_cm, [f"S{i}" for i in range(1,8)], test_cm_path)

		# CSV log
		if csv_log:
			csv_rows.append({
				"epoch": epoch,
				"train_loss": train_loss,
				"train_acc": train_acc,
				"train_prec": train_prec,
				"train_rec": train_rec,
				"train_f1": train_f1,
				"val_loss": val_loss,
				"val_acc": val_acc,
				"val_prec": val_prec,
				"val_rec": val_rec,
				"val_f1": val_f1,
				"test_loss": test_loss,
				"test_acc": test_acc,
				"test_prec": test_prec,
				"test_rec": test_rec,
				"test_f1": test_f1,
				"best_value": best_value,
				"lr": optimizer.param_groups[0]["lr"],
			})
			pd.DataFrame(csv_rows).to_csv(os.path.join(paths["logs_dir"], "metrics.csv"), index=False)

		# Early stopping
		if patience is not None and no_improve_epochs >= patience:
			logger.info(f"Early stopping after {epoch} epochs. Best {best_metric_name}={best_value:.4f}")
			break

	best_ckpt_txt = os.path.join(ckpt_dir, "best_checkpoint.txt")
	if os.path.exists(best_ckpt_txt):
		with open(best_ckpt_txt, "r", encoding="utf-8") as f:
			logger.info(f"Best checkpoint: {f.read().strip()}")


if __name__ == "__main__":
	main()
