import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


GESTURE_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


@dataclass
class SplitIndices:
	train_idx: List[int]
	test_idx: List[int]


class DeskFramesDataset(Dataset):
	def __init__(self, index_df: pd.DataFrame, transform: Optional[Callable] = None):
		self.df = index_df.reset_index(drop=True)
		self.transform = transform
		self.num_classes = len(GESTURE_NAMES)

	def __len__(self) -> int:
		return len(self.df)

	def __getitem__(self, idx: int):
		row = self.df.iloc[idx]
		img = Image.open(row["image_path"]).convert("RGB")
		if self.transform:
			img = self.transform(img)
		label = int(row["label"])
		return img, label


def build_transforms(cfg: Dict) -> Tuple[Callable, Callable]:
	image_size = int(cfg["dataset"]["image_size"])
	mean = cfg["dataset"]["mean"]
	std = cfg["dataset"]["std"]
	scale = cfg["dataset"]["augment"]["random_resized_crop_scale"]
	hflip_p = float(cfg["dataset"]["augment"]["horizontal_flip_prob"])
	cj_b, cj_c, cj_s, cj_h = cfg["dataset"]["augment"]["color_jitter"]

	train_tf = T.Compose([
		T.RandomResizedCrop(image_size, scale=tuple(scale)),
		T.RandomHorizontalFlip(p=hflip_p),
		T.ColorJitter(brightness=cj_b, contrast=cj_c, saturation=cj_s, hue=cj_h),
		T.ToTensor(),
		T.Normalize(mean=mean, std=std),
	])

	test_tf = T.Compose([
		T.Resize(int(image_size * 1.14)),
		T.CenterCrop(image_size),
		T.ToTensor(),
		T.Normalize(mean=mean, std=std),
	])

	return train_tf, test_tf


def stratified_split(df: pd.DataFrame, test_size: float, random_state: int, stratify_by: str = "label") -> SplitIndices:
	from sklearn.model_selection import train_test_split
	idx = list(range(len(df)))
	labels = df[stratify_by]
	train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, stratify=labels)
	return SplitIndices(train_idx=train_idx, test_idx=test_idx)
