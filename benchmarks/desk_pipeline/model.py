from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50ForGestures(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool = True):
		super().__init__()
		# Use the latest torchvision weights API
		if pretrained:
			weights = models.ResNet50_Weights.DEFAULT
			backbone = models.resnet50(weights=weights)
		else:
			backbone = models.resnet50(weights=None)
		# Replace the classification head
		in_features = backbone.fc.in_features
		backbone.fc = nn.Identity()
		self.backbone = backbone
		self.classifier = nn.Linear(in_features, num_classes)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		features = self.backbone(x)  # shape [B, 2048]
		logits = self.classifier(features)
		return logits, features
