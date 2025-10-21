from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50ForGestures(nn.Module):
	def __init__(self, num_classes: int, pretrained: bool = True, freeze_early_layers: bool = True):
		super().__init__()
		# Use the latest torchvision weights API
		if pretrained:
			weights = models.ResNet50_Weights.DEFAULT
			backbone = models.resnet50(weights=weights)
		else:
			backbone = models.resnet50(weights=None)
		
		# Freeze early layers (conv1, bn1, layer1, layer2) if requested
		if freeze_early_layers:
			# Freeze conv1 and bn1
			for param in backbone.conv1.parameters():
				param.requires_grad = False
			for param in backbone.bn1.parameters():
				param.requires_grad = False
			
			# Freeze layer1 and layer2 (early feature extraction)
			for param in backbone.layer1.parameters():
				param.requires_grad = False
			for param in backbone.layer2.parameters():
				param.requires_grad = False
			
			# OPTION 1: Freeze layer3 as well (more aggressive)
			for param in backbone.layer3.parameters():
				param.requires_grad = False
			
			# Keep layer3 and layer4 trainable (later feature extraction layers)
			# These are important for downstream tasks
		
		# Replace the classification head
		in_features = backbone.fc.in_features
		backbone.fc = nn.Identity()
		self.backbone = backbone
		self.classifier = nn.Linear(in_features, num_classes)

	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		features = self.backbone(x)  # shape [B, 2048]
		logits = self.classifier(features)
		return logits, features
	
	def get_trainable_params(self):
		"""Get count of trainable parameters for logging."""
		return sum(p.numel() for p in self.parameters() if p.requires_grad)
