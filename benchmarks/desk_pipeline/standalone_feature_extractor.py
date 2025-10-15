"""
Standalone feature extractor that can be run independently.
No package imports required - just run: python standalone_feature_extractor.py
"""

import argparse
import os
import sys
from typing import Union, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

GESTURE_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


class ResNet50ForGestures(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = models.ResNet50_Weights.DEFAULT
            backbone = models.resnet50(weights=weights)
        else:
            backbone = models.resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


class DESKFeatureExtractor:
    """Feature extractor that loads a trained DESK model and returns penultimate features."""
    
    def __init__(self, checkpoint_path: str, device: str = "cuda", image_size: int = 224):
        self.device = device
        self.image_size = image_size
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self._setup_transforms()
        
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model and remove classification head."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = ResNet50ForGestures(num_classes=7, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        
        # Remove classification head, keep only backbone
        model.classifier = nn.Identity()
        model.to(self.device)
        model.eval()
        
        print("Model loaded successfully, classification head removed")
        return model
    
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        self.transforms = T.Compose([
            T.Resize(int(self.image_size * 1.14)),
            T.CenterCrop(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path: str) -> torch.Tensor:
        """Extract features from a single image."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image_tensor)
        
        return features.squeeze(0)
    
    def extract_features_batch(self, image_paths: list) -> torch.Tensor:
        """Extract features from multiple images."""
        features_list = []
        
        for image_path in image_paths:
            try:
                features = self.extract_features(image_path)
                features_list.append(features)
            except Exception as e:
                print(f"Warning: Failed to process {image_path}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No images were successfully processed")
        
        return torch.stack(features_list)
    
    def extract_features_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract features from preprocessed image tensor."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            features = self.model(image_tensor)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted features."""
        return 2048


def main():
    """Example usage of the feature extractor."""
    parser = argparse.ArgumentParser(description="DESK Feature Extractor")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="features.npy", help="Output path for features")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    try:
        # Create feature extractor
        extractor = DESKFeatureExtractor(args.model, args.device)
        
        # Extract features
        features = extractor.extract_features(args.image)
        
        # Save features
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        np.save(args.output, features.cpu().numpy())
        
        print(f"Features extracted: {features.shape}")
        print(f"Features saved to: {args.output}")
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
