"""
Feature extractor class for DESK gesture classification model.
Removes classification head and returns penultimate features.
"""

import os
from typing import Union, Optional

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .model import ResNet50ForGestures
from .utils.logger import setup_logger


class DESKFeatureExtractor:
    """
    Feature extractor that loads a trained DESK model and returns penultimate features.
    """
    
    def __init__(self, checkpoint_path: str, device: str = "cuda", image_size: int = 224):
        """
        Initialize feature extractor.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run on (cuda/cpu)
            image_size: Input image size for preprocessing
        """
        self.device = device
        self.image_size = image_size
        self.logger = setup_logger("feature_extractor", "./logs", filename="feature_extractor.log")
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        self._setup_transforms()
        
    def _load_model(self, checkpoint_path: str) -> nn.Module:
        """Load model and remove classification head."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create model and load state
        model = ResNet50ForGestures(num_classes=7, pretrained=False)
        model.load_state_dict(checkpoint["model_state"])
        
        # Remove classification head, keep only backbone
        model.classifier = nn.Identity()
        model.to(self.device)
        model.eval()
        
        self.logger.info("Model loaded successfully, classification head removed")
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
        """
        Extract features from a single image.
        
        Args:
            image_path: Path to input image
            
        Returns:
            features: Extracted features tensor (shape: [2048])
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transforms(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
        
        return features.squeeze(0)  # Remove batch dimension
    
    def extract_features_batch(self, image_paths: list) -> torch.Tensor:
        """
        Extract features from multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            features: Extracted features tensor (shape: [N, 2048])
        """
        features_list = []
        
        for image_path in image_paths:
            try:
                features = self.extract_features(image_path)
                features_list.append(features)
            except Exception as e:
                self.logger.warning(f"Failed to process {image_path}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No images were successfully processed")
        
        return torch.stack(features_list)
    
    def extract_features_from_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features from preprocessed image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor (shape: [C, H, W] or [B, C, H, W])
            
        Returns:
            features: Extracted features tensor
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            features = self.model(image_tensor)
        
        return features
    
    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted features."""
        return 2048  # ResNet50 backbone output dimension


def main():
    """Example usage of the feature extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DESK Feature Extractor")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output", type=str, default="features.npy", help="Output path for features")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    args = parser.parse_args()
    
    # Create feature extractor
    extractor = DESKFeatureExtractor(args.model, args.device)
    
    # Extract features
    features = extractor.extract_features(args.image)
    
    # Save features
    import numpy as np
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.save(args.output, features.cpu().numpy())
    
    print(f"Features extracted: {features.shape}")
    print(f"Features saved to: {args.output}")


if __name__ == "__main__":
    main()
