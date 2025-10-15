"""
Standalone inference script that can be run independently.
No package imports required - just run: python standalone_inference.py
"""

import argparse
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits, features


def load_model(checkpoint_path: str, device: str = "cuda") -> ResNet50ForGestures:
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ResNet50ForGestures(num_classes=7, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model


def get_transforms(image_size: int = 224) -> T.Compose:
    """Get inference transforms."""
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_image(model: ResNet50ForGestures, image_path: str, device: str = "cuda") -> Tuple[str, float, torch.Tensor]:
    """Predict gesture for a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    transforms = get_transforms()
    image_tensor = transforms(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits, features = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    gesture_name = GESTURE_NAMES[predicted_class]
    return gesture_name, confidence, features.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="DESK Gesture Classification Inference")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--save-features", type=str, default=None, help="Path to save features (.npy file)")
    args = parser.parse_args()
    
    try:
        print(f"Loading model from {args.model}")
        model = load_model(args.model, args.device)
        print("Model loaded successfully")
        
        print(f"Processing image: {args.image}")
        gesture, confidence, features = predict_image(model, args.image, args.device)
        
        print(f"Predicted Gesture: {gesture}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Features shape: {features.shape}")
        
        if args.save_features:
            os.makedirs(os.path.dirname(args.save_features), exist_ok=True)
            np.save(args.save_features, features.cpu().numpy())
            print(f"Features saved to {args.save_features}")
            
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
