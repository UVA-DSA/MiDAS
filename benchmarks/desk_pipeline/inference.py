"""
Standalone inference script for DESK gesture classification.
Usage: python inference.py --model path/to/checkpoint.pth --image path/to/image.jpg
"""

import argparse
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from .model import ResNet50ForGestures
from .utils.config import load_config
from .utils.logger import setup_logger

GESTURE_NAMES = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def load_model(checkpoint_path: str, device: str = "cuda") -> ResNet50ForGestures:
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model and load state
    model = ResNet50ForGestures(num_classes=7, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    return model


def get_transforms(image_size: int = 224) -> T.Compose:
    """Get inference transforms (no augmentation)."""
    return T.Compose([
        T.Resize(int(image_size * 1.14)),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def predict_image(model: ResNet50ForGestures, image_path: str, device: str = "cuda") -> Tuple[str, float, torch.Tensor]:
    """
    Predict gesture for a single image.
    
    Returns:
        gesture_name: Predicted gesture (S1-S7)
        confidence: Prediction confidence
        features: Extracted features (2048-dim vector)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    transforms = get_transforms()
    image_tensor = transforms(image).unsqueeze(0).to(device)
    
    # Run inference
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
    
    # Setup logging
    logger = setup_logger("inference", "./logs", filename="inference.log")
    
    try:
        # Load model
        logger.info(f"Loading model from {args.model}")
        model = load_model(args.model, args.device)
        logger.info("Model loaded successfully")
        
        # Run inference
        logger.info(f"Processing image: {args.image}")
        gesture, confidence, features = predict_image(model, args.image, args.device)
        
        # Print results
        print(f"Predicted Gesture: {gesture}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Features shape: {features.shape}")
        
        # Save features if requested
        if args.save_features:
            import numpy as np
            os.makedirs(os.path.dirname(args.save_features), exist_ok=True)
            np.save(args.save_features, features.cpu().numpy())
            logger.info(f"Features saved to {args.save_features}")
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
