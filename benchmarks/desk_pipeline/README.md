# DESK Gesture Classification Pipeline

This project trains a ResNet50 to classify DESK Peg Transfer gestures (S1..S7) from single frames, using 1 Hz frame extraction and robust PyTorch training utilities.

## Layout
- `config.yaml`: Human-editable settings (paths, training, optim, scheduler, etc.)
- `preprocess.py`: Extract 1 Hz frames and build a consolidated `index.csv`
- `dataset.py`: Dataset and stratified split utilities
- `model.py`: ResNet50 wrapper with latest ImageNet weights
- `train.py`: Training loop with AMP, AdamW, cosine schedule, early stopping, checkpointing, metrics
- `main.py`: Orchestrates preprocess -> train -> evaluate
- `utils/`: Config loader and logger

## Quickstart
1. Place DESK dataset under `./DESK` with `gestures/` and `video/` per spec.
2. Edit `config.yaml` paths if needed.
3. Install requirements:
```bash
pip install -r desk_pipeline/requirements.txt
```
4. Preprocess frames and index:
```bash
python -m desk_pipeline.preprocess --config desk_pipeline/config.yaml
```
5. Train:
```bash
python -m desk_pipeline.train --config desk_pipeline/config.yaml
```
6. End-to-end:
```bash
python -m desk_pipeline.main --config desk_pipeline/config.yaml
```

Best checkpoint path will be reported at the end of training and written into `best_checkpoint.txt` under the checkpoints directory.

## Inference and Feature Extraction

### Standalone Inference Script
For single image prediction:
```bash
python desk_pipeline/standalone_inference.py --model path/to/best.pth --image path/to/image.jpg
```

### Standalone Feature Extractor
For extracting features from images:
```bash
python desk_pipeline/standalone_feature_extractor.py --model path/to/best.pth --image path/to/image.jpg --output features.npy
```

### Using the Feature Extractor Class
```python
from desk_pipeline.feature_extractor import DESKFeatureExtractor

# Initialize extractor
extractor = DESKFeatureExtractor("path/to/best.pth", device="cuda")

# Extract features from single image
features = extractor.extract_features("path/to/image.jpg")
print(f"Features shape: {features.shape}")  # [2048]

# Extract features from multiple images
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
batch_features = extractor.extract_features_batch(image_paths)
print(f"Batch features shape: {batch_features.shape}")  # [N, 2048]
```
