#import from models folder transtcn
from models.mmtransformer import ModularMultimodalTransformer
from models.transtcn import TransformerModel, MultimodalFusion
from models.tcn import TCNClassifier, MultiBranchTCNClassifier
from models.colintcn import EncoderDecoderNet
from models.mstcn import MS_TCN2
from models.multitranstcn import MultiTransTCN, build_default_multitranstcn
from models.simple_mlp import create_model
import torch
from datautils.ems import *
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
import csv
from functools import partial
import torch.nn.functional as F

from datautils.midas import MultimodalGestureDataset
from datautils.desk_dataset import DeskDataset
import torchvision.transforms as tfs
import numpy as np
import os
from typing import List, Tuple
from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import DataLoader, Subset

import random

import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    balanced_accuracy_score, cohen_kappa_score,
    roc_auc_score, top_k_accuracy_score
)
import json
from pathlib import Path
import traceback

from scipy.signal import butter, filtfilt
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
def lowpass_filter(data, cutoff=2.5, fs=30.0, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, axis=1)
    return y

def seed_everything(seed: int = 42):
    """
    Ensure reproducible training across Python, NumPy, and PyTorch.
    Sets seeds and some deterministic flags.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # For deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] Global seed set to {seed}")

def _labels_for_dataset_windows(ds) -> np.ndarray:
    """Build integer labels per window from ds.samples and ds.class_map (no __getitem__)."""
    return np.asarray([ds.class_map[s["gesture_code"]] for s in ds.samples], dtype=np.int64)

def stratified_gesture_train_test(
    y: np.ndarray,
    test_ratio: float,
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Return stratified (train_idx, test_idx)."""
    try:
        X = np.arange(len(y))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_idx, test_idx = next(sss.split(X, y))
        return train_idx.tolist(), test_idx.tolist()
    except Exception:
        # Fallback: manual per-class split
        rng = np.random.default_rng(seed)
        tr, te = [], []
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            rng.shuffle(idx)
            n = len(idx)
            n_te = max(1, int(round(n * test_ratio))) if n > 1 else 0
            te.extend(idx[:n_te].tolist())
            tr.extend(idx[n_te:].tolist())
        return tr, te


fusion = MultimodalFusion()

class ClassBalancedLoss(nn.Module):
    def __init__(self, beta, num_classes, class_counts):
        super(ClassBalancedLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.class_counts = torch.Tensor(class_counts)
        self.weights = (1 - beta) / (1 - beta ** self.class_counts)
        self.weights = self.weights / self.weights.sum()  # Normalize weights

    def forward(self, logits, labels):
        weights = self.weights.to(logits.device)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.nll_loss(log_probs, labels, weight=weights)
        return loss

def init_model(args, device):
    model = TransformerModel(args)
    model.to(device)

    num_classes = len(args.dataloader_params["keysteps"])
    class_counts = args.dataloader_params["train_class_stats"]
    val_class_counts = args.dataloader_params["val_class_stats"]

    print("Training class counts: ", class_counts)
    print("Validation class counts: ", val_class_counts)
    # update class_counts with missing classes from keysteps with 0 count
    for key in args.dataloader_params["keysteps"].keys():
        if key not in class_counts.keys():
            class_counts[key] = 0

    # reorganize class_counts to match the order of the keysteps
    class_counts = {key: class_counts[key] for key in args.dataloader_params["keysteps"].keys()}

    # convert dictionary values to list
    class_counts = [class_counts[key] for key in class_counts.keys()]
    class_counts = torch.Tensor([max(1, count) for count in class_counts])

    print("Class counts: ", class_counts, len(class_counts))
           
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_params["lr"], weight_decay=args.learning_params["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Class balanced loss
    # criterion = ClassBalancedLoss(beta=0.99, num_classes=num_classes, class_counts=class_counts)
        
    return model, optimizer, criterion


def preprocess(x, modality, backbone, device, task='classification'):
    global fusion
    # print("-*" * 10, "Preprocessing", "*" * 10, "=" * 10)
    # check the shape of the input tensor
    feature = None
    label = x['keystep_id']
    # print(f"\nSubject ID: {x['subject_id']}, Trial ID: {x['trial_id']}, Start Frame: {x['start_frame']}, End Frame: {x['end_frame']}, Start Time: {x['start_t']}, End Time: {x['end_t']}")

    if task == 'segmentation':
        majority_label, _ = torch.mode(label, dim=1)  # [batch_size], mode returns (values, indices)
        label = majority_label


    if('video' in modality):
        feature = None
        x = x['frames']
        # extract resnet50 features
        x = x.to(device)
        x = backbone.extract_resnet(x)
        feature = x

    elif ( 'audio' in modality and  'resnet_ego' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet = resnet.to(device)

        smartwatch = x['smartwatch'].float()
        smartwatch = smartwatch.to(device)
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        audio = x['audio']
        audio = audio.to(device)
        # print("Raw Audio shape: ", audio.shape)
        audio_feature = backbone.extract_wav2vec_features(audio, multimodal=True) # for wav2vec features
        # print("Resnet feature shape: ", resnet.shape)
        # print("Audio feature shape: ", audio_feature.shape)
        # print("Smartwatch feature shape: ", smartwatch.shape)
        # print("Resnet feature shape: ", resnet.shape)

        fusion = fusion.to(device)
        fused = fusion(audio_feature, resnet, smartwatch)  # [B, T_common, D_total]


        feature = fused.float()

        # feature = torch.cat((resnet, audio_feature, smartwatch), dim=-1).float()

    elif ( 'audio' in modality and  'resnet_ego' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet = resnet.to(device)

        audio = x['audio']
        audio = audio.to(device)
        # print("Raw Audio shape: ", audio.shape)
        # audio_feature = backbone.extract_wav2vec_features(audio, multimodal=True) # for wav2vec features
        # print("Resnet feature shape: ", resnet.shape)
        audio_feature = backbone.extract_mel_spectrogram(audio, multimodal=True) # for mel spectrogram features

        feature = torch.cat((resnet, audio_feature), dim=1).float()

    elif ( 'flow' in modality and  'rgb' in modality and  'smartwatch' in modality):

        # I3D features are already extracted
        flow = x['flow'].float()
        rgb = x['rgb'].float()
        smartwatch = x['smartwatch'].float()

        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        # concatenate all features
        feature = torch.cat((flow, rgb, smartwatch), dim=-1).float()
        
    elif ( 'flow' in modality and  'rgb' in modality):

        # I3D features are already extracted
        flow = x['flow'].float()
        rgb = x['rgb'].float()
        feature = torch.cat((flow, rgb), dim=-1).float()

    elif ('resnet_ego' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        feature = torch.cat((resnet, smartwatch), dim=-1).float()

    elif ('resnet_ego' in modality and 'resnet_exo' in modality and 'smartwatch' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet_exo = x['resnet_exo'].float()
        smartwatch = x['smartwatch'].float()
        # normalize smartwatch data (batch, seq_len, 3) (3 = x,y,z)
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()

        feature = torch.cat((resnet, resnet_exo, smartwatch), dim=-1).float()


    elif ('resnet_ego' in modality and 'resnet_exo' in modality):
        # resnet50 features are already extracted
        resnet = x['resnet_ego'].float()
        resnet_exo = x['resnet_exo'].float()
        feature = torch.cat((resnet, resnet_exo), dim=-1).float()


    elif ('resnet_ego' in modality):
        # resnet50 features are already extracted
        feature = x['resnet_ego'].float()

    elif ('resnet_exo' in modality):
        # resnet50 features are already extracted
        feature = x['resnet_exo'].float()

    elif ('clip_ego' in modality):
        # resnet50 features are already extracted
        feature = x['clip_ego'].float()
        # print("Clip ego feature shape: ", feature.shape)
    elif ('clip_exo' in modality):
        # resnet50 features are already extracted
        feature = x['clip_exo'].float()
        # print("Clip exo feature shape: ", feature.shape)

    elif ('clip_ego' in modality and 'clip_exo' in modality):
        # resnet50 features are already extracted
        feature = torch.cat((x['clip_ego'].float(), x['clip_exo'].float()), dim=-1)
        # print("Clip ego and exo feature shape: ", feature.shape)

    elif ('rgb' in modality):
        # I3D features are already extracted
        feature = x['rgb'].float()

    elif ('flow' in modality):
        # I3D features are already extracted
        feature = x['flow'].float()

    # elif ('audio' in modality):
    #     # Audio features are already extracted

    #     # Example batch of audio clips (batch, samples, channels)
    #     audio_clips = x['audio']  # Assume shape [batch, samples, channels]
    #     audio_clips = audio_clips.to(device)
    #     feature = backbone.extract_mel_spectrogram(audio_clips)

    elif ('smartwatch' in modality):
        # Audio features are already extracted
        smartwatch = x['smartwatch'].float()
        smartwatch = (smartwatch - smartwatch.mean()) / smartwatch.std()
        feature = smartwatch

    elif ('audio' in modality): # uncomment this if you want to use wav2vec features
        audio_clips = x['audio']  # Assume shape [batch, samples, channels]
        audio_clips = audio_clips.to(device)
        # feature = backbone.extract_wav2vec_features(audio_clips)
        feature = backbone.extract_mel_spectrogram(audio_clips)

        # print("Wav2Vec feature shape: ", feature.shape)

    feature_size = feature.shape[-1]
    # print("Feature shape: ", feature.shape, "\n")

    if(feature is not None):
        feature = feature.to(device)
        label = label.to(device)

    return feature, feature_size, label


# add wandb logging
def train_one_epoch(model, train_loader, criterion, optimizer, device, logger, modality, task='classification'):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:
            # print("Batch: ", i)
            print("=" * 10, "-" * 10, "=" * 10)
            input,feature_size, label = preprocess(batch, modality, model, device, task=task)

            # get more info about input
            keystep_label = batch['keystep_label'] if task == 'segmentation' else batch['keystep_label'][0]
            keystep_id = batch['keystep_id'] if task == 'segmentation' else batch['keystep_id'][0]
            start_frame = batch['start_frame'] if task == 'segmentation' else batch['start_frame'][0]
            end_frame = batch['end_frame'] if task == 'segmentation' else batch['end_frame'][0]
            start_t = batch['start_t'] if task == 'segmentation' else batch['start_t'][0]
            end_t = batch['end_t'] if task == 'segmentation' else batch['end_t'][0]
            subject_id = batch['subject_id'] if task == 'segmentation' else batch['subject_id']
            trial_id = batch['trial_id'] if task == 'segmentation' else batch['trial_id']
            window_start_frame = batch['window_start_frame'] if task == 'segmentation' else torch.tensor(-1)
            window_end_frame = batch['window_end_frame'] if task == 'segmentation' else torch.tensor(-1)

            if task == 'segmentation':
                print(f"Subject ID: {subject_id[0][0]}, Trial ID: {trial_id[0][0]}, Start Frame: {start_frame[0][0]}, End Frame: {end_frame[0][0]}, Start Time: {start_t[0][0]}, End Time: {end_t[0][0]}")
                print(f"Keystep Label: {keystep_label[0][0]}, Keystep ID: {keystep_id[0][0]}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")

            else:
                print(f"Subject ID: {subject_id}, Trial ID: {trial_id}, Start Frame: {start_frame}, End Frame: {end_frame}, Start Time: {start_t}, End Time: {end_t}")
                print(f"Keystep Label: {keystep_label}, Keystep ID: {keystep_id}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")


                    # ←—— ADDED CHECK ———→
            # if the time-dimension is zero, skip this batch
            # (inputs.shape == [B, T, F] or [B, C, T] depending on your preprocess)
            if input.size(1) == 0 or (task== 'segmentation' and input.size(1) != 150 and "audio" not in modality): 
                print(f"Skipping batch {i}: feature sequence : {input.shape}")
                continue

            if torch.isnan(input).any():
                print(f"⚠️ Skipping batch {i} because NaN")
                continue

            optimizer.zero_grad()

            output = model(input)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 1 == 0:
                print("\n")
                print("*" * 10, "=" * 10, "*" * 10)
                print(f"Pred: {torch.argmax(output, dim=1)} GT: {label}")
                logger.log({"train_loss": loss.item()})
                print(f"Batch: {i}, Loss: {loss.item()}")
                print("*" * 10, "=" * 10, "*" * 10)
                print("\n")
            # break
        
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            # print(f"Batch data: {batch}")
            continue

    return total_loss / len(train_loader)


# validate the model 
def validate(model, val_loader, criterion, device, logger, modality, task='classification'):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                input,feature_size, label = preprocess(batch, modality, model, device, task=task)

                # check if the time-dimension is zero, skip this batch
                if input.size(1) == 0:
                    print(f"Skipping batch {i}: empty feature sequence : {input.shape}")
                    continue

                if torch.isnan(input).any():
                    print(f"⚠️ Skipping batch {i} because NaN")
                    continue

                output = model(input)
                loss = criterion(output, label)
                total_loss += loss.item()
                if i % 100 == 0:
                    logger.log({"val_loss": loss.item()})
            # break
            
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                print(f"Batch data: {batch}")
                continue
            
    return total_loss / len(val_loader)

import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    balanced_accuracy_score, cohen_kappa_score
)
import json
from pathlib import Path

def test_model(model, test_loader, criterion, device, logger, epoch, results_dir, modality, task='classification', class_names=None):
    """
    Enhanced model testing with comprehensive metrics for ML papers.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on
        logger: Logger (e.g., wandb)
        epoch: Current epoch number
        results_dir: Directory to save results
        modality: Data modality
        task: Task type ('classification' or 'segmentation')
        class_names: List of class names for better visualization (optional)
    """
    model.eval()
    total_loss = 0
    gt = []
    preds = []
    preds_detail = []
    all_outputs = []  # Store all raw outputs for additional analysis

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                input, feature_size, label = preprocess(batch, modality, model, device, task=task)
                print("=" * 10, "-" * 10, "=" * 10)
                print(f"[TEST] Batch: {i}")

                # Check if the time-dimension is zero, skip this batch
                if input.size(1) == 0:
                    print(f"Skipping batch {i}: empty feature sequence : {input.shape}")
                    continue

                # Get more info about input
                keystep_label = batch['keystep_label'] if task == 'segmentation' else batch['keystep_label'][0]
                keystep_id = batch['keystep_id'] if task == 'segmentation' else batch['keystep_id'][0]
                start_frame = batch['start_frame'] if task == 'segmentation' else batch['start_frame'][0]
                end_frame = batch['end_frame'] if task == 'segmentation' else batch['end_frame'][0]
                start_t = batch['start_t'] if task == 'segmentation' else batch['start_t'][0]
                end_t = batch['end_t'] if task == 'segmentation' else batch['end_t'][0]
                subject_id = batch['subject_id'] if task == 'segmentation' else batch['subject_id']
                trial_id = batch['trial_id'] if task == 'segmentation' else batch['trial_id']
                window_start_frame = batch['window_start_frame'] if task == 'segmentation' else torch.tensor(-1)
                window_end_frame = batch['window_end_frame'] if task == 'segmentation' else torch.tensor(-1)

                if task == 'segmentation':
                    print(f"Subject ID: {subject_id[0][0]}, Trial ID: {trial_id[0][0]}, Start Frame: {start_frame[0][0]}, End Frame: {end_frame[0][0]}, Start Time: {start_t[0][0]}, End Time: {end_t[0][0]}")
                    print(f"Keystep Label: {keystep_label[0][0]}, Keystep ID: {keystep_id[0][0]}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")
                else:
                    print(f"Subject ID: {subject_id}, Trial ID: {trial_id}, Start Frame: {start_frame}, End Frame: {end_frame}, Start Time: {start_t}, End Time: {end_t}")
                    print(f"Keystep Label: {keystep_label}, Keystep ID: {keystep_id}, Window Start Frame: {window_start_frame}, Window End Frame: {window_end_frame}")  
                
                if torch.isnan(input).any():
                    print(f"⚠️ Skipping batch {i} because NaN")
                    continue
                
                output = model(input)
                pred = torch.argmax(output, dim=1)
                print(f"Model Pred: {pred.item()}")

                gt.append(label.item())
                preds.append(pred.item())
                all_outputs.append(output.cpu().numpy())

                preds_detail.append({
                    "keystep_label": keystep_label,
                    "keystep_id": keystep_id.tolist(),
                    "start_frame": start_frame.tolist(),
                    "end_frame": end_frame.tolist(),
                    "start_t": start_t.tolist(),
                    "end_t": end_t.tolist(),
                    "window_start_frame": window_start_frame.item(),
                    "window_end_frame": window_end_frame.item(),
                    "subject_id": subject_id[0],
                    "trial_id": trial_id[0],
                    "pred_keystep_id": pred.item(),
                    "true_keystep_id": label.item(),
                    "all_preds": output.tolist()
                })

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                print(f"Batch data: {batch}")
                continue

    # Calculate comprehensive metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    
    # Macro and weighted metrics
    precision_macro = precision_score(gt, preds, average='macro', zero_division=0)
    recall_macro = recall_score(gt, preds, average='macro', zero_division=0)
    f1_macro = f1_score(gt, preds, average='macro', zero_division=0)
    
    precision_weighted = precision_score(gt, preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(gt, preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(gt, preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(gt, preds, average=None, zero_division=0)
    recall_per_class = recall_score(gt, preds, average=None, zero_division=0)
    f1_per_class = f1_score(gt, preds, average=None, zero_division=0)
    
    # Additional metrics
    balanced_acc = balanced_accuracy_score(gt, preds)
    kappa = cohen_kappa_score(gt, preds)
    
    # Confusion matrix
    cm = confusion_matrix(gt, preds)
    
    # Calculate per-class accuracy from confusion matrix
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    # Get unique classes
    unique_classes = sorted(list(set(gt + preds)))
    n_classes = len(unique_classes)
    
    # Generate class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in unique_classes]
    
    # Classification report
    class_report = classification_report(gt, preds, target_names=class_names, 
                                        labels=unique_classes, zero_division=0, 
                                        output_dict=True)
    
    # Prepare results dictionary
    results = {
        "epoch": epoch,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": kappa,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }
    
    # Add per-class metrics to results
    for i, class_idx in enumerate(unique_classes):
        results[f"class_{class_idx}_accuracy"] = class_accuracy[i]
        results[f"class_{class_idx}_precision"] = precision_per_class[i]
        results[f"class_{class_idx}_recall"] = recall_per_class[i]
        results[f"class_{class_idx}_f1"] = f1_per_class[i]
    
    # Log metrics to wandb
    logger.log(results)
    
    # Create results directory if it doesn't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics to CSV
    metrics_path = f'{results_dir}/metrics_epoch_{epoch}.csv'
    with open(metrics_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Epoch", epoch])
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["Balanced Accuracy", balanced_acc])
        writer.writerow(["Cohen's Kappa", kappa])
        writer.writerow(["Precision (Macro)", precision_macro])
        writer.writerow(["Recall (Macro)", recall_macro])
        writer.writerow(["F1 Score (Macro)", f1_macro])
        writer.writerow(["Precision (Weighted)", precision_weighted])
        writer.writerow(["Recall (Weighted)", recall_weighted])
        writer.writerow(["F1 Score (Weighted)", f1_weighted])
    
    # Save per-class metrics to CSV
    class_metrics_path = f'{results_dir}/class_metrics_epoch_{epoch}.csv'
    with open(class_metrics_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class", "Class_Name", "Accuracy", "Precision", "Recall", "F1-Score", "Support"])
        for i, class_idx in enumerate(unique_classes):
            support = cm[i].sum()
            writer.writerow([
                class_idx, 
                class_names[i],
                class_accuracy[i],
                precision_per_class[i],
                recall_per_class[i],
                f1_per_class[i],
                support
            ])
    
    # Save confusion matrix as CSV
    cm_path = f'{results_dir}/confusion_matrix_epoch_{epoch}.csv'
    np.savetxt(cm_path, cm, delimiter=',', fmt='%d')
    
    # Save confusion matrix with labels
    cm_labeled_path = f'{results_dir}/confusion_matrix_labeled_epoch_{epoch}.csv'
    with open(cm_labeled_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([''] + class_names)
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + row.tolist())
    
    # Visualize and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix_epoch_{epoch}.png', dpi=300)
    plt.close()
    
    # Visualize normalized confusion matrix (by row)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    plt.title(f'Normalized Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix_normalized_epoch_{epoch}.png', dpi=300)
    plt.close()
    
    # Save classification report
    report_path = f'{results_dir}/classification_report_epoch_{epoch}.json'
    with open(report_path, 'w') as f:
        json.dump(class_report, f, indent=4)
    
    # Save classification report as text
    report_text_path = f'{results_dir}/classification_report_epoch_{epoch}.txt'
    with open(report_text_path, 'w') as f:
        f.write(classification_report(gt, preds, target_names=class_names, 
                                     labels=unique_classes, zero_division=0))
    
    # Save detailed predictions to CSV
    preds_path = f'{results_dir}/predictions_epoch_{epoch}.csv'
    print("Saving predictions to: ", preds_path)
    with open(preds_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "keystep_label", "keystep_id", "start_frame", "end_frame", 
            "start_t", "end_t", "window_start_frame", "window_end_frame", 
            "subject_id", "trial_id", "true_keystep_id", "pred_keystep_id", 
            "correct", "all_preds"
        ])
        for pred in preds_detail:
            correct = pred["true_keystep_id"] == pred["pred_keystep_id"]
            writer.writerow([
                pred["keystep_label"], pred["keystep_id"], pred["start_frame"], 
                pred["end_frame"], pred["start_t"], pred["end_t"], 
                pred["window_start_frame"], pred["window_end_frame"], 
                pred["subject_id"], pred["trial_id"], pred["true_keystep_id"],
                pred["pred_keystep_id"], correct, pred["all_preds"]
            ])
    
    # Create summary statistics
    summary_path = f'{results_dir}/summary_epoch_{epoch}.txt'
    with open(summary_path, 'w') as f:
        f.write(f"Model Evaluation Summary - Epoch {epoch}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"  Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"  Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"\nMacro-averaged Metrics:\n")
        f.write(f"  Precision: {precision_macro:.4f}\n")
        f.write(f"  Recall: {recall_macro:.4f}\n")
        f.write(f"  F1-Score: {f1_macro:.4f}\n")
        f.write(f"\nWeighted-averaged Metrics:\n")
        f.write(f"  Precision: {precision_weighted:.4f}\n")
        f.write(f"  Recall: {recall_weighted:.4f}\n")
        f.write(f"  F1-Score: {f1_weighted:.4f}\n")
        f.write(f"\nPer-class Metrics:\n")
        for i, class_idx in enumerate(unique_classes):
            f.write(f"\n  {class_names[i]} (Class {class_idx}):\n")
            f.write(f"    Accuracy: {class_accuracy[i]:.4f}\n")
            f.write(f"    Precision: {precision_per_class[i]:.4f}\n")
            f.write(f"    Recall: {recall_per_class[i]:.4f}\n")
            f.write(f"    F1-Score: {f1_per_class[i]:.4f}\n")
            f.write(f"    Support: {cm[i].sum()}\n")
    
    print(f"\n{'='*50}")
    print(f"Evaluation Complete - Epoch {epoch}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*50}\n")
    
    return results

# return train,val,test dataloaders using the VideoDataset class
def get_dataloaders(args):
    train_dataset = VideoDataset(base_path=args.dataloader_params["base_path"], fold=args.dataloader_params["fold"], skip_frames=25, transform=tfs, clip_length_in_frames=args.dataloader_params["observation_window"], train=True)
    test_dataset = VideoDataset(base_path=args.dataloader_params["base_path"], fold=args.dataloader_params["fold"], skip_frames=25, transform=tfs, clip_length_in_frames=args.dataloader_params["observation_window"], train=False)

    split_indices_path = f'{args.dataloader_params["base_path"]}/val_test_split_indices_fold_0{args.dataloader_params["fold"]}.npz'

    if os.path.exists(split_indices_path):
        # Load pre-existing indices
        split_data = np.load(split_indices_path)
        val_indices = split_data['val_indices']
        test_indices = split_data['test_indices']
    else:
        # Create new split and save the indices
        total_size = len(test_dataset)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        val_size = int(0.5 * total_size)
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]

        # Save the indices for later use
        np.savez(split_indices_path, val_indices=val_indices, test_indices=test_indices)
    
        # Subset datasets based on indices
    val_dataset = torch.utils.data.Subset(test_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)


    # Create DataLoaders for training and validation subsets
    train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

    print("train dataset size: ", len(train_dataset))
    print("val dataset size: ", len(val_dataset))
    print("test dataset size: ", len(test_dataset))

    return train_loader, val_loader, test_loader



# ''' ***** EGOEXOEMS DATASET ***** '''


# # add wandb logging
# def eee_train_one_epoch(model, train_loader, criterion, optimizer, device, logger):
#     model.train()
#     total_loss = 0
#     for i, batch in enumerate(train_loader):

#         i3d_rgb_features = batch['rgb']
#         i3d_flow_features = batch['flow']

#         # move to device
#         i3d_rgb_features = i3d_rgb_features.to(device)
#         i3d_flow_features = i3d_flow_features.to(device)

#         # get labels
#         labels = batch['keystep_id']
#         labels = labels.to(device)

#         optimizer.zero_grad()
#         output = model(i3d_rgb_features)


#         loss = criterion(output, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#         if i % 1 == 0:
#             print("\n ***** ")
#             print(batch['frames'].shape, batch['audio'].shape, batch['flow'].shape, batch['rgb'].shape, batch['keystep_label'], batch['keystep_id'], batch['start_frame'], batch['end_frame'],batch['start_t'], batch['end_t'],  batch['subject_id'], batch['trial_id'])
#             print(f"Pred: {torch.argmax(output, dim=1)} GT: {labels}")
#             logger.log({"train_loss": loss.item()})
#             print(f"Batch: {i}, Loss: {loss.item()}")
#             print(" ***** \n")

#     return total_loss / len(train_loader)


# # validate the model 
# def eee_validate(model, val_loader, criterion, device, logger):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for i, batch in enumerate(val_loader):

#             i3d_rgb_features = batch['rgb']
#             i3d_flow_features = batch['flow']

#             # move to device
#             i3d_rgb_features = i3d_rgb_features.to(device)
#             i3d_flow_features = i3d_flow_features.to(device)

#             # get labels
#             labels = batch['keystep_id']
#             labels = labels.to(device)

#             output = model(i3d_rgb_features)

#             loss = criterion(output, labels)
#             total_loss += loss.item()
#             if i % 1 == 0:
#                 logger.log({"val_loss": loss.item()})

#     return total_loss / len(val_loader)


# # test the model
# def eee_test_model(model, test_loader, criterion, device, logger, epoch, results_dir):
#     model.eval()
#     total_loss = 0


#     accuracy = 0.0
#     gt = []
#     preds = []
    

#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):

#             i3d_rgb_features = batch['rgb']
#             i3d_flow_features = batch['flow']

#             # move to device
#             i3d_rgb_features = i3d_rgb_features.to(device)
#             i3d_flow_features = i3d_flow_features.to(device)

#             # get labels
#             labels = batch['keystep_id']
#             labels = labels.to(device)

#             output = model(i3d_rgb_features)
#             pred = torch.argmax(output, dim=1)
#             gt.append(labels.item())
#             preds.append(pred.item())
    
#     # Calculate metrics
#     accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
#     precision = precision_score(gt, preds, average='macro')
#     recall = recall_score(gt, preds, average='macro')
#     f1 = f1_score(gt, preds, average='macro')

#     # Log metrics to wandb
#     logger.log({
#         "test_accuracy": accuracy,
#         "test_precision": precision,
#         "test_recall": recall,
#         "test_f1": f1,
#         "epoch": epoch
#     })
    
#     # Save metrics to CSV
#     metrics_path = f'{results_dir}/metrics.csv'
#     with open(metrics_path, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         if not os.path.isfile(metrics_path):
#             writer.writerow(["epoch", "accuracy", "precision", "recall", "f1"])
#         writer.writerow([epoch, accuracy, precision, recall, f1])
    
#     return accuracy




# # return train,val,test dataloaders using the EgoExoEMSDataset class
# def eee_get_dataloaders(args):
    
#     if(args.dataloader_params["task"] == 'classification'):
#         print("*" * 10, "=" * 10, "*" * 10)
#         print("Loading dataloader for Classification task")

#         train_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         val_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         test_dataset = EgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])


#         train_class_stats = train_dataset._get_class_stats()
#         print("Train class stats: ", train_class_stats)
#         # print number of keys in the dictionary
#         print("Train Number of classes: ", len(train_class_stats.keys()))

#         val_class_stats = val_dataset._get_class_stats()
#         print("val class stats: ", val_class_stats)
#         # print number of keys in the dictionary
#         print("Val Number of classes: ", len(val_class_stats.keys()))

#         # Create DataLoaders for training and validation subsets
#         train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True)
#         test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)
#         val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False)

#         print("train dataset size: ", len(train_dataset))
#         print("val dataset size: ", len(val_dataset))
#         print("test dataset size: ", len(test_dataset))
    
#     elif (args.dataloader_params["task"] == 'segmentation'):
#         print("*" * 10, "=" * 10, "*" * 10)
#         print("Loading dataloader for Segmentation task")
        
#         train_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["train_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         val_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["val_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         test_dataset = WindowEgoExoEMSDataset(annotation_file=args.dataloader_params["test_annotation_path"],
#                                         data_base_path='',
#                                         fps=args.dataloader_params["fps"], frames_per_clip=args.dataloader_params["observation_window"], transform=transform, data_types=args.dataloader_params["modality"], task=args.dataloader_params["task"])

#         train_class_stats = train_dataset._get_class_stats()
#         print("Train class stats: ", train_class_stats)
#         # print number of keys in the dictionary
#         print("Train Number of classes: ", len(train_class_stats.keys()))

#         val_class_stats = val_dataset._get_class_stats()
#         print("val class stats: ", val_class_stats)
#         # print number of keys in the dictionary
#         print("Val Number of classes: ", len(val_class_stats.keys()))

#         test_class_stats = test_dataset._get_class_stats()
#         print("test class stats: ", test_class_stats)
#         # print number of keys in the dictionary
#         print("Test Number of classes: ", len(test_class_stats.keys()))

        
#         # Use a partial function or lambda to pass the frames_per_clip argument
#         collate_fn_with_args = partial(window_collate_fn, frames_per_clip=args.dataloader_params["observation_window"])

#         # Create DataLoaders for training and validation subsets
#         train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True, collate_fn=collate_fn_with_args)
#         test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False, collate_fn=collate_fn_with_args)
#         val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False, collate_fn=collate_fn_with_args)

#         print("train dataset size: ", len(train_dataset))
#         print("val dataset size: ", len(val_dataset))
#         print("test dataset size: ", len(test_dataset))

#     return train_loader, val_loader, test_loader, train_class_stats, val_class_stats




# return train,val,test dataloaders using the MIDAS dataset class
def MIDAS_get_dataloaders(args):
    
    print("*" * 10, "=" * 10, "*" * 10)
    print("Loading dataloader for Classification task")

    fold = args.fold  # your chosen fold dict
    dp = args.dataloader_params

    fold_name = fold.get("name", "unknown_fold")

    print("Using fold: ", fold)
    print("Using fold name: ", fold_name)
    # ---- Trial-based path (unchanged) ----
    if "kfold" in fold_name or "loo" in fold_name or "louo" in fold_name:
        print("\n--- Using TRIAL-BASED split ---\n")
        print("Train Trials: ", args.dataloader_params["train_trials"])
        print("Val Trials: ", args.dataloader_params["val_trials"])
        print("Test Trials: ", args.dataloader_params["test_trials"])

                # ----- 1) Build ONE master class_map (contiguous) -----
        # Prefer the explicit keysteps dict if provided (stable & human-readable).
        # keysteps can be like {20: "Reach Suture", 8: "Make C Loop", ...} or {"G1": "...", ...}
        keysteps = dp.get("keysteps", None)

        if keysteps and isinstance(keysteps, dict) and len(keysteps) > 0:
            # Keep insertion order of provided keysteps
            # DeskDataset uses strings in 'gesture_code', so normalize keys to str
            master_class_map = {str(k): i for i, k in enumerate(keysteps.keys())}
            class_id_to_name  = {i: keysteps[k] for k, i in master_class_map.items()}
            class_name_to_id  = {v: i for i, v in class_id_to_name.items()}
        else:
            # Fallback: we’ll derive a union of labels from TRAIN trials only (stable for LOUO),
            # but since we can’t cheaply read CSVs here, we’ll create a tiny bootstrap dataset
            # to scan gesture codes, then rebuild with the fixed mapping.
            bootstrap_train = DeskDataset(
                base_path=dp["base_path"],
                csv_paths=dp["train_trials"],
                clip_len=dp["observation_window"],
                step=dp["step"],
                include_modalities=dp["modalities"],
                sample_rate=dp["sample_rate"],
                modality_selections=dp['selections'],
                normalize=False,            # <- no stats yet
            )
            # Build map in sorted order of observed gesture codes for determinism
            observed = sorted(map(str, bootstrap_train.df["gesture_code"].dropna().unique().tolist()))
            master_class_map = {c: i for i, c in enumerate(observed)}
            class_id_to_name = {i: str(c) for c, i in master_class_map.items()}
            class_name_to_id = {v: i for i, v in class_id_to_name.items()}
            # Drop bootstrap dataset to free memory
            del bootstrap_train

        train_dataset = MultimodalGestureDataset(
            base_path=args.dataloader_params["base_path"],
            csv_paths=args.dataloader_params["train_trials"],
            clip_len=args.dataloader_params["observation_window"],
            step=args.dataloader_params["step"],
            include_modalities=args.dataloader_params["modalities"],
            sample_rate=args.dataloader_params["sample_rate"],
            drop_neg1=args.dataloader_params.get("drop_neg1", True),
            # Allowlist patterns (fnmatch)
            modality_selections= args.dataloader_params['selections'],
            classes_to_allow= args.dataloader_params.get("keysteps", None),
            class_map=master_class_map,


            # modality_exclude={
            #     "trakstar": ["*elevation*", "*roll*"],  # just in case the allowlist was broad
            # },

            normalize=True,
        )
        # Attach helpful mappings for downstream code
        train_dataset.class_id_to_name = class_id_to_name
        train_dataset.class_name_to_id = class_name_to_id

        val_dataset = MultimodalGestureDataset(
            base_path=args.dataloader_params["base_path"],
            csv_paths=args.dataloader_params["val_trials"],
            clip_len=args.dataloader_params["observation_window"],
            step=args.dataloader_params["step"],
            include_modalities=args.dataloader_params["modalities"],
            sample_rate=args.dataloader_params["sample_rate"],
            drop_neg1=args.dataloader_params.get("drop_neg1", True),
            classes_to_allow= args.dataloader_params.get("keysteps", None),



            modality_selections= args.dataloader_params['selections'],
            class_map=master_class_map,

            # modality_exclude={
            #     "trakstar": ["*elevation*", "*roll*"],  # just in case the allowlist was broad
            # },
            normalize=True,
        )

        # Attach helpful mappings for downstream code
        val_dataset.class_id_to_name = class_id_to_name
        val_dataset.class_name_to_id = class_name_to_id

        test_dataset = MultimodalGestureDataset(
            base_path=args.dataloader_params["base_path"],
            csv_paths=args.dataloader_params["test_trials"],
            clip_len=args.dataloader_params["observation_window"],
            step=args.dataloader_params["step"],
            include_modalities=args.dataloader_params["modalities"],
            sample_rate=args.dataloader_params["sample_rate"],

            drop_neg1=args.dataloader_params.get("drop_neg1", True),

            modality_selections= args.dataloader_params['selections'],
            classes_to_allow= args.dataloader_params.get("keysteps", None),
            class_map=master_class_map,


            # modality_exclude={
            #     "trakstar": ["*elevation*", "*roll*"],  # just in case the allowlist was broad
            # },
            normalize=True,
        )

        # Attach helpful mappings for downstream code
        test_dataset.class_id_to_name = class_id_to_name
        test_dataset.class_name_to_id = class_name_to_id

        train_class_stats = train_dataset._get_class_stats()
        print("Train class stats: ", train_class_stats)

        val_class_stats = val_dataset._get_class_stats()
        print("Val class stats: ", val_class_stats)

        test_class_stats = test_dataset._get_class_stats()
        print("Test class stats: ", test_class_stats)

        train_loader = DataLoader(train_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=True,
                            collate_fn=MultimodalGestureDataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False,
                            collate_fn=MultimodalGestureDataset.collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=args.dataloader_params["batch_size"], shuffle=False,
                            collate_fn=MultimodalGestureDataset.collate_fn)
        
        return train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats , class_map

    elif fold_name == "gesture_flat" and dp["observation_window"] == -1: # simple train/val/test split for clips of gestures
        # 1) Build a TEMP dataset over all trials just to enumerate clips
        all_trials = sorted(set(fold["train_trials"]) | set(fold["test_trials"]))
        print(f"Building TEMP dataset over {len(all_trials)} trials...")

        ds_all_temp = MultimodalGestureDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            ignore_clutch=dp.get("ignore_clutch", False),
            clutch_pressed_value=dp.get("clutch_pressed_value", 1),
            drop_neg1=dp.get("drop_neg1", True),
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=False,  # IMPORTANT: don't compute stats here
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            classes_to_allow=dp.get("keysteps", None)
        )

        class_map = ds_all_temp.class_map

        # 2) Two-stage stratified split: Train/Val/Test
        y_all = _labels_for_dataset_windows(ds_all_temp)  # returns a list/np.array of class ids per clip

        # Configure split ratios (adjust these as needed)
        test_ratio = fold.get("test_ratio", 0.20)  # 20% for test
        val_ratio = fold.get("val_ratio", 0.20)    # 20% of remaining (16% of total) for validation

        # First split: separate out test set
        tr_val_idx, te_idx = stratified_gesture_train_test(
            y_all, 
            test_ratio=test_ratio, 
            seed=fold["flat_seed"]
        )

        # Second split: separate train and validation from the remaining data
        y_tr_val = y_all[tr_val_idx]  # labels for train+val subset only
        tr_idx_local, val_idx_local = stratified_gesture_train_test(
            y_tr_val, 
            test_ratio=val_ratio,  # this is ratio of (train+val), not total
            seed=fold["flat_seed"] + 1  # use different seed for reproducibility
        )

        # Convert to numpy arrays for fancy indexing
        tr_val_idx = np.array(tr_val_idx)
        tr_idx_local = np.array(tr_idx_local)
        val_idx_local = np.array(val_idx_local)

        # Map local indices back to global clip indices
        tr_idx = tr_val_idx[tr_idx_local]
        val_idx = tr_val_idx[val_idx_local]

        print(f"Data split sizes - Train: {len(tr_idx)}, Val: {len(val_idx)}, Test: {len(te_idx)}")
        print(f"Data split ratios - Train: {len(tr_idx)/len(y_all):.1%}, Val: {len(val_idx)/len(y_all):.1%}, Test: {len(te_idx)/len(y_all):.1%}")

        # 3) Build TRAIN dataset - compute normalization stats on training data only
        ds_tr = MultimodalGestureDataset(
            base_path=dp.get("base_path", None),
            csv_paths=sorted(set(fold["train_trials"]) | set(fold["test_trials"])),
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            ignore_clutch=dp.get("ignore_clutch", False),
            clutch_pressed_value=dp.get("clutch_pressed_value", 1),
            drop_neg1=dp.get("drop_neg1", True),
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,  # compute stats here
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=tr_idx,  # only training windows
            classes_to_allow=dp.get("keysteps", None)

        )

        # 4) Build VALIDATION dataset - reuse train normalization stats
        ds_val = MultimodalGestureDataset(
            base_path=dp.get("base_path", None),
            csv_paths=sorted(set(fold["train_trials"]) | set(fold["test_trials"])),
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            ignore_clutch=dp.get("ignore_clutch", False),
            clutch_pressed_value=dp.get("clutch_pressed_value", 1),
            drop_neg1=dp.get("drop_neg1", True),
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,
            normalization_stats=ds_tr.norm_stats,  # reuse TRAIN mean/std
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=val_idx,  # only validation windows
            classes_to_allow=dp.get("keysteps", None)

        )

        # 5) Build TEST dataset - reuse train normalization stats
        ds_te = MultimodalGestureDataset(
            base_path=dp.get("base_path", None),
            csv_paths=sorted(set(fold["train_trials"]) | set(fold["test_trials"])),
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            ignore_clutch=dp.get("ignore_clutch", False),
            clutch_pressed_value=dp.get("clutch_pressed_value", 1),
            drop_neg1=dp.get("drop_neg1", True),
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,
            normalization_stats=ds_tr.norm_stats,  # reuse TRAIN mean/std
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=te_idx,  # only test windows
            classes_to_allow=dp.get("keysteps", None)

        )

        # 6) Print class distribution stats for each split
        train_class_stats = ds_tr._get_class_stats()
        val_class_stats = ds_val._get_class_stats()
        test_class_stats = ds_te._get_class_stats()

        print("*" * 10, "=" * 10, "*" * 10)
        print("\nClass distribution by split:")
        print("Train class stats:", train_class_stats)
        print("Val   class stats:", val_class_stats)
        print("Test  class stats:", test_class_stats)
        print("*" * 10, "=" * 10, "*" * 10)

        # 7) Build DataLoaders
        train_loader = DataLoader(
            ds_tr, 
            batch_size=dp["batch_size"], 
            shuffle=True,  # shuffle training data
            num_workers=dp.get("num_workers", 4),
            collate_fn=MultimodalGestureDataset.collate_fn, 
            drop_last=False
        )

        val_loader = DataLoader(
            ds_val, 
            batch_size=dp["batch_size"], 
            shuffle=False,  # no shuffle for validation
            num_workers=dp.get("num_workers", 4),
            collate_fn=MultimodalGestureDataset.collate_fn, 
            drop_last=False
        )

        test_loader = DataLoader(
            ds_te, 
            batch_size=dp["batch_size"], 
            shuffle=False,  # no shuffle for test
            num_workers=dp.get("num_workers", 4),
            collate_fn=MultimodalGestureDataset.collate_fn, 
            drop_last=False
        )

        # Return all loaders and stats
        return train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats, class_map

    elif  fold_name == "gesture_flat" and dp["observation_window"] != -1: # simple train/val/test split for window segments within gestures
        # ---- 0) Build a TEMP dataset once to enumerate ALL windows across the chosen trials ----
        all_trials = sorted(set(fold["train_trials"]) | set(fold["test_trials"]))
        print(f"\n--- Using WINDOW-BASED split (no leakage) over {len(all_trials)} trials ---\n")

        ds_all = MultimodalGestureDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            ignore_clutch=dp.get("ignore_clutch", False),
            clutch_pressed_value=dp.get("clutch_pressed_value", 1),
            drop_neg1=dp.get("drop_neg1", True),
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=False,  # IMPORTANT: don't compute stats here
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            classes_to_allow=dp.get("keysteps", None)
        )

        # ---- 1) Group windows by gesture RUN (segment) to prevent leakage ----
        # Each window belongs entirely to one _run_id by construction of _make_windows().
        run_to_win = {}   # run_id -> list[window_idx]
        run_labels = []   # parallel to run_ids: class id per run (for stratification)
        run_ids = []      # list of run_ids in the same order as run_labels

        class_map = ds_all.class_map
        for wi, s in enumerate(ds_all.samples):
            start = s["start"]
            run_id = int(ds_all.df.loc[start, "_run_id"])
            if run_id not in run_to_win:
                run_to_win[run_id] = []
                run_ids.append(run_id)
                run_labels.append(class_map[s["gesture_code"]])
            run_to_win[run_id].append(wi)

        y_run = np.asarray(run_labels, dtype=np.int64)
        run_ids = np.asarray(run_ids, dtype=np.int64)

        print(f"Total gesture runs: {len(run_ids)} | Total windows: {len(ds_all)}")

        # ---- 2) Stratified split on RUNS, then expand to window indices ----
        test_ratio = float(fold.get("test_ratio", 0.20))
        val_ratio  = float(fold.get("val_ratio", 0.20))
        base_seed  = int(fold.get("flat_seed", 0))

        def expand_runs_to_windows(run_idx_array: np.ndarray) -> np.ndarray:
            out = []
            for j in run_idx_array:
                rid = int(run_ids[j])
                out.extend(run_to_win[rid])
            # unique + sorted for stability
            return np.array(sorted(set(out)), dtype=np.int64)

        # a) Train+Val vs Test on RUNS
        trval_run_idx, te_run_idx = stratified_gesture_train_test(
            y_run, test_ratio=test_ratio, seed=base_seed
        )
        trval_run_idx = np.asarray(trval_run_idx)
        te_run_idx    = np.asarray(te_run_idx)

        # b) Train vs Val on RUNS (remaining)
        y_trval = y_run[trval_run_idx]
        tr_local, val_local = stratified_gesture_train_test(
            y_trval, test_ratio=val_ratio, seed=base_seed + 1
        )
        tr_run_idx  = trval_run_idx[np.asarray(tr_local)]
        val_run_idx = trval_run_idx[np.asarray(val_local)]

        # c) Expand to WINDOW indices
        tr_idx  = expand_runs_to_windows(tr_run_idx)
        val_idx = expand_runs_to_windows(val_run_idx)
        te_idx  = expand_runs_to_windows(te_run_idx)

        # ---- 3) (Optional) ensure each split has all classes present in ds_all (best-effort retry) ----
        y_all_windows = _labels_for_dataset_windows(ds_all)
        all_classes   = set(np.unique(y_all_windows))

        def covers_all(idx):
            return set(np.unique(y_all_windows[idx])) >= all_classes

        if not (covers_all(tr_idx) and covers_all(val_idx) and covers_all(te_idx)):
            print("Note: At least one split is missing some classes. Retrying up to 25 times...")
            rng = np.random.RandomState(base_seed)
            ok = False
            for attempt in range(25):
                s1 = int(rng.randint(0, 1_000_000))
                s2 = s1 + 1
                trval_run_idx, te_run_idx = stratified_gesture_train_test(y_run, test_ratio=test_ratio, seed=s1)
                trval_run_idx = np.asarray(trval_run_idx); te_run_idx = np.asarray(te_run_idx)
                y_trval = y_run[trval_run_idx]
                tr_local, val_local = stratified_gesture_train_test(y_trval, test_ratio=val_ratio, seed=s2)
                tr_run_idx  = trval_run_idx[np.asarray(tr_local)]
                val_run_idx = trval_run_idx[np.asarray(val_local)]
                tr_idx  = expand_runs_to_windows(tr_run_idx)
                val_idx = expand_runs_to_windows(val_run_idx)
                te_idx  = expand_runs_to_windows(te_run_idx)
                if covers_all(tr_idx) and covers_all(val_idx) and covers_all(te_idx):
                    ok = True
                    print(f"Class coverage satisfied on attempt {attempt+1}.")
                    break
            if not ok:
                print("Warning: Could not satisfy full class coverage across splits. Proceeding with best effort.")

        print(f"Split sizes (windows) — Train: {len(tr_idx)}, Val: {len(val_idx)}, Test: {len(te_idx)}")

        # ---- 4) Build final datasets (compute stats on TRAIN only; reuse for VAL/TEST) ----
        ds_tr = MultimodalGestureDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            ignore_clutch=dp.get("ignore_clutch", False),
            clutch_pressed_value=dp.get("clutch_pressed_value", 1),
            drop_neg1=dp.get("drop_neg1", True),
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,  # compute stats here
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=tr_idx,
            classes_to_allow=dp.get("keysteps", None)
        )

        ds_val = MultimodalGestureDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            ignore_clutch=dp.get("ignore_clutch", False),
            clutch_pressed_value=dp.get("clutch_pressed_value", 1),
            drop_neg1=dp.get("drop_neg1", True),
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,
            normalization_stats=ds_tr.norm_stats,  # reuse TRAIN mean/std
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=val_idx,
            classes_to_allow=dp.get("keysteps", None)
        )

        ds_te = MultimodalGestureDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            ignore_clutch=dp.get("ignore_clutch", False),
            clutch_pressed_value=dp.get("clutch_pressed_value", 1),
            drop_neg1=dp.get("drop_neg1", True),
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,
            normalization_stats=ds_tr.norm_stats,  # reuse TRAIN mean/std
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=te_idx,
            classes_to_allow=dp.get("keysteps", None)
        )

        # ---- 5) Class stats + DataLoaders ----
        train_class_stats = ds_tr._get_class_stats()
        val_class_stats   = ds_val._get_class_stats()
        test_class_stats  = ds_te._get_class_stats()

        print("*" * 10, "=" * 10, "*" * 10)
        print("Class distribution by split (windows, no leakage):")
        print("Train:", train_class_stats)
        print("Val  :", val_class_stats)
        print("Test :", test_class_stats)
        print("*" * 10, "=" * 10, "*" * 10)

        train_loader = DataLoader(
            ds_tr, batch_size=dp["batch_size"], shuffle=True,
            num_workers=dp.get("num_workers", 4),
            collate_fn=MultimodalGestureDataset.collate_fn, drop_last=False
        )
        val_loader = DataLoader(
            ds_val, batch_size=dp["batch_size"], shuffle=False,
            num_workers=dp.get("num_workers", 4),
            collate_fn=MultimodalGestureDataset.collate_fn, drop_last=False
        )
        test_loader = DataLoader(
            ds_te, batch_size=dp["batch_size"], shuffle=False,
            num_workers=dp.get("num_workers", 4),
            collate_fn=MultimodalGestureDataset.collate_fn, drop_last=False
        )

        return train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats , class_map

    else:
        raise ValueError("Invalid fold configuration provided.")






# return train,val,test dataloaders using the DESK dataset class
def DESK_get_dataloaders(args):
    """
    Returns:
        train_loader, val_loader, test_loader,
        train_class_stats, val_class_stats, test_class_stats
    Side-effects:
        - Ensures a single, shared class_map across splits.
        - Uses train-only normalization stats for val/test.
        - Attaches helpers on datasets:
            .class_map            : {gesture_code(str): contiguous_id(int)}
            .class_id_to_name     : {contiguous_id(int): human_readable_name(str)}
            .class_name_to_id     : {human_readable_name(str): contiguous_id(int)}
    """
    print("*" * 10, "=" * 10, "*" * 10)
    print("Loading dataloader for DESK Classification task")

    fold = args.fold
    dp = args.dataloader_params
    fold_name = fold.get("name", "unknown_fold")

    print("Using fold: ", fold)
    print("Using fold name: ", fold_name)

    if "kfold" in fold_name or "loo" in fold_name or "louo" in fold_name:
        print("\n--- Using TRIAL-BASED split ---\n")
        print("Train Trials: ", dp["train_trials"])
        print("Val Trials: ",   dp["val_trials"])
        print("Test Trials: ",  dp["test_trials"])

        # ----- 1) Build ONE master class_map (contiguous) -----
        # Prefer the explicit keysteps dict if provided (stable & human-readable).
        # keysteps can be like {20: "Reach Suture", 8: "Make C Loop", ...} or {"G1": "...", ...}
        keysteps = dp.get("keysteps", None)

        if keysteps and isinstance(keysteps, dict) and len(keysteps) > 0:
            # Keep insertion order of provided keysteps
            # DeskDataset uses strings in 'gesture_code', so normalize keys to str
            master_class_map = {str(k): i for i, k in enumerate(keysteps.keys())}
            class_id_to_name  = {i: keysteps[k] for k, i in master_class_map.items()}
            class_name_to_id  = {v: i for i, v in class_id_to_name.items()}
        else:
            # Fallback: we’ll derive a union of labels from TRAIN trials only (stable for LOUO),
            # but since we can’t cheaply read CSVs here, we’ll create a tiny bootstrap dataset
            # to scan gesture codes, then rebuild with the fixed mapping.
            bootstrap_train = DeskDataset(
                base_path=dp["base_path"],
                csv_paths=dp["train_trials"],
                clip_len=dp["observation_window"],
                step=dp["step"],
                include_modalities=dp["modalities"],
                sample_rate=dp["sample_rate"],
                modality_selections=dp['selections'],
                normalize=False,            # <- no stats yet
            )
            # Build map in sorted order of observed gesture codes for determinism
            observed = sorted(map(str, bootstrap_train.df["gesture_code"].dropna().unique().tolist()))
            master_class_map = {c: i for i, c in enumerate(observed)}
            class_id_to_name = {i: str(c) for c, i in master_class_map.items()}
            class_name_to_id = {v: i for i, v in class_id_to_name.items()}
            # Drop bootstrap dataset to free memory
            del bootstrap_train

        # ----- 2) Build TRAIN dataset with the fixed mapping; compute normalization on TRAIN only -----
        train_dataset = DeskDataset(
            base_path=dp["base_path"],
            csv_paths=dp["train_trials"],
            clip_len=dp["observation_window"],
            step=dp["step"],
            include_modalities=dp["modalities"],
            sample_rate=dp["sample_rate"],
            modality_selections=dp['selections'],
            classes_to_allow=keysteps,         # <- filter to same class set
            class_map=master_class_map,        # <- fixed contiguous mapping
            normalize=True,                    # will compute stats on TRAIN
        )
        # Attach helpful mappings for downstream code
        train_dataset.class_id_to_name = class_id_to_name
        train_dataset.class_name_to_id = class_name_to_id

        # ----- 3) Build VAL/TEST datasets using the SAME mapping + train-only norm stats -----
        val_dataset = DeskDataset(
            base_path=dp["base_path"],
            csv_paths=dp["val_trials"],
            clip_len=dp["observation_window"],
            step=dp["step"],
            include_modalities=dp["modalities"],
            sample_rate=dp["sample_rate"],
            modality_selections=dp['selections'],
            classes_to_allow=keysteps,                # same filter
            class_map=master_class_map,               # same mapping
            normalize=True,
            normalization_stats=train_dataset.norm_stats,  # <- use train stats
        )
        val_dataset.class_id_to_name = class_id_to_name
        val_dataset.class_name_to_id = class_name_to_id

        test_dataset = DeskDataset(
            base_path=dp["base_path"],
            csv_paths=dp["test_trials"],
            clip_len=dp["observation_window"],
            step=dp["step"],
            include_modalities=dp["modalities"],
            sample_rate=dp["sample_rate"],
            modality_selections=dp['selections'],
            classes_to_allow=keysteps,                # same filter
            class_map=master_class_map,               # same mapping
            normalize=True,
            normalization_stats=train_dataset.norm_stats,  # <- use train stats
        )
        test_dataset.class_id_to_name = class_id_to_name
        test_dataset.class_name_to_id = class_name_to_id

        # ----- 4) Debug prints -----
        print("Fixed class_map (gesture_code -> id):", master_class_map)
        print("Class id -> name:", class_id_to_name)

        train_class_stats = train_dataset._get_class_stats()
        print("Train class stats: ", train_class_stats)

        val_class_stats = val_dataset._get_class_stats()
        print("Val class stats: ", val_class_stats)

        test_class_stats = test_dataset._get_class_stats()
        print("Test class stats: ", test_class_stats)

        # ----- 5) DataLoaders -----
        train_loader = DataLoader(
            train_dataset,
            batch_size=dp["batch_size"],
            shuffle=True,
            collate_fn=DeskDataset.collate_fn
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=dp["batch_size"],
            shuffle=False,
            collate_fn=DeskDataset.collate_fn
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=dp["batch_size"],
            shuffle=False,
            collate_fn=DeskDataset.collate_fn
        )

        return train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats



    elif fold_name == "gesture_flat" and dp["observation_window"] == -1: # simple train/val/test split for clips of gestures
        # 1) Build a TEMP dataset over all trials just to enumerate clips
        all_trials = sorted(set(fold["train_trials"]) | set(fold["test_trials"]))
        print(f"Building TEMP dataset over {len(all_trials)} trials...")

        ds_all_temp = DeskDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=False,  # IMPORTANT: don't compute stats here
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            classes_to_allow=dp.get("keysteps", None)
        )

        # 2) Two-stage stratified split: Train/Val/Test
        y_all = _labels_for_dataset_windows(ds_all_temp)  # returns a list/np.array of class ids per clip

        # Configure split ratios (adjust these as needed)
        test_ratio = fold.get("test_ratio", 0.20)  # 20% for test
        val_ratio = fold.get("val_ratio", 0.20)    # 20% of remaining (16% of total) for validation

        # First split: separate out test set
        tr_val_idx, te_idx = stratified_gesture_train_test(
            y_all, 
            test_ratio=test_ratio, 
            seed=fold["flat_seed"]
        )

        # Second split: separate train and validation from the remaining data
        y_tr_val = y_all[tr_val_idx]  # labels for train+val subset only
        tr_idx_local, val_idx_local = stratified_gesture_train_test(
            y_tr_val, 
            test_ratio=val_ratio,  # this is ratio of (train+val), not total
            seed=fold["flat_seed"] + 1  # use different seed for reproducibility
        )

        # Convert to numpy arrays for fancy indexing
        tr_val_idx = np.array(tr_val_idx)
        tr_idx_local = np.array(tr_idx_local)
        val_idx_local = np.array(val_idx_local)

        # Map local indices back to global clip indices
        tr_idx = tr_val_idx[tr_idx_local]
        val_idx = tr_val_idx[val_idx_local]

        print(f"Data split sizes - Train: {len(tr_idx)}, Val: {len(val_idx)}, Test: {len(te_idx)}")
        print(f"Data split ratios - Train: {len(tr_idx)/len(y_all):.1%}, Val: {len(val_idx)/len(y_all):.1%}, Test: {len(te_idx)/len(y_all):.1%}")

        # 3) Build TRAIN dataset - compute normalization stats on training data only
        ds_tr = DeskDataset(
            base_path=dp.get("base_path", None),
            csv_paths=sorted(set(fold["train_trials"]) | set(fold["test_trials"])),
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,  # compute stats here
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=tr_idx,  # only training windows
            classes_to_allow=dp.get("keysteps", None)

        )

        # 4) Build VALIDATION dataset - reuse train normalization stats
        ds_val = DeskDataset(
            base_path=dp.get("base_path", None),
            csv_paths=sorted(set(fold["train_trials"]) | set(fold["test_trials"])),
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,
            normalization_stats=ds_tr.norm_stats,  # reuse TRAIN mean/std
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=val_idx,  # only validation windows
            classes_to_allow=dp.get("keysteps", None)

        )

        # 5) Build TEST dataset - reuse train normalization stats
        ds_te = DeskDataset(
            base_path=dp.get("base_path", None),
            csv_paths=sorted(set(fold["train_trials"]) | set(fold["test_trials"])),
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,
            normalization_stats=ds_tr.norm_stats,  # reuse TRAIN mean/std
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=te_idx,  # only test windows
            classes_to_allow=dp.get("keysteps", None)

        )

        # 6) Print class distribution stats for each split
        train_class_stats = ds_tr._get_class_stats()
        val_class_stats = ds_val._get_class_stats()
        test_class_stats = ds_te._get_class_stats()

        print("*" * 10, "=" * 10, "*" * 10)
        print("\nClass distribution by split:")
        print("Train class stats:", train_class_stats)
        print("Val   class stats:", val_class_stats)
        print("Test  class stats:", test_class_stats)
        print("*" * 10, "=" * 10, "*" * 10)

        print("Building DataLoaders...")
        print("Using num_workers: ", dp.get("num_workers", 4))

        # 7) Build DataLoaders
        train_loader = DataLoader(
            ds_tr, 
            batch_size=dp["batch_size"], 
            shuffle=True,  # shuffle training data
            num_workers=dp.get("num_workers", 4),
            collate_fn=DeskDataset.collate_fn, 
            drop_last=False
        )

        val_loader = DataLoader(
            ds_val, 
            batch_size=dp["batch_size"], 
            shuffle=False,  # no shuffle for validation
            num_workers=dp.get("num_workers", 4),
            collate_fn=DeskDataset.collate_fn, 
            drop_last=False
        )

        test_loader = DataLoader(
            ds_te, 
            batch_size=dp["batch_size"], 
            shuffle=False,  # no shuffle for test
            num_workers=dp.get("num_workers", 4),
            collate_fn=DeskDataset.collate_fn, 
            drop_last=False
        )

        # Return all loaders and stats
        return train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats

    elif  fold_name == "gesture_flat" and dp["observation_window"] != -1: # simple train/val/test split for window segments within gestures
        # ---- 0) Build a TEMP dataset once to enumerate ALL windows across the chosen trials ----
        all_trials = sorted(set(fold["train_trials"]) | set(fold["test_trials"]))
        print(f"\n--- Using WINDOW-BASED split (no leakage) over {len(all_trials)} trials ---\n")

        ds_all = DeskDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=False,  # IMPORTANT: don't compute stats here
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            classes_to_allow=dp.get("keysteps", None)
        )

        # ---- 1) Group windows by gesture RUN (segment) to prevent leakage ----
        # Each window belongs entirely to one _run_id by construction of _make_windows().
        run_to_win = {}   # run_id -> list[window_idx]
        run_labels = []   # parallel to run_ids: class id per run (for stratification)
        run_ids = []      # list of run_ids in the same order as run_labels

        class_map = ds_all.class_map
        for wi, s in enumerate(ds_all.samples):
            start = s["start"]
            run_id = int(ds_all.df.loc[start, "_run_id"])
            if run_id not in run_to_win:
                run_to_win[run_id] = []
                run_ids.append(run_id)
                run_labels.append(class_map[s["gesture_code"]])
            run_to_win[run_id].append(wi)

        y_run = np.asarray(run_labels, dtype=np.int64)
        run_ids = np.asarray(run_ids, dtype=np.int64)

        print(f"Total gesture runs: {len(run_ids)} | Total windows: {len(ds_all)}")

        # ---- 2) Stratified split on RUNS, then expand to window indices ----
        test_ratio = float(fold.get("test_ratio", 0.20))
        val_ratio  = float(fold.get("val_ratio", 0.20))
        base_seed  = int(fold.get("flat_seed", 0))

        def expand_runs_to_windows(run_idx_array: np.ndarray) -> np.ndarray:
            out = []
            for j in run_idx_array:
                rid = int(run_ids[j])
                out.extend(run_to_win[rid])
            # unique + sorted for stability
            return np.array(sorted(set(out)), dtype=np.int64)

        # a) Train+Val vs Test on RUNS
        trval_run_idx, te_run_idx = stratified_gesture_train_test(
            y_run, test_ratio=test_ratio, seed=base_seed
        )
        trval_run_idx = np.asarray(trval_run_idx)
        te_run_idx    = np.asarray(te_run_idx)

        # b) Train vs Val on RUNS (remaining)
        y_trval = y_run[trval_run_idx]
        tr_local, val_local = stratified_gesture_train_test(
            y_trval, test_ratio=val_ratio, seed=base_seed + 1
        )
        tr_run_idx  = trval_run_idx[np.asarray(tr_local)]
        val_run_idx = trval_run_idx[np.asarray(val_local)]

        # c) Expand to WINDOW indices
        tr_idx  = expand_runs_to_windows(tr_run_idx)
        val_idx = expand_runs_to_windows(val_run_idx)
        te_idx  = expand_runs_to_windows(te_run_idx)

        # ---- 3) (Optional) ensure each split has all classes present in ds_all (best-effort retry) ----
        y_all_windows = _labels_for_dataset_windows(ds_all)
        all_classes   = set(np.unique(y_all_windows))

        def covers_all(idx):
            return set(np.unique(y_all_windows[idx])) >= all_classes

        if not (covers_all(tr_idx) and covers_all(val_idx) and covers_all(te_idx)):
            print("Note: At least one split is missing some classes. Retrying up to 25 times...")
            rng = np.random.RandomState(base_seed)
            ok = False
            for attempt in range(25):
                s1 = int(rng.randint(0, 1_000_000))
                s2 = s1 + 1
                trval_run_idx, te_run_idx = stratified_gesture_train_test(y_run, test_ratio=test_ratio, seed=s1)
                trval_run_idx = np.asarray(trval_run_idx); te_run_idx = np.asarray(te_run_idx)
                y_trval = y_run[trval_run_idx]
                tr_local, val_local = stratified_gesture_train_test(y_trval, test_ratio=val_ratio, seed=s2)
                tr_run_idx  = trval_run_idx[np.asarray(tr_local)]
                val_run_idx = trval_run_idx[np.asarray(val_local)]
                tr_idx  = expand_runs_to_windows(tr_run_idx)
                val_idx = expand_runs_to_windows(val_run_idx)
                te_idx  = expand_runs_to_windows(te_run_idx)
                if covers_all(tr_idx) and covers_all(val_idx) and covers_all(te_idx):
                    ok = True
                    print(f"Class coverage satisfied on attempt {attempt+1}.")
                    break
            if not ok:
                print("Warning: Could not satisfy full class coverage across splits. Proceeding with best effort.")

        print(f"Split sizes (windows) — Train: {len(tr_idx)}, Val: {len(val_idx)}, Test: {len(te_idx)}")

        # ---- 4) Build final datasets (compute stats on TRAIN only; reuse for VAL/TEST) ----
        ds_tr = DeskDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,  # compute stats here
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=tr_idx,
            classes_to_allow=dp.get("keysteps", None)
        )

        ds_val = DeskDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,
            normalization_stats=ds_tr.norm_stats,  # reuse TRAIN mean/std
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=val_idx,
            classes_to_allow=dp.get("keysteps", None)
        )

        ds_te = DeskDataset(
            base_path=dp.get("base_path", None),
            csv_paths=all_trials,
            clip_len=dp["observation_window"],
            step=dp["step"],
            sample_rate=dp["sample_rate"],
            include_modalities=dp["modalities"],
            modality_selections=dp.get("selections", {}),
            normalize=True,
            normalization_stats=ds_tr.norm_stats,  # reuse TRAIN mean/std
            video_root=dp.get("video_root", None),
            video_pattern=dp.get("video_pattern", "{trial_id}.mp4"),
            video_output_size=tuple(dp.get("video_output_size", (224, 224))),
            selected_indices=te_idx,
            classes_to_allow=dp.get("keysteps", None)
        )

        # ---- 5) Class stats + DataLoaders ----
        train_class_stats = ds_tr._get_class_stats()
        val_class_stats   = ds_val._get_class_stats()
        test_class_stats  = ds_te._get_class_stats()

        print("*" * 10, "=" * 10, "*" * 10)
        print("Class distribution by split (windows, no leakage):")
        print("Train:", train_class_stats)
        print("Val  :", val_class_stats)
        print("Test :", test_class_stats)
        print("*" * 10, "=" * 10, "*" * 10)

        train_loader = DataLoader(
            ds_tr, batch_size=dp["batch_size"], shuffle=True,
            num_workers=dp.get("num_workers", 4),
            collate_fn=DeskDataset.collate_fn, drop_last=False
        )
        val_loader = DataLoader(
            ds_val, batch_size=dp["batch_size"], shuffle=False,
            num_workers=dp.get("num_workers", 4),
            collate_fn=DeskDataset.collate_fn, drop_last=False
        )
        test_loader = DataLoader(
            ds_te, batch_size=dp["batch_size"], shuffle=False,
            num_workers=dp.get("num_workers", 4),
            collate_fn=DeskDataset.collate_fn, drop_last=False
        )

        return train_loader, val_loader, test_loader, train_class_stats, val_class_stats, test_class_stats

    else:
        raise ValueError("Invalid fold configuration provided.")









def print_one_batch(loader):
    batch = next(iter(loader))

    # print more details about batch content
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: dtype={v.dtype}, shape={tuple(v.shape)}, "
                f"min={v.min().item():.3f}, max={v.max().item():.3f}")
        elif k == "gesture_code":
            # show a few sample codes for sanity
            print(f"{k}: {len(v)} items -> {v[:5]}{' ...' if len(v) > 5 else ''}")
        elif isinstance(v, list):
            print(f"{k}: list of {len(v)} items, first 3: {v[:3]}")
        else:
            print(f"{k}: type={type(v)}, value={v}")


def initialize_multiMTRSAP_model(args, device):
    print("Initializing Multi MTRSAP model...")
    print("MultiMTRSAP Config: ", args.multiMTRSAPcfg)

    model = build_default_multitranstcn(args.multiMTRSAPcfg)
    model = model.to(device)
    print(model)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])

    # scheduler
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

def train_multiMTRSAP_one_epoch(model, train_loader, criterion, optimizer, device, logger, args):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:
            # print(f" Batch key: {list(batch.keys())}")
            inputs = build_inputs(batch, args.dataloader_params["modalities"])

            # print shape of each input tensor
            # for k, v in inputs.items():
            #     print(f"  {k}: {v.shape}")
            inputs = {k: v.to(device) for k,v in inputs.items()}

            logits = model(inputs)
            # print(f"Logits shape: {logits.shape}")
            # print(f"Batch label: {batch['label']}")


            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")



        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            continue

    return total_loss / len(train_loader)

def validate_multiMTRSAP(model, val_loader, criterion, device, logger, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                inputs = build_inputs(batch, args.dataloader_params["modalities"])
                inputs = {k: v.to(device) for k,v in inputs.items()}

                logits = model(inputs)

                loss = criterion(logits, batch['label'].to(device))
                total_loss += loss.item()
                if i % 10 == 0:
                    logger.log({"val_loss": loss.item()})

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return total_loss / len(val_loader)

def test_multiMTRSAP_model(model, test_loader, criterion, device, logger, epoch, results_dir, args):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    
    preds_detail = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                # forward
                inputs = build_inputs(batch, args.dataloader_params["modalities"])
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                logits = model(inputs)                            # [B, C]
                labels = batch["label"].to(device, non_blocking=True)  # [B]
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # predictions
                pred = torch.argmax(logits, dim=1)               # [B]

                # accumulate scalar lists
                gt.extend(batch["label"].cpu().tolist())         # extend with B items
                preds.extend(pred.cpu().tolist())                # extend with B items

                # optional: probs if you need them
                probs = torch.softmax(logits, dim=1).detach().cpu().tolist()

                # detailed per-sample records
                B = pred.shape[0]
                trial_ids      = batch.get("trial_id",      [None]*B)   # might be list[str] or tensor
                subject_ids    = batch.get("subject_id",    [None]*B)
                gesture_codes  = batch.get("gesture_code",  [None]*B)   # usually list[str]

                for i in range(B):
                    preds_detail.append({
                        "trial_id":      trial_ids[i] if not torch.is_tensor(trial_ids) else trial_ids[i].item(),
                        "subject_id":    subject_ids[i] if not torch.is_tensor(subject_ids) else subject_ids[i].item(),
                        "gesture_code":  gesture_codes[i] if isinstance(gesture_codes, list) else gesture_codes[i],
                        "pred_label":    int(pred[i].cpu().item()),
                        "logits":        logits[i].detach().cpu().tolist(),
                        "probs":         probs[i],
                    })

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # print(f"Batch data: {batch}")
                continue

            # break
            
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "epoch": epoch
    }
    # Log metrics to wandb
    logger.log(results)
    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch",  "precision", "recall", "f1", "accuracy"])
        writer.writerow([epoch,  precision, recall, f1, accuracy])  


# MMT Model initialization

def initialize_mmt_model(args, device):
    print("Initializing MMT model...")
    print("MMT Config: ", args.mmtransformercfg)
    model = ModularMultimodalTransformer(args.mmtransformercfg)
    model = model.to(device)
    print(model)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])

    # scheduler
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

def build_inputs(batch, active_modalities):
    # if images in active modalities, m is 'images_feat'
    if "images" in active_modalities:
        active_modalities.remove("images")
        active_modalities.append("images_feat")  # use pre-extracted features
    return {m: batch[m] for m in active_modalities if m in batch}


def train_mmt_one_epoch(model, train_loader, criterion, optimizer, device, logger, args):
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:

            
            inputs = build_inputs(batch, args.dataloader_params["modalities"])

            # print shape of each input tensor
            for k, v in inputs.items():
                print(f"  {k}: {v.shape}")
            inputs = {k: v.to(device) for k,v in inputs.items()}

            logits = model(inputs)
            print(f"MMT logits shape: {logits.shape}, Batch label: {batch['label']}")

            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")



        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            continue

    return total_loss / len(train_loader)



def validate_mmt(model, val_loader, criterion, device, logger, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                inputs = build_inputs(batch, args.dataloader_params["modalities"])
                inputs = {k: v.to(device) for k,v in inputs.items()}

                logits = model(inputs)

                loss = criterion(logits, batch['label'].to(device))
                total_loss += loss.item()
                if i % 10 == 0:
                    logger.log({"val_loss": loss.item()})

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return total_loss / len(val_loader)

def test_mmt_model(model, test_loader, criterion, device, logger, epoch, results_dir, args):
    model.eval()
    total_loss = 0


    accuracy = 0.0
    gt = []
    preds = []
    
    preds_detail = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                # forward
                inputs = build_inputs(batch, args.dataloader_params["modalities"])
                inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

                logits = model(inputs)                            # [B, C]
                labels = batch["label"].to(device, non_blocking=True)  # [B]
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # predictions
                pred = torch.argmax(logits, dim=1)               # [B]

                # accumulate scalar lists
                gt.extend(batch["label"].cpu().tolist())         # extend with B items
                preds.extend(pred.cpu().tolist())                # extend with B items

                # optional: probs if you need them
                probs = torch.softmax(logits, dim=1).detach().cpu().tolist()

                # detailed per-sample records
                B = pred.shape[0]
                trial_ids      = batch.get("trial_id",      [None]*B)   # might be list[str] or tensor
                subject_ids    = batch.get("subject_id",    [None]*B)
                gesture_codes  = batch.get("gesture_code",  [None]*B)   # usually list[str]

                for i in range(B):
                    preds_detail.append({
                        "trial_id":      trial_ids[i] if not torch.is_tensor(trial_ids) else trial_ids[i].item(),
                        "subject_id":    subject_ids[i] if not torch.is_tensor(subject_ids) else subject_ids[i].item(),
                        "gesture_code":  gesture_codes[i] if isinstance(gesture_codes, list) else gesture_codes[i],
                        "pred_label":    int(pred[i].cpu().item()),
                        "logits":        logits[i].detach().cpu().tolist(),
                        "probs":         probs[i],
                    })

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                # print(f"Batch data: {batch}")
                continue

            # break
            
    # Calculate metrics
    accuracy = sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)
    precision = precision_score(gt, preds, average='macro')
    recall = recall_score(gt, preds, average='macro')
    f1 = f1_score(gt, preds, average='macro')

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "epoch": epoch
    }
    # Log metrics to wandb
    logger.log(results)
    
    # Save metrics to CSV
    metrics_path = f'{results_dir}/metrics.csv'
    with open(metrics_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["epoch",  "precision", "recall", "f1", "accuracy"])
        writer.writerow([epoch,  precision, recall, f1, accuracy])

    # Save detailed predictions to CSV
    preds_path = f'{results_dir}/preds.csv'
    print("Saving predictions to: ", preds_path)
    with open(preds_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["trial_id", "subject_id", "gesture_code", "pred_label","probs"])
        for pred in preds_detail:
            writer.writerow([pred["trial_id"], pred["subject_id"], pred["gesture_code"], pred["pred_label"], pred["probs"]])
    return results

def mmt_preprocess(batch, args, backbone, device):
    """
    Build [B, T_max, F_total] by concatenating features from the modalities
    listed in args.dataloader_params["modalities"].

    Rules for image data:
      - If "images_feat" is listed in modalities AND present in batch, use it directly.
      - Else if "images" is listed AND present, run backbone.extract_resnet on frames to get per-frame features.
      - "images" frames themselves are NOT concatenated to model inputs; they are for viz.
    """
    if not hasattr(args, "dataloader_params") or "modalities" not in args.dataloader_params:
        raise ValueError("args.dataloader_params['modalities'] not found.")

    wanted_mods = list(args.dataloader_params["modalities"])
    # print("Wanted modalities for MMT input:", wanted_mods)

    # print("Batch keys:", list(batch.keys()))

    # Expand the effective list with image logic:
    # Prefer precomputed features if requested/present; otherwise fall back to frames->backbone.
    # print("batch", batch)
    effective_mods = []
    for m in wanted_mods:
        # print("Processing modality:", m)
        if m == "images_feat":
            if "images_feat" in batch:
                effective_mods.append("images_feat")
            # if images_feat was requested but not present, we do NOT silently
            # substitute frames; keep behavior strict and explicit.
        elif m == "images":
            if "images_feat" in batch:
                # If user asked for "images" but precomputed features are already available,
                # prefer features for the model (frames are for viz). Add only once.
                if "images_feat" not in effective_mods:
                    effective_mods.append("images_feat")
            elif "images" in batch:
                # We will compute features from frames using the provided backbone.
                effective_mods.append("images")  # means: derive features from frames below
        else:
            if m in batch:
                effective_mods.append(m)

    if not effective_mods:
        print("Probklems with batch keys:", list(batch.keys()))
        raise ValueError(
            "No active modalities found in batch matching args.dataloader_params['modalities'] "
            "(nothing to concatenate)."
        )

    mod_tensors = []  # list of (name, tensor[B,T,F])
    T_max = 0
    B_ref = None

    for m in effective_mods:
        if m == "images_feat":
            x = batch["images_feat"]  # [B, T, F_img]
            if not isinstance(x, torch.Tensor) or x.dim() != 3:
                raise ValueError(f"'images_feat' must be [B,T,F], got {type(x)} with shape {getattr(x,'shape',None)}")
            B, T, F = x.shape
            if B_ref is None:
                B_ref = B
            elif B != B_ref:
                raise ValueError(f"Batch size mismatch across modalities: expected {B_ref}, got {B} for 'images_feat'")

            # Respect images_mask if present (zero out invalid timesteps)
            img_mask = batch.get("images_mask", None)  # [B,T] bool
            if img_mask is not None:
                if not isinstance(img_mask, torch.Tensor) or img_mask.shape[:2] != (B, T):
                    raise ValueError(f"images_mask must be [B,T] bool, got {type(img_mask)} with shape {getattr(img_mask,'shape',None)}")
                mask = img_mask.to(device=device, dtype=x.dtype).unsqueeze(-1)  # [B,T,1]
                x = x.to(device, non_blocking=True) * mask
            else:
                x = x.to(device, non_blocking=True)

            # print(f"Projected image features shape: {x.shape}")
            mod_tensors.append((m, x))
            T_max = max(T_max, T)
            continue

        if m == "images":
            # Derive features from frames using the provided backbone
            imgs = batch["images"]  # [B, T, C, H, W], float in [0,1]
            if not isinstance(imgs, torch.Tensor) or imgs.dim() != 5:
                raise ValueError(f"'images' must be [B,T,C,H,W], got {type(imgs)} with shape {getattr(imgs,'shape',None)}")

            B, T, C, H, W = imgs.shape
            if B_ref is None:
                B_ref = B
            elif B != B_ref:
                raise ValueError(f"Batch size mismatch across modalities: expected {B_ref}, got {B} for 'images'")

            imgs = imgs.to(device, non_blocking=True)

            # Backbone is user-provided; expected to output per-frame features.
            # We flatten B,T -> (B*T) for feature extraction, then reshape back.
            imgs_btchw = imgs.view(B * T, C, H, W)

            with torch.no_grad():
                feats = backbone.extract_resnet(imgs_btchw)  # expected [B*T, D] or [B*T, D, 1, 1]

            if feats.dim() == 4:
                # pool to [N, D]
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
            elif feats.dim() == 2:
                pass
            else:
                feats = feats.view(feats.size(0), -1)

            feats = feats.view(B, T, -1)  # [B, T, Fimg]

            # Respect images_mask if present (zero out invalid timesteps)
            img_mask = batch.get("images_mask", None)  # [B,T] bool
            if img_mask is not None:
                if not isinstance(img_mask, torch.Tensor) or img_mask.shape != (B, T):
                    raise ValueError(f"images_mask must be [B,T] bool, got {type(img_mask)} with shape {getattr(img_mask,'shape',None)}")
                mask = img_mask.to(device=device, dtype=feats.dtype).unsqueeze(-1)  # [B,T,1]
                feats = feats * mask

            mod_tensors.append(("images_feat", feats))  # store as images_feat going forward
            T_max = max(T_max, T)
            continue

        # ---- Non-image modalities: expected [B, T, F] ----
        x = batch[m]
        if not isinstance(x, torch.Tensor) or x.dim() != 3:
            raise ValueError(f"Expected [B,T,F] for modality '{m}', got {type(x)} with shape {getattr(x,'shape',None)}")

        B, T, F = x.shape
        if B_ref is None:
            B_ref = B
        elif B != B_ref:
            raise ValueError(f"Batch size mismatch across modalities: got {B_ref} and {B} for '{m}'")

        T_max = max(T_max, T)
        mod_tensors.append((m, x.to(device, non_blocking=True)))

    # Right-pad each [B,T,F] to T_max along time, then concat on feature dim
    padded = []
    for _, x in mod_tensors:
        B, T, F = x.shape
        if T < T_max:
            pad = x.new_zeros((B, T_max - T, F))
            x = torch.cat([x, pad], dim=1)
        padded.append(x)

    X = torch.cat(padded, dim=-1)  # [B, T_max, sum(F)]
    return X



def get_feature_dim(loader, args, model, device):
    batch = next(iter(loader))
    preprocessed_inputs = mmt_preprocess(batch, args, model, device)
    return preprocessed_inputs.size(-1)


def train_transtcn_one_epoch(model, train_loader, criterion, optimizer, device, logger, args):
    
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:

            preprocessed_inputs = mmt_preprocess(batch, args, model, device)  # [B, T, F_total]
            # print(f" Preprocessed input shape: {preprocessed_inputs.shape}")
            logits = model(preprocessed_inputs)
            # print("prediction argmax: ", torch.argmax(logits, dim=1))
            # print(F" Batch label: {batch['label']}")
            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")



        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            continue

    return total_loss / len(train_loader)


def validate_transtcn(model, val_loader, criterion, device, logger, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
                logits = model(preprocessed_inputs)

                loss = criterion(logits, batch['label'].to(device))
                total_loss += loss.item()
                if i % 10 == 0:
                    logger.log({"val_loss": loss.item()})

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return total_loss / len(val_loader)

def test_transtcn_model(model, test_loader, criterion, device, logger, epoch, results_dir, args, class_names=None):
    """
    Enhanced TransTCN model testing with comprehensive metrics for ML papers.
    
    Args:
        model: The TransTCN model to evaluate
        test_loader: DataLoader for test data
        criterion: Loss function
        device: Device to run on
        logger: Logger (e.g., wandb)
        epoch: Current epoch number
        results_dir: Directory to save results
        args: Arguments object containing model configuration
        class_names: Optional; dict {contiguous_id->name} or list ["name_for_0", ...]
    """
    model.eval()
    total_loss = 0
    gt = []
    preds = []
    preds_detail = []
    all_logits = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            try:
                # Forward pass
                preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
                logits = model(preprocessed_inputs)                              # [B, C]
                labels = batch["label"].to(device, non_blocking=True)           # [B]
                loss = criterion(logits, labels)
                total_loss += loss.item()

                # Predictions
                pred = torch.argmax(logits, dim=1)      # [B]
                probs = torch.softmax(logits, dim=1)    # [B, C]

                # Accumulate for metrics
                gt.extend(labels.cpu().tolist())
                preds.extend(pred.cpu().tolist())
                all_logits.extend(logits.detach().cpu().tolist())
                all_probs.extend(probs.detach().cpu().tolist())

                # Extract batch metadata
                B = pred.shape[0]
                trial_ids = batch.get("trial_id", [None]*B)
                subject_ids = batch.get("subject_id", [None]*B)
                gesture_codes = batch.get("gesture_code", [None]*B)

                # Store detailed per-sample records
                for i in range(B):
                    true_label = int(labels[i].cpu().item())
                    pred_label = int(pred[i].cpu().item())
                    preds_detail.append({
                        "trial_id": trial_ids[i] if not torch.is_tensor(trial_ids) else trial_ids[i].item(),
                        "subject_id": subject_ids[i] if not torch.is_tensor(subject_ids) else subject_ids[i].item(),
                        "gesture_code": gesture_codes[i] if isinstance(gesture_codes, list) else gesture_codes[i],
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "correct": true_label == pred_label,
                        "logits": logits[i].detach().cpu().tolist(),
                        "probs": probs[i].detach().cpu().tolist(),
                        "confidence": float(probs[i, pred_label].cpu().item()),
                        "true_class_prob": float(probs[i, true_label].cpu().item()),
                    })

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                traceback.print_exc()
                continue

    # Averages
    avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0.0

    # Overall metrics
    accuracy = (sum(1 for x, y in zip(preds, gt) if x == y) / len(gt)) if len(gt) > 0 else 0.0
    precision_macro = precision_score(gt, preds, average='macro', zero_division=0)
    recall_macro    = recall_score(gt, preds, average='macro', zero_division=0)
    f1_macro        = f1_score(gt, preds, average='macro', zero_division=0)
    precision_weighted = precision_score(gt, preds, average='weighted', zero_division=0)
    recall_weighted    = recall_score(gt, preds, average='weighted', zero_division=0)
    f1_weighted        = f1_score(gt, preds, average='weighted', zero_division=0)
    balanced_acc = balanced_accuracy_score(gt, preds)
    kappa        = cohen_kappa_score(gt, preds)

    # Class sets
    classes_gt   = sorted(set(gt))               # only classes present in ground truth
    classes_all  = sorted(set(gt) | set(preds))  # union of GT and predictions
    n_classes_all = len(classes_all)

    # Top-k accuracy (only if we can align labels to score columns; may be skipped)
    top_k_acc = {}
    if n_classes_all >= 3:
        for k in [3, 5]:
            if k < n_classes_all:
                try:
                    # Note: y_score must correspond to the order of `labels`; if not perfectly aligned, we skip.
                    top_k_acc[f"top_{k}_accuracy"] = top_k_accuracy_score(
                        gt, np.array(all_probs), k=k, labels=classes_all
                    )
                except Exception:
                    top_k_acc[f"top_{k}_accuracy"] = None

    # ROC-AUC
    try:
        if len(classes_all) == 2:
            roc_auc = roc_auc_score(gt, np.array(all_probs)[:, 1])
        else:
            roc_auc = roc_auc_score(gt, np.array(all_probs), multi_class='ovr', average='macro')
    except Exception:
        roc_auc = None

    # ---------- Class name helpers ----------
    def _names_for(cls_ids, names):
        if names is None:
            return [f"Class {i}" for i in cls_ids]
        if isinstance(names, dict):
            return [names.get(i, f"Class {i}") for i in cls_ids]
        # assume list-like aligned to contiguous ids
        out = []
        for i in cls_ids:
            out.append(names[i] if 0 <= i < len(names) else f"Class {i}")
        return out

    class_names_all = _names_for(classes_all, class_names)
    class_names_gt  = _names_for(classes_gt,  class_names)

    # ---------- Confusion matrix on union labels ----------
    cm = confusion_matrix(gt, preds, labels=classes_all)

    # Per-class accuracy (row recall) with safe division
    row_sums = cm.sum(axis=1, keepdims=True)  # (N,1)
    class_accuracy_all = np.divide(
        np.diag(cm).reshape(-1, 1), row_sums,
        out=np.zeros((n_classes_all, 1), dtype=float), where=row_sums != 0
    ).squeeze(1)

    # ---------- Per-class metrics on GT-only labels; map back to union ----------
    if classes_gt:
        precision_gt = precision_score(gt, preds, labels=classes_gt, average=None, zero_division=0)
        recall_gt    = recall_score(gt, preds,    labels=classes_gt, average=None, zero_division=0)
        f1_gt        = f1_score(gt, preds,        labels=classes_gt, average=None, zero_division=0)
    else:
        precision_gt = recall_gt = f1_gt = np.array([])

    idx_all = {cid: i for i, cid in enumerate(classes_all)}
    precision_all_vec = np.zeros(n_classes_all, dtype=float)
    recall_all_vec    = np.zeros(n_classes_all, dtype=float)
    f1_all_vec        = np.zeros(n_classes_all, dtype=float)
    for j, cid in enumerate(classes_gt):
        i_all = idx_all[cid]
        precision_all_vec[i_all] = precision_gt[j]
        recall_all_vec[i_all]    = recall_gt[j]
        f1_all_vec[i_all]        = f1_gt[j]

    # ---------- Classification report (GT-only) ----------
    try:
        class_report = classification_report(
            gt, preds,
            labels=classes_gt,
            target_names=class_names_gt,
            zero_division=0,
            output_dict=True
        )
    except Exception:
        class_report = classification_report(gt, preds, zero_division=0, output_dict=True)

    # ---------- Confidence stats ----------
    confidences = [p["confidence"] for p in preds_detail]
    correct_confidences   = [p["confidence"] for p in preds_detail if p["correct"]]
    incorrect_confidences = [p["confidence"] for p in preds_detail if not p["correct"]]
    confidence_stats = {
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "std_confidence":  float(np.std(confidences))  if confidences else 0.0,
        "mean_confidence_correct": float(np.mean(correct_confidences))   if correct_confidences else 0.0,
        "mean_confidence_incorrect": float(np.mean(incorrect_confidences)) if incorrect_confidences else 0.0,
    }

    # ---------- Results dict ----------
    results = {
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": kappa,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }
    if roc_auc is not None:
        results["roc_auc"] = roc_auc
    results.update(confidence_stats)
    results.update(top_k_acc)

    # Per-class metrics (aligned with classes_all)
    for i, class_idx in enumerate(classes_all):
        results[f"class_{class_idx}_accuracy"]  = float(class_accuracy_all[i])
        results[f"class_{class_idx}_precision"] = float(precision_all_vec[i])
        results[f"class_{class_idx}_recall"]    = float(recall_all_vec[i])
        results[f"class_{class_idx}_f1"]        = float(f1_all_vec[i])

    # Log metrics to wandb
    logger.log(results)

    # ---------- Save artifacts ----------
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Overall metrics CSV
    metrics_path = f'{results_dir}/metrics_epoch_{epoch}.csv'
    with open(metrics_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Epoch", epoch])
        writer.writerow(["Loss", avg_loss])
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["Balanced Accuracy", balanced_acc])
        writer.writerow(["Cohen's Kappa", kappa])
        if roc_auc is not None:
            writer.writerow(["ROC-AUC", roc_auc])
        for k, v in top_k_acc.items():
            if v is not None:
                writer.writerow([k.replace('_', ' ').title(), v])
        writer.writerow(["Precision (Macro)", precision_macro])
        writer.writerow(["Recall (Macro)", recall_macro])
        writer.writerow(["F1 Score (Macro)", f1_macro])
        writer.writerow(["Precision (Weighted)", precision_weighted])
        writer.writerow(["Recall (Weighted)", recall_weighted])
        writer.writerow(["F1 Score (Weighted)", f1_weighted])
        writer.writerow(["Mean Confidence", confidence_stats["mean_confidence"]])
        writer.writerow(["Mean Confidence (Correct)", confidence_stats["mean_confidence_correct"]])
        writer.writerow(["Mean Confidence (Incorrect)", confidence_stats["mean_confidence_incorrect"]])

    # Per-class metrics CSV (aligned with classes_all)
    class_metrics_path = f'{results_dir}/class_metrics_epoch_{epoch}.csv'
    with open(class_metrics_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Class_ID", "Class_Name", "Accuracy", "Precision", "Recall", "F1-Score", "Support"])
        for i, class_idx in enumerate(classes_all):
            support = int(cm[i].sum())
            writer.writerow([
                class_idx,
                class_names_all[i],
                f"{class_accuracy_all[i]:.4f}",
                f"{precision_all_vec[i]:.4f}",
                f"{recall_all_vec[i]:.4f}",
                f"{f1_all_vec[i]:.4f}",
                support
            ])

    # Confusion matrix CSVs
    np.savetxt(f'{results_dir}/confusion_matrix_epoch_{epoch}.csv', cm, delimiter=',', fmt='%d')
    with open(f'{results_dir}/confusion_matrix_labeled_epoch_{epoch}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True\\Predicted'] + class_names_all)
        for i, row in enumerate(cm):
            writer.writerow([class_names_all[i]] + row.tolist())

    # Confusion matrix plot
    plt.figure(figsize=(max(10, n_classes_all), max(8, n_classes_all * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_all, yticklabels=class_names_all,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Normalized confusion matrix (row recall) — SAFE DIVIDE, no broadcasting
    row_sums = cm.sum(axis=1, keepdims=True)  # (N,1)
    cm_normalized = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    )
    plt.figure(figsize=(max(10, n_classes_all), max(8, n_classes_all * 0.8)))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names_all, yticklabels=class_names_all,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    plt.title(f'Normalized Confusion Matrix (Recall) - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix_normalized_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Per-class bars
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics_to_plot = [
        (class_accuracy_all, 'Accuracy', axes[0, 0]),
        (precision_all_vec,  'Precision', axes[0, 1]),
        (recall_all_vec,     'Recall', axes[1, 0]),
        (f1_all_vec,         'F1-Score', axes[1, 1])
    ]
    for metric_values, metric_name, ax in metrics_to_plot:
        vals = np.nan_to_num(metric_values, nan=0.0)
        bars = ax.bar(range(len(class_names_all)), vals, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_xlabel('Class', fontsize=11)
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'Per-Class {metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(class_names_all)))
        ax.set_xticklabels(class_names_all, rotation=45, ha='right')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/per_class_metrics_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Confidence plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(confidences, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    if confidences:
        axes[0].axvline(np.mean(confidences), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(confidences):.3f}')
    axes[0].set_xlabel('Prediction Confidence', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Prediction Confidence', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    if correct_confidences and incorrect_confidences:
        axes[1].hist([correct_confidences, incorrect_confidences], bins=20, 
                     label=['Correct', 'Incorrect'], color=['green', 'red'], 
                     alpha=0.6, edgecolor='black')
        axes[1].set_xlabel('Prediction Confidence', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Confidence by Prediction Correctness', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confidence_distribution_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Classification report JSON/TXT (GT-only)
    with open(f'{results_dir}/classification_report_epoch_{epoch}.json', 'w') as f:
        json.dump(class_report, f, indent=4)
    try:
        report_text = classification_report(
            gt, preds, labels=classes_gt, target_names=class_names_gt, zero_division=0
        )
    except Exception:
        report_text = classification_report(gt, preds, zero_division=0)
    with open(f'{results_dir}/classification_report_epoch_{epoch}.txt', 'w') as f:
        f.write(report_text)

    # Detailed predictions
    preds_path = f'{results_dir}/predictions_epoch_{epoch}.csv'
    print("Saving predictions to:", preds_path)
    with open(preds_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "trial_id", "subject_id", "gesture_code", "true_label", 
            "pred_label", "correct", "confidence", "true_class_prob", 
            "logits", "probs"
        ])
        for pred in preds_detail:
            writer.writerow([
                pred["trial_id"], 
                pred["subject_id"], 
                pred["gesture_code"], 
                pred["true_label"],
                pred["pred_label"], 
                pred["correct"],
                f"{pred['confidence']:.4f}",
                f"{pred['true_class_prob']:.4f}",
                pred["logits"],
                pred["probs"]
            ])

    misclassified = [p for p in preds_detail if not p["correct"]]
    if misclassified:
        with open(f'{results_dir}/misclassified_epoch_{epoch}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "trial_id", "subject_id", "gesture_code", "true_label", 
                "pred_label", "confidence", "true_class_prob"
            ])
            for pred in misclassified:
                writer.writerow([
                    pred["trial_id"], pred["subject_id"], pred["gesture_code"],
                    pred["true_label"], pred["pred_label"],
                    f"{pred['confidence']:.4f}", f"{pred['true_class_prob']:.4f}"
                ])

    # Summary
    with open(f'{results_dir}/summary_epoch_{epoch}.txt', 'w') as f:
        f.write(f"TransTCN Model Evaluation Summary - Epoch {epoch}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Accuracy: {accuracy:.4f}\n")
        f.write(f"\nMacro-averaged Metrics:\n")
        f.write(f"  Precision: {precision_macro:.4f}\n")
        f.write(f"  Recall: {recall_macro:.4f}\n")
        f.write(f"  F1-Score: {f1_macro:.4f}\n")
        f.write(f"  Loss: {avg_loss:.4f}\n")
        f.write(f"  Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"  Cohen's Kappa: {kappa:.4f}\n")
        if roc_auc is not None:
            f.write(f"  ROC-AUC: {roc_auc:.4f}\n")
        if top_k_acc:
            f.write(f"\nTop-K Accuracy:\n")
            for k, v in top_k_acc.items():
                if v is not None:
                    f.write(f"  {k.replace('_', '-').title()}: {v:.4f}\n")
        f.write(f"\nWeighted-averaged Metrics:\n")
        f.write(f"  Precision: {precision_weighted:.4f}\n")
        f.write(f"  Recall: {recall_weighted:.4f}\n")
        f.write(f"  F1-Score: {f1_weighted:.4f}\n")
        f.write(f"\nConfidence Statistics:\n")
        f.write(f"  Mean Confidence: {confidence_stats['mean_confidence']:.4f}\n")
        f.write(f"  Std Confidence: {confidence_stats['std_confidence']:.4f}\n")
        f.write(f"  Mean Confidence (Correct): {confidence_stats['mean_confidence_correct']:.4f}\n")
        f.write(f"  Mean Confidence (Incorrect): {confidence_stats['mean_confidence_incorrect']:.4f}\n")
        f.write(f"\nPer-Class Metrics (union of GT & preds):\n")
        for i, class_idx in enumerate(classes_all):
            f.write(f"\n  {class_names_all[i]} (Class {class_idx}):\n")
            f.write(f"    Accuracy: {class_accuracy_all[i]:.4f}\n")
            f.write(f"    Precision: {precision_all_vec[i]:.4f}\n")
            f.write(f"    Recall: {recall_all_vec[i]:.4f}\n")
            f.write(f"    F1-Score: {f1_all_vec[i]:.4f}\n")
            f.write(f"    Support: {int(cm[i].sum())}\n")
        f.write(f"\n" + "=" * 70 + "\n")
        f.write(f"Total Samples: {len(gt)}\n")
        f.write(f"Correct Predictions: {sum(1 for x, y in zip(preds, gt) if x == y)}\n")
        f.write(f"Incorrect Predictions: {len(misclassified)}\n")
        f.write(f"Number of Classes (union): {n_classes_all}\n")

    print(f"\n{'='*70}")
    print(f"TransTCN Evaluation Complete - Epoch {epoch}")
    print(f"Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | F1 (Macro): {f1_macro:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f} | Kappa: {kappa:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

    return results



def initialize_tcn_model(args, input_dim, device, num_classes):
    print("Initializing TCN model...")
    print("TCN Config: input_dim=", input_dim, ", num_classes=", num_classes)

    args.tcn_model_params['input_dim'] = input_dim
    args.tcn_model_params['encoder_params']['input_size'] = input_dim    
    
    model_params = {
        "class_num": num_classes,
        "decoder_params": args.tcn_model_params['decoder_params'],
        "encoder_params": args.tcn_model_params['encoder_params'],
        "fc_size": 32,
        "mid_lstm_params": {
            "hidden_size": 64,
            "input_size": 128,
            "layer_num": 1
        }
    }
    model = EncoderDecoderNet(**model_params)
    model = model.to(device)
    print(model)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])

    # scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    return model, optimizer, criterion


def initialize_multi_tcn_model(args, device, feature_dim):
    print("Initializing Multi-TCN model...")
    print("Multi-TCN Config: ", args.multi_tcncfg)
    model = MultiBranchTCNClassifier(modality_input_dims=args.tc, num_classes=args.num_classes, **args.multi_tcncfg)
    model = model.to(device)
    print(model)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])

    # scheduler
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion



def train_TCN_one_epoch(model, train_loader, criterion, optimizer, device, logger, args):
    
    model.train()
    total_loss = 0
    for i, batch in enumerate(train_loader):

        try:

            preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
            # skip batch if sequence length is less than 10
            if preprocessed_inputs.shape[1] < 10:
                continue

            logits = model(preprocessed_inputs)


            loss = criterion(logits, batch['label'].to(device))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Batch {i}, Loss: {loss.item()}")



        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # print stack trace
            import traceback
            traceback.print_exc()
            continue

    return total_loss / len(train_loader)


def validate_TCN(model, val_loader, criterion, device, logger, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                preprocessed_inputs = mmt_preprocess(batch, args, None, device)  # [B, T, F_total]
                            # skip batch if sequence length is less than 10
                if preprocessed_inputs.shape[1] < 10:
                    continue
                logits = model(preprocessed_inputs)

                loss = criterion(logits, batch['label'].to(device))
                total_loss += loss.item()
                if i % 10 == 0:
                    logger.log({"val_loss": loss.item()})

            except Exception as e:
                print(f"Error in batch {i}: {e}")
                import traceback
                traceback.print_exc()
                continue

    return total_loss / len(val_loader)
def test_TCN_model(model, test_loader, criterion, device, logger, epoch, results_dir, args, class_names=None):
    import os, csv, json
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    from sklearn.metrics import (
        precision_score, recall_score, f1_score, balanced_accuracy_score,
        cohen_kappa_score, jaccard_score, confusion_matrix,
        classification_report, roc_auc_score, top_k_accuracy_score
    )

    Path(results_dir).mkdir(parents=True, exist_ok=True)

    model.eval()
    total_loss = 0.0

    gt = []
    preds = []
    all_probs = []          # list of per-sample probability vectors
    all_logits = []         # optional, not strictly needed
    preds_detail = []

    with torch.no_grad():
        for bidx, batch in enumerate(test_loader):
            try:
                # Forward
                x_BTF = mmt_preprocess(batch, args, None, device)          # [B, T, F_total]
                if x_BTF.shape[1] < 10:                                     # keep your early skip
                    continue
                logits = model(x_BTF)                                       # [B, C]
                labels = batch["label"].to(device, non_blocking=True)       # [B]
                loss = criterion(logits, labels)
                total_loss += float(loss.item())

                # Predictions / probs
                pred = torch.argmax(logits, dim=1)                           # [B]
                probs = torch.softmax(logits, dim=1)                         # [B, C]

                # Accumulate
                gt.extend(labels.detach().cpu().tolist())
                preds.extend(pred.detach().cpu().tolist())
                all_probs.extend(probs.detach().cpu().tolist())
                all_logits.extend(logits.detach().cpu().tolist())

                # Metadata (tolerate list or tensor)
                B = pred.shape[0]
                trial_ids     = batch.get("trial_id", [None]*B)
                subject_ids   = batch.get("subject_id", [None]*B)
                gesture_codes = batch.get("gesture_code", [None]*B)

                if torch.is_tensor(trial_ids):     trial_ids = trial_ids.cpu().tolist()
                if torch.is_tensor(subject_ids):   subject_ids = subject_ids.cpu().tolist()
                if torch.is_tensor(gesture_codes): gesture_codes = gesture_codes.cpu().tolist()

                for i in range(B):
                    t_id  = trial_ids[i] if not torch.is_tensor(trial_ids) else trial_ids[i].item()
                    s_id  = subject_ids[i] if not torch.is_tensor(subject_ids) else subject_ids[i].item()
                    gcode = gesture_codes[i] if isinstance(gesture_codes, list) else gesture_codes[i]
                    plab  = int(pred[i].cpu().item())
                    tlab  = int(labels[i].cpu().item())
                    conf  = float(probs[i, plab].cpu().item())
                    tprob = float(probs[i, tlab].cpu().item()) if (0 <= tlab < probs.shape[1]) else 0.0

                    preds_detail.append({
                        "trial_id": t_id,
                        "subject_id": s_id,
                        "gesture_code": gcode,
                        "true_label": tlab,
                        "pred_label": plab,
                        "correct": (tlab == plab),
                        "confidence": conf,
                        "true_class_prob": tprob,
                        "logits": logits[i].detach().cpu().tolist(),
                        "probs": probs[i].detach().cpu().tolist(),
                    })

            except Exception as e:
                print(f"[TEST] Error in batch {bidx}: {e}")
                import traceback; traceback.print_exc()
                continue

    # ----- If no data collected -----
    if len(gt) == 0:
        print("Warning: no test samples were processed.")
        results = {"epoch": epoch, "loss": 0.0, "accuracy": 0.0,
                   "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0}
        logger.log(results)
        return results

    # ----- LOUO-safe class set: derive from y_true only -----
    unique_classes = sorted(list(set(gt)))
    n_classes = len(unique_classes)

    # Compute core metrics
    accuracy = sum(int(p == t) for p, t in zip(preds, gt)) / len(gt)

    precision_macro  = precision_score(gt, preds, average='macro', zero_division=0)
    recall_macro     = recall_score(gt, preds, average='macro', zero_division=0)
    f1_macro         = f1_score(gt, preds, average='macro', zero_division=0)

    precision_weighted = precision_score(gt, preds, average='weighted', zero_division=0)
    recall_weighted    = recall_score(gt, preds, average='weighted', zero_division=0)
    f1_weighted        = f1_score(gt, preds, average='weighted', zero_division=0)

    precision_per_class = precision_score(gt, preds, average=None, labels=unique_classes, zero_division=0)
    recall_per_class    = recall_score(gt, preds, average=None, labels=unique_classes, zero_division=0)
    f1_per_class        = f1_score(gt, preds, average=None, labels=unique_classes, zero_division=0)

    balanced_acc = balanced_accuracy_score(gt, preds)
    kappa        = cohen_kappa_score(gt, preds)
    jaccard_macro   = jaccard_score(gt, preds, average='macro', zero_division=0)
    jaccard_weighted= jaccard_score(gt, preds, average='weighted', zero_division=0)

    # Top-k (optional)
    top_k_acc = {}
    if n_classes >= 3:
        try:
            probs_arr = np.asarray(all_probs)  # [N, C]
            # restrict nothing for top-k (works fine as long as C>=k and labels passed)
            for k in (3, 5):
                if k < probs_arr.shape[1]:
                    top_k_acc[f"top_{k}_accuracy"] = top_k_accuracy_score(gt, probs_arr, k=k, labels=unique_classes)
        except Exception:
            for k in (3, 5):
                top_k_acc[f"top_{k}_accuracy"] = None

    # ROC-AUC (optional)
    roc_auc = None
    try:
        probs_arr = np.asarray(all_probs)  # [N, C]
        if n_classes == 2:
            # pick column for positive class (assume the larger label is positive)
            pos_col = unique_classes[-1]
            roc_auc = roc_auc_score(gt, probs_arr[:, pos_col])
        elif n_classes > 2:
            # subset columns to unique_classes order
            cols = np.array(unique_classes, dtype=int)
            roc_auc = roc_auc_score(gt, probs_arr[:, cols], multi_class='ovr', average='macro')
    except Exception:
        roc_auc = None

    # Confusion matrix & per-class accuracy
    cm = confusion_matrix(gt, preds, labels=unique_classes)
    row_sums = cm.sum(axis=1, keepdims=True) + 1e-10
    class_accuracy = (cm.diagonal().astype(np.float64) / row_sums.squeeze(1))

    # ----- Class-name handling aligned to unique_classes -----
    if class_names is None:
        class_names_list = [f"Class {c}" for c in unique_classes]
    elif isinstance(class_names, dict):
        class_names_list = [class_names.get(c, f"Class {c}") for c in unique_classes]
    else:
        # list/tuple indexed by contiguous id
        class_names_list = [
            class_names[c] if (isinstance(c, int) and c < len(class_names)) else f"Class {c}"
            for c in unique_classes
        ]

    # Average loss
    avg_loss = total_loss / max(1, len(test_loader))

    # Confidence stats
    confidences = [p["confidence"] for p in preds_detail]
    correct_conf = [p["confidence"] for p in preds_detail if p["correct"]]
    wrong_conf   = [p["confidence"] for p in preds_detail if not p["correct"]]
    confidence_stats = {
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "std_confidence":  float(np.std(confidences)) if confidences else 0.0,
        "mean_confidence_correct": float(np.mean(correct_conf)) if correct_conf else 0.0,
        "mean_confidence_incorrect": float(np.mean(wrong_conf)) if wrong_conf else 0.0,
    }

    # ----- Results dict -----
    results = {
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": kappa,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "jaccard_macro": jaccard_macro,
        "jaccard_weighted": jaccard_weighted,
        **{k: v for k, v in top_k_acc.items() if v is not None},
    }
    if roc_auc is not None:
        results["roc_auc"] = roc_auc
    results.update(confidence_stats)

    # Per-class into results
    for i, class_idx in enumerate(unique_classes):
        results[f"class_{class_idx}_accuracy"]  = float(class_accuracy[i])
        results[f"class_{class_idx}_precision"] = float(precision_per_class[i])
        results[f"class_{class_idx}_recall"]    = float(recall_per_class[i])
        results[f"class_{class_idx}_f1"]        = float(f1_per_class[i])

    # ----- Log -----
    logger.log(results)

    # ===== Saving =====
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Overall metrics CSV
    metrics_path = f'{results_dir}/metrics_epoch_{epoch}.csv'
    with open(metrics_path, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Epoch", epoch])
        w.writerow(["Loss", avg_loss])
        w.writerow(["Accuracy", accuracy])
        w.writerow(["Balanced Accuracy", balanced_acc])
        w.writerow(["Cohen's Kappa", kappa])
        if roc_auc is not None:
            w.writerow(["ROC-AUC", roc_auc])
        for k, v in top_k_acc.items():
            if v is not None:
                w.writerow([k.replace('_', ' ').title(), v])
        w.writerow(["Precision (Macro)", precision_macro])
        w.writerow(["Recall (Macro)", recall_macro])
        w.writerow(["F1 Score (Macro)", f1_macro])
        w.writerow(["Precision (Weighted)", precision_weighted])
        w.writerow(["Recall (Weighted)", recall_weighted])
        w.writerow(["F1 Score (Weighted)", f1_weighted])
        w.writerow(["Jaccard (Macro)", jaccard_macro])
        w.writerow(["Jaccard (Weighted)", jaccard_weighted])
        w.writerow(["Mean Confidence", confidence_stats["mean_confidence"]])
        w.writerow(["Mean Confidence (Correct)", confidence_stats["mean_confidence_correct"]])
        w.writerow(["Mean Confidence (Incorrect)", confidence_stats["mean_confidence_incorrect"]])

    # Per-class metrics CSV
    class_metrics_path = f'{results_dir}/class_metrics_epoch_{epoch}.csv'
    with open(class_metrics_path, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Class_ID", "Class_Name", "Accuracy", "Precision", "Recall", "F1-Score", "Support"])
        for i, class_idx in enumerate(unique_classes):
            support = int(cm[i].sum())
            w.writerow([
                class_idx,
                class_names_list[i],
                f"{class_accuracy[i]:.4f}",
                f"{precision_per_class[i]:.4f}",
                f"{recall_per_class[i]:.4f}",
                f"{f1_per_class[i]:.4f}",
                support,
            ])

    # Confusion matrices
    cm_path = f'{results_dir}/confusion_matrix_epoch_{epoch}.csv'
    np.savetxt(cm_path, cm, delimiter=',', fmt='%d')

    cm_labeled_path = f'{results_dir}/confusion_matrix_labeled_epoch_{epoch}.csv'
    with open(cm_labeled_path, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['True\\Predicted'] + class_names_list)
        for i, row in enumerate(cm):
            w.writerow([class_names_list[i]] + row.tolist())

    # Heatmaps
    fig_w = max(10, n_classes)
    fig_h = max(8, int(n_classes * 0.8))
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list,
                cbar_kws={'label': 'Count'})
    plt.title(f'Confusion Matrix - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    plt.figure(figsize=(fig_w, fig_h))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names_list, yticklabels=class_names_list,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    plt.title(f'Normalized Confusion Matrix (Recall) - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/confusion_matrix_normalized_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Per-class bar charts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics_to_plot = [
        (class_accuracy, 'Accuracy', axes[0, 0]),
        (precision_per_class, 'Precision', axes[0, 1]),
        (recall_per_class, 'Recall', axes[1, 0]),
        (f1_per_class, 'F1-Score', axes[1, 1]),
    ]
    for metric_values, metric_name, ax in metrics_to_plot:
        bars = ax.bar(range(len(class_names_list)), metric_values, color='skyblue', edgecolor='navy', alpha=0.7)
        ax.set_xlabel('Class'); ax.set_ylabel(metric_name)
        ax.set_title(f'Per-Class {metric_name}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(class_names_list)))
        ax.set_xticklabels(class_names_list, rotation=45, ha='right')
        ax.set_ylim([0, 1.1]); ax.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, metric_values)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/per_class_metrics_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Classification report
    class_report = classification_report(gt, preds,
                                         target_names=class_names_list,
                                         labels=unique_classes,
                                         zero_division=0,
                                         output_dict=True)
    with open(f'{results_dir}/classification_report_epoch_{epoch}.json', 'w') as f:
        json.dump(class_report, f, indent=4)
    with open(f'{results_dir}/classification_report_epoch_{epoch}.txt', 'w') as f:
        f.write(classification_report(gt, preds,
                                      target_names=class_names_list,
                                      labels=unique_classes,
                                      zero_division=0))

    # Save detailed predictions
    preds_path = f'{results_dir}/predictions_epoch_{epoch}.csv'
    print("Saving predictions to:", preds_path)
    with open(preds_path, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["trial_id","subject_id","gesture_code","true_label","pred_label",
                    "correct","confidence","true_class_prob","logits","probs"])
        for r in preds_detail:
            w.writerow([
                r["trial_id"], r["subject_id"], r["gesture_code"],
                r["true_label"], r["pred_label"], r["correct"],
                f'{r["confidence"]:.6f}', f'{r["true_class_prob"]:.6f}',
                r["logits"], r["probs"]
            ])

    print(f"\n{'='*70}")
    print(f"TCN Evaluation Complete - Epoch {epoch}")
    print(f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | F1 (Macro): {f1_macro:.4f}")
    print(f"Balanced Acc: {balanced_acc:.4f} | Kappa: {kappa:.4f} | Jaccard (Macro): {jaccard_macro:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.4f}")
    for k, v in top_k_acc.items():
        if v is not None:
            print(f"{k.replace('_',' ').title()}: {v:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

    return results


# =========================
# MS-TCN++ utilities
# =========================

IGNORE_INDEX = -100  # make sure your dataloader uses this for pad frames if you have variable-length sequences

def initialize_mstcn_model(args, input_dim, device, num_classes):
    print("Initializing MS-TCN++ model...")
    print("MSTCN++ Config: input_dim=", input_dim, ", num_classes=", num_classes)

    model = MS_TCN2(
        num_layers_PG=args.mstcn_model_params['num_layers_PG'],
        num_layers_R=args.mstcn_model_params['num_layers_R'],
        num_R=args.mstcn_model_params['num_R'],
        num_f_maps=args.mstcn_model_params['num_f_maps'],
        dim=input_dim,
        num_classes=num_classes
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_params['lr'],
        weight_decay=args.learning_params['weight_decay']
    )

    # Loss fns + weights
    loss_fns = {
        "ce": nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, label_smoothing=0.1),
        "mse": nn.MSELoss(reduction="none"),
        # args.learning_params is a dict in your code – use .get(...)
        "lambda_smooth": args.learning_params.get("lambda_smooth", 0.15),
        "num_classes": num_classes
    }

    return model, optimizer, loss_fns


@torch.no_grad()
def _masked_accuracy_from_logits(logits_BCT, target_BT, mask_B1T):
    """
    logits_BCT: (B, C, T)
    target_BT:  (B, T) long
    mask_B1T:   (B, 1, T) float {0,1}
    """
    # valid positions exclude IGNORE_INDEX and respect mask
    valid = (mask_B1T.squeeze(1) > 0.5) & (target_BT != IGNORE_INDEX)
    pred = logits_BCT.argmax(dim=1)  # (B, T)
    correct = (pred.eq(target_BT) & valid).sum().item()
    total = valid.sum().item()
    return (correct / total) if total > 0 else 0.0


def _as_stage_list(predictions):
    """
    Normalize model output into a list of (B,C,T).
    MS-TCN++ may return a list of tensors or a stacked tensor (S,B,C,T).
    """
    if isinstance(predictions, (list, tuple)):
        return list(predictions)
    # assume (S,B,C,T)
    return [predictions[s] for s in range(predictions.shape[0])]


def mstcn_compute_loss(predictions, target_BT, mask_B1T, loss_fns):
    """
    predictions: list[(B,C,T)] or (S,B,C,T)
    target_BT:   (B,T) long with class indices; pads should be IGNORE_INDEX
    mask_B1T:    (B,1,T) float {0,1} (1 for valid frames)
    """
    stages = _as_stage_list(predictions)
    ce = loss_fns["ce"]
    mse = loss_fns["mse"]
    lam = float(loss_fns.get("lambda_smooth", 0.15))

    total_loss = 0.0
    for p in stages:  # p: (B,C,T)
        # -------- Cross-Entropy per stage --------
        total_loss += ce(p, target_BT)  # CE expects (N,C,*) + target (N,*)

        # -------- Temporal smoothing on log-probs --------
        T = p.size(-1)
        if lam > 0.0 and T > 1:
            logp_t   = F.log_softmax(p[:, :, 1:],   dim=1)          # (B,C,T-1)
            logp_tm1 = F.log_softmax(p.detach()[:, :, :-1], dim=1)  # (B,C,T-1), stop-grad
            per_elem = mse(logp_t, logp_tm1)                        # (B,C,T-1)
            per_elem = torch.clamp(per_elem, min=0.0, max=16.0)
            m = mask_B1T[:, :, 1:]                                  # (B,1,T-1)
            smoothed = (per_elem * m).mean()
            total_loss += lam * smoothed

    return total_loss


# mstcn seq to one codes
# =========================
# MS-TCN++ — Sequence Classification (Fix B)
# =========================
import os, csv, json, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    balanced_accuracy_score, cohen_kappa_score,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Helpers
# -------------------------
def _as_stage_list(preds):
    """
    Normalize model output into a list of tensors (B, C, T) per stage.
    Accepts:
      - list/tuple of (B, C, T)
      - tensor of shape (S, B, C, T)
      - single tensor (B, C, T)
    """
    if isinstance(preds, (list, tuple)):
        return preds
    if torch.is_tensor(preds):
        if preds.dim() == 4:   # (S, B, C, T)
            return [preds[s] for s in range(preds.size(0))]
        elif preds.dim() == 3: # (B, C, T)
            return [preds]
    raise ValueError(f"Unexpected preds type/shape: {type(preds)}, {getattr(preds, 'shape', None)}")

def _masked_seq_mean(final_logits_BCT, mask_B1T=None):
    """
    final_logits_BCT: (B, C, T)
    mask_B1T: (B, 1, T) with {0,1} or None
    Returns seq_logits (B, C) via (masked) mean over T.
    """
    if mask_B1T is None:
        return final_logits_BCT.mean(dim=-1)
    w = mask_B1T.float()                      # (B,1,T)
    num = (final_logits_BCT * w).sum(dim=-1)  # (B,C)
    den = w.sum(dim=-1).clamp_min(1.0)        # (B,1)
    return num / den

def _get_mask_from_batch_or_default(batch, T, device):
    """
    Prefer explicit mask if present, else all-ones.
    Note: In clip-label mode we don't have framewise labels to derive a mask from.
    """
    if 'mask' in batch:
        mask = batch['mask'].to(device)
        if mask.dim() == 2: 
            mask = mask.unsqueeze(1)          # (B,1,T)
        if mask.size(-1) != T:
            raise ValueError(f"Mask time length {mask.size(-1)} != T {T}")
        return mask
    return torch.ones((batch['label'].shape[0], 1, T), device=device, dtype=torch.float32)

def _save_confusion_and_plots(cm, class_names, results_dir, epoch, title_prefix=""):
    fig_size = max(10, len(class_names) * 0.8)
    # Raw counts
    plt.figure(figsize=(fig_size, fig_size * 0.9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{title_prefix} Confusion Matrix - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{title_prefix.lower().replace(' ','_')}_confusion_matrix_epoch_{epoch}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    # Normalized by true (recall)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    plt.figure(figsize=(fig_size, fig_size * 0.9))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
    plt.title(f'{title_prefix} Confusion Matrix (Recall-Normalized) - Epoch {epoch}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"{title_prefix.lower().replace(' ','_')}_confusion_matrix_normalized_epoch_{epoch}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------
# TRAIN (sequence classification)
# -------------------------
def train_mstcn_seqclf_one_epoch(model, train_loader, optimizer, device, logger, args):
    """
    MS-TCN++ sequence classification:
      - Inputs: (B, T, F_total) -> (B, C=F_total, T)
      - Model: produces stages; use the final stage (B, C_classes, T)
      - Pool over time (masked mean if available) -> (B, C_classes)
      - Loss: CE(seq_logits, labels_B)
    """
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches = 0

    ce_loss = torch.nn.CrossEntropyLoss()

    for i, batch in enumerate(train_loader):
        try:
            # Inputs
            x_BTF = mmt_preprocess(batch, args, None, device)  # must return (B, T, F)
            if x_BTF.dim() != 3:
                raise ValueError(f"mmt_preprocess must return (B,T,F), got {tuple(x_BTF.shape)}")
            x_BCT = x_BTF.permute(0, 2, 1).contiguous()        # (B, C_in, T)
            B, C, T = x_BCT.shape

            # Labels: (B,) clip labels only
            labels_B = batch['label'].to(device)
            if labels_B.dim() != 1 or labels_B.size(0) != B:
                raise ValueError(f"Expect clip labels of shape (B,), got {tuple(labels_B.shape)}")

            # Optional: skip super-short clips (your original behavior)
            if T < 30:
                continue

            # Mask (optional)
            mask = _get_mask_from_batch_or_default(batch, T, device)  # (B,1,T)

            # Forward
            optimizer.zero_grad()
            preds = model(x_BCT)                       # list[(B,C_out,T)] or (S,B,C_out,T)
            stages = _as_stage_list(preds)
            final_logits_BCT = stages[-1]              # (B, C_classes, T)

            # Sequence logits via masked mean over time
            seq_logits_BC = _masked_seq_mean(final_logits_BCT, mask)  # (B, C_classes)

            # Loss & step
            loss = ce_loss(seq_logits_BC, labels_B)   # (B,)
            loss.backward()
            optimizer.step()

            # Accuracy (sequence level)
            seq_pred = seq_logits_BC.argmax(dim=1)    # (B,)
            acc = (seq_pred == labels_B).float().mean().item()

            running_loss += float(loss.item())
            running_acc  += float(acc)
            n_batches += 1

            if i % 10 == 0:
                print(f"[TRAIN-SEQ] Batch {i}: loss={loss.item():.4f}, acc={acc:.4f}")

        except Exception as e:
            print(f"[TRAIN-SEQ] Error in batch {i}: {e}")
            traceback.print_exc()
            continue

    if n_batches == 0:
        return 0.0, 0.0

    epoch_loss = running_loss / n_batches
    epoch_acc  = running_acc  / n_batches
    if logger is not None:
        try:
            logger.info(f"[TRAIN-SEQ] loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
        except Exception:
            pass
    return epoch_loss, epoch_acc

# -------------------------
# VALIDATE (sequence classification)
# -------------------------
def validate_mstcn_seqclf(model, val_loader, device, logger, args):
    model.eval()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches    = 0

    ce_loss = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                x_BTF = mmt_preprocess(batch, args, None, device)  # (B, T, F)
                if x_BTF.dim() != 3:
                    raise ValueError(f"mmt_preprocess must return (B,T,F), got {tuple(x_BTF.shape)}")
                x_BCT = x_BTF.permute(0, 2, 1).contiguous()
                B, C, T = x_BCT.shape

                labels_B = batch['label'].to(device)   # (B,)
                if labels_B.dim() != 1 or labels_B.size(0) != B:
                    raise ValueError(f"Expect clip labels of shape (B,), got {tuple(labels_B.shape)}")

                if T < 30:
                    continue

                mask = _get_mask_from_batch_or_default(batch, T, device)

                preds = model(x_BCT)
                stages = _as_stage_list(preds)
                final_logits_BCT = stages[-1]              # (B, C_classes, T)

                seq_logits_BC = _masked_seq_mean(final_logits_BCT, mask)
                loss  = ce_loss(seq_logits_BC, labels_B)

                seq_pred = seq_logits_BC.argmax(dim=1)
                acc = (seq_pred == labels_B).float().mean().item()

                running_loss += float(loss.item())
                running_acc  += float(acc)
                n_batches    += 1

                if i % 10 == 0:
                    print(f"[VAL-SEQ] Batch {i}: loss={loss.item():.4f}, acc={acc:.4f}")

            except Exception as e:
                print(f"[VAL-SEQ] Error in batch {i}: {e}")
                traceback.print_exc()
                continue

    if n_batches == 0:
        if logger is not None:
            try:
                logger.warning("[VAL-SEQ] Zero successful batches.")
            except Exception:
                pass
        return {"loss": 0.0, "acc": 0.0}

    val_loss = running_loss / n_batches
    val_acc  = running_acc  / n_batches

    if logger is not None:
        try:
            logger.info(f"[VAL-SEQ] loss={val_loss:.4f}, acc={val_acc:.4f}")
        except Exception:
            pass

    return {"loss": val_loss, "acc": val_acc}

# -------------------------
# TEST (sequence classification + rich reporting)
# -------------------------
def test_mstcn_seqclf(model, test_loader, device, logger, epoch, results_dir, args, class_names=None):
    """
    Sequence classification evaluation:
      - Pools final stage logits over time for (B,C)
      - Computes clip-level metrics (accuracy, macro/weighted P/R/F1, kappa, balanced acc)
      - Saves confusion matrices, per-class metrics, and per-sample artifacts
      - Also saves framewise predictions/probs for inspection (even though training is seq-level)
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval()

    save_figs = False

    ce_loss = torch.nn.CrossEntropyLoss()
    running_loss = 0.0
    n_batches = 0

    # Aggregates for metrics (sequence-level)
    all_gt_seq, all_pred_seq = [], []

    # Optional per-sample artifacts
    per_sample_records = []
    per_sample_detailed = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                x_BTF = mmt_preprocess(batch, args, None, device)  # (B,T,F)
                if x_BTF.dim() != 3:
                    raise ValueError(f"mmt_preprocess must return (B,T,F), got {tuple(x_BTF.shape)}")
                x_BCT = x_BTF.permute(0, 2, 1).contiguous()
                B, C, T = x_BCT.shape

                labels_B = batch['label'].to(device)  # (B,)
                if labels_B.dim() != 1 or labels_B.size(0) != B:
                    raise ValueError(f"Expect clip labels of shape (B,), got {tuple(labels_B.shape)}")

                # if T < 30:
                #     print(f"[TEST-SEQ] Skipping batch {i}: T={T} < 30")
                #     continue

                mask = _get_mask_from_batch_or_default(batch, T, device)

                preds = model(x_BCT)
                stages = _as_stage_list(preds)
                final_logits_BCT = stages[-1]                  # (B, C_classes, T)
                seq_logits_BC    = _masked_seq_mean(final_logits_BCT, mask)

                loss  = ce_loss(seq_logits_BC, labels_B)
                running_loss += float(loss.item())
                n_batches    += 1

                # Predictions (sequence-level)
                seq_pred_B = seq_logits_BC.argmax(dim=1)       # (B,)
                all_gt_seq.append(labels_B.detach().cpu().numpy())
                all_pred_seq.append(seq_pred_B.detach().cpu().numpy())

                # Save per-sample artifacts (also framewise stuff for inspection)
                probs_BCT = torch.softmax(final_logits_BCT, dim=1).detach().cpu().numpy()  # (B,C,T)
                pred_BT   = final_logits_BCT.argmax(dim=1).detach().cpu().numpy()          # (B,T)

                trial_ids     = batch.get("trial_id",    [None]*B)
                subject_ids   = batch.get("subject_id",  [None]*B)
                gesture_codes = batch.get("gesture_code",[None]*B)
                if torch.is_tensor(trial_ids):     trial_ids     = trial_ids.cpu().tolist()
                if torch.is_tensor(subject_ids):   subject_ids   = subject_ids.cpu().tolist()
                if torch.is_tensor(gesture_codes): gesture_codes = gesture_codes.cpu().tolist()

                pred_save_path = os.path.join(results_dir, "predictions")
                os.makedirs(pred_save_path, exist_ok=True)

                for b in range(B):
                    stem = str(trial_ids[b]) if trial_ids[b] is not None else f"sample_{i}_{b}"
                    # Framewise predictions (optional, for debugging/plots)
                    np.save(os.path.join(pred_save_path, f"{stem}_frame_pred.npy"), pred_BT[b])      # (T,)
                    if getattr(args, "save_probs", False):
                        np.save(os.path.join(pred_save_path, f"{stem}_frame_probs.npy"), probs_BCT[b]) # (C,T)
                    # Sequence prediction
                    np.save(os.path.join(pred_save_path, f"{stem}_seq_pred.npy"), seq_pred_B[b].cpu().numpy())
                    np.save(os.path.join(pred_save_path, f"{stem}_seq_gt.npy"),   labels_B[b].cpu().numpy())

                    per_sample_records.append({
                        "trial_id": trial_ids[b],
                        "subject_id": subject_ids[b],
                        "gesture_code": gesture_codes[b],
                        "T": int(T),
                        "seq_pred": int(seq_pred_B[b].item()),
                        "seq_gt":   int(labels_B[b].item()),
                    })

                    per_sample_detailed.append({
                        "trial_id": trial_ids[b],
                        "subject_id": subject_ids[b],
                        "gesture_code": gesture_codes[b],
                        "T": int(T),
                        "seq_pred": int(seq_pred_B[b].item()),
                        "seq_gt":   int(labels_B[b].item()),
                        "frame_pred": pred_BT[b].tolist()
                    })

                if i % 10 == 0:
                    logger.log({"test_seq_batch_loss": loss.item(), "batch": i})

            except Exception as e:
                print(f"[TEST-SEQ] Error in batch {i}: {e}")
                traceback.print_exc()
                continue

    # ---------- Aggregate & Report ----------
    if n_batches == 0:
        print("[TEST-SEQ] Warning: No valid batches processed")
        results = {
            "epoch": epoch, "loss": 0.0,
            "accuracy": 0.0, "balanced_accuracy": 0.0, "cohen_kappa": 0.0,
            "precision_macro": 0.0, "recall_macro": 0.0, "f1_macro": 0.0,
            "precision_weighted": 0.0, "recall_weighted": 0.0, "f1_weighted": 0.0,
            "total_samples": 0
        }
        logger.log(results)
        return results

    avg_loss = running_loss / n_batches

    gt_all   = np.concatenate(all_gt_seq,  axis=0)
    pred_all = np.concatenate(all_pred_seq, axis=0)

    accuracy = float(accuracy_score(gt_all, pred_all))
    precision_macro  = float(precision_score(gt_all, pred_all, average='macro',   zero_division=0))
    recall_macro     = float(recall_score(gt_all,    pred_all, average='macro',   zero_division=0))
    f1_macro         = float(f1_score(gt_all,        pred_all, average='macro',   zero_division=0))
    precision_weight = float(precision_score(gt_all, pred_all, average='weighted',zero_division=0))
    recall_weight    = float(recall_score(gt_all,    pred_all, average='weighted',zero_division=0))
    f1_weight        = float(f1_score(gt_all,        pred_all, average='weighted',zero_division=0))
    balanced_acc     = float(balanced_accuracy_score(gt_all, pred_all))
    kappa            = float(cohen_kappa_score(gt_all, pred_all))

    unique_classes = sorted(list(set(gt_all.tolist())))
    n_classes = len(unique_classes)

    # Class names
    if class_names is None:
        class_names = [f"Class_{c}" for c in unique_classes]
    elif len(class_names) < n_classes:
        class_names = class_names + [f"Class_{c}" for c in range(len(class_names), n_classes)]

    # Per-class metrics
    precision_per_class = precision_score(gt_all, pred_all, average=None, zero_division=0, labels=unique_classes)
    recall_per_class    = recall_score(gt_all,    pred_all, average=None, zero_division=0, labels=unique_classes)
    f1_per_class        = f1_score(gt_all,        pred_all, average=None, zero_division=0, labels=unique_classes)

    # Confusion matrix
    cm = confusion_matrix(gt_all, pred_all, labels=unique_classes)
    class_support = cm.sum(axis=1)

    # Results dict
    results = {
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": kappa,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weight,
        "recall_weighted": recall_weight,
        "f1_weighted": f1_weight,
        "total_samples": int(gt_all.shape[0]),
    }
    for i, cls in enumerate(unique_classes):
        results[f"class_{cls}_precision"] = float(precision_per_class[i])
        results[f"class_{cls}_recall"]    = float(recall_per_class[i])
        results[f"class_{cls}_f1"]        = float(f1_per_class[i])
        results[f"class_{cls}_support"]   = int(class_support[i])

    # Log to logger (e.g., wandb)
    logger.log(results)

    # ----- Save overall metrics -----
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, f"seq_metrics_epoch_{epoch}.csv")
    with open(metrics_path, mode='w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["Epoch", epoch])
        w.writerow(["Loss", avg_loss])
        w.writerow(["Accuracy", accuracy])
        w.writerow(["Balanced Accuracy", balanced_acc])
        w.writerow(["Cohen's Kappa", kappa])
        w.writerow(["Precision (Macro)", precision_macro])
        w.writerow(["Recall (Macro)", recall_macro])
        w.writerow(["F1 (Macro)", f1_macro])
        w.writerow(["Precision (Weighted)", precision_weight])
        w.writerow(["Recall (Weighted)", recall_weight])
        w.writerow(["F1 (Weighted)", f1_weight])
        w.writerow(["Total Samples", results["total_samples"]])

    # ----- Save per-class metrics -----
    if n_classes > 0:
        class_metrics_path = os.path.join(results_dir, f"seq_class_metrics_epoch_{epoch}.csv")
        with open(class_metrics_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Class_ID", "Class_Name", "Precision", "Recall", "F1", "Support"])
            for i, cls in enumerate(unique_classes):
                writer.writerow([
                    cls, class_names[i],
                    f"{precision_per_class[i]:.4f}",
                    f"{recall_per_class[i]:.4f}",
                    f"{f1_per_class[i]:.4f}",
                    int(class_support[i])
                ])

        # Save CM CSVs & plots
        cm_path = os.path.join(results_dir, f"seq_confusion_matrix_epoch_{epoch}.csv")
        np.savetxt(cm_path, cm, delimiter=',', fmt='%d')

        cm_labeled_path = os.path.join(results_dir, f"seq_confusion_matrix_labeled_epoch_{epoch}.csv")
        with open(cm_labeled_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['True\\Predicted'] + class_names)
            for i, row in enumerate(cm):
                writer.writerow([class_names[i]] + row.tolist())

        if save_figs:
            _save_confusion_and_plots(cm, class_names, results_dir, epoch, title_prefix="Seq")

        # Classification report
        report_path = os.path.join(results_dir, f"seq_classification_report_epoch_{epoch}.json")
        with open(report_path, 'w') as f:
            json.dump(classification_report(gt_all, pred_all, target_names=class_names,
                                            labels=unique_classes, zero_division=0, output_dict=True), f, indent=4)

        report_txt = os.path.join(results_dir, f"seq_classification_report_epoch_{epoch}.txt")
        with open(report_txt, 'w') as f:
            f.write(classification_report(gt_all, pred_all, target_names=class_names,
                                          labels=unique_classes, zero_division=0))

    # ----- Save per-sample summaries -----
    samples_path = os.path.join(results_dir, f"seq_samples_epoch_{epoch}.csv")
    with open(samples_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["trial_id","subject_id","gesture_code","T","seq_pred","seq_gt"])
        writer.writeheader()
        for rec in per_sample_records:
            writer.writerow(rec)

    detailed_path = os.path.join(results_dir, f"seq_samples_detailed_epoch_{epoch}.json")
    with open(detailed_path, 'w') as f:
        json.dump(per_sample_detailed, f, indent=2)

    # ----- Summary text -----
    summary_path = os.path.join(results_dir, f"seq_summary_epoch_{epoch}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"MS-TCN++ Sequence Classification Summary - Epoch {epoch}\n")
        f.write("="*70 + "\n\n")
        f.write(f"Loss: {avg_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"\nMacro:\n  P={precision_macro:.4f}  R={recall_macro:.4f}  F1={f1_macro:.4f}\n")
        f.write(f"Weighted:\n  P={precision_weight:.4f}  R={recall_weight:.4f}  F1={f1_weight:.4f}\n")
        if n_classes > 0:
            f.write("\nPer-Class:\n")
            for i, cls in enumerate(unique_classes):
                f.write(f"  {class_names[i]} (ID {cls}): "
                        f"P={precision_per_class[i]:.4f}, R={recall_per_class[i]:.4f}, "
                        f"F1={f1_per_class[i]:.4f}, Support={int(class_support[i])}\n")
        f.write("\n" + "="*70 + "\n")
        f.write(f"Total Samples: {results['total_samples']}\n")
        f.write(f"Number of Classes: {n_classes}\n")

    print(f"\n{'='*70}")
    print(f"[TEST-SEQ] Complete - Epoch {epoch}")
    print(f"Loss: {avg_loss:.4f} | Acc: {accuracy:.4f} | F1(macro): {f1_macro:.4f}")
    print(f"Balanced Acc: {balanced_acc:.4f} | Kappa: {kappa:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

    return results


def train_mstcn_one_epoch(model, train_loader, loss_fns, optimizer, device, logger, args):
    """
    Expects each batch dict to have:
      - 'label' : (B,) or (B,T) long
      - optionally 'mask': (B,1,T) float {0,1}; if missing, we create all-ones
    And your mmt_preprocess returns: (B, T, F_total). We convert to (B, F_total, T).
    """
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches = 0

    for i, batch in enumerate(train_loader):
        try:
            # ----- Inputs -----
            x_BTF = mmt_preprocess(batch, args, None, device)  # (B, T, F_total)
            if x_BTF.dim() != 3:
                raise ValueError(f"mmt_preprocess must return (B,T,F), got shape {tuple(x_BTF.shape)}")
            x_BCT = x_BTF.permute(0, 2, 1).contiguous()        # (B, C=F_total, T)

            # ----- Targets -----
            labels = batch['label'].to(device)                 # (B,) or (B,T)
            B, C, T = x_BCT.shape
            
            # skip batches with time length < 30
            # if T < 30:
            #     continue

            if labels.dim() == 1:
                # Expand single label per sequence across time to match MS-TCN++ interface
                labels = labels.unsqueeze(1).expand(-1, T).contiguous()  # (B,T)
            elif labels.dim() == 2:
                if labels.size(1) != T:
                    raise ValueError(f"Label time length {labels.size(1)} != input T {T}")
            else:
                raise ValueError(f"'label' must be (B,) or (B,T); got {tuple(labels.shape)}")

            # ----- Mask (optional) -----
            if 'mask' in batch:
                mask = batch['mask'].to(device)                # (B,1,T) preferred
                # be lenient: accept (B,T) and upgrade
                if mask.dim() == 2:
                    mask = mask.unsqueeze(1)
                if mask.size(-1) != T:
                    raise ValueError(f"Mask time length {mask.size(-1)} != T {T}")
            else:
                mask = torch.ones((B, 1, T), device=device, dtype=torch.float32)
            
            # print(f"Batch {i}: x_BCT shape={x_BCT.shape}, labels shape={labels.shape}, mask shape={mask.shape}")
            # print(f"Labels: {labels}")
            # ----- Forward / Backward -----
            optimizer.zero_grad()
            preds = model(x_BCT)                                # list[(B,C,T)] or (S,B,C,T)
            loss = mstcn_compute_loss(preds, labels, mask, loss_fns)
            loss.backward()
            optimizer.step()

            # ----- Metrics from final stage -----
            stages = _as_stage_list(preds)
            final_logits = stages[-1]                           # (B,C,T)
            acc = _masked_accuracy_from_logits(final_logits, labels, mask)

            running_loss += float(loss.item())
            running_acc  += float(acc)
            n_batches += 1

            if i % 10 == 0:
                print(f"Batch {i}: loss={loss.item():.4f}, acc={acc:.4f}")

        except Exception as e:
            print(f"Error in batch {i}: {e}")
            import traceback; traceback.print_exc()
            continue

    if n_batches == 0:
        return 0.0

    epoch_loss = running_loss / n_batches
    epoch_acc  = running_acc  / n_batches
    if logger is not None:
        try:
            logger.info(f"Train epoch: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
        except Exception:
            pass
    return epoch_loss


def validate_mstcn(model, val_loader, loss_fns, device, logger, args):
    model.eval()
    running_loss = 0.0
    running_acc  = 0.0
    n_batches    = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            try:
                # ----- Inputs -----
                x_BTF = mmt_preprocess(batch, args, None, device)  # (B, T, F_total)
                if x_BTF.dim() != 3:
                    raise ValueError(f"mmt_preprocess must return (B,T,F), got shape {tuple(x_BTF.shape)}")
                x_BCT = x_BTF.permute(0, 2, 1).contiguous()        # (B, C=F_total, T)

                # ----- Targets -----
                labels = batch['label'].to(device)                 # (B,) or (B,T)
                B, C, T = x_BCT.shape

                # skip batches with time length < 30
                # if T < 30:
                #     continue

                if labels.dim() == 1:
                    labels = labels.unsqueeze(1).expand(-1, T).contiguous()  # (B,T)
                elif labels.dim() == 2:
                    if labels.size(1) != T:
                        raise ValueError(f"Label time length {labels.size(1)} != input T {T}")
                else:
                    raise ValueError(f"'label' must be (B,) or (B,T); got {tuple(labels.shape)}")

                # ----- Mask (optional) -----
                if 'mask' in batch:
                    mask = batch['mask'].to(device)                # (B,1,T) preferred
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1)
                    if mask.size(-1) != T:
                        raise ValueError(f"Mask time length {mask.size(-1)} != T {T}")
                else:
                    mask = torch.ones((B, 1, T), device=device, dtype=torch.float32)

                # ----- Forward & loss -----
                preds = model(x_BCT)                                # list[(B,C,T)] or (S,B,C,T)
                loss  = mstcn_compute_loss(preds, labels, mask, loss_fns)

                # ----- Metrics from final stage -----
                stages = _as_stage_list(preds)
                final_logits = stages[-1]                           # (B,C,T)
                acc = _masked_accuracy_from_logits(final_logits, labels, mask)

                running_loss += float(loss.item())
                running_acc  += float(acc)
                n_batches    += 1

                if i % 10 == 0:
                    print(f"[VAL] Batch {i}: loss={loss.item():.4f}, acc={acc:.4f}")

            except Exception as e:
                print(f"[VAL] Error in batch {i}: {e}")
                import traceback; traceback.print_exc()
                continue

    if n_batches == 0:
        if logger is not None:
            try:
                logger.warning("Validation had zero successful batches.")
            except Exception:
                pass
        return {"loss": 0.0, "acc": 0.0}

    val_loss = running_loss / n_batches
    val_acc  = running_acc  / n_batches

    if logger is not None:
        try:
            logger.info(f"Validation: loss={val_loss:.4f}, acc={val_acc:.4f}")
        except Exception:
            pass

    return {"loss": val_loss, "acc": val_acc}


import torch
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    balanced_accuracy_score, cohen_kappa_score,
    jaccard_score
)
import json
import os
from pathlib import Path
import traceback
from collections import defaultdict
def test_MSTCN_model(model, test_loader, loss_fns, device, logger, epoch, results_dir, args, class_names=None):
    """
    Enhanced frame-wise test loop for MS-TCN++ with comprehensive metrics for ML papers.
    """
    os.makedirs(results_dir, exist_ok=True)

    model.eval()
    running_loss = 0.0
    n_batches = 0

    # For global frame-level metrics
    all_gt_frames = []
    all_pred_frames = []
    
    # Per-sample temporal metrics
    per_sample_records = []
    per_sample_detailed = []
    
    # Per-class frame counts
    class_frame_counts = defaultdict(int)
    class_correct_counts = defaultdict(int)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            try:
                # ----- Inputs -----
                x_BTF = mmt_preprocess(batch, args, None, device)  # (B,T,F)
                if x_BTF.dim() != 3:
                    raise ValueError(f"mmt_preprocess must return (B,T,F), got shape {tuple(x_BTF.shape)}")
                x_BCT = x_BTF.permute(0, 2, 1).contiguous()        # (B,C=F,T)
                B, C, T = x_BCT.shape

                # ----- Targets -----
                labels = batch['label'].to(device)                 # (B,) or (B,T)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1).expand(-1, T).contiguous()  # (B,T)
                elif labels.dim() == 2 and labels.size(1) != T:
                    raise ValueError(f"Label time length {labels.size(1)} != input T {T}")
                elif labels.dim() != 2:
                    raise ValueError(f"'label' must be (B,) or (B,T); got {tuple(labels.shape)}")

                # ----- Mask (optional) -----
                if 'mask' in batch:
                    mask = batch['mask'].to(device)                # (B,1,T) preferred
                    if mask.dim() == 2:
                        mask = mask.unsqueeze(1)
                    if mask.size(-1) != T:
                        raise ValueError(f"Mask time length {mask.size(-1)} != T {T}")
                else:
                    mask = torch.ones((B, 1, T), device=device, dtype=torch.float32)

                # ----- Forward & Loss -----
                preds = model(x_BCT)                                # list[(B,C,T)] or (S,B,C,T)
                loss = mstcn_compute_loss(preds, labels, mask, loss_fns)
                running_loss += float(loss.item())
                n_batches += 1

                # ----- Final-stage logits → predictions -----
                stages = _as_stage_list(preds)
                final_logits = stages[-1]                           # (B,C,T)
                pred_BT = final_logits.argmax(dim=1)               # (B,T)

                # ----- Collect masked frames for global metrics -----
                valid = (mask.squeeze(1) > 0.5) & (labels != IGNORE_INDEX)
                gt_flat = labels[valid].detach().cpu().numpy()
                pred_flat = pred_BT[valid].detach().cpu().numpy()
                
                if gt_flat.size > 0:
                    all_gt_frames.append(gt_flat)
                    all_pred_frames.append(pred_flat)
                    
                    # Count per-class frames and correct predictions
                    for gt_class, pred_class in zip(gt_flat, pred_flat):
                        class_frame_counts[int(gt_class)] += 1
                        if gt_class == pred_class:
                            class_correct_counts[int(gt_class)] += 1

                # ----- Per-sample metrics and saving -----
                probs_BCT = torch.softmax(final_logits, dim=1).detach().cpu().numpy()  # (B,C,T)
                trial_ids = batch.get("trial_id", [None]*B)
                subject_ids = batch.get("subject_id", [None]*B)
                gesture_codes = batch.get("gesture_code", [None]*B)

                # Make iterables uniform
                if torch.is_tensor(trial_ids):
                    trial_ids = trial_ids.cpu().tolist()
                if torch.is_tensor(subject_ids):
                    subject_ids = subject_ids.cpu().tolist()
                if torch.is_tensor(gesture_codes):
                    gesture_codes = gesture_codes.cpu().tolist()

                for b in range(B):
                    # Get valid frames for this sample
                    valid_b = valid[b].cpu().numpy()
                    gt_b = labels[b].detach().cpu().numpy()
                    pred_b = pred_BT[b].detach().cpu().numpy()
                    
                    # Calculate per-sample metrics
                    valid_frames = int(valid_b.sum())
                    if valid_frames > 0:
                        gt_valid = gt_b[valid_b]
                        pred_valid = pred_b[valid_b]
                        sample_accuracy = float((gt_valid == pred_valid).mean())
                        
                        # Temporal metrics
                        edit_dist = compute_edit_distance(gt_valid, pred_valid)
                        f1_scores = compute_f1_at_k(gt_valid, pred_valid, k_list=[10, 25, 50])
                    else:
                        sample_accuracy = 0.0
                        edit_dist = 0
                        f1_scores = {10: 0.0, 25: 0.0, 50: 0.0}
                    
                    # Derive filename stem
                    stem = str(trial_ids[b]) if trial_ids[b] is not None else f"sample_{i}_{b}"
                    
                    # Save predictions and (optionally) probabilities
                    pred_save_path = os.path.join(results_dir, "predictions")
                    os.makedirs(pred_save_path, exist_ok=True)
                    
                    np.save(os.path.join(pred_save_path, f"{stem}_pred.npy"), pred_b)
                    np.save(os.path.join(pred_save_path, f"{stem}_gt.npy"), gt_b)
                    np.save(os.path.join(pred_save_path, f"{stem}_mask.npy"), valid_b)
                    
                    if getattr(args, "save_probs", False):
                        np.save(os.path.join(pred_save_path, f"{stem}_probs.npy"), probs_BCT[b])

                    # Compact summary for CSV
                    per_sample_records.append({
                        "trial_id": trial_ids[b],
                        "subject_id": subject_ids[b],
                        "gesture_code": gesture_codes[b],
                        "T": int(T),
                        "valid_frames": valid_frames,
                        "accuracy": sample_accuracy,
                        "edit_distance": edit_dist,
                        "f1@10": f1_scores[10],
                        "f1@25": f1_scores[25],
                        "f1@50": f1_scores[50],
                    })
                    
                    # Detailed record with predictions
                    per_sample_detailed.append({
                        "trial_id": trial_ids[b],
                        "subject_id": subject_ids[b],
                        "gesture_code": gesture_codes[b],
                        "T": int(T),
                        "valid_frames": valid_frames,
                        "accuracy": sample_accuracy,
                        "predictions": pred_b.tolist(),
                        "ground_truth": gt_b.tolist(),
                        "valid_mask": valid_b.tolist(),
                    })

                if i % 10 == 0:
                    logger.log({"test_batch_loss": loss.item(), "batch": i})

            except Exception as e:
                print(f"[TEST] Error in batch {i}: {e}")
                traceback.print_exc()
                continue

    # ----- Aggregate metrics -----
    if n_batches == 0:
        print("Warning: No valid batches processed")
        results = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "epoch": epoch}
        logger.log(results)
        return results

    avg_loss = running_loss / n_batches

    if len(all_gt_frames) == 0:
        print("Warning: No valid frames collected")
        accuracy = precision_macro = recall_macro = f1_macro = 0.0
        precision_weighted = recall_weighted = f1_weighted = 0.0
        balanced_acc = kappa = 0.0
        precision_per_class = recall_per_class = f1_per_class = np.array([])
        unique_classes = []
        cm = np.array([[]])
        class_accuracy = np.array([])
        jaccard_macro = jaccard_weighted = 0.0
        gt_all = pred_all = np.array([])
    else:
        gt_all = np.concatenate(all_gt_frames, axis=0)
        pred_all = np.concatenate(all_pred_frames, axis=0)

        # ----- Use only classes present in y_true (LOUO-safe) -----
        unique_classes = sorted(list(set(gt_all.tolist())))

        # Overall metrics
        accuracy = float((gt_all == pred_all).mean())
        
        # Macro and weighted metrics
        precision_macro = float(precision_score(gt_all, pred_all, average='macro', zero_division=0))
        recall_macro = float(recall_score(gt_all, pred_all, average='macro', zero_division=0))
        f1_macro = float(f1_score(gt_all, pred_all, average='macro', zero_division=0))
        
        precision_weighted = float(precision_score(gt_all, pred_all, average='weighted', zero_division=0))
        recall_weighted = float(recall_score(gt_all, pred_all, average='weighted', zero_division=0))
        f1_weighted = float(f1_score(gt_all, pred_all, average='weighted', zero_division=0))
        
        # Per-class metrics (restricted to unique_classes)
        precision_per_class = precision_score(gt_all, pred_all, average=None, zero_division=0, labels=unique_classes)
        recall_per_class = recall_score(gt_all, pred_all, average=None, zero_division=0, labels=unique_classes)
        f1_per_class = f1_score(gt_all, pred_all, average=None, zero_division=0, labels=unique_classes)
        
        # Additional metrics
        balanced_acc = float(balanced_accuracy_score(gt_all, pred_all))
        kappa = float(cohen_kappa_score(gt_all, pred_all))
        
        # Jaccard score (IoU)
        jaccard_macro = float(jaccard_score(gt_all, pred_all, average='macro', zero_division=0))
        jaccard_weighted = float(jaccard_score(gt_all, pred_all, average='weighted', zero_division=0))
        
        # Confusion matrix on unique_classes
        cm = confusion_matrix(gt_all, pred_all, labels=unique_classes)
        
        # Per-class accuracy from confusion matrix (safe divide)
        row_sums = cm.sum(axis=1, keepdims=True)
        class_accuracy = (cm.diagonal().astype(np.float64) / (row_sums.squeeze(1) + 1e-10))

    # ----- Class-name handling aligned with unique_classes -----
    n_classes = len(unique_classes)
    if class_names is None:
        class_names_list = [f"Class {c}" for c in unique_classes]
    elif isinstance(class_names, dict):
        class_names_list = [class_names.get(c, f"Class {c}") for c in unique_classes]
    else:
        # list/tuple: index by contiguous id
        class_names_list = [
            class_names[c] if (isinstance(c, int) and c < len(class_names)) else f"Class {c}"
            for c in unique_classes
        ]

    # Average temporal metrics
    avg_edit_distance = np.mean([rec["edit_distance"] for rec in per_sample_records]) if per_sample_records else 0.0
    avg_f1_10 = np.mean([rec["f1@10"] for rec in per_sample_records]) if per_sample_records else 0.0
    avg_f1_25 = np.mean([rec["f1@25"] for rec in per_sample_records]) if per_sample_records else 0.0
    avg_f1_50 = np.mean([rec["f1@50"] for rec in per_sample_records]) if per_sample_records else 0.0

    # Prepare results dictionary
    results = {
        "epoch": epoch,
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": kappa,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "jaccard_macro": jaccard_macro,
        "jaccard_weighted": jaccard_weighted,
        "avg_edit_distance": avg_edit_distance,
        "avg_f1@10": avg_f1_10,
        "avg_f1@25": avg_f1_25,
        "avg_f1@50": avg_f1_50,
        "total_frames": len(gt_all) if len(all_gt_frames) > 0 else 0,
        "total_samples": len(per_sample_records),
    }

    # Add per-class metrics to results
    for i, class_idx in enumerate(unique_classes):
        results[f"class_{class_idx}_accuracy"] = float(class_accuracy[i])
        results[f"class_{class_idx}_precision"] = float(precision_per_class[i])
        results[f"class_{class_idx}_recall"] = float(recall_per_class[i])
        results[f"class_{class_idx}_f1"] = float(f1_per_class[i])
        results[f"class_{class_idx}_frames"] = int(class_frame_counts[class_idx])

    # Log metrics to wandb
    logger.log(results)

    # ----- Save overall metrics to CSV -----
    metrics_path = os.path.join(results_dir, f"metrics_epoch_{epoch}.csv")
    with open(metrics_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Epoch", epoch])
        writer.writerow(["Loss", avg_loss])
        writer.writerow(["Accuracy", accuracy])
        writer.writerow(["Balanced Accuracy", balanced_acc])
        writer.writerow(["Cohen's Kappa", kappa])
        writer.writerow(["Precision (Macro)", precision_macro])
        writer.writerow(["Recall (Macro)", recall_macro])
        writer.writerow(["F1 Score (Macro)", f1_macro])
        writer.writerow(["Precision (Weighted)", precision_weighted])
        writer.writerow(["Recall (Weighted)", recall_weighted])
        writer.writerow(["F1 Score (Weighted)", f1_weighted])
        writer.writerow(["Jaccard (Macro)", jaccard_macro])
        writer.writerow(["Jaccard (Weighted)", jaccard_weighted])
        writer.writerow(["Avg Edit Distance", avg_edit_distance])
        writer.writerow(["Avg F1@10", avg_f1_10])
        writer.writerow(["Avg F1@25", avg_f1_25])
        writer.writerow(["Avg F1@50", avg_f1_50])
        writer.writerow(["Total Frames", results["total_frames"]])
        writer.writerow(["Total Samples", results["total_samples"]])

    # ----- Save per-class metrics to CSV -----
    if n_classes > 0:
        class_metrics_path = os.path.join(results_dir, f"class_metrics_epoch_{epoch}.csv")
        with open(class_metrics_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Class_ID", "Class_Name", "Accuracy", "Precision", "Recall", "F1-Score", "Frame_Count"])
            for i, class_idx in enumerate(unique_classes):
                writer.writerow([
                    class_idx,
                    class_names_list[i],
                    f"{class_accuracy[i]:.4f}",
                    f"{precision_per_class[i]:.4f}",
                    f"{recall_per_class[i]:.4f}",
                    f"{f1_per_class[i]:.4f}",
                    class_frame_counts[class_idx]
                ])

        # ----- Save confusion matrix -----
        cm_path = os.path.join(results_dir, f"confusion_matrix_epoch_{epoch}.csv")
        np.savetxt(cm_path, cm, delimiter=',', fmt='%d')

        # Save confusion matrix with labels
        cm_labeled_path = os.path.join(results_dir, f"confusion_matrix_labeled_epoch_{epoch}.csv")
        with open(cm_labeled_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['True\\Predicted'] + class_names_list)
            for i, row in enumerate(cm):
                writer.writerow([class_names_list[i]] + row.tolist())

        # ----- Visualize confusion matrix -----
        fig_size = max(10, n_classes * 0.8)
        plt.figure(figsize=(fig_size, fig_size * 0.9))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names_list, yticklabels=class_names_list,
                    cbar_kws={'label': 'Frame Count'})
        plt.title(f'Confusion Matrix (Frame-level) - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_matrix_epoch_{epoch}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # Normalized confusion matrix (safe row-wise division)
        row_sums = cm.sum(axis=1, keepdims=True) + 1e-10
        cm_normalized = cm.astype('float') / row_sums
        plt.figure(figsize=(fig_size, fig_size * 0.9))
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=class_names_list, yticklabels=class_names_list,
                    cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        plt.title(f'Normalized Confusion Matrix (Recall) - Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"confusion_matrix_normalized_epoch_{epoch}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ----- Visualize per-class metrics -----
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        metrics_to_plot = [
            (class_accuracy, 'Accuracy', axes[0, 0]),
            (precision_per_class, 'Precision', axes[0, 1]),
            (recall_per_class, 'Recall', axes[1, 0]),
            (f1_per_class, 'F1-Score', axes[1, 1])
        ]

        for metric_values, metric_name, ax in metrics_to_plot:
            bars = ax.bar(range(len(class_names_list)), metric_values, color='skyblue', edgecolor='navy', alpha=0.7)
            ax.set_xlabel('Class', fontsize=11)
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'Per-Class {metric_name} (Frame-level)', fontsize=12, fontweight='bold')
            ax.set_xticks(range(len(class_names_list)))
            ax.set_xticklabels(class_names_list, rotation=45, ha='right')
            ax.set_ylim([0, 1.1])
            ax.grid(axis='y', alpha=0.3)

            for bar, val in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"per_class_metrics_epoch_{epoch}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ----- Visualize frame distribution per class -----
        plt.figure(figsize=(12, 6))
        frame_counts = [class_frame_counts[c] for c in unique_classes]
        bars = plt.bar(range(len(class_names_list)), frame_counts, color='coral', edgecolor='darkred', alpha=0.7)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Frame Count', fontsize=12)
        plt.title('Frame Distribution by Class', fontsize=14, fontweight='bold')
        plt.xticks(range(len(class_names_list)), class_names_list, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, frame_counts):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(frame_counts + [1]) * 0.01,
                    f'{count}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"frame_distribution_epoch_{epoch}.png"), dpi=300, bbox_inches='tight')
        plt.close()

        # ----- Classification report -----
        if len(gt_all) > 0:
            class_report = classification_report(
                gt_all, pred_all,
                target_names=class_names_list,
                labels=unique_classes,
                zero_division=0,
                output_dict=True
            )
            
            report_path = os.path.join(results_dir, f"classification_report_epoch_{epoch}.json")
            with open(report_path, 'w') as f:
                json.dump(class_report, f, indent=4)

            report_text_path = os.path.join(results_dir, f"classification_report_epoch_{epoch}.txt")
            with open(report_text_path, 'w') as f:
                f.write(classification_report(
                    gt_all, pred_all,
                    target_names=class_names_list,
                    labels=unique_classes,
                    zero_division=0
                ))

    # ----- Save per-sample summary CSV -----
    samples_path = os.path.join(results_dir, f"samples_epoch_{epoch}.csv")
    with open(samples_path, mode='w', newline='') as f:
        fieldnames = ["trial_id", "subject_id", "gesture_code", "T", "valid_frames",
                     "accuracy", "edit_distance", "f1@10", "f1@25", "f1@50"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in per_sample_records:
            writer.writerow(rec)

    # ----- Save detailed per-sample predictions -----
    detailed_path = os.path.join(results_dir, f"samples_detailed_epoch_{epoch}.json")
    with open(detailed_path, 'w') as f:
        json.dump(per_sample_detailed, f, indent=2)

    # ----- Summary -----
    summary_path = os.path.join(results_dir, f"summary_epoch_{epoch}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"MS-TCN++ Model Evaluation Summary - Epoch {epoch}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"  Loss: {avg_loss:.4f}\n")
        f.write(f"  Frame Accuracy: {accuracy:.4f}\n")
        f.write(f"  Balanced Accuracy: {balanced_acc:.4f}\n")
        f.write(f"  Cohen's Kappa: {kappa:.4f}\n")
        f.write(f"\nMacro-averaged Metrics:\n")
        f.write(f"  Precision: {precision_macro:.4f}\n")
        f.write(f"  Recall: {recall_macro:.4f}\n")
        f.write(f"  F1-Score: {f1_macro:.4f}\n")
        f.write(f"  Jaccard (IoU): {jaccard_macro:.4f}\n")
        f.write(f"\nWeighted-averaged Metrics:\n")
        f.write(f"  Precision: {precision_weighted:.4f}\n")
        f.write(f"  Recall: {recall_weighted:.4f}\n")
        f.write(f"  F1-Score: {f1_weighted:.4f}\n")
        f.write(f"  Jaccard (IoU): {jaccard_weighted:.4f}\n")
        f.write(f"\nTemporal Metrics:\n")
        f.write(f"  Average Edit Distance: {avg_edit_distance:.2f}\n")
        f.write(f"  Average F1@10: {avg_f1_10:.4f}\n")
        f.write(f"  Average F1@25: {avg_f1_25:.4f}\n")
        f.write(f"  Average F1@50: {avg_f1_50:.4f}\n")
        if n_classes > 0:
            f.write(f"\nPer-Class Metrics:\n")
            for i, class_idx in enumerate(unique_classes):
                f.write(f"\n  {class_names_list[i]} (Class {class_idx}):\n")
                f.write(f"    Accuracy: {class_accuracy[i]:.4f}\n")
                f.write(f"    Precision: {precision_per_class[i]:.4f}\n")
                f.write(f"    Recall: {recall_per_class[i]:.4f}\n")
                f.write(f"    F1-Score: {f1_per_class[i]:.4f}\n")
                f.write(f"    Frame Count: {class_frame_counts[class_idx]}\n")
        f.write(f"\n" + "=" * 70 + "\n")
        f.write(f"Total Frames: {results['total_frames']}\n")
        f.write(f"Total Samples: {results['total_samples']}\n")
        f.write(f"Number of Classes: {n_classes}\n")

    print(f"\n{'='*70}")
    print(f"MS-TCN++ Evaluation Complete - Epoch {epoch}")
    print(f"Loss: {avg_loss:.4f} | Frame Accuracy: {accuracy:.4f} | F1 (Macro): {f1_macro:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f} | Kappa: {kappa:.4f}")
    print(f"Edit Distance: {avg_edit_distance:.2f} | F1@25: {avg_f1_25:.4f}")
    print(f"Results saved to: {results_dir}")
    print(f"{'='*70}\n")

    return results


def compute_edit_distance(seq1, seq2):
    """
    Compute Levenshtein edit distance between two sequences.
    Normalized by the length of the longer sequence.
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[n][m]


def compute_f1_at_k(gt_seq, pred_seq, k_list=[10, 25, 50]):
    """
    Compute F1 score at different overlap thresholds (F1@k).
    This measures segmental overlap - whether predicted segments overlap
    with ground truth segments by at least k% of frames.
    
    Args:
        gt_seq: Ground truth sequence (1D array)
        pred_seq: Predicted sequence (1D array)
        k_list: List of overlap thresholds (percentages)
    
    Returns:
        Dictionary of F1 scores at each threshold
    """
    def get_segments(seq):
        """Convert frame sequence to list of segments (label, start, end)"""
        if len(seq) == 0:
            return []
        
        segments = []
        current_label = seq[0]
        start = 0
        
        for i in range(1, len(seq)):
            if seq[i] != current_label:
                segments.append((current_label, start, i-1))
                current_label = seq[i]
                start = i
        segments.append((current_label, start, len(seq)-1))
        
        return segments
    
    def compute_overlap(gt_seg, pred_seg):
        """Compute overlap ratio between two segments"""
        gt_label, gt_start, gt_end = gt_seg
        pred_label, pred_start, pred_end = pred_seg
        
        # Only consider segments with same label
        if gt_label != pred_label:
            return 0.0
        
        overlap_start = max(gt_start, pred_start)
        overlap_end = min(gt_end, pred_end)
        
        if overlap_start > overlap_end:
            return 0.0
        
        overlap_frames = overlap_end - overlap_start + 1
        gt_frames = gt_end - gt_start + 1
        
        return overlap_frames / gt_frames
    
    gt_segments = get_segments(gt_seq)
    pred_segments = get_segments(pred_seq)
    
    f1_scores = {}
    
    for k in k_list:
        threshold = k / 100.0
        
        # Count true positives
        tp = 0
        for gt_seg in gt_segments:
            for pred_seg in pred_segments:
                if compute_overlap(gt_seg, pred_seg) >= threshold:
                    tp += 1
                    break
        
        # False positives and false negatives
        fp = len(pred_segments) - tp
        fn = len(gt_segments) - tp
        
        # Calculate F1
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)
        
        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        f1_scores[k] = f1
    
    return f1_scores


def _as_stage_list(preds):
    """
    Convert MS-TCN output to list of stage predictions.
    Handles both list and tensor formats.
    """
    if isinstance(preds, list):
        return preds
    elif isinstance(preds, torch.Tensor):
        if preds.dim() == 4:  # (S, B, C, T)
            return [preds[s] for s in range(preds.size(0))]
        else:
            return [preds]
    else:
        raise ValueError(f"Unexpected preds type: {type(preds)}")

import os
import csv
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# def test_MSTCN_model(model, test_loader, loss_fns, device, logger, epoch, results_dir, args):
#     """
#     Frame-wise test loop for MS-TCN++.
#     - Computes average loss over batches using mstcn_compute_loss
#     - Aggregates frame-level predictions vs. labels (mask + IGNORE_INDEX respected)
#     - Reports macro precision/recall/F1 on valid frames
#     - Saves metrics.csv (appends) and per-sample predictions as npy files

#     Expects each batch to contain:
#       - 'label': (B,) or (B,T) long (if (B,), we expand to (B,T))
#       - optional 'mask': (B,1,T) float in {0,1}; if absent, uses all-ones
#       - optional metadata: 'trial_id', 'subject_id', 'gesture_code'
#     """
#     os.makedirs(results_dir, exist_ok=True)

#     model.eval()
#     running_loss = 0.0
#     n_batches    = 0

#     # For global metrics (flatten after masking)
#     all_gt_frames   = []
#     all_pred_frames = []

#     # Optional: keep a compact per-sample artifact (npy) rather than huge CSV rows
#     per_sample_records = []  # a small summary per sample

#     with torch.no_grad():
#         for i, batch in enumerate(test_loader):
#             try:
#                 # ----- Inputs -----
#                 x_BTF = mmt_preprocess(batch, args, None, device)  # (B,T,F)
#                 if x_BTF.dim() != 3:
#                     raise ValueError(f"mmt_preprocess must return (B,T,F), got shape {tuple(x_BTF.shape)}")
#                 x_BCT = x_BTF.permute(0, 2, 1).contiguous()        # (B,C=F,T)
#                 B, C, T = x_BCT.shape

#                 # skip batches with time length < 30
#                 if T < 30:
#                     continue

#                 # ----- Targets -----
#                 labels = batch['label'].to(device)                 # (B,) or (B,T)
#                 if labels.dim() == 1:
#                     labels = labels.unsqueeze(1).expand(-1, T).contiguous()  # (B,T)
#                 elif labels.dim() == 2 and labels.size(1) != T:
#                     raise ValueError(f"Label time length {labels.size(1)} != input T {T}")
#                 elif labels.dim() != 2:
#                     raise ValueError(f"'label' must be (B,) or (B,T); got {tuple(labels.shape)}")

#                 # ----- Mask (optional) -----
#                 if 'mask' in batch:
#                     mask = batch['mask'].to(device)                # (B,1,T) preferred
#                     if mask.dim() == 2:
#                         mask = mask.unsqueeze(1)
#                     if mask.size(-1) != T:
#                         raise ValueError(f"Mask time length {mask.size(-1)} != T {T}")
#                 else:
#                     mask = torch.ones((B, 1, T), device=device, dtype=torch.float32)

#                 # ----- Forward & Loss -----
#                 preds = model(x_BCT)                                # list[(B,C,T)] or (S,B,C,T)
#                 loss  = mstcn_compute_loss(preds, labels, mask, loss_fns)
#                 running_loss += float(loss.item())
#                 n_batches    += 1

#                 # ----- Final-stage logits → predictions -----
#                 stages = _as_stage_list(preds)
#                 final_logits = stages[-1]                           # (B,C,T)
#                 pred_BT = final_logits.argmax(dim=1)               # (B,T)

#                 # ----- Collect masked frames for global metrics -----
#                 # valid = mask & label != IGNORE_INDEX
#                 valid = (mask.squeeze(1) > 0.5) & (labels != IGNORE_INDEX)
#                 gt_flat   = labels[valid].detach().cpu().numpy()
#                 pred_flat = pred_BT[valid].detach().cpu().numpy()
#                 if gt_flat.size > 0:
#                     all_gt_frames.append(gt_flat)
#                     all_pred_frames.append(pred_flat)

#                 # ----- Optional: save per-sample predictions/probs -----
#                 # (safer to store per-sample npy files than giant CSV rows)
#                 probs_BCT = torch.softmax(final_logits, dim=1).detach().cpu().numpy()  # (B,C,T)
#                 trial_ids     = batch.get("trial_id",    [None]*B)
#                 subject_ids   = batch.get("subject_id",  [None]*B)
#                 gesture_codes = batch.get("gesture_code",[None]*B)

#                 # Make iterables uniform
#                 if torch.is_tensor(trial_ids):     trial_ids     = trial_ids.cpu().tolist()
#                 if torch.is_tensor(subject_ids):   subject_ids   = subject_ids.cpu().tolist()
#                 if torch.is_tensor(gesture_codes): gesture_codes = gesture_codes.cpu().tolist()

#                 for b in range(B):
#                     # derive a filename stem
#                     stem = str(trial_ids[b]) if trial_ids[b] is not None else f"sample_{i}_{b}"
#                     # save predictions and (optionally) probabilities
#                     np.save(os.path.join(results_dir, f"{stem}_pred.npy"),
#                             pred_BT[b].detach().cpu().numpy())
#                     # probs can be large; save only if you want them
#                     if getattr(args, "save_probs", False):
#                         np.save(os.path.join(results_dir, f"{stem}_probs.npy"),
#                                 probs_BCT[b])

#                     # short summary for CSV
#                     per_sample_records.append({
#                         "trial_id":     trial_ids[b],
#                         "subject_id":   subject_ids[b],
#                         "gesture_code": gesture_codes[b],
#                         "T":            int(T),
#                         "valid_frames": int(valid[b].sum().item()),
#                     })

#                 if i % 10 == 0:
#                     logger.log({"test_batch_loss": loss.item()})

#             except Exception as e:
#                 print(f"[TEST] Error in batch {i}: {e}")
#                 import traceback; traceback.print_exc()
#                 continue

#     # ----- Aggregate metrics -----
#     if n_batches == 0:
#         results = {"loss": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "epoch": epoch}
#         logger.log(results)
#         return results

#     avg_loss = running_loss / n_batches

#     if len(all_gt_frames) == 0:
#         # No valid frames collected
#         accuracy = precision = recall = f1 = 0.0
#     else:
#         gt_all   = np.concatenate(all_gt_frames, axis=0)
#         pred_all = np.concatenate(all_pred_frames, axis=0)
#         accuracy  = float((gt_all == pred_all).mean())
#         precision = float(precision_score(gt_all, pred_all, average='macro', zero_division=0))
#         recall    = float(recall_score(gt_all, pred_all, average='macro', zero_division=0))
#         f1        = float(f1_score(gt_all, pred_all, average='macro', zero_division=0))

#     results = {
#         "epoch":     epoch,
#         "loss":      avg_loss,
#         "precision": precision,
#         "recall":    recall,
#         "f1":        f1,
#         "accuracy":  accuracy,
#     }
#     logger.log(results)

#     # ----- Save metrics to CSV (append) -----
#     metrics_path = os.path.join(results_dir, "metrics.csv")
#     fresh_file = not os.path.exists(metrics_path)
#     with open(metrics_path, mode='a', newline='') as f:
#         w = csv.writer(f)
#         if fresh_file:
#             w.writerow(["epoch", "loss", "precision", "recall", "f1", "accuracy"])
#         w.writerow([epoch, avg_loss, precision, recall, f1, accuracy])

#     # ----- Save a compact per-sample summary CSV -----
#     # (detailed framewise preds are saved as .npy per sample above)
#     samples_path = os.path.join(results_dir, "samples.csv")
#     fresh_file = not os.path.exists(samples_path)
#     with open(samples_path, mode='a', newline='') as f:
#         w = csv.DictWriter(f, fieldnames=["trial_id", "subject_id", "gesture_code", "T", "valid_frames"])
#         if fresh_file:
#             w.writeheader()
#         for rec in per_sample_records:
#             w.writerow(rec)

#     return results


def initialize_multimodal_model(args, device):
    print("Initializing Multi MTRSAP model...")
    print("MultiMTRSAP Config: ", args.multimodeltransformer_cfg)

    model = build_default_multitransformer(args.multimodeltransformer_cfg)
    model = model.to(device)
    print(model)

    # optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])

    # scheduler
    criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion


def initialize_simple_model(args, device):
    model = create_model(model_type="lstm",device=device,args=args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_params['lr'], weight_decay=args.learning_params['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion  
