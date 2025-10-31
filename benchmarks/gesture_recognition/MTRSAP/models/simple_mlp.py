import torch
import torch.nn as nn
import torch.nn.functional as F

import os, json
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

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
        print("Problematic batch:", batch)
        
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

@torch.no_grad()
def _pool_clip_features(preprocessed_inputs, mask=None, pool="mean"):
    """
    preprocessed_inputs: [B, T, F]
    mask: [B, T] boolean (True for valid), optional
    returns: [B, F]
    """
    x = preprocessed_inputs  # [B,T,F]
    if mask is not None:
        # avoid div-by-zero
        valid_counts = mask.sum(dim=1, keepdim=True).clamp_min(1)  # [B,1]
        x = (x * mask.unsqueeze(-1)).sum(dim=1) / valid_counts
    else:
        x = x.mean(dim=1)
    return x

import os, json
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, balanced_accuracy_score
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

@torch.no_grad()
def _pool_clip_features(preprocessed_inputs, mask=None, agg="meanstd", add_delta=False):
    """
    preprocessed_inputs: [B, T, F]
    mask: [B, T] bool
    agg:
      - 'mean'    -> [B, F]
      - 'meanstd' -> [B, 2F]  (recommended)
      - 'max'     -> [B, F]
    add_delta: if True, appends mean of first-difference over time (another [B, F])
    """
    x = preprocessed_inputs  # [B,T,F]
    if mask is not None:
        valid = mask.unsqueeze(-1)  # [B,T,1]
        count = valid.sum(dim=1).clamp_min(1)  # [B,1,1]
        mean = (x * valid).sum(dim=1) / count  # [B,F]
        if agg == "mean":
            pooled = mean
        elif agg == "max":
            x_masked = x.clone()
            x_masked[~mask] = -1e30
            pooled = x_masked.max(dim=1).values
        elif agg == "meanstd":
            var = ((x - mean.unsqueeze(1))**2 * valid).sum(dim=1) / count
            std = var.clamp_min(1e-12).sqrt()
            pooled = torch.cat([mean, std], dim=-1)
        else:
            raise ValueError("agg must be one of {'mean','max','meanstd'}")
    else:
        if agg == "mean":
            pooled = x.mean(dim=1)
        elif agg == "max":
            pooled = x.max(dim=1).values
        elif agg == "meanstd":
            mean = x.mean(dim=1)
            std = x.std(dim=1)
            pooled = torch.cat([mean, std], dim=-1)
        else:
            raise ValueError("agg must be one of {'mean','max','meanstd'}")

    if add_delta and x.shape[1] > 1:
        dx = x[:, 1:] - x[:, :-1]                  # [B, T-1, F]
        if mask is not None:
            m2 = mask[:, 1:] & mask[:, :-1]        # [B, T-1]
            cnt2 = m2.sum(dim=1, keepdim=True).clamp_min(1)
            dmean = (dx * m2.unsqueeze(-1)).sum(dim=1) / cnt2  # [B,F]
        else:
            dmean = dx.mean(dim=1)
        pooled = torch.cat([pooled, dmean], dim=-1)

    return pooled  # [B, D']

@torch.no_grad()
def _collect_loader_embeddings(loader, args, device, agg="meanstd", add_delta=False):
    X_list, y_list, meta = [], [], []
    for batch in loader:
        pre = mmt_preprocess(batch, args, None, device)  # [B,T,F]
        mask = batch.get("obs_mask", None)
        if isinstance(mask, torch.Tensor):
            mask = mask.to(device)
        pooled = _pool_clip_features(pre, mask, agg=agg, add_delta=add_delta)  # [B,D']
        X_list.append(pooled.cpu().numpy())
        y_list.append(batch["label"].cpu().numpy())

        B = pooled.shape[0]
        trial_ids = batch.get("trial_id", [None]*B)
        gesture_codes = batch.get("gesture_code", [None]*B)
        for i in range(B):
            meta.append({
                "trial_id": trial_ids[i] if isinstance(trial_ids, list) else trial_ids[i],
                "gesture_code": gesture_codes[i] if isinstance(gesture_codes, list) else gesture_codes[i],
            })
    X = np.concatenate(X_list, axis=0) if X_list else np.zeros((0, 0))
    y = np.concatenate(y_list, axis=0) if y_list else np.zeros((0,), dtype=np.int64)
    return X, y, meta

def _tiny_cv_grid_search(X, y, model_type="rbf", random_state=0):
    """
    Super small 3-fold CV over a tiny grid. Returns best estimator and params.
    model_type: 'rbf' for SVC(kernel='rbf'), 'linear' for LinearSVC
    """
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    if model_type == "rbf":
        C_grid = [0.5, 1.0, 5.0, 10.0]
        gamma_grid = ["scale", 1e-3, 1e-2]
        best = (None, -1, {})
        for C in C_grid:
            for gamma in gamma_grid:
                clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", probability=True, random_state=random_state)
                scores = []
                for tr, va in skf.split(X, y):
                    clf.fit(X[tr], y[tr])
                    scores.append(balanced_accuracy_score(y[va], clf.predict(X[va])))
                score = float(np.mean(scores))
                if score > best[1]:
                    best = (SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced", probability=True, random_state=random_state),
                            score, {"C": C, "gamma": gamma})
        return best[0], best[2]
    elif model_type == "linear":
        C_grid = [0.1, 0.5, 1.0, 5.0]
        best = (None, -1, {})
        for C in C_grid:
            clf = LinearSVC(C=C, class_weight="balanced", random_state=0)
            scores = []
            for tr, va in skf.split(X, y):
                clf.fit(X[tr], y[tr])
                scores.append(balanced_accuracy_score(y[va], clf.predict(X[va])))
            score = float(np.mean(scores))
            if score > best[1]:
                best = (LinearSVC(C=C, class_weight="balanced", random_state=0), score, {"C": C})
        return best[0], best[2]
    else:
        raise ValueError("model_type must be 'rbf' or 'linear'")

def svm_baseline_from_loaders(train_loader, val_loader, test_loader, args, device,
                              class_names=None, results_dir="./svm_baseline",
                              use_pca=True, pca_dim=128,
                              random_state=0, agg="meanstd", add_delta=True,
                              model_type="rbf"):
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # ---- collect ----
    Xtr, ytr, _ = _collect_loader_embeddings(train_loader, args, device, agg=agg, add_delta=add_delta)
    Xva, yva, _ = _collect_loader_embeddings(val_loader, args, device, agg=agg, add_delta=add_delta) if val_loader else (None, None, None)
    Xte, yte, meta_te = _collect_loader_embeddings(test_loader, args, device, agg=agg, add_delta=add_delta)

    if Xtr.size == 0 or Xte.size == 0:
        raise RuntimeError("Empty features from loaders. Check mmt_preprocess / inputs.")

    # ---- scale (+ PCA) ----
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva) if Xva is not None else None
    Xte_s = scaler.transform(Xte)

    if use_pca:
        pca = PCA(n_components=min(pca_dim, Xtr_s.shape[1]), random_state=random_state)
        Xtr_s = pca.fit_transform(Xtr_s)
        if Xva_s is not None: Xva_s = pca.transform(Xva_s)
        Xte_s = pca.transform(Xte_s)
    else:
        pca = None

    # ---- tiny CV to pick SVM ----
    clf, best_params = _tiny_cv_grid_search(Xtr_s, ytr, model_type=model_type, random_state=random_state)
    clf.fit(Xtr_s, ytr)

    def _evaluate(split_name, Xs, ys):
        yhat = clf.predict(Xs)
        acc = accuracy_score(ys, yhat)
        bacc = balanced_accuracy_score(ys, yhat)
        cm = confusion_matrix(ys, yhat)
        uniq = np.unique(np.concatenate([ys, yhat], axis=0))
        if class_names is None:
            names = [f"Class {i}" for i in uniq]
        elif isinstance(class_names, dict):
            names = [class_names.get(int(i), f"Class {int(i)}") for i in uniq]
        else:
            names = [class_names[int(i)] if int(i) < len(class_names) else f"Class {int(i)}" for i in uniq]

        rpt_txt = classification_report(ys, yhat, labels=uniq, target_names=names, zero_division=0)
        rpt = classification_report(ys, yhat, labels=uniq, target_names=names, zero_division=0, output_dict=True)
        with open(os.path.join(results_dir, f"{split_name}_report.txt"), "w") as f: f.write(rpt_txt)
        with open(os.path.join(results_dir, f"{split_name}_report.json"), "w") as f: json.dump(rpt, f, indent=2)

        plt.figure(figsize=(max(8, len(names)), max(6, len(names)*0.6)))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=names, yticklabels=names, cbar_kws={'label': 'Count'})
        plt.title(f"SVM Confusion Matrix - {split_name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, f"{split_name}_cm.png"), dpi=200); plt.close()

        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        plt.figure(figsize=(max(8, len(names)), max(6, len(names)*0.6)))
        sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                    xticklabels=names, yticklabels=names, vmin=0, vmax=1, cbar_kws={'label': 'Proportion'})
        plt.title(f"SVM Confusion Matrix (Recall) - {split_name}")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout(); plt.savefig(os.path.join(results_dir, f"{split_name}_cm_norm.png"), dpi=200); plt.close()

        return {"split": split_name, "accuracy": float(acc), "balanced_accuracy": float(bacc),
                "confusion_matrix": cm.tolist(), "labels": [int(i) for i in uniq], "class_names": names, "report": rpt}

    results = {"train": _evaluate("train", Xtr_s, ytr)}
    if Xva is not None:
        results["val"] = _evaluate("val", Xva_s, yva)
    results["test"] = _evaluate("test", Xte_s, yte)

    # save quick artifacts
    np.save(os.path.join(results_dir, "test_preds.npy"), clf.predict(Xte_s))
    if hasattr(clf, "predict_proba"):
        np.save(os.path.join(results_dir, "test_probs.npy"), clf.predict_proba(Xte_s))
    np.save(os.path.join(results_dir, "test_labels.npy"), yte)
    with open(os.path.join(results_dir, "svm_results_summary.json"), "w") as f:
        json.dump({"best_params": best_params, **results}, f, indent=2)

    print(f"[SVM/{model_type}] best_params={best_params} | "
          f"Train acc={results['train']['accuracy']:.3f} | "
          f"Test acc={results['test']['accuracy']:.3f} "
          f"(bAcc={results['test']['balanced_accuracy']:.3f})")

    return {"scaler": scaler, "pca": pca, "svm": clf, "results": results}


# Option 1: Simple MLP with Temporal Pooling (Best for small datasets)
class TemporalPoolingMLP(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, num_classes=7, dropout=0.5, use_batchnorm=False):
        super().__init__()
        self.feature_dim = feature_dim

        self.use_batchnorm = use_batchnorm
        # MLP layers
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        if use_batchnorm:
            self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        if use_batchnorm:
            self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        
        # Average pooling over temporal dimension
        x = x.mean(dim=1)  # (batch_size, feature_dim)
        
        # MLP
        x = self.fc1(x)
        if self.use_batchnorm:
            x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        if self.use_batchnorm:
            x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        return x


# Option 2: LSTM-based classifier (Better for temporal patterns)
class LSTMClassifier(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=256, num_layers=2, 
                 num_classes=7, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = h_n[-1]  # (batch_size, hidden_dim)
        
        # Classification
        out = self.fc(last_hidden)
        return out


# Option 3: Temporal Convolutional Network (Good balance)
class TemporalConvNet(nn.Module):
    def __init__(self, feature_dim=2048, hidden_dim=512, num_classes=7, dropout=0.5):
        super().__init__()
        
        # 1D convolutions over time
        self.conv1 = nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        
        # Transpose for Conv1d: (batch_size, feature_dim, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Global average pooling
        x = x.mean(dim=2)  # (batch_size, hidden_dim // 2)
        
        x = self.fc(x)
        return x


# Training example
def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_loss)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
        
        print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.2f}%')
    
    return best_val_acc

def create_model(model_type='mlp', device='cuda', args=None):
    kwargs = {
        'feature_dim': 2048,
        'hidden_dim': 512,
        'num_classes': args.num_classes}
    
    if model_type == 'mlp':
        model = TemporalPoolingMLP(**kwargs)
        model = model.to(device)
        return model
    elif model_type == 'lstm':
        model = LSTMClassifier(**kwargs)
        model = model.to(device)
        return model
    elif model_type == 'tcn':
        model = TemporalConvNet(**kwargs)
        model = model.to(device)
        return model
    else:
        raise ValueError(f"Unknown model type: {model_type}")



# Usage example
if __name__ == '__main__':
    # Initialize model (choose one)
    model = TemporalPoolingMLP(feature_dim=2048, hidden_dim=512, num_classes=7)
    # model = LSTMClassifier(feature_dim=2048, hidden_dim=256, num_classes=7)
    # model = TemporalConvNet(feature_dim=2048, hidden_dim=512, num_classes=7)
    
    # Test with dummy data
    batch_size = 4
    seq_len = 30  # Variable length clips
    feature_dim = 2048  # ResNet features
    
    x = torch.randn(batch_size, seq_len, feature_dim)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # (batch_size, num_classes)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")