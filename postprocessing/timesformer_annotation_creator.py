import os
import random
from collections import defaultdict, Counter

# --- config ---
video_clips_dir = "/standard/UVA-DSA/MIDAS/Organized/Bootcamp/Suturing/Processed/Gesture_Clips/"
test_size = 0.20        # 20% to val
seed = 42               # reproducible split
valid_exts = {".mp4", ".avi", ".mov", ".mkv"}

train_text_path = os.path.join(video_clips_dir, "train.txt")
train_text_path_mvit = os.path.join(video_clips_dir, "train_mvit.txt")
val_text_path   = os.path.join(video_clips_dir, "val.txt")
val_text_path_mvit   = os.path.join(video_clips_dir, "val_mvit.txt")

# classes to keep (keys are INT gesture IDs)
keep_classes = {
    2: "Cold Cut",
    5: "Grasp Long End of Suture",
    6: "Loop around Free Arm-DoubleLoop",
    7: "Loop around Free Arm-SingleLoop",
    9: "Orient Needle",
    13: "Pull Needle",
    14: "Pull Suture into Slip Knot and Cinch",
    15: "Pull Suture-BothHands",
    16: "Pull Suture-Fulcrum",
    17: "Pull Suture-OneHand",
    18: "Push Needle",
    20: "Square Knot-1: Grasp and Pull Short End Through Loop",
    21: "Square Knot-2",
    22: "Target Needle",
    23: "Two-hand Spread",
}

random.seed(seed)

def infer_label(root_relpath: str, filename: str) -> str:
    """
    Prefer the top-level folder under video_clips_dir as the class label.
    Falls back to 'third token in filename' only if needed.
    """
    parts = root_relpath.replace("\\", "/").split("/")
    if parts and parts[0] not in (".", "", None):
        return parts[0]
    toks = filename.split("_")
    return toks[2] if len(toks) >= 3 else toks[0]

# 1) Discover clips and their labels
# Store as tuples: (rel_path, label_str)
by_class = defaultdict(list)   # label_str -> [(rel_path, label_str), ...]

print("Keeping classes:", sorted(keep_classes.keys()))

for root, _, files in os.walk(video_clips_dir):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext not in valid_exts:
            continue

        rel_root = os.path.relpath(root, video_clips_dir)
        label_str = infer_label(rel_root, file)

        # keep only classes we want (labels are numeric strings)
        try:
            label_int = int(label_str)
        except ValueError:
            # skip non-numeric labels
            continue

        if label_int not in keep_classes:
            continue

        rel_path = f"{rel_root}/{file}".replace("\\", "/")
        by_class[label_str].append((rel_path, label_str))

# 2) Per-class split with coverage guarantees when possible
train_items, val_items = [], []
warnings = []

for label_str, items in by_class.items():
    n = len(items)
    random.shuffle(items)

    if n == 1:
        # impossible to have this label in both splits; keep in train and warn
        train_items.extend(items)
        warnings.append(f"[WARN] Label '{label_str}' has only 1 sample -> kept in TRAIN; it will be missing from VAL.")
        continue

    # at least 2 -> enforce at least 1 in each split
    n_val = round(test_size * n)
    n_val = max(1, min(n_val, n - 1))   # clamp to [1, n-1]
    val_items.extend(items[:n_val])
    train_items.extend(items[n_val:])

# Optional: shuffle overall order
random.shuffle(train_items)
random.shuffle(val_items)

# 3) Stats
train_counts = Counter([lbl for _, lbl in train_items])
val_counts   = Counter([lbl for _, lbl in val_items])

print("Train gesture counts:")
for g in sorted(train_counts.keys(), key=lambda x: int(x)):
    print(f"  {g}: {train_counts[g]}")

print("\nValidation gesture counts:")
for g in sorted(val_counts.keys(), key=lambda x: int(x)):
    print(f"  {g}: {val_counts[g]}")

# Report labels present/missing
all_labels = set(by_class.keys())
train_labels = set(train_counts.keys())
val_labels = set(val_counts.keys())

missing_in_train = sorted(all_labels - train_labels, key=lambda x: int(x)) if all_labels - train_labels else []
missing_in_val   = sorted(all_labels - val_labels, key=lambda x: int(x)) if all_labels - val_labels else []

if missing_in_train:
    print("\n[INFO] Labels missing in TRAIN (should be empty if n>=2 rule held):", missing_in_train)
if missing_in_val:
    print("\n[INFO] Labels missing in VAL (usually those with only 1 sample):", missing_in_val)

# 4) Write TAB-separated lists: "path<TAB>label"
with open(train_text_path, "w") as f:
    for rel_path, lbl in train_items:
        f.write(f"{rel_path} {lbl}\n")

with open(val_text_path, "w") as f:
    for rel_path, lbl in val_items:
        f.write(f"{rel_path} {lbl}\n")

# 4) Write comma-separated lists: "path,label"
with open(train_text_path_mvit, "w") as f:
    for rel_path, lbl in train_items:
        f.write(f"{rel_path}\n")

with open(val_text_path_mvit, "w") as f:
    for rel_path, lbl in val_items:
        f.write(f"{rel_path}\n")

# 5) Show any warnings
for w in warnings:
    print(w)

print(f"\nWrote {len(train_items)} rows to {train_text_path}")
print(f"Wrote {len(val_items)} rows to {val_text_path}")
