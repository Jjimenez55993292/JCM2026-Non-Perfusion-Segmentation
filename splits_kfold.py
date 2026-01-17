# splits_kfold.py

import os
from sklearn.model_selection import KFold

def list_common_filenames(data_root, img_dir, label_dir, valid_dir):
    """
    Finds filenames that exist in all three folders.
    Example: 01.png ... 15.png
    """
    img_path = os.path.join(data_root, img_dir)
    lab_path = os.path.join(data_root, label_dir)
    val_path = os.path.join(data_root, valid_dir)

    imgs = set(os.listdir(img_path))
    labs = set(os.listdir(lab_path))
    vals = set(os.listdir(val_path))

    common = sorted(list(imgs & labs & vals))
    common = [f for f in common if f.lower().endswith(".png")]
    return common

def make_kfold_splits(filenames, k_folds=5, seed=42):
    """
    Returns list of (train_files, val_files) for each fold.
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
    splits = []
    for train_idx, val_idx in kf.split(filenames):
        train_files = [filenames[i] for i in train_idx]
        val_files = [filenames[i] for i in val_idx]
        splits.append((train_files, val_files))
    return splits
