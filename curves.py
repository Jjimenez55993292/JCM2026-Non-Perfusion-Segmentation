# curves.py
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from dataset_primefp20 import PrimeFP20Dataset


def compute_pixel_scores(cfg, filenames, fold_id=0):
    """
    Loads:
      - GT label using SAME preprocessing as training (so it is 512x512)
      - predicted probability .npy (512x512)
    Returns flattened arrays y_true, y_score for ROC/PR.
    """
    fold_dir = os.path.join(cfg.pred_dir, f"{cfg.model_name}_fold{fold_id}")
    prob_dir = os.path.join(fold_dir, "probs")

    ds = PrimeFP20Dataset(
        cfg.data_root,
        cfg.img_dir,
        cfg.label_dir,
        cfg.valid_mask_dir,
        filenames,
        out_size=cfg.out_size,
        crop_margin=cfg.crop_margin,
        normalize_mode=cfg.normalize_mode,
    )

    y_true_all = []
    y_score_all = []

    for i in range(len(ds)):
        sample = ds[i]
        fn = sample["filename"]
        prob_path = os.path.join(prob_dir, fn.replace(".png", ".npy"))

        if not os.path.exists(prob_path):
            print("Missing prob file:", prob_path)
            continue

        gt = sample["label"].cpu().numpy()[0].astype(np.uint8)  # (H,W) 0/1
        probs = np.load(prob_path).astype(np.float32)           # (H,W) [0,1]

        if probs.shape != gt.shape:
            print(f"Shape mismatch for {fn}: probs={probs.shape} gt={gt.shape} (skipping)")
            continue

        y_true_all.append(gt.flatten())
        y_score_all.append(probs.flatten())

    if not y_true_all:
        return np.array([], dtype=np.uint8), np.array([], dtype=np.float32)

    y_true = np.concatenate(y_true_all).astype(np.uint8)
    y_score = np.concatenate(y_score_all).astype(np.float32)
    return y_true, y_score


def save_roc_pr_curves(cfg, filenames, fold_id=0):
    out_dir = os.path.join(cfg.pred_dir, f"{cfg.model_name}_fold{fold_id}")
    os.makedirs(out_dir, exist_ok=True)

    y_true, y_score = compute_pixel_scores(cfg, filenames, fold_id=fold_id)
    if y_true.size == 0:
        print("❌ No valid prob/gt pairs found for ROC/PR.")
        return 0.0, 0.0

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.title(f"ROC Curve (fold {fold_id}) AUC={roc_auc:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    plt.figure()
    plt.plot(recall, precision)
    plt.title(f"Precision-Recall Curve (fold {fold_id}) AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    pr_path = os.path.join(out_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()

    print(f"✅ Saved ROC curve: {roc_path}")
    print(f"✅ Saved PR curve:  {pr_path}")

    return float(roc_auc), float(ap)
