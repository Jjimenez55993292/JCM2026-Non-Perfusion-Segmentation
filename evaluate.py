# evaluate.py

import os
import numpy as np

from preprocess_pipeline import read_gray, binarize_mask

def eval_predictions(cfg, filenames, fold_id=0):
    pred_dir = os.path.join(cfg.pred_dir, f"{cfg.model_name}_fold{fold_id}")

    dices = []
    ious = []

    for fn in filenames:
        pred_path = os.path.join(pred_dir, fn)
        gt_path = os.path.join(cfg.data_root, cfg.label_dir, fn)

        if not os.path.exists(pred_path):
            print("Missing prediction:", pred_path)
            continue

        pred = read_gray(pred_path)
        gt = read_gray(gt_path)

        pred = binarize_mask(pred, 0.5)
        gt = binarize_mask(gt, 0.5)

        inter = (pred * gt).sum()
        union = (pred + gt - pred * gt).sum()

        dice = (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)

        dices.append(dice)
        ious.append(iou)

    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    print(f"✅ Results ({cfg.model_name} fold {fold_id})")
    print(f"Mean Dice = {mean_dice:.4f}")
    print(f"Mean IoU  = {mean_iou:.4f}")

    return mean_dice, mean_iou
