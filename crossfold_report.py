# crossfold_report.py

import numpy as np
from evaluate import eval_predictions
from curves import save_roc_pr_curves

def run_crossfold_report(cfg, splits):
    dice_list = []
    iou_list = []
    auc_list = []
    ap_list = []

    for fold_id, (train_files, val_files) in enumerate(splits):
        print("\n" + "="*60)
        print(f"📌 FOLD {fold_id}")
        print("="*60)

        # Evaluate binary masks
        dice, iou = eval_predictions(cfg, val_files, fold_id=fold_id)
        dice_list.append(dice)
        iou_list.append(iou)

        # ROC + PR curves from probability maps
        roc_auc, ap = save_roc_pr_curves(cfg, val_files, fold_id=fold_id)
        auc_list.append(roc_auc)
        ap_list.append(ap)

    # Summary
    print("\n" + "="*60)
    print("✅ CROSS-FOLD SUMMARY")
    print("="*60)

    def stats(name, arr):
        arr = np.array(arr, dtype=np.float32)
        print(f"{name}: mean={arr.mean():.4f}  std={arr.std():.4f}")

    stats("Dice", dice_list)
    stats("IoU ", iou_list)
    stats("AUC ", auc_list)
    stats("AP  ", ap_list)

    return {
        "dice": dice_list,
        "iou": iou_list,
        "auc": auc_list,
        "ap": ap_list,
    }
