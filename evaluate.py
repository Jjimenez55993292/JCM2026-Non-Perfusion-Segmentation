# evaluate.py
import os
import numpy as np
import torch

from preprocess_pipeline import read_gray, binarize_mask
from dataset_primefp20 import PrimeFP20Dataset


def _resize_like(x: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize a 2D numpy array to target (H,W) using torch bilinear."""
    th, tw = target_hw
    if x.shape == (th, tw):
        return x
    t = torch.from_numpy(x).float().unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    t = torch.nn.functional.interpolate(t, size=(th, tw), mode="bilinear", align_corners=False)
    return t.squeeze(0).squeeze(0).cpu().numpy()


def eval_predictions(cfg, filenames, fold_id=0):
    """
    Computes Dice/IoU using GT that has the SAME preprocessing as training:
    crop -> valid mask -> resize to cfg.out_size -> binarize.
    """
    pred_dir = os.path.join(cfg.pred_dir, f"{cfg.model_name}_fold{fold_id}")

    # Build dataset ONLY to get preprocessed/resized GT labels
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

    dices = []
    ious = []

    for i in range(len(ds)):
        sample = ds[i]
        fn = sample["filename"]

        pred_path = os.path.join(pred_dir, fn)
        if not os.path.exists(pred_path):
            print("Missing prediction:", pred_path)
            continue

        # Prediction is saved as PNG (0..255). read_gray converts to 0..1.
        pred = read_gray(pred_path)
        pred = binarize_mask(pred, 0.5).astype(np.float32)

        # GT label is already (1,H,W) torch tensor, binary
        gt = sample["label"].cpu().numpy()[0].astype(np.float32)

        # Safety: if anything mismatches, resize pred to gt size
        pred = _resize_like(pred, gt.shape)

        inter = (pred * gt).sum()
        union = (pred + gt - pred * gt).sum()

        dice = (2 * inter + 1e-6) / (pred.sum() + gt.sum() + 1e-6)
        iou = (inter + 1e-6) / (union + 1e-6)

        dices.append(float(dice))
        ious.append(float(iou))

    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0

    print(f"✅ Results ({cfg.model_name} fold {fold_id})")
    print(f"Mean Dice = {mean_dice:.4f}")
    print(f"Mean IoU  = {mean_iou:.4f}")

    return mean_dice, mean_iou
