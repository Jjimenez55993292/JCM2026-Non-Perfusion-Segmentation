# train.py

import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from monai.losses import DiceCELoss

from dataset_primefp20 import PrimeFP20Dataset
from models import build_model
from metrics import dice_score, iou_score, sigmoid_to_binary

def train_one_fold(cfg, train_files, val_files, fold_id=0):
    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    train_ds = PrimeFP20Dataset(
        cfg.data_root, cfg.img_dir, cfg.label_dir, cfg.valid_mask_dir,
        train_files, out_size=cfg.out_size, crop_margin=cfg.crop_margin,
        normalize_mode=cfg.normalize_mode
    )
    val_ds = PrimeFP20Dataset(
        cfg.data_root, cfg.img_dir, cfg.label_dir, cfg.valid_mask_dir,
        val_files, out_size=cfg.out_size, crop_margin=cfg.crop_margin,
        normalize_mode=cfg.normalize_mode
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(cfg.model_name, img_size=cfg.out_size, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = DiceCELoss(sigmoid=True)  # binary segmentation, stable for vessels

    best_val_dice = -1.0
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_fold{fold_id}.pt")

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Fold {fold_id} | Epoch {epoch+1}/{cfg.epochs}"):
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))

        # ---- validation ----
        model.eval()
        dices = []
        ious = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                logits = model(x)
                pred_bin = sigmoid_to_binary(logits)

                d = dice_score(pred_bin, y)
                i = iou_score(pred_bin, y)
                dices.append(d)
                ious.append(i)

        val_dice = sum(dices) / len(dices)
        val_iou = sum(ious) / len(ious)

        print(f"[Fold {fold_id}] Epoch {epoch+1}: loss={avg_loss:.4f} | val_dice={val_dice:.4f} | val_iou={val_iou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({"model_state": model.state_dict()}, ckpt_path)
            print(f"✅ Saved best model: {ckpt_path} (val_dice={best_val_dice:.4f})")

    return ckpt_path
