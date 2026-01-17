# infer.py

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import imageio.v2 as imageio

from dataset_primefp20 import PrimeFP20Dataset
from models import build_model
from overlay_utils import make_overlay

def run_inference(cfg, filenames, fold_id=0):
    device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    ds = PrimeFP20Dataset(
        cfg.data_root, cfg.img_dir, cfg.label_dir, cfg.valid_mask_dir,
        filenames, out_size=cfg.out_size, crop_margin=cfg.crop_margin,
        normalize_mode=cfg.normalize_mode
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_fold{fold_id}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(cfg.model_name, img_size=cfg.out_size, device=device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = os.path.join(cfg.pred_dir, f"{cfg.model_name}_fold{fold_id}")
    os.makedirs(out_dir, exist_ok=True)

    overlay_dir = os.path.join(out_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    prob_dir = os.path.join(out_dir, "probs")
    os.makedirs(prob_dir, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)     # (1,1,H,W)
            fn = batch["filename"][0]

            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]  # (H,W) float [0,1]
            pred_bin = (probs > 0.5).astype(np.uint8)          # (H,W) 0/1

            # Save binary prediction mask
            pred_u8 = (pred_bin * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(out_dir, fn), pred_u8)

            # Save probability map as .npy (for ROC/PR)
            np.save(os.path.join(prob_dir, fn.replace(".png", ".npy")), probs)

            # Create + save overlay (prediction on top of original)
            # NOTE: x is normalized; overlay will still look fine as a visibility preview
            img_for_overlay = x.cpu().numpy()[0, 0]
            overlay_rgb = make_overlay(img_for_overlay, pred_bin, alpha=0.75)
            imageio.imwrite(os.path.join(overlay_dir, fn), overlay_rgb)

    print(f"✅ Inference done.")
    print(f"   Masks saved to: {out_dir}")
    print(f"   Overlays saved to: {overlay_dir}")
    print(f"   Prob maps saved to: {prob_dir}")
