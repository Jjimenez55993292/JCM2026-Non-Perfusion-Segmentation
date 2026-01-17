# dataset_primefp20.py

import os
import numpy as np
import torch

from preprocess_pipeline import (
    read_gray,
    binarize_mask,
    crop_to_valid_region,
    apply_valid_mask,
    normalize
)

def resize_numpy(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Simple resize using torch interpolation (no OpenCV needed).
    """
    x = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    x = torch.nn.functional.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False)
    return x.squeeze(0).squeeze(0).numpy()

class PrimeFP20Dataset(torch.utils.data.Dataset):
    """
    Returns:
      image:  (1,H,W) float32
      label:  (1,H,W) float32 (binary vessel map)
    """
    def __init__(self, data_root, img_dir, label_dir, valid_dir, filenames,
                 out_size=(512,512), crop_margin=8, normalize_mode="zscore"):
        self.data_root = data_root
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.valid_dir = valid_dir
        self.filenames = filenames

        self.out_h, self.out_w = out_size
        self.crop_margin = crop_margin
        self.normalize_mode = normalize_mode

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]

        img_path = os.path.join(self.data_root, self.img_dir, fn)
        lab_path = os.path.join(self.data_root, self.label_dir, fn)
        val_path = os.path.join(self.data_root, self.valid_dir, fn)

        img = read_gray(img_path)
        label = read_gray(lab_path)
        valid = read_gray(val_path)

        label = binarize_mask(label, 0.5)
        valid = binarize_mask(valid, 0.5)

        # ---- crop based on valid region ----
        img_crop, valid_crop = crop_to_valid_region(img, valid, margin=self.crop_margin)
        label_crop, _ = crop_to_valid_region(label, valid, margin=self.crop_margin)

        # ---- apply valid mask ----
        img_crop = apply_valid_mask(img_crop, valid_crop)
        label_crop = apply_valid_mask(label_crop, valid_crop)

        # ---- resize ----
        img_rs = resize_numpy(img_crop, self.out_h, self.out_w)
        lab_rs = resize_numpy(label_crop, self.out_h, self.out_w)

        # ---- normalize ----
        img_rs = normalize(img_rs, mode=self.normalize_mode)

        # final tensor shape (1,H,W)
        img_t = torch.tensor(img_rs, dtype=torch.float32).unsqueeze(0)
        lab_t = torch.tensor(lab_rs, dtype=torch.float32).unsqueeze(0)

        # clamp label for safety
        lab_t = (lab_t > 0.5).float()

        return {"image": img_t, "label": lab_t, "filename": fn}
