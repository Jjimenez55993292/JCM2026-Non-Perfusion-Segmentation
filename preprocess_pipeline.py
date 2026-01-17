# preprocess_pipeline.py

import numpy as np
import imageio.v2 as imageio

def read_gray(path: str) -> np.ndarray:
    """Read an image as grayscale float32 in range [0,1]."""
    img = imageio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]  # simple channel pick if RGB
    img = img.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img

def binarize_mask(mask: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    """Make a binary mask 0/1."""
    return (mask > thresh).astype(np.float32)

def crop_to_valid_region(img: np.ndarray, valid_mask: np.ndarray, margin: int = 8):
    """
    Crop image + label to bounding box of valid region mask.
    """
    ys, xs = np.where(valid_mask > 0.5)

    # If mask is empty, return original
    if len(xs) == 0 or len(ys) == 0:
        return img, valid_mask

    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    y0 = max(0, y0 - margin)
    x0 = max(0, x0 - margin)
    y1 = min(img.shape[0] - 1, y1 + margin)
    x1 = min(img.shape[1] - 1, x1 + margin)

    cropped_img = img[y0:y1+1, x0:x1+1]
    cropped_mask = valid_mask[y0:y1+1, x0:x1+1]
    return cropped_img, cropped_mask

def apply_valid_mask(img: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Zero out outside valid region."""
    return img * valid_mask

def normalize(img: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """
    Normalize image for training stability.
    """
    if mode == "minmax":
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8:
            return img
        return (img - mn) / (mx - mn)

    # z-score
    mu = img.mean()
    std = img.std()
    if std < 1e-8:
        return img - mu
    return (img - mu) / std
