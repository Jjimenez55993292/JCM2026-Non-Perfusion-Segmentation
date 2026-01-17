# overlay_utils.py

import numpy as np

def to_uint8(img01: np.ndarray) -> np.ndarray:
    """Convert float [0,1] or arbitrary float to uint8."""
    x = img01.copy()
    x = x - x.min()
    if x.max() > 1e-8:
        x = x / x.max()
    return (x * 255).astype(np.uint8)

def make_overlay(gray_img: np.ndarray, pred_mask: np.ndarray, alpha=0.6) -> np.ndarray:
    """
    gray_img: (H,W) float or uint8
    pred_mask: (H,W) 0/1 float or bool
    returns: RGB overlay uint8
    """
    g = gray_img.astype(np.float32)
    if g.max() > 1.0:
        g = g / 255.0

    base = np.stack([g, g, g], axis=-1)  # RGB gray
    overlay = base.copy()

    # RED channel highlight where pred is 1
    red = overlay[..., 0]
    red[pred_mask > 0.5] = np.clip(red[pred_mask > 0.5] + alpha, 0, 1)
    overlay[..., 0] = red

    return (overlay * 255).astype(np.uint8)
