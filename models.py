# models.py

import torch
from monai.networks.nets import SwinUNETR, UNETR

def build_model(model_name: str, img_size=(512, 512), device="cuda"):
    """
    2D SwinUNETR / UNETR for single-channel input, single-class output.
    """
    if model_name.lower() == "swinunetr":
        model = SwinUNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=1,
            feature_size=48,
            spatial_dims=2
        )

    elif model_name.lower() == "unetr":
        model = UNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=1,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            spatial_dims=2
        )

    else:
        raise ValueError("model_name must be 'swinunetr' or 'unetr'")

    return model.to(device)
