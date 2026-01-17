# models.py
from monai.networks.nets import SwinUNETR, UNETR


def build_model(model_name: str, img_size=(512, 512), device="cpu"):
    """
    Build a 2D segmentation model using MONAI.

    SwinUNETR: does NOT accept img_size
    UNETR: REQUIRES img_size
    """

    name = model_name.lower().strip()

    if name == "swinunetr":
        model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            feature_size=48,
            spatial_dims=2,
        )

    elif name == "unetr":
        model = UNETR(
            img_size=img_size,
            in_channels=1,
            out_channels=1,
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            spatial_dims=2,
        )

    else:
        raise ValueError("model_name must be 'swinunetr' or 'unetr'")

    return model.to(device)
