# config.py

import os

# ---- DATASET PATH ----
# Example:
# data_root = r"C:\Users\YOU\Desktop\PHD\...\JCM2020-Non-Perfusion-Segmentation\facnp8"
data_root = r"./facnp8"

img_dir = "Done"
label_dir = "NP"
valid_mask_dir = "Valid"

# ---- PREPROCESS ----
out_size = (512, 512)   # resize target (H, W)
crop_margin = 8         # adds margin around valid region box
normalize_mode = "zscore"  # "zscore" or "minmax"

# ---- TRAINING ----
k_folds = 5
seed = 42

epochs = 80
batch_size = 2
lr = 1e-4

num_workers = 0  # Windows safe (set 2 if stable)
device = "cuda"  # or "cpu"

# ---- MODEL ----
# choose: "swinunetr" or "unetr"
model_name = "swinunetr"

# model checkpoint folder
ckpt_dir = "./checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# prediction outputs
pred_dir = "./predictions"
os.makedirs(pred_dir, exist_ok=True)
