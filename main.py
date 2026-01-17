# main.py

import argparse
import random
import numpy as np
import torch

import config as cfg
from splits_kfold import list_common_filenames, make_kfold_splits
from train import train_one_fold
from infer import run_inference
from evaluate import eval_predictions
from crossfold_report import run_crossfold_report

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "infer", "metrics", "report"],
        help="train | infer | metrics | report"
    )
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    set_seed(cfg.seed)

    filenames = list_common_filenames(cfg.data_root, cfg.img_dir, cfg.label_dir, cfg.valid_mask_dir)
    if len(filenames) == 0:
        print("❌ No matching PNG files found across Done/NP/Valid")
        return

    splits = make_kfold_splits(filenames, k_folds=cfg.k_folds, seed=cfg.seed)

    if args.mode == "report":
        # Cross-fold summary requires you already ran inference for each fold
        run_crossfold_report(cfg, splits)
        return

    fold_id = args.fold
    if fold_id < 0 or fold_id >= len(splits):
        print(f"❌ fold must be 0..{len(splits)-1}")
        return

    train_files, val_files = splits[fold_id]

    if args.mode == "train":
        train_one_fold(cfg, train_files, val_files, fold_id=fold_id)

    elif args.mode == "infer":
        run_inference(cfg, val_files, fold_id=fold_id)

    elif args.mode == "metrics":
        eval_predictions(cfg, val_files, fold_id=fold_id)

if __name__ == "__main__":
    main()
