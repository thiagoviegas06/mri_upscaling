import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import MRIPatchDataset
from preprocessing import make_pairs, split_pairs


def fg_frac(hf_patch: torch.Tensor, thresh: float = 0.05) -> float:
    """
    hf_patch: torch tensor [1,H,W] in (ideally) [0,1]
    Returns fraction of pixels above thresh (rough proxy for "foreground").
    """
    x = hf_patch.squeeze(0).detach().cpu().numpy()
    return float((x > thresh).mean())


if __name__ == "__main__":
    patch_size = 96
    stack_size = 7

    pairs = make_pairs(
        "/scratch/tjv235/pytorch-example/mri_upscaling/mri_resolution/train/low_field",
        "/scratch/tjv235/pytorch-example/mri_upscaling/mri_resolution/train/high_field",
    )
    train_pairs, val_pairs = split_pairs(pairs, val_frac=0.2, seed=42)

    train_ds = MRIPatchDataset(
        train_pairs,
        patch_size=patch_size,
        patches_per_volume=64,
        cache_volumes=True,
        stack_size=stack_size,
    )
    val_ds = MRIPatchDataset(
        val_pairs,
        patch_size=patch_size,
        patches_per_volume=16,
        cache_volumes=True,
        stack_size=stack_size,
    )

    # -------------------------
    # Quick sanity checks
    # -------------------------
    print(f"num pairs total: {len(pairs)}")
    print(f"train pairs: {len(train_pairs)} | val pairs: {len(val_pairs)}")

    # overlap check (should be 0)
    train_set = set(train_pairs)
    val_set = set(val_pairs)
    print(f"overlap train/val pairs: {len(train_set & val_set)}")

    # sample a few random patches from each split
    for name, ds in [("train", train_ds), ("val", val_ds)]:
        print("\n====================")
        print(f"{name} samples:")
        for _ in range(5):
            lf_t, hf_t = ds[random.randint(0, len(ds) - 1)]
            print(
                f"  lf {tuple(lf_t.shape)}  hf {tuple(hf_t.shape)}  "
                f"lf[min,max]=({lf_t.min():.4f},{lf_t.max():.4f})  "
                f"hf[min,max]=({hf_t.min():.4f},{hf_t.max():.4f})  "
                f"hf mean={hf_t.mean():.4f} std={hf_t.std():.4f}  "
                f"fg@0.05={fg_frac(hf_t):.3f}"
            )

    # optional: foreground fraction distribution over many samples (train)
    fgs = []
    for _ in range(500):
        _, hf_t = train_ds[random.randint(0, len(train_ds) - 1)]
        fgs.append(fg_frac(hf_t))
    fgs = np.asarray(fgs, dtype=np.float32)

    print(
        "\nFG frac stats over 500 train patches: "
        f"mean={fgs.mean():.3f}  "
        f"p10={np.percentile(fgs,10):.3f}  "
        f"p50={np.percentile(fgs,50):.3f}  "
        f"p90={np.percentile(fgs,90):.3f}"
    )