import os
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import MRIPatchDataset
from data_utils import make_pairs, split_pairs


def fg_frac(hf_patch: torch.Tensor, thresh: float = 0.05) -> float:
    """
    hf_patch: torch tensor [1,H,W] in (ideally) [0,1]
    Returns fraction of pixels above thresh (rough proxy for "foreground").
    """
    x = hf_patch.squeeze(0).detach().cpu().numpy()
    return float((x > thresh).mean())


def patch_energy(hf_patch: torch.Tensor) -> float:
    """
    Simple edge/texture energy proxy.
    hf_patch: [1,H,W] (float), ideally in [0,1]
    Higher => more structure/edges; lower => flatter/background-ish.
    """
    x = hf_patch.squeeze(0).detach().cpu().numpy().astype(np.float32)
    gx = np.abs(np.diff(x, axis=0)).mean()
    gy = np.abs(np.diff(x, axis=1)).mean()
    return float(gx + gy)


if __name__ == "__main__":
    patch_size = 96
    stack_size = 7

    pairs = make_pairs(
        "./mri_resolution/train/low_field",
        "./mri_resolution/train/high_field",
    )
    train_pairs, val_pairs = split_pairs(pairs, val_frac=0.2, seed=42)

    train_ds = MRIPatchDataset(
        train_pairs,
        patch_size=patch_size,
        patches_per_volume=64,
        cache_volumes=True,
        stack_size=stack_size,
        sample_strategy="filtered",
        debug=True,
    )
    val_ds = MRIPatchDataset(
        val_pairs,
        patch_size=patch_size,
        patches_per_volume=16,
        cache_volumes=True,
        stack_size=stack_size,
        debug=True,
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
    # sample a few random patches from each split
    for name, ds in [("train", train_ds), ("val", val_ds)]:
        print("\n====================")
        print(f"{name} samples:")
        for _ in range(5):
            lf_t, hf_t, debug_dict = ds[random.randint(0, len(ds) - 1)]
            if debug_dict is not None:
                print(f"  reason={debug_dict['reason']}  tries={debug_dict['tries']}")
            print(
                f"  lf {tuple(lf_t.shape)}  hf {tuple(hf_t.shape)}  "
                f"lf[min,max]=({lf_t.min():.4f},{lf_t.max():.4f})  "
                f"hf[min,max]=({hf_t.min():.4f},{hf_t.max():.4f})  "
                f"hf mean={hf_t.mean():.4f} std={hf_t.std():.4f}  "
                f"fg@0.05={fg_frac(hf_t):.3f}  "
                f"energy={patch_energy(hf_t):.6f}"
            )

    # distributions over many samples (train)
    from collections import Counter
    reason_counts = Counter()
    tries_list = []
    energies = []
    fgs = []
    for _ in range(1000):
        lf_t, hf_t, debug_dict = train_ds[random.randint(0, len(train_ds) - 1)]
        fgs.append(fg_frac(hf_t))
        energies.append(patch_energy(hf_t))
        if debug_dict is not None:
            reason_counts[debug_dict['reason']] += 1
            tries_list.append(debug_dict['tries'])

    fgs = np.asarray(fgs, dtype=np.float32)
    energies = np.asarray(energies, dtype=np.float32)

    print(
        "\nEnergy stats over 1000 train patches: "
        f"mean={energies.mean():.6f}  "
        f"p10={np.percentile(energies,10):.6f}  "
        f"p50={np.percentile(energies,50):.6f}  "
        f"p90={np.percentile(energies,90):.6f}"
    )
    print(
        "FG frac stats over 1000 train patches: "
        f"mean={fgs.mean():.3f}  "
        f"p10={np.percentile(fgs,10):.3f}  "
        f"p50={np.percentile(fgs,50):.3f}  "
        f"p90={np.percentile(fgs,90):.3f}"
    )
    print("Reason counts:", dict(reason_counts))
    if tries_list:
        print(f"Avg tries: {sum(tries_list)/len(tries_list):.2f}  max tries: {max(tries_list)}")