"""Table 1: Frobenius-norm growth of fc / conv / BN across a donor's pretraining epochs.

For each donor trajectory it loads every epoch checkpoint and records ||fc||_F (weight+bias),
the mean ||conv||_F over the six conv layers, and ||BN affine||_F, then prints a pandas
DataFrame of the start norm, end norm and their ratio. Only the FC readout grows substantially
-- it starts tiny by construction (N(0, 0.01)) -- reproducing the paper's Table 1:
FC x23.70 (random label) / x14.40 (random pixel).

Consumes: the upstream donor checkpoints under CHECKPOINT_DIR (produced by
``train/train.py --stage upstream``, or downloaded from the GitHub Release). CPU only.
Corresponds to: Table 1.

    python tables/norm_growth.py
    python tables/norm_growth.py --randlabel-dir <dir> --randpixel-dir <dir>
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.checkpoints import extract_state
from common.config import CHECKPOINT_DIR
from common.model import WideCNN_BN


def epoch_of(path: Path) -> int:
    return int(re.search(r"epoch(\d+)", path.stem).group(1))


def _cat_norm(*tensors) -> float:
    """Frobenius norm of several tensors concatenated (double precision)."""
    return float(torch.cat([t.detach().double().flatten() for t in tensors]).norm())


def norms_for(model: WideCNN_BN) -> tuple[float, float, float]:
    """(||fc||, mean ||conv||, ||BN affine||). Each group uses its whole parameter block:
    FC = [weight; bias], each conv = [weight; bias], BN = [gamma; beta]."""
    fc = _cat_norm(model.fc.weight, model.fc.bias)
    convs = [_cat_norm(m.weight, m.bias) for m in model.modules() if isinstance(m, nn.Conv2d)]
    bn_parts = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_parts += [m.weight.detach().double().flatten(), m.bias.detach().double().flatten()]
    return fc, sum(convs) / len(convs), float(torch.cat(bn_parts).norm())


def norm_trajectory(ckpt_dir: Path) -> pd.DataFrame:
    """One row per epoch checkpoint: epoch, ||fc||, mean ||conv||, ||BN||."""
    ckpts = sorted(ckpt_dir.glob("epoch*.pth"), key=epoch_of)
    if not ckpts:
        raise FileNotFoundError(f"no epoch*.pth checkpoints under {ckpt_dir}")
    model = WideCNN_BN()
    rows = []
    for p in ckpts:
        model.load_state_dict(extract_state(torch.load(p, map_location="cpu")), strict=False)
        fc, conv, bn = norms_for(model)
        rows.append({"epoch": epoch_of(p), "fc": fc, "conv_mean": conv, "bn": bn})
    return pd.DataFrame(rows)


def growth_row(name: str, df: pd.DataFrame) -> dict:
    first, last = df.iloc[0], df.iloc[-1]
    return {
        "upstream": name,
        "fc_init": first.fc, "fc_final": last.fc, "fc_ratio": last.fc / first.fc,
        "conv_ratio": last.conv_mean / first.conv_mean, "bn_ratio": last.bn / first.bn,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Table 1: Frobenius-norm growth ratios.")
    p.add_argument("--randlabel-dir", type=Path,
                   default=CHECKPOINT_DIR / "upstream_randlabel_seed21_labelseed21" / "checkpoints")
    p.add_argument("--randpixel-dir", type=Path,
                   default=CHECKPOINT_DIR / "upstream_randpixel_seed21_labelseed21" / "checkpoints")
    args = p.parse_args(argv)

    rows = []
    for name, d in [("random label", args.randlabel_dir), ("random pixel", args.randpixel_dir)]:
        if d.exists():
            rows.append(growth_row(name, norm_trajectory(d)))
        else:
            print(f"[skip] {name}: {d} not found "
                  f"(train the upstream donor with train/train.py, or fetch the Release)")
    if rows:
        df = pd.DataFrame(rows).set_index("upstream")
        pd.set_option("display.float_format", lambda x: f"{x:.2f}")
        print()
        print(df.to_string())


if __name__ == "__main__":
    main()
