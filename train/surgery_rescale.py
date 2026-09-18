"""Rescale control: does a kept component transfer by STRUCTURE or just by SCALE?

Weight surgery + downstream fine-tune, built on the init-reset knockout (see
``surgery_initreset.py``). Keep ONE component and reset the other two to the donor's epoch-0
original init, then rescale the KEPT component's [weight; bias] back to its epoch-0 Frobenius
norm -- one scalar factor, so its DIRECTION is untouched and only its magnitude returns to
init. If the rescaled init still transfers, the component's contribution is structural; if it
collapses onto the from-scratch floor, the "transfer" was just scale (a larger norm acts like a
larger effective learning rate on that component).

    --condition fc     keep FC   (reset conv + gamma/beta), rescale the whole readout [w; b]
                       by ONE factor to the epoch-0 readout norm. The FC readout grows ~20x
                       over pretraining from its normal_(0, 0.01) init, so this is the load-
                       bearing control.
    --condition conv   keep conv (reset gamma/beta + FC), rescale EACH conv layer's [W; b] by
                       one per-layer factor to that layer's epoch-0 norm.

All targets are read from the donor's own ``epoch0000.pth`` (the init magnitude), so no seeded
model is reconstructed. This is the init-reset variant: the reset components land on the donor's
epoch-0 values, not a fresh Kaiming draw (see ``surgery_initreset.py`` for why).

Produces one run per (condition, donor epoch) under
``CHECKPOINT_DIR/<out-name>/<donor>/<condition>_rescale/epoch<k>/``. Consumes a donor trajectory
(randlabel seed21/labelseed21, or randpixel seed21/labelseed21/dataseed2026). Feeds the rescaled-
keep overlays of the init-reset weight-ablation figures ``weight_ablation_{randlabel,randpixel}_init``
(Fig 4 and its random-pixel appendix counterpart): rescaled-FC goes flat (FC transfer is scale),
rescaled-conv tracks keep-conv (conv transfer is direction).

    python train/surgery_rescale.py --donor randlabel --condition fc
    python train/surgery_rescale.py --donor randlabel --condition conv --epochs 0 10 20 30 40
    python train/surgery_rescale.py --donor randpixel --condition conv --epochs 0 5 10 15
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # so `surgery_initreset` (a sibling) imports
sys.path.insert(0, str(HERE.parent))     # so `common` imports

from common.checkpoints import load_ckpt_into_model
from common.config import CHECKPOINT_DIR
from common.data import CIFAR10RandomLabels, cifar_transform, seed_everything
from common.engine import fit_model
from common.model import WideCNN_BN
from common.svd_ops import ALL_CONV_PATHS, resolve_conv
from surgery_initreset import (RESET_FACTORS, add_common_args, apply_initreset,
                               load_init_state, resolve_donor_dir)

# condition -> which keep-only reset base it rides on (reusing the knockout factor sets)
BASE_CONDITION = {"fc": "keep_fc", "conv": "keep_conv"}


def _wb_norm(weight: torch.Tensor, bias) -> float:
    """||[weight; bias]||_2 -- the parameter group treated as one vector (bias included)."""
    n = weight.detach().flatten().pow(2).sum()
    if bias is not None:
        n = n + bias.detach().flatten().pow(2).sum()
    return float(n.sqrt())


@torch.no_grad()
def rescale_fc(model: WideCNN_BN, target_norm: float) -> dict:
    """Rescale the whole readout [fc.weight; fc.bias] to ``target_norm`` with one scalar factor."""
    before = _wb_norm(model.fc.weight, model.fc.bias)
    factor = target_norm / before
    model.fc.weight.mul_(factor)
    model.fc.bias.mul_(factor)
    return {"readout_norm_before": before, "readout_norm_after": _wb_norm(model.fc.weight, model.fc.bias),
            "target_norm": target_norm, "rescale_factor": factor}


@torch.no_grad()
def rescale_conv(model: WideCNN_BN, target_norms: dict[str, float]) -> dict:
    """Rescale each conv layer's [W; b] to its own epoch-0 norm with one per-layer factor."""
    info: dict[str, dict] = {}
    for path in ALL_CONV_PATHS:
        conv = resolve_conv(model, path)
        before = _wb_norm(conv.weight, conv.bias)
        factor = target_norms[path] / before
        conv.weight.mul_(factor)
        if conv.bias is not None:
            conv.bias.mul_(factor)
        info[path] = {"norm_before": before, "norm_after": _wb_norm(conv.weight, conv.bias),
                      "target_norm": target_norms[path], "rescale_factor": factor}
    return info


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(p)
    p.add_argument("--condition", choices=sorted(BASE_CONDITION), default="fc",
                   help="fc: rescale the kept FC readout; conv: rescale each kept conv layer")
    p.add_argument("--out-name", default="surgery_rescale_init_labelseed42",
                   help="checkpoint sub-directory under CHECKPOINT_DIR")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base = BASE_CONDITION[args.condition]
    factors = RESET_FACTORS[base]  # the two components reset to epoch-0 init; the kept one is rescaled
    donor_dir = resolve_donor_dir(args.donor, args.donor_ckpt_dir)
    init_state = load_init_state(donor_dir / "epoch0000.pth")

    # rescale targets = the kept component's epoch-0 (init) Frobenius norm, read from epoch0000
    if args.condition == "fc":
        fc_target = _wb_norm(init_state["fc.weight"], init_state["fc.bias"])
        conv_targets = None
    else:
        fc_target = None
        conv_targets = {p: _wb_norm(init_state[f"{p}.weight"], init_state[f"{p}.bias"])
                        for p in ALL_CONV_PATHS}

    tf = cifar_transform()
    train_ds = CIFAR10RandomLabels(train=True, transform=tf, label_seed=args.label_seed)
    test_ds = CIFAR10RandomLabels(train=False, transform=tf)

    for epoch in args.epochs:
        donor_ckpt = donor_dir / f"epoch{epoch:04d}.pth"
        model = WideCNN_BN().to(device)
        load_ckpt_into_model(model, donor_ckpt, str(device))
        reset = apply_initreset(model, factors, init_state)  # reset non-kept -> epoch-0 init
        rescale = (rescale_fc(model, fc_target) if args.condition == "fc"
                   else rescale_conv(model, conv_targets))    # rescale the kept component

        subdir = f"{args.condition}_rescale"
        run_dir = CHECKPOINT_DIR / args.out_name / args.donor / subdir / f"epoch{epoch:04d}"
        config = {
            "experiment": "init_reset_rescale_control",
            "surgery": f"{args.condition}_rescale_init",
            "donor": args.donor,
            "donor_ckpt": str(donor_ckpt),
            "upstream_epoch": int(epoch),
            "condition": args.condition,
            "base_reset": base,
            "reset_factors": list(factors),
            "reset_target": "donor_epoch0000_original_init",
            "params_reset": reset,
            "rescale": rescale,
            "seed": args.seed,
            "label_seed": args.label_seed,
            "target_dataset": "cifar10_random_labels",
            "model": "WideCNN_BN",
        }
        print(f"[rescale] {args.donor}/{subdir} epoch{epoch:04d} -> {run_dir}")
        fit_model(model, train_ds, test_ds, seed=args.seed, ckpt_dir=run_dir, device=device,
                  config=config, batch_size=args.batch_size, num_workers=args.num_workers,
                  lr=args.lr, momentum=0.9, weight_decay=0.0, max_epochs=args.max_epochs,
                  early_stop_acc=args.early_stop_acc,
                  ckpt_every=args.ckpt_every or (args.max_epochs + 1))


if __name__ == "__main__":
    main()
