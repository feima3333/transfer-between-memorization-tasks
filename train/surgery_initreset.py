"""Init-reset factorial knockout: which donor component actually carries the transfer?

Weight surgery + downstream fine-tune. Load a donor checkpoint, keep ONE component
(conv weights / BN affine gamma,beta / FC readout) and reset the other two to the donor's
own epoch-0 ORIGINAL init, then fine-tune on the CIFAR-10 random-label task (label_seed 42)
and record the t0-vs-pretraining-epoch trajectory.

Three keep-only conditions (the name is the component that SURVIVES; the tuple is what is
reset). Source factorial-knockout equivalents are noted for cross-referencing older runs:

    keep_conv   reset gamma/beta + FC   -> "conv only"        (source knockout_GF)
    keep_bn     reset conv + FC         -> "gamma/beta only"  (source knockout_CF)
    keep_fc     reset conv + gamma/beta -> "FC only"          (source knockout_CG)

Why reset to the epoch-0 ORIGINAL init (seed 21) and not a fresh Kaiming draw. Both land on
the same init distribution (conv: kaiming_normal_(relu); FC: normal_(0, 0.01); BN affine:
(1, 0)), so they differ only by the random draw. Copying the donor's actual epoch-0 values
removes the "did the kept part adapt to the reset part's SPECIFIC init draw" confounder and
isolates "does this component carry information". BN affine (1, 0) already equals init, so
resetting it is draw-independent. The epoch-0 init is read from the donor trajectory's own
``epoch0000.pth`` sibling, which also sidesteps the cross-machine SVD sign-gauge trap of
reconstructing a seeded model.

Produces one run per (condition, donor epoch) under
``CHECKPOINT_DIR/<out-name>/<donor>/<condition>/epoch<k>/``. Consumes a donor trajectory
(randlabel seed21/labelseed21, or randpixel seed21/labelseed21/dataseed2026). Feeds the
init-reset weight-ablation figures ``weight_ablation_{randlabel,randpixel}_init`` (Fig 4 and
its random-pixel appendix counterpart); ``surgery_align.py`` builds the aligned overlay on
top of these same conditions.

    python train/surgery_initreset.py --donor randlabel --condition keep_conv
    python train/surgery_initreset.py --donor randlabel --condition keep_fc --epochs 0 10 20 30 40
    python train/surgery_initreset.py --donor randpixel --condition keep_conv --epochs 0 5 10 15
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# make `common` importable when run as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.checkpoints import extract_state, load_ckpt_into_model
from common.config import CHECKPOINT_DIR
from common.data import CIFAR10RandomLabels, cifar_transform, seed_everything
from common.engine import fit_model
from common.model import WideCNN_BN
from common.svd_ops import ALL_CONV_PATHS, resolve_conv

# condition -> the components RESET to the donor's epoch-0 init (the kept component is the
# complement). Reused by surgery_rescale.py and surgery_align.py, which add a step on top.
RESET_FACTORS: dict[str, tuple[str, ...]] = {
    "keep_conv": ("gammabeta", "fc"),
    "keep_bn": ("conv", "fc"),
    "keep_fc": ("conv", "gammabeta"),
}
KEEP_LABEL = {"keep_conv": "conv only", "keep_bn": "gamma/beta only", "keep_fc": "FC only"}

# donor -> the upstream run whose ``checkpoints/`` holds epoch0000..epochNN (train.py's naming)
DONOR_RUN = {
    "randlabel": "upstream_randlabel_seed21_labelseed21",
    "randpixel": "upstream_randpixel_seed21_labelseed21",
}


def load_init_state(init_ckpt: Path) -> dict:
    """State dict of the donor's epoch-0 original-init checkpoint (its ``epoch0000.pth``)."""
    if not init_ckpt.exists():
        raise FileNotFoundError(
            f"epoch-0 init checkpoint not found: {init_ckpt}\n"
            f"(expected the epoch0000.pth sibling of the donor trajectory)"
        )
    return extract_state(torch.load(init_ckpt, map_location="cpu"))


@torch.no_grad()
def apply_initreset(model: WideCNN_BN, factors: tuple[str, ...], init_state: dict) -> dict[str, int]:
    """Reset the named components of a loaded donor to their epoch-0 original-init values.

    conv / fc weights and biases are copied from ``init_state``; BN affine gamma/beta go to
    (1, 0), which equals init. Returns {component: n_params_reset} for provenance.
    """
    n: dict[str, int] = {}
    if "conv" in factors:
        for path in ALL_CONV_PATHS:
            conv = resolve_conv(model, path)
            conv.weight.copy_(init_state[f"{path}.weight"].to(conv.weight))
            if conv.bias is not None:  # ConvBNReLU convs keep bias=True
                conv.bias.copy_(init_state[f"{path}.bias"].to(conv.bias))
        n["conv"] = sum(resolve_conv(model, p).weight.numel() for p in ALL_CONV_PATHS)
    if "gammabeta" in factors:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1.0)
                m.bias.zero_()
        n["gammabeta"] = sum(m.weight.numel() + m.bias.numel()
                             for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    if "fc" in factors:
        model.fc.weight.copy_(init_state["fc.weight"].to(model.fc.weight))
        model.fc.bias.copy_(init_state["fc.bias"].to(model.fc.bias))
        n["fc"] = model.fc.weight.numel() + model.fc.bias.numel()
    return n


def resolve_donor_dir(donor: str, donor_ckpt_dir: str | None) -> Path:
    """The directory holding the donor's ``epoch<k>.pth`` (and its ``epoch0000.pth``)."""
    if donor_ckpt_dir is not None:
        return Path(donor_ckpt_dir)
    return CHECKPOINT_DIR / DONOR_RUN[donor] / "checkpoints"


def add_common_args(p: argparse.ArgumentParser) -> None:
    """The flags every surgery script shares (donor selection + downstream fit hyperparameters)."""
    p.add_argument("--donor", choices=sorted(DONOR_RUN), default="randlabel")
    p.add_argument("--donor-ckpt-dir", default=None,
                   help="directory holding the donor epoch<k>.pth (default: the upstream run's "
                        "checkpoints/ for the chosen --donor)")
    p.add_argument("--epochs", type=int, nargs="+",
                   default=[0, 5, 10, 15, 20, 25, 30, 35, 40],
                   help="donor pretraining epochs to run surgery on (randpixel donors usually only "
                        "have 0 5 10 15)")
    p.add_argument("--seed", type=int, default=21, help="downstream init + dataloader shuffle seed")
    p.add_argument("--label-seed", type=int, default=42, help="downstream random-label seed")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--early-stop-acc", type=float, default=0.9999)
    p.add_argument("--ckpt-every", type=int, default=0,
                   help="periodic checkpoint interval; 0 (default) writes only the init, stop and "
                        "last checkpoints -- the figures read metrics.json, not the trajectory")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(p)
    p.add_argument("--condition", choices=sorted(RESET_FACTORS), default="keep_conv")
    p.add_argument("--out-name", default="surgery_initreset_labelseed42",
                   help="checkpoint sub-directory under CHECKPOINT_DIR")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    factors = RESET_FACTORS[args.condition]
    donor_dir = resolve_donor_dir(args.donor, args.donor_ckpt_dir)
    init_state = load_init_state(donor_dir / "epoch0000.pth")

    # downstream is always the CIFAR-10 random-label task; the test split keeps real labels
    tf = cifar_transform()
    train_ds = CIFAR10RandomLabels(train=True, transform=tf, label_seed=args.label_seed)
    test_ds = CIFAR10RandomLabels(train=False, transform=tf)

    for epoch in args.epochs:
        donor_ckpt = donor_dir / f"epoch{epoch:04d}.pth"
        model = WideCNN_BN().to(device)
        load_ckpt_into_model(model, donor_ckpt, str(device))
        reset = apply_initreset(model, factors, init_state)

        run_dir = CHECKPOINT_DIR / args.out_name / args.donor / args.condition / f"epoch{epoch:04d}"
        config = {
            "experiment": "init_reset_factorial_knockout",
            "surgery": "initreset",
            "donor": args.donor,
            "donor_ckpt": str(donor_ckpt),
            "upstream_epoch": int(epoch),
            "condition": args.condition,
            "keep": KEEP_LABEL[args.condition],
            "reset_factors": list(factors),
            "reset_target": "donor_epoch0000_original_init",
            "params_reset": reset,
            "seed": args.seed,
            "label_seed": args.label_seed,
            "target_dataset": "cifar10_random_labels",
            "model": "WideCNN_BN",
        }
        print(f"[initreset] {args.donor}/{args.condition} ({KEEP_LABEL[args.condition]}) "
              f"epoch{epoch:04d} -> {run_dir}")
        fit_model(model, train_ds, test_ds, seed=args.seed, ckpt_dir=run_dir, device=device,
                  config=config, batch_size=args.batch_size, num_workers=args.num_workers,
                  lr=args.lr, momentum=0.9, weight_decay=0.0, max_epochs=args.max_epochs,
                  early_stop_acc=args.early_stop_acc,
                  ckpt_every=args.ckpt_every or (args.max_epochs + 1))


if __name__ == "__main__":
    main()
