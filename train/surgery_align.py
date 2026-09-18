"""Alignment control: can a Procrustes rotation recover the transfer an ablation removed?

Weight surgery + downstream fine-tune, built on the init-reset knockout (see
``surgery_initreset.py``). Two steps, strictly in this order:

  (1) full ablation : keep ONE component, reset the other two to the donor's epoch-0 original
                      init -- exactly the ``surgery_initreset.py`` model.
  (2) alignment     : with that ablated model fixed, align it to the PRISTINE donor. For each
                      conv layer j = 1..6, measure the post-BN-ReLU representation drift against
                      the donor as a cross-covariance M_j = sum Phi_donor^T Phi_ablated, take the
                      orthogonal Procrustes rotation A_j = U V^T of M_j, and push A_j FORWARD
                      into the next layer -- the (j+1)-th conv is rotated on its input channels
                      (einsum), the FC (j = 6) is right-multiplied (theta @ A_6). We push forward,
                      not backward, because layer j's output basis rotated, so the layer that
                      CONSUMES it must be counter-rotated to see the donor's representation again.

Unlike the SVD surgery -- where a U-swap is the experiment and the alignment legitimately undoes
just that rotation -- here EVERY reset is the ablation, so there is no U-swap: ``ref`` is the
untouched donor throughout and only the Procrustes step is applied. keep_conv stays approximately
identity (conv is kept; only the reset BN affine perturbs the activations), keep_bn / keep_fc do
not. Expected result across all three: the aligned curve tracks its no-align sibling -- alignment
cannot recover transfer, which confirms the ablation conclusions.

The cross-covariance is measured on REAL CIFAR-10 images (the downstream fine-tuning distribution)
for BOTH donors, since that is the input the aligned model will be trained on. Labels are unused.

Note: this deliberately does NOT call ``common.alignment.align`` -- that pipeline inlines a fresh-U'
swap and reloads the donor into the model, which is the SVD-surgery experiment. Here the model is
the already-ablated one, so the same primitives (``collect_cross_covariance`` / ``resolve_block`` /
``snapshot_bn`` / ``restore_bn``) are composed directly, with no U-swap.

Produces one run per (condition, donor epoch) under
``CHECKPOINT_DIR/<out-name>/<donor>/<condition>_aligned/epoch<k>/``. Consumes a donor trajectory
(randlabel seed21/labelseed21, or randpixel seed21/labelseed21/dataseed2026). Feeds the appendix
aligned-overlay figures ``weight_ablation_{randlabel,randpixel}_init_aligned`` (A1 / A2).

    python train/surgery_align.py --donor randlabel --condition keep_conv
    python train/surgery_align.py --donor randlabel --condition keep_fc --epochs 0 10 20 30 40
    python train/surgery_align.py --donor randpixel --condition keep_conv --epochs 0 5 10 15
"""
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))            # so `surgery_initreset` (a sibling) imports
sys.path.insert(0, str(HERE.parent))     # so `common` imports

from common.alignment import collect_cross_covariance, resolve_block, restore_bn, snapshot_bn
from common.checkpoints import load_ckpt_into_model
from common.config import CHECKPOINT_DIR
from common.data import CIFAR10RandomLabels, cifar_transform, seed_everything
from common.engine import fit_model
from common.model import WideCNN_BN
from common.svd_ops import ALL_CONV_PATHS, resolve_conv
from surgery_initreset import (RESET_FACTORS, KEEP_LABEL, add_common_args, apply_initreset,
                               load_init_state, resolve_donor_dir)

ALIGN_BATCH_SIZE = 256  # cross-covariance is a read-only pass; larger batches just run faster


def make_align_loader(device: torch.device, num_workers: int) -> DataLoader:
    """Read-only loader over real CIFAR-10 train images (shuffle=False) for the Procrustes pass."""
    ds = CIFAR10RandomLabels(train=True, transform=cifar_transform(), label_seed=42)  # labels unused
    return DataLoader(ds, batch_size=ALIGN_BATCH_SIZE, shuffle=False, num_workers=num_workers,
                      pin_memory=(device.type == "cuda"))


@torch.no_grad()
def align_to_donor(ref: WideCNN_BN, mod: WideCNN_BN, loader: DataLoader,
                   device: torch.device) -> dict[str, torch.Tensor]:
    """Per-layer Procrustes: align the already-ablated ``mod`` to the pristine donor ``ref``.

    Pushes each A_j forward into the next conv's input channels (einsum) / the FC (theta @ A).
    BN running stats are snapshotted/restored so the train-mode activation collection cannot
    perturb them. Returns {conv_path: A_hat (C x C, float64 on CPU)}.
    """
    snap_ref, snap_mod = snapshot_bn(ref), snapshot_bn(mod)
    paths = list(ALL_CONV_PATHS)
    ahats: dict[str, torch.Tensor] = {}
    for j, path in enumerate(paths):
        C = resolve_conv(mod, path).out_channels
        M = collect_cross_covariance(ref, mod, resolve_block(ref, path), resolve_block(mod, path),
                                     loader, device, bn_eval=False, C=C)  # train-mode batch stats
        U, _, Vt = torch.linalg.svd(M)
        a_hat = U @ Vt  # (C, C) orthogonal
        ahats[path] = a_hat
        if j < len(paths) - 1:  # rotate the next conv's input channels: R = A_j (not A_j^T)
            nxt = resolve_conv(mod, paths[j + 1])
            w_next_new = torch.einsum("oihw,ij->ojhw", nxt.weight.double().cpu(), a_hat)
            nxt.weight.data.copy_(w_next_new.reshape(nxt.weight.shape).to(device, torch.float32))
        else:  # last conv feeds the FC: theta' = theta @ A_6
            fc_new = mod.fc.weight.double().cpu() @ a_hat
            mod.fc.weight.data.copy_(fc_new.to(device, torch.float32))
    restore_bn(ref, snap_ref)
    restore_bn(mod, snap_mod)
    return ahats


@torch.no_grad()
def build_aligned_model(donor_ckpt: Path, factors: tuple[str, ...], init_state: dict,
                        loader: DataLoader, device: torch.device) -> tuple[WideCNN_BN, dict]:
    """(1) full ablation to epoch-0 init, then (2) align the whole ablated model to the donor."""
    ref = WideCNN_BN().to(device)
    load_ckpt_into_model(ref, donor_ckpt, str(device))       # pristine donor reference
    mod = WideCNN_BN().to(device)
    load_ckpt_into_model(mod, donor_ckpt, str(device))
    apply_initreset(mod, factors, init_state)                # (1) full ablation
    ahats = align_to_donor(ref, mod, loader, device)         # (2) align vs pristine donor
    eye = {p: torch.eye(a.shape[0], dtype=a.dtype) for p, a in ahats.items()}
    meta = {"ahat_max_dev_from_identity": {p: float((ahats[p] - eye[p]).abs().max()) for p in ahats}}
    return mod, meta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_common_args(p)
    p.add_argument("--condition", choices=sorted(RESET_FACTORS), default="keep_conv")
    p.add_argument("--out-name", default="surgery_align_labelseed42",
                   help="checkpoint sub-directory under CHECKPOINT_DIR")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    factors = RESET_FACTORS[args.condition]
    donor_dir = resolve_donor_dir(args.donor, args.donor_ckpt_dir)
    init_state = load_init_state(donor_dir / "epoch0000.pth")
    align_loader = make_align_loader(device, args.num_workers)

    tf = cifar_transform()
    train_ds = CIFAR10RandomLabels(train=True, transform=tf, label_seed=args.label_seed)
    test_ds = CIFAR10RandomLabels(train=False, transform=tf)

    for epoch in args.epochs:
        donor_ckpt = donor_dir / f"epoch{epoch:04d}.pth"
        model, meta = build_aligned_model(donor_ckpt, factors, init_state, align_loader, device)

        subdir = f"{args.condition}_aligned"
        run_dir = CHECKPOINT_DIR / args.out_name / args.donor / subdir / f"epoch{epoch:04d}"
        config = {
            "experiment": "init_reset_aligned_control",
            "surgery": "initreset_aligned",
            "donor": args.donor,
            "donor_ckpt": str(donor_ckpt),
            "upstream_epoch": int(epoch),
            "condition": args.condition,
            "keep": KEEP_LABEL[args.condition],
            "reset_factors": list(factors),
            "reset_target": "donor_epoch0000_original_init",
            "alignment": "procrustes_vs_pristine_donor",
            "ahat_max_dev_from_identity": meta["ahat_max_dev_from_identity"],
            "seed": args.seed,
            "label_seed": args.label_seed,
            "target_dataset": "cifar10_random_labels",
            "model": "WideCNN_BN",
        }
        print(f"[align] {args.donor}/{subdir} ({KEEP_LABEL[args.condition]}) "
              f"epoch{epoch:04d} -> {run_dir}")
        fit_model(model, train_ds, test_ds, seed=args.seed, ckpt_dir=run_dir, device=device,
                  config=config, batch_size=args.batch_size, num_workers=args.num_workers,
                  lr=args.lr, momentum=0.9, weight_decay=0.0, max_epochs=args.max_epochs,
                  early_stop_acc=args.early_stop_acc,
                  ckpt_every=args.ckpt_every or (args.max_epochs + 1))


if __name__ == "__main__":
    main()
