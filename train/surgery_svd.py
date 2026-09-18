"""SVD factor-replacement surgery for Figure 5, plus the factorial knockout it is read against.

Weight surgery + downstream fine-tune. Each run loads a donor checkpoint (an upstream
random-label / random-pixel trajectory at pretraining epoch k), edits its weights, fine-tunes
on the CIFAR-10 random-label task (label_seed 42) and records the t0-vs-pretraining-epoch
trajectory in metrics.json. Two output families come out of this one script:

(1) FACTORIAL KNOCKOUT  ->  CHECKPOINT_DIR/surgery_svd_t0_labelseed42/<donor>/<condition>/epoch<k>/
    The 2x2x2 knockout over three donor components. Component letters:
        C = conv weights (all six 3x3 conv .weight/.bias)
        G = gamma/beta   (BatchNorm affine)
        F = FC readout   (fc.weight/.bias)
    ``knockout_X`` RESETS the listed components to a fresh Kaiming model (``seed_everything(12345)``
    then a fresh ``WideCNN_BN``, whose weights supply the reset values) and KEEPS the rest at the
    donor. ``original`` resets nothing (the raw donor). Eight conditions:
        original, knockout_C, knockout_G, knockout_F, knockout_CF, knockout_CG, knockout_GF,
        knockout_CGF.
    This is Fig 4's Original reference, and supplies Fig 5's two anchors: ``knockout_GF`` is the
    "conv only" ceiling (donor conv, everything else fresh) and ``knockout_CGF`` is the "nothing"
    floor (every component fresh -> must land on the from-scratch band).

(2) SVD SURGERY + FC/BN RESET + ALIGNMENT
    ->  CHECKPOINT_DIR/surgery_svd_fcbn_reset_labelseed42/<donor>/<condition>/epoch<k>/
    Fig 5(init)'s three lines. Each conv weight is measured through the reduced SVD of
    ``W.reshape(Cout, -1) = U diag(Sigma) V^T``; the surgery swaps ONE side of that factorization
    for the donor's own EPOCH-0 ORIGINAL INIT (seed 21) factors and keeps the other, then undoes
    the resulting representation drift with a per-layer orthogonal Procrustes rotation. The FC and
    BN affine are reset to the same fresh Kaiming model as (1) AFTER the alignment, so the conv
    weights are the only surviving donor channel and this family shares (1)'s ``knockout_GF`` /
    ``knockout_CGF`` anchors. Three conditions:
        conv_only               keep the donor conv whole (no SVD surgery); reset gamma/beta + FC.
                                Identical to ``knockout_GF`` -- the conv-only ceiling.
        reset_u_uinit_aligned   keep the donor Sigma and V; replace U with the epoch-0 init U0
                                ("keep S&V"), then align; then reset gamma/beta + FC.
        reset_sv_svinit_aligned keep the donor U; replace Sigma and V with the epoch-0 init
                                Sigma0/V0 ("keep U"), then align; then reset gamma/beta + FC.

Why the epoch-0 ORIGINAL init and not a fresh Kaiming draw for the SVD factors. The question the
two lines ask is which learned SVD factor carries the transfer, so the replacement must be the
donor's UNLEARNED counterpart of that same factor, i.e. its epoch-0 init. Reading U0 / (Sigma0,V0)
from the donor's own ``epoch0000.pth`` (rather than reconstructing a seed-21 model) also keeps it on
this machine and sidesteps the SVD sign-gauge reproducibility trap.

Why keep Sigma,V vs keep U. reset_u_uinit isolates the LEFT singular subspace (output basis): it
puts back the init output directions while keeping the learned spectrum and input structure.
reset_sv_svinit isolates the U side symmetrically: it keeps the learned output basis and puts back
the init spectrum and right vectors. Together they factor the conv transfer into "which side of the
SVD learned the useful thing".

Why align forward into the NEXT layer. Swapping a factor rotates layer j's output basis, so the
layer that CONSUMES that output (the next conv's input channels, or the FC after global pooling)
must be counter-rotated to see the donor's representation again. Per layer the drift is measured as
a post-BN-ReLU cross-covariance M_j = sum Phi_donor^T Phi_modified against the PRISTINE donor; the
orthogonal Procrustes rotation A_j = U V^T of M_j is pushed forward: conv_{j+1} <- rotate input
channels by A_j, FC <- theta @ A_6.

Alignment mechanics reuse ``common.alignment`` / ``common.svd_ops``:
  * reset_u_uinit_aligned is EXACTLY ``common.alignment.align`` -- that function inlines the U-swap
    (keep the loaded donor's Sigma,V, substitute u_prime) and interleaves it with the per-layer
    Procrustes pass, which is the canonical, tested pipeline. It is called directly with
    u_prime = U0 (the epoch-0 init U).
  * reset_sv_svinit_aligned CANNOT use ``align`` (it only swaps U). The Sigma,V-reset discards V,
    which is where the compensation lives, so it must reset all six conv layers FIRST and only then
    run the Procrustes pass (``reset_then_align``), or each A_j pushed into layer j+1 is overwritten
    by that layer's own reset. It is composed here from the same primitives ``align`` uses
    (``svd_ops.reset_layer`` for the Sigma,V-swap; ``collect_cross_covariance`` / ``resolve_block``
    / ``snapshot_bn`` / ``restore_bn`` for the forward push).
``svd_ops.reconstruct_layer`` (the spectrum-MIX family: base-init U/V + donor spectrum) is a
different experiment and is deliberately not used here.

Consumes a donor trajectory (randlabel seed21/labelseed21, or randpixel seed21/labelseed21).
Downstream fine-tune is fixed at seed 21, label_seed 42, matching the other surgery scripts.

    python train/surgery_svd.py --donor randlabel --condition original
    python train/surgery_svd.py --donor randlabel --condition knockout_GF
    python train/surgery_svd.py --donor randlabel --condition reset_u_uinit_aligned
    python train/surgery_svd.py --donor randlabel --condition reset_sv_svinit_aligned --epochs 0 10 20 30 40
    python train/surgery_svd.py --donor randpixel --condition conv_only --epochs 0 5 10 15
"""
import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# make `common` importable when run as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.alignment import (
    align,
    collect_cross_covariance,
    resolve_block,
    restore_bn,
    snapshot_bn,
)
from common.checkpoints import load_ckpt_into_model
from common.config import CHECKPOINT_DIR
from common.data import CIFAR10RandomLabels, cifar_transform, seed_everything
from common.engine import fit_model
from common.model import WideCNN_BN
from common.svd_ops import (
    ALL_CONV_PATHS,
    compute_sv_prime,
    compute_u_prime,
    reset_layer,
    resolve_conv,
)

# The fresh Kaiming model whose weights supply every reset value (knockout conv/FC, and the FC/BN
# reset of the SVD family). MUST differ from the training seed 21: a seed-21 Kaiming model is
# byte-identical to every donor's epoch0000, which would make the knockout a no-op.
FRESH_RESET_SEED = 12345

# donor -> the upstream run whose checkpoints/ holds epoch0000..epochNN
DONOR_RUN = {
    "randlabel": "upstream_randlabel_seed21_labelseed21",
    "randpixel": "upstream_randpixel_seed21_labelseed21",
}

# (1) Factorial knockout: condition -> the components RESET to the fresh model (kept = complement).
# C = conv, G = gamma/beta (BN affine), F = FC. `original` resets nothing.
KNOCKOUT_FACTORS: dict[str, tuple[str, ...]] = {
    "original": (),
    "knockout_C": ("conv",),
    "knockout_G": ("gammabeta",),
    "knockout_F": ("fc",),
    "knockout_CF": ("conv", "fc"),
    "knockout_CG": ("conv", "gammabeta"),
    "knockout_GF": ("gammabeta", "fc"),
    "knockout_CGF": ("conv", "gammabeta", "fc"),
}

# (2) SVD surgery family. All three reset gamma/beta + FC (below); the two reset_* also do an SVD
# conv surgery + alignment first.
SVD_CONDITIONS = ("conv_only", "reset_u_uinit_aligned", "reset_sv_svinit_aligned")

# The FC/BN reset every SVD condition applies AFTER the alignment: it is exactly `knockout_GF`, so
# `conv_only` reproduces that anchor and the two SVD lines sit on the same baseline. BN running
# buffers are intentionally NOT reset -- train() normalizes by batch statistics and never reads
# them, so they can only move the init_eval number, not the trajectory.
FCBN_RESET_FACTORS: tuple[str, ...] = ("gammabeta", "fc")

# Default checkpoint sub-directory per family (overridable with --out-name).
KNOCKOUT_OUT = "surgery_svd_t0_labelseed42"
SVD_OUT = "surgery_svd_fcbn_reset_labelseed42"

ALL_CONDITIONS = tuple(KNOCKOUT_FACTORS) + SVD_CONDITIONS
ALIGN_BATCH_SIZE = 256  # the Procrustes pass is a read-only forward pass; larger batches run faster


def resolve_donor_dir(donor: str, donor_ckpt_dir: str | None) -> Path:
    """The directory holding the donor's ``epoch<k>.pth`` (and its ``epoch0000.pth``)."""
    if donor_ckpt_dir is not None:
        return Path(donor_ckpt_dir)
    return CHECKPOINT_DIR / DONOR_RUN[donor] / "checkpoints"


@torch.no_grad()
def build_fresh_state(device: str) -> dict[str, torch.Tensor]:
    """The state dict of a fresh Kaiming ``WideCNN_BN`` seeded ``FRESH_RESET_SEED``.

    One model is the single source of every reset value: its conv weights (kaiming_normal + bias 0),
    its FC readout (normal(0, 0.01) + bias 0) and its BN affine ((1, 0)). Both the knockout and the
    SVD family's FC/BN reset read from it, so ``knockout_GF`` and ``conv_only`` are bit-identical.
    """
    seed_everything(FRESH_RESET_SEED)
    fresh = WideCNN_BN().to(device)
    return {k: v.detach().clone() for k, v in fresh.state_dict().items()}


@torch.no_grad()
def apply_reset(model: WideCNN_BN, factors: tuple[str, ...], fresh_state: dict) -> dict[str, int]:
    """Reset the named components of a loaded donor to the fresh Kaiming model's values.

    conv resets both weight and bias; gammabeta copies the BN affine (which the fresh model holds at
    (1, 0)); fc resets weight and bias. Returns {component: n_params_reset} for provenance.
    """
    n: dict[str, int] = {}
    if "conv" in factors:
        for path in ALL_CONV_PATHS:
            conv = resolve_conv(model, path)
            conv.weight.copy_(fresh_state[f"{path}.weight"].to(conv.weight))
            if conv.bias is not None:  # ConvBNReLU convs keep bias=True
                conv.bias.copy_(fresh_state[f"{path}.bias"].to(conv.bias))
        n["conv"] = sum(resolve_conv(model, p).weight.numel() for p in ALL_CONV_PATHS)
    if "gammabeta" in factors:
        for name, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.copy_(fresh_state[f"{name}.weight"].to(m.weight))
                m.bias.copy_(fresh_state[f"{name}.bias"].to(m.bias))
        n["gammabeta"] = sum(m.weight.numel() + m.bias.numel()
                             for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    if "fc" in factors:
        model.fc.weight.copy_(fresh_state["fc.weight"].to(model.fc.weight))
        model.fc.bias.copy_(fresh_state["fc.bias"].to(model.fc.bias))
        n["fc"] = model.fc.weight.numel() + model.fc.bias.numel()
    return n


@torch.no_grad()
def load_init_replacements(init_ckpt: Path, device: str):
    """(u_init, sv_init) from the donor's epoch-0 original-init model: U0 and (Sigma0, V0^T).

    Read from the donor trajectory's ``epoch0000.pth`` sibling. reset_u_uinit consumes u_init;
    reset_sv_svinit consumes sv_init.
    """
    if not init_ckpt.exists():
        raise FileNotFoundError(
            f"epoch-0 init checkpoint not found: {init_ckpt}\n"
            f"(expected the epoch0000.pth sibling of the donor trajectory)"
        )
    m = WideCNN_BN().to(device)
    load_ckpt_into_model(m, init_ckpt, device)
    return compute_u_prime(m), compute_sv_prime(m)


def make_align_loader(device: torch.device, num_workers: int) -> DataLoader:
    """Read-only loader over real CIFAR-10 train images (shuffle=False) for the Procrustes pass.

    The alignment measures the drift on the downstream fine-tuning distribution (real images);
    labels are never read.
    """
    ds = CIFAR10RandomLabels(train=True, transform=cifar_transform(), label_seed=42)
    return DataLoader(ds, batch_size=ALIGN_BATCH_SIZE, shuffle=False, num_workers=num_workers,
                      pin_memory=(device.type == "cuda"))


@torch.no_grad()
def build_reset_sv_aligned(donor_ckpt: Path, sv_init: dict, loader: DataLoader,
                           device: torch.device) -> tuple[WideCNN_BN, list[dict[str, Any]]]:
    """reset='sv' + reset_then_align: keep the donor U, replace Sigma,V with the init Sigma0,V0.

    ``common.alignment.align`` only swaps U, so this path is composed from the same primitives. The
    Sigma,V-reset discards V (where the compensation lives), so ALL six conv layers are reset before
    any rotation is pushed forward; otherwise the A_j pushed into layer j+1 would be overwritten by
    that layer's own reset one iteration later. Returns (modified model, per-layer meta).
    """
    ref = WideCNN_BN().to(device)                      # pristine donor: the alignment target
    load_ckpt_into_model(ref, donor_ckpt, str(device))
    mod = WideCNN_BN().to(device)                      # cumulatively modified
    load_ckpt_into_model(mod, donor_ckpt, str(device))

    snap_ref, snap_mod = snapshot_bn(ref), snapshot_bn(mod)

    # (1) reset every conv layer first (keep donor U; substitute the init Sigma0, V0^T)
    meta: list[dict[str, Any]] = [
        reset_layer(resolve_conv(mod, path), path, "sv", sv_init[path], str(device))
        for path in ALL_CONV_PATHS
    ]

    # (2) per-layer Procrustes vs the pristine donor, each A_j pushed into a layer never reset again
    for j, path in enumerate(ALL_CONV_PATHS):
        C = resolve_conv(mod, path).out_channels
        M = collect_cross_covariance(ref, mod, resolve_block(ref, path), resolve_block(mod, path),
                                     loader, device, bn_eval=False, C=C)
        U, _, Vt = torch.linalg.svd(M)
        a_hat = U @ Vt                                  # (C, C) orthogonal, float64 CPU
        if j < len(ALL_CONV_PATHS) - 1:                 # rotate the next conv's input channels
            nxt = resolve_conv(mod, ALL_CONV_PATHS[j + 1])
            w_next = torch.einsum("oihw,ij->ojhw", nxt.weight.double().cpu(), a_hat)
            nxt.weight.data.copy_(w_next.reshape(nxt.weight.shape).to(device, torch.float32))
        else:                                           # last conv feeds the FC: theta' = theta @ A_6
            fc_new = mod.fc.weight.double().cpu() @ a_hat
            mod.fc.weight.data.copy_(fc_new.to(device, torch.float32))
        meta[j]["ahat_ortho_err_max"] = float(
            (a_hat.transpose(0, 1) @ a_hat - torch.eye(C, dtype=a_hat.dtype)).abs().max()
        )
        meta[j]["ahat_det"] = float(torch.det(a_hat))

    restore_bn(ref, snap_ref)
    restore_bn(mod, snap_mod)
    return mod, meta


def build_surgery_model(condition: str, donor_ckpt: Path, fresh_state: dict, init_reps,
                        loader, device: torch.device
                        ) -> tuple[WideCNN_BN, dict[str, Any]]:
    """Dispatch a single (condition, donor epoch) to its surgery; return (model, surgery meta)."""
    surgery: dict[str, Any] = {}

    if condition in KNOCKOUT_FACTORS:
        factors = KNOCKOUT_FACTORS[condition]
        model = WideCNN_BN().to(device)
        load_ckpt_into_model(model, donor_ckpt, str(device))
        surgery["reset_factors"] = list(factors)
        surgery["params_reset"] = apply_reset(model, factors, fresh_state)
        return model, surgery

    # SVD family: (optional) conv surgery + alignment, THEN the FC/BN reset (== knockout_GF).
    if condition == "conv_only":
        model = WideCNN_BN().to(device)
        load_ckpt_into_model(model, donor_ckpt, str(device))
    elif condition == "reset_u_uinit_aligned":
        u_init, _ = init_reps
        # This IS the canonical pipeline: align inlines the U-swap (keep the donor Sigma,V, put in
        # U0) and interleaves it with the per-layer Procrustes push. bn_eval=False -> train batch
        # stats, matching the donor's own forward statistics.
        model, per_layer, _ = align(donor_ckpt, u_init, loader, device, False,
                                    apply_comp=True, verbose=False)
        surgery["per_layer"] = per_layer
    elif condition == "reset_sv_svinit_aligned":
        _, sv_init = init_reps
        model, per_layer = build_reset_sv_aligned(donor_ckpt, sv_init, loader, device)
        surgery["per_layer"] = per_layer
    else:
        raise ValueError(f"unknown condition: {condition!r}")

    surgery["fc_bn_reset_factors"] = list(FCBN_RESET_FACTORS)
    surgery["fc_bn_reset_params"] = apply_reset(model, FCBN_RESET_FACTORS, fresh_state)
    return model, surgery


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--donor", choices=sorted(DONOR_RUN), default="randlabel")
    p.add_argument("--donor-ckpt-dir", default=None,
                   help="directory holding the donor epoch<k>.pth (default: the upstream run's "
                        "checkpoints/ for the chosen --donor)")
    p.add_argument("--condition", choices=list(ALL_CONDITIONS), default="original")
    p.add_argument("--epochs", type=int, nargs="+",
                   default=[0, 5, 10, 15, 20, 25, 30, 35, 40],
                   help="donor pretraining epochs to run surgery on (randpixel donors usually only "
                        "have 0 5 10 15)")
    p.add_argument("--out-name", default=None,
                   help="checkpoint sub-directory under CHECKPOINT_DIR (default: "
                        f"{KNOCKOUT_OUT} for the knockout family, {SVD_OUT} for the SVD family)")
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
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_svd = args.condition in SVD_CONDITIONS
    out_name = args.out_name or (SVD_OUT if is_svd else KNOCKOUT_OUT)
    donor_dir = resolve_donor_dir(args.donor, args.donor_ckpt_dir)

    fresh_state = build_fresh_state(str(device))
    # The epoch-0 init SVD factors and the alignment loader are only needed by the two reset_* SVD
    # lines; conv_only and the knockouts need neither.
    needs_align = args.condition in ("reset_u_uinit_aligned", "reset_sv_svinit_aligned")
    init_reps = load_init_replacements(donor_dir / "epoch0000.pth", str(device)) if needs_align else None
    loader = make_align_loader(device, args.num_workers) if needs_align else None

    # downstream is always the CIFAR-10 random-label task; the test split keeps real labels
    tf = cifar_transform()
    train_ds = CIFAR10RandomLabels(train=True, transform=tf, label_seed=args.label_seed)
    test_ds = CIFAR10RandomLabels(train=False, transform=tf)

    for epoch in args.epochs:
        donor_ckpt = donor_dir / f"epoch{epoch:04d}.pth"
        model, surgery = build_surgery_model(args.condition, donor_ckpt, fresh_state,
                                             init_reps, loader, device)

        run_dir = CHECKPOINT_DIR / out_name / args.donor / args.condition / f"epoch{epoch:04d}"
        config = {
            "experiment": "svd_surgery" if is_svd else "factorial_knockout",
            "family": "svd_fcbn_reset" if is_svd else "knockout",
            "donor": args.donor,
            "donor_ckpt": str(donor_ckpt),
            "upstream_epoch": int(epoch),
            "condition": args.condition,
            "reset_target": "fresh_kaiming_seed12345",
            "fresh_reset_seed": FRESH_RESET_SEED,
            "svd_init_source": ("donor_epoch0000_original_init" if needs_align else None),
            "surgery": surgery,
            "seed": args.seed,
            "label_seed": args.label_seed,
            "target_dataset": "cifar10_random_labels",
            "model": "WideCNN_BN",
        }
        print(f"[surgery_svd] {args.donor}/{args.condition} epoch{epoch:04d} -> {run_dir}")
        fit_model(model, train_ds, test_ds, seed=args.seed, ckpt_dir=run_dir, device=device,
                  config=config, batch_size=args.batch_size, num_workers=args.num_workers,
                  lr=args.lr, momentum=0.9, weight_decay=0.0, max_epochs=args.max_epochs,
                  early_stop_acc=args.early_stop_acc,
                  ckpt_every=args.ckpt_every or (args.max_epochs + 1))


if __name__ == "__main__":
    main()
