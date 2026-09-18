"""Weight-SVD Procrustes alignment: undo an SVD surgery's representation drift.

For each conv layer j = 1..6 of a pretrained donor, the surgery (``common.svd_ops``) replaces the
LEFT singular vectors U_j of the (reshaped) conv weight with those of a fresh Kaiming model, U'_j,
keeping the donor's learned spectrum Sigma_j and right-singular vectors V_j^T:

    W'_j = U'_j diag(Sigma_j) V_j^T        (reshape (Cout, Cin*k*k) <-> (Cout,Cin,k,k))

That rotates the layer's output basis. ``align`` measures the resulting per-layer representation
drift on the training set and undoes it with a per-layer orthogonal Procrustes rotation pushed
FORWARD into the next layer:

    M_j = sum_batch  Phi_A^T Phi_B'         (C x C, Phi = post-BN-ReLU activation, (N*H*W, C))
    A_j = U V^T   with  U S V^T = svd(M_j)   (orthogonal)
    conv_{j+1}:  W <- einsum('oihw,ij->ojhw', W, A_j)       (rotate C_in; R = A_j, no transpose)
    FC (j = 6):  theta' = theta @ A_6                        (same R = A_6)

The next conv effectively operates on (A_j @ a'); we want A_j @ a' ~ a, and the orthogonal
Procrustes solution over M = Phi_A^T Phi_B' is R = U V^T = A_j (NOT A_j^T). Conv and FC use the SAME
rotation A_j -- an inverted transpose provably widens the logit gap to the donor rather than closing
it.

Phi_A = pristine donor activations; Phi_B' = activations of the progressively-modified model. Only
the 6 conv ``.weight`` and ``fc.weight`` are ever written; conv biases, BN affine + running buffers,
and fc.bias stay at the donor's values (BN buffers are snapshotted before / restored after activation
collection so train-mode batch-stats do not perturb them).

U' comes from a fresh Kaiming model seeded with FRESH_U_SEED (default 12345), which is DISTINCT from
the training seed 21 -- a seed-21 Kaiming model is byte-identical to the donor's epoch0000 init, so
U' would collapse onto the init basis. FRESH_U_SEED gives a genuinely independent random output
basis.
"""
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.checkpoints import load_ckpt_into_model
from common.model import WideCNN_BN
from common.svd_ops import (
    ALL_CONV_PATHS,
    SVD_DEVICE,
    SVD_DTYPE,
    conv_reduced_svd,
    resolve_conv,
)


def _new_model(device: str) -> WideCNN_BN:
    return WideCNN_BN().to(device)


def resolve_block(model: WideCNN_BN, conv_path: str) -> nn.Module:
    """'block2.0.conv' -> model.block2[0]  (the whole ConvBNReLU; its output is post-ReLU)."""
    block_name, idx, leaf = conv_path.split(".")
    assert leaf == "conv", f"unexpected conv path: {conv_path}"
    return getattr(model, block_name)[int(idx)]


def snapshot_bn(model: nn.Module) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Clone every BN running buffer so activation collection in train() mode can be rolled back."""
    snap = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            snap[name] = (
                m.running_mean.detach().clone(),
                m.running_var.detach().clone(),
                m.num_batches_tracked.detach().clone(),
            )
    return snap


@torch.no_grad()
def restore_bn(model: nn.Module, snap: dict) -> None:
    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and name in snap:
            rm, rv, nbt = snap[name]
            m.running_mean.copy_(rm)
            m.running_var.copy_(rv)
            m.num_batches_tracked.copy_(nbt)


@torch.no_grad()
def collect_cross_covariance(
    ref: nn.Module, mod: nn.Module, block_ref: nn.Module, block_mod: nn.Module,
    loader: DataLoader, device: str, bn_eval: bool, C: int,
) -> torch.Tensor:
    """M = sum_batch Phi_A^T Phi_B'  (C x C, float64 on CPU), Phi = post-ReLU (N*H*W, C)."""
    ref.eval() if bn_eval else ref.train()
    mod.eval() if bn_eval else mod.train()

    holder: dict[str, torch.Tensor] = {}
    h_ref = block_ref.register_forward_hook(lambda m, i, o: holder.__setitem__("A", o))
    h_mod = block_mod.register_forward_hook(lambda m, i, o: holder.__setitem__("B", o))

    M = torch.zeros(C, C, dtype=torch.float64)
    try:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            ref(x)
            mod(x)
            phi_a = holder["A"].permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
            phi_b = holder["B"].permute(0, 2, 3, 1).reshape(-1, C)
            M += (phi_a.transpose(0, 1) @ phi_b).double().cpu()
    finally:
        h_ref.remove()
        h_mod.remove()
    return M


@torch.no_grad()
def align(
    donor_ckpt: Path, u_prime: dict[str, torch.Tensor], loader: DataLoader,
    device: str, bn_eval: bool, apply_comp: bool = True, verbose: bool = True,
) -> tuple[WideCNN_BN, list[dict[str, Any]], dict[str, torch.Tensor]]:
    """Return (aligned model, per-layer meta, {path: A_hat}).

    The U-swap is inlined here rather than delegated to ``svd_ops.reset_layer`` on purpose: this is
    the canonical alignment pipeline, and interleaving the swap with the per-layer Procrustes
    collection (and its own meta/norm bookkeeping) is what the surgery scripts regress against.
    apply_comp=False skips the compensation (U'-swap only) -- the control the fidelity diagnostic
    reads against.
    """
    ref = _new_model(device)
    load_ckpt_into_model(ref, donor_ckpt, device)
    mod = _new_model(device)
    load_ckpt_into_model(mod, donor_ckpt, device)

    snap_ref, snap_mod = snapshot_bn(ref), snapshot_bn(mod)
    meta: list[dict[str, Any]] = []
    ahats: dict[str, torch.Tensor] = {}

    for j, path in enumerate(ALL_CONV_PATHS):
        conv_mod = resolve_conv(mod, path)
        C = conv_mod.out_channels

        # Replace U_j -> U'_j, keeping Sigma_j, V_j^T from the CURRENT mod weight.
        _, s_j, vh_j = conv_reduced_svd(conv_mod.weight)
        up = u_prime[path].to(dtype=SVD_DTYPE, device=SVD_DEVICE)
        assert up.shape == (C, s_j.shape[0]), f"{path}: U' {tuple(up.shape)} vs (C={C}, r={s_j.shape[0]})"
        w_new = (up * s_j.unsqueeze(0)) @ vh_j
        conv_mod.weight.data.copy_(w_new.reshape(conv_mod.weight.shape).to(device, torch.float32))

        # Procrustes rotation A_j from the post-ReLU cross-covariance against the pristine donor.
        M = collect_cross_covariance(
            ref, mod, resolve_block(ref, path), resolve_block(mod, path),
            loader, device, bn_eval, C,
        )
        U, _, Vt = torch.linalg.svd(M)
        a_hat = U @ Vt                                            # (C, C) orthogonal, float64 CPU
        ahats[path] = a_hat
        ortho_err = float((a_hat.transpose(0, 1) @ a_hat - torch.eye(C, dtype=a_hat.dtype)).abs().max())

        # Push A_j forward: the next conv operates on (A_j @ a'), and R = A_j (not A_j^T) is what
        # makes A_j @ a' ~ a. The FC (last) is right-multiplied by the same A_j.
        if apply_comp:
            if j < len(ALL_CONV_PATHS) - 1:
                nxt = resolve_conv(mod, ALL_CONV_PATHS[j + 1])
                w_next = nxt.weight.double().cpu()
                w_next_new = torch.einsum("oihw,ij->ojhw", w_next, a_hat)
                nxt.weight.data.copy_(w_next_new.reshape(nxt.weight.shape).to(device, torch.float32))
            else:
                fc_new = mod.fc.weight.double().cpu() @ a_hat     # theta' = theta @ A_6 (same R)
                mod.fc.weight.data.copy_(fc_new.to(device, torch.float32))

        meta.append({
            "layer": path,
            "C": int(C),
            "rank": int(s_j.shape[0]),
            "sigma_fro": float(s_j.norm()),
            "w_new_fro": float(w_new.norm()),
            "ahat_ortho_err_max": ortho_err,
            "ahat_det": float(torch.det(a_hat)),
        })
        if verbose:
            m = meta[-1]
            print(f"    [{path:<14}] C={C:<4} r={m['rank']:<4} "
                  f"||S||={m['sigma_fro']:.3f} ||W'||={m['w_new_fro']:.3f} "
                  f"orthoErr={ortho_err:.1e} det(A)={m['ahat_det']:+.3f}")

    restore_bn(ref, snap_ref)
    restore_bn(mod, snap_mod)
    return mod, meta, ahats
