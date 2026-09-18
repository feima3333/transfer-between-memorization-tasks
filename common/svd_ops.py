"""SVD primitives and the U / (Sigma, V) weight-replacement operators for Figure 5.

Every conv weight is measured through ONE reshape -- ``W.reshape(Cout, -1)`` -- and its reduced SVD
``W = U diag(S) Vh``. Two families of surgery are built on that factorization and share these
operators:

  * spectrum replacement (``reconstruct_layer``): mix the SVD factors of a base (Kaiming) init and a
    pretrained donor -- replace the spectrum only, the spectrum plus U, or the spectrum plus V.
  * factor reset (``reset_layer``): keep the donor's own current factors and swap in a fresh model's
    U' (keeping donor Sigma, V) or its Sigma'/V' (keeping donor U ONLY).

The SVD runs in float64 on CPU (``SVD_DTYPE`` / ``SVD_DEVICE``) so the factorization is deterministic
across machines; the rebuilt weight is cast back to float32 as it lands in a module. These are pure
tensor operations -- which checkpoint plays donor, and the seed of the fresh model, are experiment
choices that stay in the training / surgery scripts.
"""
from typing import Any

import torch
import torch.nn as nn

from common.model import WideCNN_BN

# The six conv layers in forward order, addressed as "<block>.<idx>.conv".
ALL_CONV_PATHS = (
    "stem.0.conv",
    "stem.1.conv",
    "block2.0.conv",
    "block2.1.conv",
    "block3.0.conv",
    "block3.1.conv",
)

SVD_DTYPE = torch.float64   # double precision for the SVD + reconstruction
SVD_DEVICE = "cpu"          # CPU SVD is deterministic across platforms (the sign gauge aside)


def resolve_conv(model: WideCNN_BN, path: str) -> nn.Conv2d:
    """'block2.0.conv' -> model.block2[0].conv"""
    block_name, idx, leaf = path.split(".")
    assert leaf == "conv", f"unexpected conv path: {path}"
    return getattr(model, block_name)[int(idx)].conv


@torch.no_grad()
def conv_reduced_svd(w4d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduced SVD of a conv weight under the analysis reshape (Cout, -1).

    Returns U (Cout, r), S (r,), Vh (r, N) in SVD_DTYPE on SVD_DEVICE, with r = min(Cout, N).
    """
    w2d = w4d.reshape(w4d.shape[0], -1).to(device=SVD_DEVICE, dtype=SVD_DTYPE)
    u, s, vh = torch.linalg.svd(w2d, full_matrices=False)
    return u, s, vh


@torch.no_grad()
def reconstruct_layer(
    condition: str,
    u0: torch.Tensor, s0: torch.Tensor, vh0: torch.Tensor,
    uc: torch.Tensor, sc: torch.Tensor, vhc: torch.Tensor,
    orig_shape: tuple[int, ...],
    norm_preserve: bool,
) -> torch.Tensor:
    """Rebuild one conv weight by mixing base (``*0``) and donor (``*c``) SVD factors.

    condition:
      spectrum     U0 diag(Sc') Vh0   -- replace the spectrum only
      spectrum_U   Uc diag(Sc') Vh0   -- replace spectrum + U (keep base V)
      spectrum_V   U0 diag(Sc') Vhc   -- replace spectrum + V (keep base U)

    norm_preserve rescales the donor spectrum so ``||W_new||_F == ||W0||_F``; by default the donor's
    raw spectrum is used, since the warm-start convention never rescales weight norms. Returns the 4D
    weight in SVD_DTYPE on SVD_DEVICE; the caller casts to float32.
    """
    if norm_preserve:
        sc_norm = sc.norm()
        scale = (s0.norm() / sc_norm) if float(sc_norm) > 0 else torch.ones((), dtype=sc.dtype)
        sc_prime = sc * scale
    else:
        sc_prime = sc

    if condition == "spectrum":
        u, s, vh = u0, sc_prime, vh0
    elif condition == "spectrum_U":
        u, s, vh = uc, sc_prime, vh0
    elif condition == "spectrum_V":
        u, s, vh = u0, sc_prime, vhc
    else:
        raise ValueError(f"reconstruct_layer does not handle condition={condition!r}")

    w2d = (u * s.unsqueeze(0)) @ vh
    return w2d.reshape(orig_shape)


@torch.no_grad()
def compute_u_prime(model: WideCNN_BN) -> dict[str, torch.Tensor]:
    """Left singular vectors U (Cout, r) of every conv layer of ``model`` (SVD_DTYPE/SVD_DEVICE)."""
    return {path: conv_reduced_svd(resolve_conv(model, path).weight)[0] for path in ALL_CONV_PATHS}


@torch.no_grad()
def compute_sv_prime(model: WideCNN_BN) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """(Sigma, Vh) of every conv layer of ``model`` -- the mirror of ``compute_u_prime``."""
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for path in ALL_CONV_PATHS:
        _, s, vh = conv_reduced_svd(resolve_conv(model, path).weight)
        result[path] = (s, vh)
    return result


@torch.no_grad()
def make_replacements(
    fresh_model: WideCNN_BN,
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Split ONE fresh model's SVD factors into both replacement sets: (u_prime, sv_prime).

    Taking both from the same model is the invariant the two resets rely on -- reset='u' and
    reset='sv' are then a complementary split of the SAME model's factors. Which seed draws that
    fresh Kaiming model (and the guard that it must differ from the training seed, or the draw
    collapses onto the donor's own init) is the caller's decision.
    """
    return compute_u_prime(fresh_model), compute_sv_prime(fresh_model)


def replacement_for(reset: str, u_prime: dict, sv_prime: dict) -> dict:
    """Pick the replacement set a reset consumes: U' for 'u', (Sigma',V') for 'sv', none for 'none'."""
    if reset == "none":
        return {}
    return u_prime if reset == "u" else sv_prime


@torch.no_grad()
def reset_layer(
    conv: nn.Conv2d, path: str, reset: str, replacement_j: Any, device: str,
) -> dict[str, Any]:
    """Rebuild ``conv.weight`` in place from the SVD of its CURRENT value; return per-layer meta.

    reset='u'  keeps the current Sigma_j and V_j, substituting U'_j.
    reset='sv' keeps the current U_j ONLY -- the current Sigma_j and V_j are discarded and replaced
               by the fresh model's Sigma'_j and V'_j (the two ``_`` below are that discard).
    """
    C = conv.out_channels
    if reset == "u":
        _, s_used, vh_used = conv_reduced_svd(conv.weight)
        u_used = replacement_j.to(dtype=SVD_DTYPE, device=SVD_DEVICE)
        if tuple(u_used.shape) != (C, s_used.shape[0]):
            raise ValueError(f"{path}: U' {tuple(u_used.shape)} vs (C={C}, r={s_used.shape[0]})")
        donor_sigma_fro = float(s_used.norm())
    elif reset == "sv":
        u_used, s_donor, _ = conv_reduced_svd(conv.weight)
        s_used, vh_used = (t.to(dtype=SVD_DTYPE, device=SVD_DEVICE) for t in replacement_j)
        if s_used.shape != s_donor.shape:
            raise ValueError(f"{path}: S' {tuple(s_used.shape)} vs donor {tuple(s_donor.shape)}")
        if tuple(vh_used.shape) != (s_used.shape[0], conv.weight.numel() // C):
            raise ValueError(f"{path}: Vh' {tuple(vh_used.shape)} unexpected")
        donor_sigma_fro = float(s_donor.norm())
    else:
        raise ValueError(f"unknown reset: {reset!r}")

    w_new = (u_used * s_used.unsqueeze(0)) @ vh_used            # (Cout, Cin*kh*kw)
    conv.weight.data.copy_(w_new.reshape(conv.weight.shape).to(device, torch.float32))

    return {
        "layer": path,
        "C": int(C),
        "rank": int(s_used.shape[0]),
        "sigma_fro": float(s_used.norm()),        # the spectrum actually used
        "donor_sigma_fro": donor_sigma_fro,       # the donor's, for reference
        "w_new_fro": float(w_new.norm()),
    }
