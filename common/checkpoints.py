"""Checkpoint and JSON IO shared by training and the surgery scripts.

A checkpoint written here is ``{"model_state", "epoch", "history", "extra"?}``; ``extract_state``
also accepts the bare and legacy layouts other tools have produced, so any of the project's
checkpoints loads into a fresh model without a per-file adapter.
"""
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def extract_state(ckpt_obj: Any) -> dict[str, torch.Tensor]:
    """Pull the model state_dict out of a checkpoint, whatever wrapper key it used.

    Accepts our own ``{"model_state": ...}`` payloads, the common ``state_dict`` / ``model`` /
    ``net`` / ``params`` variants, and a bare state_dict saved with no wrapper at all.
    """
    if isinstance(ckpt_obj, dict):
        for key in ("model_state", "state_dict", "model", "net", "params"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        if ckpt_obj:
            first_key = next(iter(ckpt_obj.keys()))
            if isinstance(first_key, str) and (
                "weight" in first_key or "bias" in first_key or "." in first_key
            ):
                return ckpt_obj
    raise TypeError("Checkpoint does not contain a recognizable model state dict.")


def load_ckpt_into_model(model: nn.Module, ckpt_path: Path, device: str) -> Any:
    """Load ``ckpt_path`` into ``model`` in place and return the raw checkpoint object.

    Loading is strict=False but any missing/unexpected key is fatal: a silent partial load would
    leave some layers at their fresh init and quietly corrupt every downstream measurement.
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = extract_state(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) != 0 or len(unexpected) != 0:
        raise RuntimeError(
            f"state_dict mismatch for {ckpt_path}\nmissing={missing}\nunexpected={unexpected}"
        )
    return ckpt


def save_model_ckpt(
    path: Path,
    model: nn.Module,
    epoch: int,
    history: dict[str, Any],
    extra: dict | None = None,
) -> None:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "epoch": int(epoch),
        "history": history,
    }
    if extra is not None:
        payload["extra"] = extra
    torch.save(payload, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
