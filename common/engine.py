"""Training engine shared by train.py and the surgery scripts.

``train_one_epoch`` / ``evaluate`` are the inner loops; ``fit_model`` runs a model to the
early-stop accuracy, writing the checkpoint trajectory (``epoch{e:04d}.pth``) plus the
``metrics.json`` a run is judged by. There is no data augmentation and, by the paper's
convention, no lr schedule -- so batch order is the only run-to-run variation (see
:func:`common.data.build_train_loader`).
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common.checkpoints import save_model_ckpt, write_json
from common.config import CHECKPOINT_DIR, FIGURE_DATA_DIR
from common.data import build_train_loader, make_train_eval_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One optimization pass; returns (mean loss, accuracy) over the epoch."""
    model.train()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Loss and accuracy of ``model`` over ``loader`` (no grad, eval mode)."""
    model.eval()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == y).sum().item()
        total += bs
    return total_loss / total, total_correct / total


def fit_model(model, train_ds, test_ds, *, seed, ckpt_dir, device, config,
              batch_size=128, num_workers=0, lr=0.01, momentum=0.9, weight_decay=0.0,
              max_epochs=1000, early_stop_acc=0.999, ckpt_every=5,
              resample=None, seed_schedule=None):
    """Fit ``model`` on ``train_ds`` to ``early_stop_acc`` train accuracy.

    ``test_ds=None`` skips test evaluation -- used by the D1/D2 split stages, which have no test
    set. Writes ``metrics.json`` at ``ckpt_dir`` and the checkpoint trajectory under
    ``ckpt_dir/checkpoints/``: epoch0000 (init) + every ``ckpt_every`` epochs + the stop epoch as
    ``epoch{e:04d}.pth``, plus a final ``last.pth``.

    ``resample`` drives the two resampling tasks: "labels" redraws random labels each epoch,
    "images" redraws random pixels. ``seed_schedule[epoch-1]`` gives that epoch's seed (for
    "labels" a None schedule falls back to an os.urandom seed, which is recorded per epoch).
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_sub = ckpt_dir / "checkpoints"
    ckpt_sub.mkdir(parents=True, exist_ok=True)
    # metrics.json goes to the committed figure_data/ tree; the weights stay in checkpoints/ (local).
    try:
        metrics_path = FIGURE_DATA_DIR / ckpt_dir.relative_to(CHECKPOINT_DIR) / "metrics.json"
    except ValueError:
        metrics_path = ckpt_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pin = device.type == "cuda"
    train_loader = build_train_loader(train_ds, seed, batch_size, num_workers, pin)
    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                                weight_decay=weight_decay)

    history = {"config": dict(config), "epochs": []}

    # epoch-0 init snapshot; read the training set through the eval-only view so this pass does
    # not consume a shuffle permutation and shift every later epoch (see make_train_eval_loader).
    init_loss, init_acc = evaluate(model, make_train_eval_loader(train_loader), criterion, device)
    history["init_eval"] = {"train_loss": init_loss, "train_acc": init_acc}
    save_model_ckpt(ckpt_sub / "epoch0000.pth", model, 0, history)

    best = 0.0
    started = time.time()
    for epoch in range(1, max_epochs + 1):
        row = {"epoch": epoch}
        if resample == "labels":
            sd = seed_schedule[epoch - 1] if seed_schedule is not None else None
            row["train_label_seed"] = train_ds.resample_labels(label_seed=sd)
        elif resample == "images":
            row["train_data_seed"] = train_ds.resample_images(seed_schedule[epoch - 1])

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        row.update(train_loss=train_loss, train_acc=train_acc)
        msg = f"epoch {epoch:04d} | train_acc {train_acc:.4f}"
        if test_loader is not None:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            row.update(test_loss=test_loss, test_acc=test_acc)
            msg += f" | test_acc {test_acc:.4f}"
        history["epochs"].append(row)
        best = max(best, train_acc)
        print(msg)

        stop = train_acc >= early_stop_acc
        if stop or epoch % ckpt_every == 0:
            save_model_ckpt(ckpt_sub / f"epoch{epoch:04d}.pth", model, epoch, history)
        if stop:
            print(f"early stop at epoch {epoch} (train_acc {train_acc:.4f})")
            break

    last_epoch = history["epochs"][-1]["epoch"]
    save_model_ckpt(ckpt_sub / "last.pth", model, last_epoch, history)
    write_json(metrics_path, history)
    print(f"done in {(time.time() - started) / 60:.1f} min | best train_acc {best:.4f} | -> {ckpt_dir}")
    return history
