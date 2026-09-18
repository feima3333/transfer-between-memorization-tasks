"""One entry point for every training run behind the paper.

Stages:
  upstream     -- train from a fresh Kaiming init (the donor trajectories)
  downstream   -- warm-start from an upstream checkpoint and fine-tune (the t0-vs-epoch curves)
  split_stage1 -- train on the D1 half of a seeded CIFAR-10 split (Fig 2b, random labels)
  split_stage2 -- warm-start from a split_stage1 checkpoint and fine-tune on the D2 half

Tasks (which training set): randlabel, randpixel, resample_label, resample_pixel, truelabel.
Every seed the paper uses is an explicit flag with the paper's value as its default (upstream
label seed 21, downstream 42, random-pixel data seed 2026, split seeds 37/11/29/53).

Examples:
  python train/train.py --stage upstream   --task randlabel --seed 21 --label-seed 21 --ckpt-every 5
  python train/train.py --stage downstream --task randlabel --label-seed 42 \
      --warmstart checkpoints/upstream_randlabel_seed21_labelseed21/epoch0040.pth
  python train/train.py --stage split_stage1 --task randlabel --split-seed 37 --max-epochs 35 --early-stop-acc 2.0
  python train/train.py --stage split_stage2 --task randlabel --split-seed 37 \
      --warmstart checkpoints/split_stage1_randlabel_splitseed37/epoch0035.pth
"""
import argparse
import sys
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Subset

# make `common` importable when this file is run as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.checkpoints import load_ckpt_into_model
from common.config import CHECKPOINT_DIR, DATA_DIR
from common.data import (CIFAR10RandomLabels, RandomPixelCIFAR10, build_data_seed_schedule,
                         build_split_indices, cifar_transform, pixel_transform, seed_everything)
from common.engine import fit_model
from common.model import WideCNN_BN

TASKS = ("randlabel", "randpixel", "resample_label", "resample_pixel", "truelabel")
# task -> fit_model resample mode (the non-resampling tasks are absent -> None)
RESAMPLE = {"resample_label": "labels", "resample_pixel": "images"}
WARMSTART_STAGES = ("downstream", "split_stage2")
SPLIT_STAGES = ("split_stage1", "split_stage2")


def build_datasets(task, label_seed, data_seed, train_size=50_000, test_size=10_000):
    """(train_ds, test_ds) for a whole-dataset task. Test splits keep the real CIFAR-10 labels."""
    if task in ("randlabel", "resample_label"):
        tf = cifar_transform()
        return (CIFAR10RandomLabels(train=True, transform=tf, label_seed=label_seed),
                CIFAR10RandomLabels(train=False, transform=tf))
    if task in ("randpixel", "resample_pixel"):
        tf = pixel_transform()
        return (RandomPixelCIFAR10(train=True, data_seed=data_seed, label_seed=label_seed,
                                   size=train_size, transform=tf),
                RandomPixelCIFAR10(train=False, data_seed=data_seed + 1, label_seed=label_seed,
                                   size=test_size, transform=tf))
    if task == "truelabel":
        tf = cifar_transform()
        root = str(DATA_DIR)
        return (torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tf),
                torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tf))
    raise ValueError(f"unknown task: {task}")


def build_split_datasets(stage, label_seed, split_seed, d1_size):
    """(train subset, None, d1_idx, d2_idx) for a split stage.

    D1/D2 share one random-label map (drawn before the split); stage 1 trains on D1, stage 2 on
    D2. No test set. Random labels only -- the split control (Fig 2b) is a random-label result.
    """
    full = CIFAR10RandomLabels(train=True, transform=cifar_transform(), label_seed=label_seed)
    d1, d2 = build_split_indices(len(full), d1_size, split_seed)
    idx = d1 if stage == "split_stage1" else d2
    return Subset(full, idx.tolist()), None, d1, d2


def default_run_name(args):
    if args.stage == "downstream":
        # name after the upstream run + the warm-start epoch, e.g.
        # downstream_randlabel_labelseed42_from_upstream_randlabel_seed21_labelseed21_epoch0040
        if args.warmstart:
            src = f"{args.warmstart.parent.parent.name}_{args.warmstart.stem}"
        else:
            src = "warmstart"
        return f"downstream_{args.task}_labelseed{args.label_seed}_from_{src}"
    if args.stage == "split_stage1":
        return f"split_stage1_{args.task}_splitseed{args.split_seed}"
    if args.stage == "split_stage2":
        stem = args.warmstart.parent.name if args.warmstart else "warmstart"
        return f"split_stage2_{args.task}_splitseed{args.split_seed}_from_{stem}"
    return f"upstream_{args.task}_seed{args.seed}_labelseed{args.label_seed}"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train upstream / downstream / split memorization runs.")
    p.add_argument("--stage", required=True,
                   choices=("upstream", "downstream", "split_stage1", "split_stage2"))
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--seed", type=int, default=21, help="init + dataloader shuffle seed")
    p.add_argument("--label-seed", type=int, default=21, help="random-label seed (downstream: 42)")
    p.add_argument("--data-seed", type=int, default=2026, help="random-pixel seed")
    p.add_argument("--split-seed", type=int, default=37, help="D1/D2 split seed (split stages)")
    p.add_argument("--d1-size", type=int, default=25_000, help="size of the D1 half (split stages)")
    p.add_argument("--warmstart", type=Path, default=None,
                   help="downstream / split_stage2: checkpoint to warm-start from")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--max-epochs", type=int, default=1000)
    p.add_argument("--early-stop-acc", type=float, default=0.999,
                   help="stop when train acc reaches this; set > 1.0 to train the full max_epochs")
    p.add_argument("--ckpt-every", type=int, default=5)
    p.add_argument("--run-name", default=None, help="checkpoint sub-directory (auto if omitted)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WideCNN_BN().to(device)
    if args.stage in WARMSTART_STAGES:
        if args.warmstart is None:
            raise SystemExit(f"--warmstart is required for --stage {args.stage}")
        load_ckpt_into_model(model, args.warmstart, str(device))

    run_dir = CHECKPOINT_DIR / (args.run_name or default_run_name(args))
    resample = RESAMPLE.get(args.task)
    seed_schedule = build_data_seed_schedule(args.data_seed, args.max_epochs) if resample else None

    if args.stage in SPLIT_STAGES:
        if args.task != "randlabel":
            raise SystemExit("split stages support --task randlabel only")
        train_ds, test_ds, d1, d2 = build_split_datasets(
            args.stage, args.label_seed, args.split_seed, args.d1_size)
        if args.stage == "split_stage1":
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"split_seed": args.split_seed, "label_seed": args.label_seed,
                        "d1_indices": d1, "d2_indices": d2}, run_dir / "split_indices.pt")
    else:
        train_ds, test_ds = build_datasets(args.task, args.label_seed, args.data_seed)

    # JSON-serializable copy of the run config for metrics.json
    run_cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    fit_model(model, train_ds, test_ds, seed=args.seed, ckpt_dir=run_dir, device=device,
              config=run_cfg, batch_size=args.batch_size, num_workers=args.num_workers,
              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
              max_epochs=args.max_epochs, early_stop_acc=args.early_stop_acc,
              ckpt_every=args.ckpt_every, resample=resample, seed_schedule=seed_schedule)


if __name__ == "__main__":
    main()
