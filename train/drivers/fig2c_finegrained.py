"""Driver: Fig 2(c) fine-grained random-pixel -> random-label transfer.

Train the random-pixel donor with a checkpoint EVERY epoch (0..18, early stop disabled so all 19
exist), then warm-start downstream (random labels, label_seed 42) from each epoch. Downstream runs
land under downstream_randlabel_from_randpixel_finegrained/init{k}, which the figure reads.

    python train/drivers/fig2c_finegrained.py
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "train" / "train.py"
PY = sys.executable

EPOCHS = list(range(0, 19))   # warm-start from every donor epoch 0..18


def run(args):
    print(">>> train.py " + " ".join(args), flush=True)
    subprocess.run([PY, str(TRAIN), *args], check=True)


def main():
    up_run = "upstream_randpixel_finegrained_seed21_labelseed21"
    # 1. random-pixel donor: checkpoint every epoch, 19 epochs, early stop off
    run(["--stage", "upstream", "--task", "randpixel", "--seed", "21", "--label-seed", "21",
         "--data-seed", "2026", "--ckpt-every", "1", "--max-epochs", "19", "--early-stop-acc", "2.0",
         "--run-name", up_run])
    up_ckpts = REPO / "checkpoints" / up_run / "checkpoints"
    # 2. downstream warm-start from each donor epoch (downstream random labels, label_seed 42)
    for k in EPOCHS:
        run(["--stage", "downstream", "--task", "randlabel", "--label-seed", "42",
             "--warmstart", str(up_ckpts / f"epoch{k:04d}.pth"),
             "--run-name", f"downstream_randlabel_from_randpixel_finegrained/init{k:04d}"])
    print("fig2c finegrained done", flush=True)


if __name__ == "__main__":
    main()
