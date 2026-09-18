"""Driver: Fig 2(a) upstream label-seed sweep + downstream warm-start.

For each upstream label_seed in {21, 7, 33, 55}: train the from-scratch randlabel donor (seed 21,
checkpoint every 5 epochs), then warm-start downstream (random labels, label_seed 42) from each of
its epoch checkpoints (0, 5, ..., 40). Downstream runs land under the grouped naming the figure
reads: downstream_randlabel_from_labelseed{ls}/init{k}.

    python train/drivers/fig2a_upstream_sweep.py

Serial (one process saturates the GPU). Each downstream run writes its own metrics.json.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "train" / "train.py"
PY = sys.executable

UPSTREAM_LABEL_SEEDS = [21, 7, 33, 55]   # only the random-label assignment changes; seed stays 21
EPOCHS = list(range(0, 41, 5))           # warm-start from these upstream epochs


def run(args):
    print(">>> train.py " + " ".join(args), flush=True)
    subprocess.run([PY, str(TRAIN), *args], check=True)


def main():
    for ls in UPSTREAM_LABEL_SEEDS:
        up_run = f"upstream_randlabel_seed21_labelseed{ls}"
        # 1. upstream donor, from scratch
        run(["--stage", "upstream", "--task", "randlabel", "--seed", "21", "--label-seed", str(ls),
             "--ckpt-every", "5", "--run-name", up_run])
        up_ckpts = REPO / "checkpoints" / up_run / "checkpoints"
        # 2. downstream warm-start from each upstream epoch (downstream label_seed 42)
        for k in EPOCHS:
            run(["--stage", "downstream", "--task", "randlabel", "--label-seed", "42",
                 "--warmstart", str(up_ckpts / f"epoch{k:04d}.pth"),
                 "--run-name", f"downstream_randlabel_from_labelseed{ls}/init{k:04d}"])
    print("fig2a upstream sweep done", flush=True)


if __name__ == "__main__":
    main()
