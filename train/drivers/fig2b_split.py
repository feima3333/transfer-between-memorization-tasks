"""Driver: the Fig 2(b) split-seed sweep runs that are missing from disk (split_seed 11/29/53).

For each split seed: stage 1 trains the D1 half for a fixed 35 epochs (early stop disabled), then
stage 2 warm-starts from each stage-1 epoch checkpoint (0,5,...,35) and fine-tunes the D2 half.
Seeds: seed 21, label_seed 21, d1_size 25000 -- D1/D2 share one random-label map (drawn before the
split). This is exactly what produced the shipped split_seed 37 run.

    python train/drivers/fig2b_split.py

Runs are serial (one process saturates the GPU). Each stage-2 run writes its own metrics.json.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAIN = REPO / "train" / "train.py"
PY = sys.executable

SPLIT_SEEDS = [11, 29, 53]
INIT_EPOCHS = [0, 5, 10, 15, 20, 25, 30, 35]  # stage-1 epochs to warm-start stage 2 from


def run(args):
    print(">>> train.py " + " ".join(args), flush=True)
    subprocess.run([PY, str(TRAIN), *args], check=True)


def main():
    for ss in SPLIT_SEEDS:
        # stage 1: D1 half, fixed 35 epochs (early stop disabled with a >1 threshold)
        run(["--stage", "split_stage1", "--task", "randlabel", "--split-seed", str(ss),
             "--max-epochs", "35", "--early-stop-acc", "2.0", "--ckpt-every", "5"])
        stage1_ckpts = REPO / "checkpoints" / f"split_stage1_randlabel_splitseed{ss}" / "checkpoints"

        # stage 2: warm-start from each stage-1 epoch, fine-tune the D2 half to early stop
        for e in INIT_EPOCHS:
            ckpt = stage1_ckpts / f"epoch{e:04d}.pth"
            run(["--stage", "split_stage2", "--task", "randlabel", "--split-seed", str(ss),
                 "--warmstart", str(ckpt),
                 "--run-name", f"split_stage2_randlabel_splitseed{ss}/init{e:04d}"])
    print("split sweep 11/29/53 done", flush=True)


if __name__ == "__main__":
    main()
