# Localizing Transfer Between Memorization Tasks

Reproduction code for the paper **"Localizing Transfer Between Memorization Tasks."** In this paper, we investigate transfer learning under memorization. This enables us to generate many synthetic tasks like image classification with random labels or random pixels. We identify *equivalent transfer*, where one additional pre-training epoch acts approximately like one fine-tuning epoch and *non-equivalent transfer*, where pre-training on a severely mismatched data distribution, for a short period, can be even more *efficient* than directly training on the downstream task. Also we decompose and localize the transfer effect into two separate effects: a *"trivial"* magnitude-driven transfer (in the last layer) and a *"non-trivial"* structure-driven transfer (in the other layers), which can be partially attributed to the weight covariance.

 
The whole paper — **7 figures + 1 table** — regenerates from this repo with no path edits.

---

## Installation

```bash
pip install -r requirements.txt
```
Tested with **torch 2.9.1 + CUDA 12.6** (Python 3.11); CPU-only also works (training is just slower).
For a CUDA-matched wheel, follow the selector at <https://pytorch.org>.

## Data

CIFAR-10 is downloaded automatically by torchvision into `data/` on first run — no manual step.

> The default source (`www.cs.toronto.edu`) can be slow (~150 KB/s on some networks). If so, download
> `cifar-10-python.tar.gz` once (e.g. from a HuggingFace mirror) and drop it in `data/`; torchvision
> will verify and skip the download.

## Quick start

Everything reads paths from `common/config.py` (relative by default). Figures read the small
`metrics.json` files committed under `figure_data/`, so **you can regenerate every figure without
retraining**:

```bash
# figures -> outputs/*.png   (each notebook is self-contained; a few minutes each, they fit sigmoids)
jupyter nbconvert --to notebook --execute --inplace figures/fig1_t0_definition.ipynb
#   ... likewise fig2..fig5, figA1, figA2

# Table 1 -> printed DataFrame (FC norm growth x23.7 / x14.4)
python tables/norm_growth.py
```

To reproduce the training runs themselves, see **[Reproducing from scratch](#reproducing-from-scratch)**.

## Repository structure

```
common/      shared modules — config (paths), model, data, fit_t0, alignment, svd_ops, checkpoints, engine
train/       train.py (upstream / downstream / split) + surgery_*.py (weight surgery) + drivers/
figures/     one notebook per paper figure (figN_*.ipynb), outputs cleared
tables/      norm_growth.py (Table 1)
figure_data/ committed metrics.json + cache CSVs the figures read (a few MB)
checkpoints/ model weights (*.pth) only — gitignored, local (donor trajectories on the Release)
outputs/     figures land here (gitignored)
data/        CIFAR-10 (gitignored, auto-downloaded)
```

## The 7 figures and 1 table

| Paper item | Entry point | Reads |
|---|---|---|
| Fig 1 — t0 definition | `figures/fig1_t0_definition.ipynb` | randlabel donor metrics |
| Fig 2 — linear transfer (3 panels) | `figures/fig2_linear_transfer.ipynb` | upstream-seed sweep + split sweep + finegrained |
| Fig 3 — resampling | `figures/fig3_resampling.ipynb` | label/pixel resampling |
| Fig 4 — weight ablation | `figures/fig4_weight_ablation.ipynb` | init-reset knockouts + rescale |
| Fig 5 — SVD surgery + alignment | `figures/fig5_svd_alignment.ipynb` | conv-only / keep S&V / keep U |
| Fig A1 — ablation + alignment | `figures/figA1_weight_ablation_aligned.ipynb` | + Procrustes-aligned overlay |
| Fig A2 — ablation (random pixel) | `figures/figA2_weight_ablation_randpixel.ipynb` | random-pixel upstream |
| Table 1 — Frobenius norm growth | `tables/norm_growth.py` | donor checkpoints |

## Reproducing from scratch

Every seed the paper uses is an explicit flag defaulting to the paper's value.

```bash
# upstream donors (Fig1, Table1, and the start of every downstream curve)
python train/train.py --stage upstream --task randlabel --seed 21 --label-seed 21
python train/train.py --stage upstream --task randpixel --seed 21 --label-seed 21 --data-seed 2026

# downstream fine-tune, warm-started from an upstream epoch
python train/train.py --stage downstream --task randlabel --label-seed 42 \
    --warmstart checkpoints/upstream_randlabel_seed21_labelseed21/checkpoints/epoch0040.pth

# Fig 2 panels via drivers (each loops seeds/epochs; grouped run naming under figure_data/)
python train/drivers/fig2a_upstream_sweep.py   # Fig 2a: upstream label seeds 21/7/33/55
python train/drivers/fig2b_split.py            # Fig 2b: split seeds 37/11/29/53
python train/drivers/fig2c_finegrained.py      # Fig 2c: randpixel donor, every epoch 0-18

# weight surgery (Fig 4/5/A1/A2): e.g.
python train/surgery_initreset.py --donor randlabel --condition keep_conv
python train/surgery_svd.py       --donor randlabel --condition reset_sv_svinit_aligned
```

**Seeds used** (all in the paper's config):
- upstream: `seed 21`, `label_seed 21`, random-pixel `data_seed 2026`
- downstream / surgery: `seed 21`, `label_seed 42`
- Fig 2a upstream label-seed sweep: `{21, 7, 33, 55}`
- Fig 2b split seeds: `{37, 11, 29, 53}` (D1 fixed at 35 epochs)
- SVD-surgery fresh reset seed: `12345`

## Reproducibility notes

- **Committed metrics, not weights.** The `metrics.json` files (+ cache CSVs) the figures read live
  in `figure_data/` and are committed (a few MB of JSON); the model weights (`*.pth`, ~30 GB total)
  stay in `checkpoints/`, gitignored. Upstream donor checkpoints are on the GitHub Release for anyone
  who wants to skip upstream training.
- **Cross-machine noise.** A re-trained run lands on a batch order one permutation off the published
  one (data order is the only source of variation — no augmentation, fixed lr). Trends are unaffected;
  an individual t0 moves by ~1 epoch, and Table 1's ratios by a few percent (e.g. FC ×24.3 vs the
  paper's ×23.7). This is expected — the figures sell trends, and the noise floor is quoted where it
  matters.

## License

MIT — see [LICENSE](LICENSE).
