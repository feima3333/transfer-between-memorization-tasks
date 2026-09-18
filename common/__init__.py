"""Shared config and measurement instruments for the paper's experiments.

- ``common.config``    — paths (CHECKPOINT_DIR / OUTPUT_DIR / DATA_DIR) and constants
- ``common.model``     — the 6-conv + FC WideCNN_BN
- ``common.checkpoints`` — checkpoint + JSON IO shared by training and the surgery scripts
- ``common.data``      — CIFAR-10 with random labels / random pixels / resampling variants
- ``common.engine``    — training loop: one epoch, evaluate, and fit-to-convergence + checkpoints
- ``common.fit_t0``    — sigmoid fit of the training-accuracy curve and the inflection point t0
- ``common.alignment`` — cross-covariance Procrustes alignment applied to the next layer
- ``common.svd_ops``   — the U / (Sigma, V) replacement operations used in Figure 5
"""
