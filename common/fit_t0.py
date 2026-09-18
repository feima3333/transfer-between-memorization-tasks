"""Sigmoid fit of a training-accuracy curve and the t0 speed metric.

This module is the single implementation of the paper's t0 convention. A
memorization curve (train accuracy vs. epoch on CIFAR-10 with random labels) is
summarized by a fixed-floor/ceiling logistic

    y = b + l * sigmoid((t - t0) / s)

whose inflection epoch ``t0`` is the paper's headline "how fast did it fit"
number. The floor ``b`` and span ``l`` are fixed rather than fitted, so that t0
is comparable across every run: only the location t0 and the scale s are free.

It defines t0 for Figure 1 (the t0-definition panel) and produces the t0 used by
every downstream "t0 vs. pre-training epoch" figure. Kept as a pure numerical
tool: it depends only on numpy/torch and never imports the model or data code,
so any notebook can fit a curve it loaded from a ``metrics.json``.

Fit convention (see :func:`fit_t0`): Adam, lr 0.03, 5000 steps, over epochs
1..N with train accuracy expressed as a fraction in [0, 1] (never percent).
"""
from dataclasses import dataclass

import numpy as np
import torch

from common.config import NUM_CLASSES

# Floor and span are FIXED, never fitted, so t0 and s mean the same thing in
# every run and stay comparable across the whole paper. The floor is chance
# accuracy (1/NUM_CLASSES = 0.1 for CIFAR-10) and the span carries the curve the
# rest of the way to perfect memorization, so b + l == 1.0.
SIG_B = 1.0 / NUM_CLASSES
SIG_L = 1.0 - SIG_B

# Adam settings for the two-parameter (t0, s) fit; long/small enough to converge
# smoothly on a ~50-point curve. The fit draws no random numbers, so these are
# the only knobs and the result is deterministic.
FIT_STEPS = 5000
FIT_LR = 0.03


def sigmoid_curve(epochs, t0, s, b=SIG_B, l=SIG_L):
    """Evaluate y = b + l*sigmoid((t - t0)/s) on ``epochs`` (numpy, any shape).

    Kept separate from the fit so that plotting and residual code share one
    formula: Figure 1 draws the fitted curve on a dense grid with this, and the
    fit itself uses it to score the result.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    return b + l / (1.0 + np.exp(-(epochs - t0) / s))


def estimate_s_init(epochs, train_acc, b=SIG_B, l=SIG_L):
    """Data-driven starting value for the logistic scale s.

    A good s0 keeps Adam away from the flat regions where the (t0, s) loss
    barely changes. We measure the width in epochs between the 10% and 90%
    points of the *normalized* curve (y mapped from [b, b + l] onto [0, 1]); for
    a logistic that rise spans exactly ``2*ln(9)*s``, so we invert it to get
    ``s0 = width / (2*ln(9))``. If the curve never spreads out (nearly flat, or
    too few points) we fall back to a fifth of the epoch range.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    train_acc = np.asarray(train_acc, dtype=np.float64)
    y_norm = np.clip((train_acc - b) / max(l, 1e-8), 1e-4, 1.0 - 1e-4)
    lo = int(np.argmin(np.abs(y_norm - 0.1)))
    hi = int(np.argmin(np.abs(y_norm - 0.9)))
    width = float(abs(epochs[hi] - epochs[lo]))
    if width > 0:
        return max(width / (2.0 * np.log(9.0)), 1.0)
    return max(float(epochs.max() - epochs.min()) / 5.0, 1.0)


@dataclass(frozen=True)
class T0Fit:
    """Result of one sigmoid fit; ``t0`` is the headline speed metric.

    Attributes:
        t0: Inflection epoch of the fitted logistic (the paper's t0).
        s: Logistic scale; larger s means a slower, gentler transition.
        b: The fixed floor the fit used (echoed back for the caller).
        l: The fixed span the fit used (echoed back for the caller).
        mse: Mean squared error of the fit on the input points.
        r2: Coefficient of determination against the total variance of y.
    """

    t0: float
    s: float
    b: float
    l: float
    mse: float
    r2: float

    @property
    def midpoint_acc(self):
        """Accuracy at t0 (b + l/2); Figure 1 marks the red t0 dot here."""
        return self.b + 0.5 * self.l

    def curve(self, epochs):
        """The fitted accuracy curve on ``epochs`` (e.g. a dense plotting grid)."""
        return sigmoid_curve(epochs, self.t0, self.s, self.b, self.l)

    def residuals(self, epochs, train_acc):
        """Data minus fit.

        Figure 1 quotes max|residual| to make explicit that t0 is a summary
        statistic, not a claim that the curve *is* a logistic (the fit sits on
        its floor while the data already creeps up, and it undershoots the tail).
        """
        return np.asarray(train_acc, dtype=np.float64) - self.curve(epochs)


def fit_t0(epochs, train_acc, b=SIG_B, l=SIG_L, steps=FIT_STEPS, lr=FIT_LR):
    """Fit y = b + l*sigmoid((t - t0)/s) to a training-accuracy curve.

    Only t0 (inflection) and s (scale) are free; b and l stay fixed. ``epochs``
    and ``train_acc`` are 1-D array-likes over epochs 1..N, with ``train_acc`` a
    fraction in [0, 1]. Returns a :class:`T0Fit`; read ``.t0`` for the metric
    and use ``.curve(...)`` to draw the fitted line.

    The fit runs on CPU (two parameters, ~50 points, trivially cheap) and is
    deterministic: t0 starts at the epoch nearest the midpoint, s at
    :func:`estimate_s_init`, and nothing is sampled, so no seeding is needed.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    train_acc = np.asarray(train_acc, dtype=np.float64)

    t = torch.as_tensor(epochs, dtype=torch.float32)
    y = torch.as_tensor(train_acc, dtype=torch.float32)

    t0_init = t[torch.argmin(torch.abs(y - (b + 0.5 * l)))]
    s0 = estimate_s_init(epochs, train_acc, b, l)

    # s is optimized in log space so it stays strictly positive without a hard
    # constraint; t0 is a plain epoch coordinate.
    raw_log_s = torch.nn.Parameter(torch.log(torch.tensor(float(s0))))
    raw_t0 = torch.nn.Parameter(t0_init.clone())
    opt = torch.optim.Adam([raw_log_s, raw_t0], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        s = torch.exp(raw_log_s).clamp(1e-4, 1e4)
        pred = b + l * torch.sigmoid((t - raw_t0) / s)
        loss = torch.mean((pred - y) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_log_s, raw_t0], 10.0)
        opt.step()

    with torch.no_grad():
        s = float(torch.exp(raw_log_s).clamp(1e-4, 1e4))
        t0 = float(raw_t0)

    y_hat = sigmoid_curve(epochs, t0, s, b, l)
    ss_res = float(np.sum((train_acc - y_hat) ** 2))
    ss_tot = float(np.sum((train_acc - train_acc.mean()) ** 2))
    return T0Fit(
        t0=t0,
        s=s,
        b=b,
        l=l,
        mse=float(np.mean((y_hat - train_acc) ** 2)),
        r2=1.0 - ss_res / ss_tot,
    )


def crossing_epoch(epochs, train_acc, level):
    """First epoch at which train accuracy reaches ``level`` (a fraction).

    An empirical threshold crossing on the raw curve -- the definition the
    paper's speed tables use for t25/t75 -- read off the data, not the sigmoid
    fit. Returns ``None`` if the level is never reached.
    """
    epochs = np.asarray(epochs, dtype=np.float64)
    train_acc = np.asarray(train_acc, dtype=np.float64)
    hit = np.nonzero(train_acc >= level)[0]
    return float(epochs[hit[0]]) if hit.size else None


def speed_metrics(epochs, train_acc, b=SIG_B, l=SIG_L, steps=FIT_STEPS, lr=FIT_LR):
    """The paper's speed-table row for one curve: t25, t0, t75, ES, s.

    t0 and s come from the sigmoid fit (:func:`fit_t0`); t25/t75 are empirical
    crossings of 25%/75% train accuracy (:func:`crossing_epoch`); ES ("early
    stop") is the last recorded epoch. Grouped here because downstream tables
    report all five together -- the Figure 1 definition panel itself uses only
    t0 and s.
    """
    fit = fit_t0(epochs, train_acc, b, l, steps, lr)
    epochs = np.asarray(epochs, dtype=np.float64)
    return {
        "t25": crossing_epoch(epochs, train_acc, 0.25),
        "t0": fit.t0,
        "t75": crossing_epoch(epochs, train_acc, 0.75),
        "es": float(epochs[-1]),
        "s": fit.s,
    }
