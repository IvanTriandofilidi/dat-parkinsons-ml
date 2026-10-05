"""Constrained logit blending, matching the final five-branch Colab experiment."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import log_loss

from .contracts import binary_labels, probabilities


def blend(preds, weights, eps=1e-6, output_clip=1e-4):
    p = probabilities(preds)
    w = np.asarray(weights, dtype=float)
    if p.ndim != 2 or w.ndim != 1 or p.shape[1] != len(w):
        raise ValueError("Prediction columns must match the ordered model weights")
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Weights must be finite, nonnegative, with a positive sum")
    if not 0 < eps < 0.5 or not 0 <= output_clip < 0.5:
        raise ValueError("Invalid probability clipping bounds")
    result = expit(logit(np.clip(p, eps, 1 - eps)) @ (w / w.sum()))
    return np.clip(result, output_clip, 1 - output_clip)


def fit_blend(preds, labels):
    """Fit weights only. Metrics on this same matrix are fitting diagnostics."""
    p, y = probabilities(preds), binary_labels(labels)
    if p.ndim != 2 or p.shape[0] != len(y) or p.shape[1] < 1:
        raise ValueError("Expected an N x M prediction matrix matching the labels")
    if len(np.unique(y)) != 2:
        raise ValueError("Weight fitting requires both classes")
    count = p.shape[1]
    result = minimize(
        lambda w: log_loss(y, blend(p, w), labels=[0, 1]),
        np.ones(count) / count,
        method="SLSQP",
        bounds=[(0, 1)] * count,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
    )
    if not result.success:
        raise RuntimeError(f"Blend optimization failed: {result.message}")
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()
