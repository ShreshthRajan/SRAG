"""
Calibration drift as a self-improvement-loop health signal.

The original STAR run's Phase-1 achieved ECE 0.00026, then the self-improvement phase
degraded it ~2000x (->0.63) while reporting success. Calibration drift is therefore a cheap,
interpretable witness that a loop is degrading the model even when its own reward looks fine.

confidence(solution) = exp(mean token log-prob) under the solver  (the model's own likelihood)
correctness(solution) = held-out pass-rate on GROUND-TRUTH tests
ECE = sum_b (n_b/N) * |mean_confidence_b - mean_correctness_b|  over equal-width bins.
"""
import math
import numpy as np


def expected_calibration_error(confidences, correctness, n_bins=10):
    c = np.asarray(confidences, dtype=np.float64)
    y = np.asarray(correctness, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    N = len(c)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (c > lo) & (c <= hi) if i > 0 else (c >= lo) & (c <= hi)
        if not mask.any():
            continue
        ece += (mask.sum() / N) * abs(c[mask].mean() - y[mask].mean())
    return float(ece)


def confidence_from_logprob(mean_logprob):
    """Map a mean per-token log-prob to a [0,1] confidence (sequence geometric-mean prob)."""
    try:
        return float(math.exp(mean_logprob))
    except OverflowError:
        return 0.0
