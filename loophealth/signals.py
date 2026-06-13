"""
LoopHealth — early-warning signals for self-improvement / RLVR / self-training loops.

Each signal maps state a training loop already has into a scalar per iteration. Grounded in:
  - intra-group reward variance / effective-sample fraction  (DAPO, arXiv:2503.14476)
  - policy entropy                                           (Cui et al., arXiv:2505.22617)
  - spectral entropy of representation/Gram covariance       (SIGMA, arXiv:2601.03385)
  - calibration drift (ECE of confidence vs correctness)     (fresh for self-improvement; our addition)
  - trainable-weight movement                                (our addition)
  - output diversity (distinct-token / set distance)
  - classical EWS: rolling variance + lag-1 autocorrelation  (Scheffer CSD) — included to SHOW it
    FAILS for the discontinuous collapse regime (Entropy-Collapse, arXiv:2512.12381).

Signals are pure + dependency-light (numpy only) so they unit-test and run anywhere.
"""
import numpy as np


# ---- per-iteration scalar signals -----------------------------------------
def reward_variance(groups):
    """Mean intra-group reward variance over groups (list[list[float]]). Low => no learning signal."""
    vs = [float(np.var(g)) for g in groups if len(g) > 1]
    return float(np.mean(vs)) if vs else 0.0


def effective_sample_fraction(groups, eps=1e-8):
    """Fraction of groups with nonzero reward variance (=> nonzero GRPO advantage)."""
    vs = [float(np.var(g)) for g in groups if len(g) >= 1]
    return float(np.mean([v >= eps for v in vs])) if vs else 0.0


def policy_entropy_from_logits(logits):
    """Mean token predictive entropy from logits (T, V)."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=-1, keepdims=True)
    p = np.exp(logits - m); p /= p.sum(axis=-1, keepdims=True)
    ent = -(p * np.log(p + 1e-12)).sum(axis=-1)
    return float(ent.mean())


def spectral_entropy(matrix):
    """Normalized spectral entropy of a covariance/Gram matrix (SIGMA-style). 1=full rank, ->0=collapse."""
    M = np.asarray(matrix, dtype=np.float64)
    if M.ndim == 1:  # already eigenvalues
        ev = np.clip(M, 0, None)
    else:
        M = 0.5 * (M + M.T)
        ev = np.clip(np.linalg.eigvalsh(M), 0, None)
    s = ev.sum()
    if s <= 0 or len(ev) <= 1:
        return 0.0
    p = ev / s
    H = -(p * np.log(p + 1e-12)).sum()
    return float(H / np.log(len(ev)))


def expected_calibration_error(confidences, correctness, n_bins=10):
    """ECE of model self-confidence vs held-out correctness. Rises when the loop becomes confidently wrong."""
    c = np.asarray(confidences, float); y = np.asarray(correctness, float)
    if len(c) == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1); N = len(c); e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (c > lo) & (c <= hi) if i > 0 else (c >= lo) & (c <= hi)
        if m.any():
            e += (m.sum() / N) * abs(c[m].mean() - y[m].mean())
    return float(e)


def weight_delta(prev_vec, cur_vec):
    """L2 movement of trainable weights between iterations (e.g., flattened lora_B)."""
    a = np.asarray(prev_vec, float); b = np.asarray(cur_vec, float)
    return float(np.linalg.norm(b - a))


def output_diversity(items):
    """Mean pairwise token-set Jaccard distance of generated outputs (collapse => ->0)."""
    toks = [set(str(x).split()) for x in items]
    if len(toks) < 2:
        return 0.0
    d = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            u = len(toks[i] | toks[j])
            d.append(1.0 - (len(toks[i] & toks[j]) / u if u else 1.0))
    return float(np.mean(d))


# ---- per-prompt coverage signals (the leading-indicator thesis) -----------
def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al. 2021): prob >=1 of k sampled (w/o replacement) is correct,
    given c of n samples correct. n=#samples, c=#fully-correct, k<=n."""
    from math import comb
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return float(1.0 - comb(n - c, k) / comb(n, k))


def answer_entropy(keys):
    """Normalized Shannon entropy over distinct answer-keys within ONE prompt's samples.
    1 = all samples distinct & uniform (diverse); 0 = all identical (collapsed). keys = hashable per sample."""
    from collections import Counter
    n = len(keys)
    if n <= 1:
        return 0.0
    p = np.array(list(Counter(keys).values()), dtype=np.float64) / n
    H = -(p * np.log(p + 1e-12)).sum()
    return float(max(0.0, H / np.log(n)))


def unique_fraction(keys):
    """Fraction of distinct answer-keys within one prompt's samples (#distinct / #samples). ->0 = collapse."""
    return float(len(set(keys)) / len(keys)) if keys else 0.0


def mean_per_prompt(fn, groups):
    """Average a within-group signal fn over per-prompt groups (list of per-prompt lists). The FIX for the
    pooling bug: compute diversity/spectral/answer-entropy WITHIN each prompt's samples, then average."""
    vals = [fn(g) for g in groups if len(g) >= 1]
    return float(np.mean(vals)) if vals else 0.0


# ---- classical early-warning signals (to demonstrate they FAIL on discontinuous collapse) ----
def rolling_variance(series, t, window=5):
    s = np.asarray(series[max(0, t - window + 1):t + 1], float)
    return float(np.var(s)) if len(s) > 1 else 0.0


def lag1_autocorr(series, t, window=8):
    s = np.asarray(series[max(0, t - window + 1):t + 1], float)
    if len(s) < 3 or np.std(s) < 1e-12:
        return 0.0
    s = s - s.mean()
    denom = (s[:-1] @ s[:-1])
    return float((s[1:] @ s[:-1]) / denom) if denom > 1e-12 else 0.0


# Registry: name -> (direction, description). direction=-1 means "drop = trouble", +1 means "rise = trouble".
SIGNAL_DIRECTION = {
    "reward_variance": -1, "effective_sample_fraction": -1, "policy_entropy": -1,
    "spectral_entropy": -1, "output_diversity": -1, "weight_delta": -1,
    # per-prompt (within-group) coverage signals — the ones that catch silent collapse the pooled versions miss
    "perprompt_diversity": -1, "perprompt_answer_entropy": -1, "perprompt_unique_fraction": -1,
    "ece": +1, "reward_heldout_gap": +1, "rolling_variance": +1, "lag1_autocorr": +1,
}
