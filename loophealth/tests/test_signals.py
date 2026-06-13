"""Smoke tests for loophealth signals + monitor. Run: python loophealth/tests/test_signals.py  (or pytest)."""
import numpy as np
from loophealth import signals as S
from loophealth import LoopHealthMonitor


def test_pass_at_k():
    assert abs(S.pass_at_k(128, 1, 1) - 1 / 128) < 1e-9
    assert S.pass_at_k(128, 0, 8) == 0.0
    assert S.pass_at_k(8, 8, 8) == 1.0
    assert S.pass_at_k(8, 1, 8) == 1.0  # n-c < k -> certain


def test_answer_signals_collapse():
    assert S.answer_entropy([str(i) for i in range(8)]) > 0.99      # all distinct
    assert S.answer_entropy(["x"] * 8) < 1e-6                       # collapsed
    assert S.unique_fraction(["x"] * 8) == 0.125
    assert S.unique_fraction([str(i) for i in range(8)]) == 1.0


def test_spectral_entropy_rank():
    rs = np.random.RandomState(0)
    A = rs.randn(40, 64); Ac = A - A.mean(0, keepdims=True)
    full = S.spectral_entropy(Ac @ Ac.T)
    d = rs.randn(64); B = np.outer(rs.randn(40), d) + 0.02 * rs.randn(40, 64); Bc = B - B.mean(0, keepdims=True)
    low = S.spectral_entropy(Bc @ Bc.T)
    assert full > 0.8 and low < 0.1                                 # full-rank vs rank-1


def test_monitor_catches_perprompt_not_pooled():
    """The headline behavior: per-prompt diversity collapse fires; pooled (held flat) does not."""
    m = LoopHealthMonitor(baseline_iters=3)
    fired_pp = False
    for t in range(15):
        # per-prompt groups collapse over time (diverse -> identical); pooled set stays diverse across prompts
        if t < 4:
            groups = [[f"p{p}_sol{j}_{t}" for j in range(8)] for p in range(6)]   # all distinct within group
        else:
            groups = [[f"p{p}_MODE"] * 8 for p in range(6)]                        # each prompt collapsed to 1 answer
        pooled = [o for g in groups for o in g]                                    # pooled across prompts (still 6 modes)
        panel = m.update(output_groups=groups, outputs=pooled, group_rewards=[[1.0, 0.0]] * 6, iteration=t)
        if any(w.startswith("perprompt") for w in panel["warnings"]):
            fired_pp = True
    # per-prompt collapse must have fired; pooled_diversity should NOT have fired (6 distinct modes => stays high)
    assert fired_pp, "per-prompt collapse was not detected"
    assert m.fired_iteration("output_diversity") is None, "pooled diversity false-fired (should be blind here)"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"PASS {name}")
    print("ALL TESTS PASS")
