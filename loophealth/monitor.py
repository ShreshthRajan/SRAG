"""
LoopHealthMonitor — drop-in live monitor for self-improvement / RLVR loops.

Call .update(...) each training iteration with whatever you have (group rewards, held-out score,
confidences+correctness, policy entropy, representation covariance, trainable-weight vector, outputs).
It computes the signal panel, tracks an early baseline, and fires WARNINGS when a signal crosses its
baseline-relative threshold in the trouble direction — flagging STALL / COLLAPSE / REWARD-HACKING
before the held-out metric moves.
"""
import numpy as np
from . import signals as S


class LoopHealthMonitor:
    def __init__(self, baseline_iters=3, drop_frac=0.5, rise_abs=0.15, ece_drift=0.10):
        self.baseline_iters = baseline_iters      # iters to establish per-signal baseline
        self.drop_frac = drop_frac                 # "drop" signal fires if value < drop_frac * baseline
        self.rise_abs = rise_abs                   # "rise" signal fires if value > baseline + rise_abs
        self.ece_drift = ece_drift
        self.history = []                          # list of panel dicts
        self._series = {}                          # signal -> list of values
        self._baseline = {}

    def update(self, *, group_rewards=None, heldout=None, confidences=None, correctness=None,
               policy_entropy=None, repr_cov=None, weight_vec=None, outputs=None, output_groups=None,
               key_fn=None, loop_reward=None, iteration=None):
        it = iteration if iteration is not None else len(self.history)
        panel = {"iteration": it, "heldout": heldout, "loop_reward": loop_reward}

        if group_rewards is not None:
            panel["reward_variance"] = S.reward_variance(group_rewards)
            panel["effective_sample_fraction"] = S.effective_sample_fraction(group_rewards)
        if policy_entropy is not None:
            panel["policy_entropy"] = float(policy_entropy)
        if repr_cov is not None:
            panel["spectral_entropy"] = S.spectral_entropy(repr_cov)
        if confidences is not None and correctness is not None:
            panel["ece"] = S.expected_calibration_error(confidences, correctness)
        if outputs is not None:
            panel["output_diversity"] = S.output_diversity(outputs)        # POOLED across prompts (hides per-prompt collapse)
        if output_groups is not None:
            # PER-PROMPT (within-group) coverage — the signals that catch collapse pooled metrics miss.
            # output_groups: list of per-prompt sample lists. key_fn maps a sample -> hashable answer key.
            kf = key_fn or (lambda x: str(x))
            divs = [S.output_diversity(g) for g in output_groups if len(g) > 1]
            ents = [S.answer_entropy([kf(o) for o in g]) for g in output_groups if g]
            uniq = [S.unique_fraction([kf(o) for o in g]) for g in output_groups if g]
            panel["perprompt_diversity"] = float(np.mean(divs)) if divs else 0.0
            panel["perprompt_answer_entropy"] = float(np.mean(ents)) if ents else 0.0
            panel["perprompt_unique_fraction"] = float(np.mean(uniq)) if uniq else 0.0
        if weight_vec is not None:
            prev = self._series.get("_wvec", [None])[-1]
            panel["weight_delta"] = S.weight_delta(prev, weight_vec) if prev is not None else 0.0
            self._series.setdefault("_wvec", []).append(np.asarray(weight_vec, float))
        if loop_reward is not None and heldout is not None:
            panel["reward_heldout_gap"] = float(loop_reward - heldout)

        # update series + baselines
        for k, v in panel.items():
            if k in ("iteration", "heldout", "loop_reward") or k.startswith("_") or v is None:
                continue
            self._series.setdefault(k, []).append(float(v))
            if k not in self._baseline and len(self._series[k]) >= self.baseline_iters:
                self._baseline[k] = float(np.mean(self._series[k][:self.baseline_iters]))

        # fire warnings
        warnings, status = [], "HEALTHY"
        for k, base in self._baseline.items():
            v = panel.get(k)
            if v is None:
                continue
            direction = S.SIGNAL_DIRECTION.get(k, 0)
            fired = False
            if direction < 0 and base > 0 and v < self.drop_frac * base:
                fired = True
            if direction > 0:
                thr = (base + self.ece_drift) if k == "ece" else (base + self.rise_abs)
                if v > thr:
                    fired = True
            if fired:
                warnings.append(f"{k}={v:.4f} crossed baseline {base:.4f} ({'drop' if direction<0 else 'rise'})")
                if k in ("ece", "reward_heldout_gap"):
                    status = "HACKING_OR_DEGRADING"
                elif status == "HEALTHY":
                    status = "STALL_OR_COLLAPSE_RISK"
        panel["status"] = status
        panel["warnings"] = warnings
        self.history.append(panel)
        return panel

    def fired_iteration(self, signal):
        """First iteration this signal crossed its threshold (or None)."""
        for h in self.history:
            if signal in [w.split("=")[0] for w in h["warnings"]]:
                return h["iteration"]
        return None
