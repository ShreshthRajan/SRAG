"""
LoopHealthMonitor — a cheap, training-internal health panel for self-improvement loops.

Drop into any GRPO/RLVR/self-play training loop. Each step, call `.update(...)` with the group
rewards, held-out correctness, and model self-confidences (plus optional grad-norm / weight-delta of
the trainable params). It returns a panel of near-free signals and an ACTIONABLE status that the
loop's own metrics (loss, mean reward, "convergence") are blind to:

  reward_variance / effective_fraction : is there any learning signal? (zero-variance groups give
                                         zero GRPO gradient — DAPO 2503.14476)
  grad_norm / weight_delta             : are the trainable weights actually moving?
  ece (calibration drift)              : is the loop DEGRADING the model? (the canary)
  reward_gap = loop_reward - heldout   : is reward decoupling from true performance?
                                         (reward-hacking / reward-reality gap — arXiv 2605.02964)

status: HEALTHY | STALLING | HACKING_OR_DEGRADING, with human-readable warnings.

All inputs are things a training loop already has after a rollout, so cost is negligible. The
component mechanisms are established; the contribution is packaging them into one cheap
*leading-indicator* panel — usable at small (1.5B) scale where it fires before standard metrics.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import calibration as cal


class LoopHealthMonitor:
    def __init__(self, var_floor=1e-8, eff_frac_warn=0.5, gap_warn=0.15, ece_drift_warn=0.10,
                 entropy_collapse_frac=0.5, ece_bins=10):
        self.var_floor = var_floor
        self.eff_frac_warn = eff_frac_warn
        self.gap_warn = gap_warn
        self.ece_drift_warn = ece_drift_warn
        self.entropy_collapse_frac = entropy_collapse_frac  # fraction of baseline entropy below which we warn
        self.ece_bins = ece_bins
        self.history = []
        self._baseline_ece = None
        self._baseline_entropy = None

    def update(self, *, group_rewards, heldout_correct, confidences,
               grad_norm=None, weight_delta=None, entropy=None, iteration=None):
        """group_rewards: list[list[float]] (one list per prompt/group);
        heldout_correct, confidences: flat lists over all rollouts (same length)."""
        variances = [float(np.var(g)) for g in group_rewards if len(g) > 0]
        reward_var = float(np.mean(variances)) if variances else 0.0
        eff = (sum(1 for v in variances if v >= self.var_floor) / len(variances)) if variances else 0.0
        flat = [r for g in group_rewards for r in g]
        loop_reward = float(np.mean(flat)) if flat else 0.0
        heldout = float(np.mean(heldout_correct)) if heldout_correct else 0.0
        ece = cal.expected_calibration_error(confidences, heldout_correct, self.ece_bins)
        if self._baseline_ece is None:
            self._baseline_ece = ece
        ece_drift = ece - self._baseline_ece
        gap = loop_reward - heldout

        warnings, status = [], "HEALTHY"
        stalling = (eff < self.eff_frac_warn) or (grad_norm is not None and grad_norm < 1e-6)
        degrading = (gap > self.gap_warn) or (ece_drift > self.ece_drift_warn)
        if stalling:
            status = "STALLING"
            warnings.append(f"effective-sample fraction {eff:.2f} (<{self.eff_frac_warn}) / grad~0: "
                            f"reward variance has collapsed — little or no learning signal.")
        if degrading:  # takes precedence — worse failure
            status = "HACKING_OR_DEGRADING"
            warnings.append(f"reward decoupling from held-out (gap {gap:+.3f}) and/or calibration drift "
                            f"(ECE {ece:.3f} vs baseline {self._baseline_ece:.3f}, +{ece_drift:.3f}): "
                            f"loop 'success' is not real improvement.")

        panel = {
            "iteration": iteration if iteration is not None else len(self.history),
            "loop_reward": loop_reward, "heldout_passrate": heldout, "reward_gap": round(gap, 4),
            "reward_variance": reward_var, "effective_fraction": round(eff, 3),
            "ece": ece, "ece_drift": round(ece_drift, 4),
            "grad_norm": grad_norm, "weight_delta": weight_delta,
            "status": status, "warnings": warnings,
        }
        self.history.append(panel)
        return panel

    def summary(self):
        if not self.history:
            return {}
        statuses = [h["status"] for h in self.history]
        return {
            "iterations": len(self.history),
            "ever_stalled": any(s == "STALLING" for s in statuses),
            "ever_degraded": any(s == "HACKING_OR_DEGRADING" for s in statuses),
            "final_status": statuses[-1],
            "max_reward_gap": max(h["reward_gap"] for h in self.history),
            "max_ece_drift": max(h["ece_drift"] for h in self.history),
        }
