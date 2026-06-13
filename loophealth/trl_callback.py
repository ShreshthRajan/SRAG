"""
LoopHealthCallback — drop into trl.GRPOTrainer (or any HF Trainer) to get live loop-health warnings.

Usage:
    from loophealth.trl_callback import LoopHealthCallback
    trainer = GRPOTrainer(..., callbacks=[LoopHealthCallback(heldout_fn=my_eval)])

It reads what TRL logs each step (reward, reward_std, entropy, kl, completions) and optionally calls a
user `heldout_fn()` (returns held-out score) and `confidence_fn()` for ECE — feeding the LoopHealthMonitor.
Emits a warning to the logger when a signal crosses its baseline threshold (STALL/COLLAPSE/HACK risk),
typically BEFORE held-out degrades. trl/transformers imported lazily so the package imports without them.
"""
import logging
from .monitor import LoopHealthMonitor

logger = logging.getLogger("loophealth")


def _make_base():
    try:
        from transformers import TrainerCallback
        return TrainerCallback
    except Exception:
        class _Stub:  # allows import without transformers; instantiation will warn
            pass
        return _Stub


class LoopHealthCallback(_make_base()):
    def __init__(self, heldout_fn=None, confidence_fn=None, outputs_key="completions",
                 warn_every=1, **monitor_kwargs):
        self.monitor = LoopHealthMonitor(**monitor_kwargs)
        self.heldout_fn = heldout_fn            # () -> float held-out score (optional)
        self.confidence_fn = confidence_fn      # () -> (confidences, correctness) (optional)
        self.outputs_key = outputs_key
        self.warn_every = warn_every
        self._it = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        # TRL GRPOTrainer commonly logs: reward, reward_std, entropy, kl, completion_length
        reward_std = logs.get("reward_std") or logs.get("rewards/std")
        entropy = logs.get("entropy") or logs.get("policy_entropy")
        loop_reward = logs.get("reward") or logs.get("rewards/mean")
        outputs = logs.get(self.outputs_key)
        heldout = self.heldout_fn() if self.heldout_fn else None
        conf = corr = None
        if self.confidence_fn:
            try:
                conf, corr = self.confidence_fn()
            except Exception:
                pass
        # reward_std -> a 1-group variance proxy; effective-sample needs per-group rewards (not in logs)
        group_rewards = [[0.0, float(reward_std), 2 * float(reward_std)]] if reward_std is not None else None
        panel = self.monitor.update(
            group_rewards=group_rewards, heldout=heldout, confidences=conf, correctness=corr,
            policy_entropy=entropy, outputs=outputs, loop_reward=loop_reward, iteration=self._it,
        )
        self._it += 1
        if panel["warnings"] and (self._it % self.warn_every == 0):
            logger.warning(f"[LoopHealth] iter {panel['iteration']} status={panel['status']} :: "
                           + "; ".join(panel["warnings"]))
        return control

    def summary(self):
        return self.monitor.history
