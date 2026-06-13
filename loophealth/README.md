# LoopHealth

A self-improvement loop can lose its per-prompt exploration while every standard metric still looks fine. Within
each prompt the samples collapse onto one answer and reward variance goes to zero, so the loop stops exploring.
Pooled diversity, held-out pass@1, and mean reward barely move and never flag it. LoopHealth computes the
per-prompt signals that do, at almost no cost, from rollouts the loop already produces. It ships as a library and
a TRL callback.

## What the experiments show

Qwen2.5-Coder-1.5B-Instruct on MBPP, GRPO, single L4. Under a no-ground-truth self-consensus reward, held-out
pass@8 fell from 0.91 to 0.46 and pass@4 from 0.73 to 0.45, while pass@1 stayed at 0.32 and pooled diversity
stayed near 0.8. Per-prompt answer-entropy and unique-answer fraction fell to zero, and they fell first. The
monitor's first warning fires while pass@8 is still 0.91, so stopping there keeps the coverage that the unstopped
loop destroys. A grounded control loop holds pass@k flat. Full write-up and the honest limits are in
[the top-level README](../README.md).

## What is and is not novel

The phenomenon is owned, and the report cites it rather than claiming it: self-consensus collapse to a single
mode (TTRL, SRT), pass@k shrinkage under RLVR (Yue et al., Invisible Leash), the zero-advantage degenerate group
and entropy collapse (DAPO, Cui et al.), and divergence-to-reference governing it (DPH-RL). What is ours is
narrower. The explicit point that pooled diversity and pass@1 stay flat while per-prompt diversity collapses, so
aggregate dashboards are blind to it. The packaging of those per-prompt signals as a free training-time monitor
rather than an objective or an expensive aggregate. And the observation that early-stopping at the monitor's
warning preserves coverage, which makes it a control and not only a diagnostic. This is a measurement and tooling
contribution, not a new mechanism.

## Install

```
pip install -e .            # core dependency is numpy
# extras: .[trl] for the TRL callback, .[experiments] to run the RLVR harness
```

## Use the monitor

Feed it per-prompt groups, not a pooled set.

```python
from loophealth import LoopHealthMonitor
m = LoopHealthMonitor()
for step in range(num_steps):
    panel = m.update(output_groups=rollouts_by_prompt,   # list of per-prompt sample lists
                     group_rewards=rewards_by_prompt,
                     key_fn=normalize_answer,             # maps a sample to an answer key; default is str
                     heldout=heldout_score,               # optional
                     iteration=step)
    if panel["status"] != "HEALTHY":
        print(step, panel["status"], panel["warnings"])
```

The panel carries the per-prompt signals (`perprompt_diversity`, `perprompt_answer_entropy`,
`perprompt_unique_fraction`) next to the pooled `output_diversity` that stays flat and misses the collapse, so
the gap is visible as it opens. For TRL: `from loophealth.trl_callback import LoopHealthCallback`.

Tests: `python loophealth/tests/test_signals.py`. They validate the signals and check that the monitor fires on
per-prompt collapse while pooled diversity stays blind.

## The experiment harness

`coverage.py` runs the RLVR loop, logs per-prompt and pooled signals every step, and evaluates pass@k on a
held-out split scored by ground-truth tests. `coverage_analyze.py` produces the lead-time table, the
early-stop-preserves-coverage number, and the figures. See [the top-level README](../README.md) for the results,
including the no-anchor experiment where removing the ground-truth anchor makes the collapse harmful (pass@8
falls while pass@1 holds).

## Layout

`signals.py` per-prompt and classical signals. `monitor.py` the live monitor. `trl_callback.py` TRL integration.
`coverage.py` the RLVR loop with per-prompt and pooled logging and pass@k eval. `coverage_analyze.py` the
analysis. `regimes.py` and `benchmark.py` an earlier synthetic benchmark. `tests/`, `results/`.

## Status

Signals, monitor, and the batched log-prob path are unit-validated, the batched path matching the per-sample one
to within 5e-7. The KL trust-region fix is validated live: with no KL the loop collapses, with KL it holds. The
headline result is from a single real run and is preliminary: one seed, one scale, eleven held-out problems. The
qualitative result is solid, the precise lead time is not. See [the top-level README](../README.md) for the full account.
