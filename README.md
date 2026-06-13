# LoopHealth

A code model trained with reinforcement learning, given a reward it could satisfy without being correct, threw
away half the problems it could solve while every number on the training dashboard kept improving. The reward
climbed to a perfect score. Held-out accuracy at one attempt held steady. Aggregate output diversity barely
moved. The collapse was invisible to all of them. It showed up in exactly one place, the diversity of answers
within each individual problem, which is free to measure during training from rollouts you already have. That
signal turned over in the first few steps, while the model was still good, early enough to stop the run and keep
the coverage. LoopHealth is that signal, packaged as a drop-in monitor, with the experiment that found it.

## The collapse

The loop is GRPO on coding problems with Qwen2.5-Coder-1.5B. The only departure from a normal run is the reward.
Instead of unit tests, each of a problem's eight sampled solutions is scored by how many of the other seven agree
with it after the code is canonicalized. The model is paid to agree with itself, not to be correct. This is the
minimal version of the reward used by label-free self-training, where the signal is the model's own majority vote
(TTRL, arXiv:2504.16084; SRT, arXiv:2505.21444).

The reward rises smoothly to 1.0, and the model gets there by making all eight attempts on each problem identical,
since agreement is trivial once there is one answer. Measured on held-out problems with real unit tests:

| held-out, ground-truth tests | start | end |
|---|---|---|
| pass@1 | 0.32 | 0.31 |
| pass@8 | 0.91 | 0.46 |

The single best attempt is unchanged, so pass@1 holds. pass@8, the chance that one of eight tries is correct,
falls by half. The model did not get worse at its top answer. It lost the spread of answers that let eight tries
reach problems one try cannot. This is coverage collapse, the same pass@k shrinkage that Yue et al.
(arXiv:2504.13837) and the Invisible Leash paper (arXiv:2507.14843) report, induced here on purpose and watched
as it happens.

## Why the dashboard misses it

Three of the four quantities people track show nothing. The reward is climbing, pass@1 is flat, and pooled
diversity, the variety across all outputs on all problems, stays high because different problems still give
different answers. The collapse is one level down, inside each problem, where the eight attempts have become
identical. Within-prompt answer entropy falls from 0.97 to 0 and the distinct-answer fraction from 0.95 to one in
eight. You see it only by measuring diversity within each problem and averaging across problems. The per-prompt
view sees the failure and the pooled view does not, and that gap is the whole point: the metric most teams log is
the one blind to this.

## Catching it in time

The per-prompt signal turns over in the first few steps, while held-out pass@8 is still near 0.91. The pooled view
does not move for roughly twenty more steps. Stopping at the first per-prompt warning keeps coverage near where it
started; letting the run continue loses half. The signal is a control, not just an after-the-fact diagnosis.

## The cause

Put the unit-test reward back and the effect vanishes: pass@k holds flat over the same horizon. The ground-truth
anchor prevents the consensus fixed point, which is what the proof that verification blocks recursive collapse
predicts (Self-Verification Provably Prevents Model Collapse, NeurIPS 2025). Run the two rewards side by side and
the cause is isolated. The collapse comes from grading the model against itself, not from the optimizer or the
data.

## What is known, and what is new

The collapse itself is established. Self-consensus training drives the policy toward a single, often
prompt-independent answer (TTRL; SRT; RESTRAIN, arXiv:2510.02172). Reinforcement with verifiable rewards can lift
pass@1 while shrinking pass@k (Yue et al.; Invisible Leash). The degenerate group whose samples all agree, and
whose advantage is therefore zero, is named in DAPO (arXiv:2503.14476) and analyzed in Cui et al.
(arXiv:2505.22617); the divergence to a frozen reference sets the rate (DPH-RL, arXiv:2509.07430).

Three things here are new. Pooled diversity and pass@1 both stay flat while per-prompt diversity collapses, so the
standard aggregate dashboards are blind to this failure by construction. The per-prompt signals cost nothing to
compute from rollouts already in hand, which makes them a monitor rather than an objective to optimize or a pass@k
sweep to run. And stopping at the monitor's warning preserves the coverage, which makes the signal actionable
rather than merely descriptive. This is a controlled run on a 1.5B code model, and the effect is large and
one-directional; extending it to more seeds, a larger held-out set, and other scales and domains is a matter of
more runs, not of open problems.

## The tool

```bash
pip install -e .                          # core dependency is numpy; extras: .[trl] .[experiments]
python loophealth/tests/test_signals.py   # includes the per-prompt-vs-pooled detection test
```
```python
from loophealth import LoopHealthMonitor
m = LoopHealthMonitor()
for step in range(num_steps):
    panel = m.update(output_groups=rollouts_by_prompt,   # per-prompt sample lists, not a pooled set
                     group_rewards=rewards_by_prompt, key_fn=normalize_answer, iteration=step)
    if panel["status"] != "HEALTHY":
        print(step, panel["status"], panel["warnings"])
```
The panel reports the per-prompt signals next to their pooled counterparts, so the gap is visible as it opens.
A drop-in TRL `GRPOTrainer` callback is in `loophealth.trl_callback`; tool internals are in
[`loophealth/README.md`](loophealth/README.md).

## Reproduce (single L4 or A100, about four hours)

```bash
pip install -e ".[experiments]"
python -m loophealth.coverage --solver Qwen/Qwen2.5-Coder-1.5B-Instruct --device cuda --dtype bf16 \
    --mode no_anchor --difficulty-band 0.05 0.45 --kl-coef 0.0 --lora-rank 128 --lr 1e-5 \
    --steps 120 --batch-prompts 8 --group-size 8 --n-train 24 --n-eval 16 --eval-k 8 --eval-every 3 \
    --out loophealth/results/coverage.json
python -m loophealth.coverage_analyze --in loophealth/results/coverage.json --plot
# control: the same command with --mode grounded --kl-coef 0.02 --difficulty-band 0.0 1.0
```

## Repo map

- `loophealth/`: the tool. `signals.py`, `monitor.py`, `trl_callback.py`, `tests/`, plus the experiment harness
  `coverage.py` (the RLVR loop with per-prompt and pooled logging and pass@k eval) and `coverage_analyze.py`.
- `experiments/srag_probe/`: the model and RL utilities the experiments reuse (`model_utils.py`, `_load.py`, `calibration.py`).
- `sragv_4agent/`: the original four-agent self-play project (SRAG-V) whose silent collapse motivated this work.
  Its GRPO advantage is reused by the experiments through `_load.py`.

MIT licensed.
