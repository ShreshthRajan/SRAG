# Running the variance-gate experiments on Colab Pro (A100)

The harness is built on SRAG-V's own code (`compute_group_advantages`, `RoleConditionedRewardSystem`,
`GRPOConfig`). It runs on CPU with a tiny model (smoke) and on a Colab A100 with Qwen2.5-1.5B (paper figures).

## 0. Setup cell
```python
!git clone https://github.com/ShreshthRajan/srag.git    # or your repo URL
%cd srag
!pip install -q -r experiments/srag_probe/requirements.txt
```

## 1. Figure 1 — dead verifiers (no GPU, runs anywhere)
```python
!python experiments/srag_probe/fig1_dead_adapters.py --plot
```
Expect: `verification_generator` and `meta_verifier` = 100% lora_B zero (DEAD); `problem_generator` learned.

## 2. Exp A — variance gates the gradient (Qwen, ~2 min on A100)
```python
!python experiments/srag_probe/exp_a_variance_gate.py \
    --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --dtype bf16 \
    --group-size 8 --reward sragv --plot
```
Expect: `grad_norm_B` and `weight_delta_B` EXACTLY 0 at frac=0; nonzero for frac>0; `scale_invariant_advantages = True`; loss flat (uninformative). This is the paper's Exp A figure.

Sanity (synthetic rewards, model-agnostic):
```python
!python experiments/srag_probe/exp_a_variance_gate.py --model Qwen/Qwen2.5-1.5B-Instruct --device cuda --reward synthetic --plot
```

## 3. Exp B — silent collapse + DAPO baseline (~few min on A100; minutes on CPU too)
```python
# naive co-evolving loop
!python experiments/srag_probe/exp_b_silent_collapse.py --plot
# DAPO dynamic-sampling baseline — delays but does not prevent collapse
!python experiments/srag_probe/exp_b_silent_collapse.py --dynamic-sampling --plot
```
Expect: UPPER metrics (loss/quality/convergence) stay healthy; LOWER metrics (reward variance,
effective-sample fraction) decay to 0. DAPO holds effective-fraction up longer, then collapses.
Two silent regimes: exact-degenerate (grad→0) and near-degenerate (1/σ noise amplification).

> NOTE: the smoke models verifier homogenization with a decaying reward spread. For the paper's
> Exp B, replace `verifier_rewards()` with SRAG-V's real `verification_generator` player scoring the
> solver's rollouts via `RoleConditionedRewardSystem` (the homogenization then emerges, not modeled).
> This is the version to run for the figure; the toggle and metrics are identical.

## 4. Exp C — grounding restores variance (the fix arm)
Switch the verifier reward from the homogenizing heuristic to SRAG-V's execution-grounded
`SolutionGeneratorReward._evaluate_correctness` (already in repo; verified to yield reward spread:
correct→0.53, wrong→0.47, garbage→0.0). Expect intra-group reward variance and lora_B gradient to
revive where the heuristic verifier collapsed.

## Checkpointing against preemption
Each script writes JSON after completion; for long Exp B/C paper runs, set `--iters` modestly and
re-run — state is cheap. Keep sessions < the A100 wall clock; results are saved to
`experiments/srag_probe/out/`.
