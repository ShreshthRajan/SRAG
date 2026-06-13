"""
Real-model regime runner (Colab) — produces the trajectories that benchmark.py --trajectories consumes.

Runs a small Qwen self-improvement loop under 4 regimes and logs the per-iteration signal series:
  healthy  : execution-grounded reward (held-out should rise)
  stall    : easy problems / competent solver -> reward variance -> 0 (gradient gate)
  collapse : recursive self-training on the model's OWN generated data (no real anchor) -> entropy/
             diversity collapse + held-out drop
  hack     : weak/gameable verifier -> loop reward up, held-out flat, ECE up

Reuses the de-risked srag_probe harness (model_utils, _load, exp_b executor/generation). REAL-MODEL,
GPU — run on Colab. Writes results/trajectories.json keyed by regime with the LoopHealth signal series.

Colab:
  !python -m loophealth.regimes --solver Qwen/Qwen2.5-Coder-1.5B-Instruct --device cuda --iters 30
  !python -m loophealth.benchmark --trajectories loophealth/results/trajectories.json --plot
"""
import argparse
import json
import pathlib
import sys

import numpy as np

# wire in the de-risked harness
_HERE = pathlib.Path(__file__).resolve().parents[1]  # repo root (loophealth/ sits at the root)
sys.path.insert(0, str(_HERE / "experiments" / "srag_probe"))

MBPP_PREAMBLE = "import math\nimport re\nimport collections\nfrom collections import *\nimport itertools\nimport functools\nimport heapq\n"


def load_problems(dataset, n, seed=0):
    """Return problems as {name, func, prompt, gt_tests, preamble}. 'embedded'=toy (saturated for 1.5B);
    'mbpp'=real benchmark with HEADROOM (1.5B coder ~40-55%). Keeps only problems whose reference passes."""
    import exp_b_real_verifier as eb
    if dataset == "embedded":
        return eb.PROBLEMS
    import re as _re, random
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    rng = random.Random(seed); idxs = list(range(len(ds))); rng.shuffle(idxs)
    probs = []
    for i in idxs:
        ex = ds[i]; code = ex["code"]; tests = ex["test_list"]
        m = _re.search(r"def\s+(\w+)\s*\(", code)
        if not m:
            continue
        func = m.group(1)
        sig = _re.search(r"(def\s+" + _re.escape(func) + r"\s*\([^)]*\)\s*:)", code)
        sigline = sig.group(1) if sig else f"def {func}():"
        preamble = MBPP_PREAMBLE + "\n".join(ex.get("test_imports", []) or [])
        if eb.passrate(preamble + "\n" + code, tests) != 1.0:
            continue  # keep only problems with clean executable ground truth
        probs.append({"name": f"mbpp_{ex['task_id']}", "func": func,
                      "prompt": ex["prompt"].strip() + "\n" + sigline + "\n",
                      "gt_tests": tests, "preamble": preamble})
        if len(probs) >= n:
            break
    return probs


def run():
    import torch
    import _load, model_utils as mu, calibration as cal
    import exp_b_real_verifier as eb
    from loophealth import signals as S

    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--dataset", default="mbpp", choices=["mbpp", "embedded"])
    ap.add_argument("--n-problems", type=int, default=12)
    ap.add_argument("--regimes", default="healthy,stall,collapse,hack")
    ap.add_argument("--out", default="loophealth/results/trajectories.json")
    args = ap.parse_args()
    hp = _load.srag_lora_hparams()
    dev = args.device
    traj = {}

    def score(prob, sol):
        return eb.passrate(prob.get("preamble", "") + "\n" + sol, prob["gt_tests"])

    pool = load_problems(args.dataset, args.n_problems)
    print(f"loaded {len(pool)} problems ({args.dataset})")

    for regime in args.regimes.split(","):
        tok, solver = mu.load_real_model(args.solver, dev, args.dtype)
        solver = mu.attach_lora(solver, hp).to(dev)
        from torch.optim import AdamW
        problems = pool[:max(2, len(pool) // 4)] if regime == "stall" else pool
        series = {k: [] for k in ("heldout", "pass_at_1", "pass_at_k", "reward_variance",
                                  "effective_sample_fraction", "policy_entropy", "spectral_entropy",
                                  "output_diversity", "ece", "loop_reward", "reward_heldout_gap")}
        for t in range(args.iters):
            opt = AdamW([p for _, p in mu.lora_named(solver)], lr=hp["lr"], weight_decay=0.01)
            opt.zero_grad()
            groups, heldout_all, conf, ent_all, reprs, outs = [], [], [], [], [], []
            loop_rewards, pass1_all, passk_all = [], [], []
            for prob in problems:
                sols = eb.gen_solutions(solver, tok, prob, args.group_size, dev)
                outs += sols
                scores = [score(prob, s) for s in sols]      # partial pass-rate on ground-truth tests
                heldout_all += scores
                full = [1.0 if x >= 0.999 else 0.0 for x in scores]   # fully-correct (all tests pass)
                pass1_all.append(float(np.mean(full)))                # expected pass@1
                passk_all.append(1.0 if max(full) > 0 else 0.0)       # solved by >=1 of the k samples (coverage)
                conf_s, ent_s = [], []
                with torch.no_grad():
                    for s in sols:
                        lp, ent, hid = mu.probe_from_text(solver, tok, prob["prompt"], "\n" + s, device=dev)
                        conf_s.append(cal.confidence_from_logprob(lp)); ent_s.append(ent); reprs.append(hid)
                conf += conf_s; ent_all += ent_s
                if regime == "hack":
                    # PURELY gameable reward (verbosity), fully decoupled from correctness -> loop reward up, held-out flat
                    rewards = [min(1.0, len([l for l in s.split("\n") if l.strip()]) / 8.0) for s in sols]
                elif regime == "collapse":
                    # recursive self-reward: reinforce the model's OWN high-confidence outputs (no ground anchor)
                    # after a short warmup -> mode amplification -> diversity/entropy/pass@k collapse
                    rewards = scores if t < 2 else conf_s
                else:  # healthy / stall
                    rewards = scores
                groups.append(rewards); loop_rewards += rewards
                v = float(np.var(rewards))
                if v >= 1e-8:
                    adv = _load.srag_group_advantages(list(rewards), [1.0] * len(rewards))
                    for s, a in zip(sols, adv):
                        lp = mu.logprob_from_text(solver, tok, prob["prompt"], "\n" + s, device=dev)
                        (-(torch.tensor(float(a), device=dev, dtype=lp.dtype) * lp) / (len(sols) * len(problems))).backward()
            opt.step()
            R = np.stack(reprs) if len(reprs) > 1 else None
            spec = 0.0
            if R is not None:
                Rc = R - R.mean(0, keepdims=True)
                spec = S.spectral_entropy(Rc @ Rc.T)          # effective rank of representations (SIGMA-style)
            correctness = [1.0 if x >= 0.999 else 0.0 for x in heldout_all]
            series["reward_variance"].append(S.reward_variance(groups))
            series["effective_sample_fraction"].append(S.effective_sample_fraction(groups))
            series["heldout"].append(float(np.mean(heldout_all)))
            series["pass_at_1"].append(float(np.mean(pass1_all)))
            series["pass_at_k"].append(float(np.mean(passk_all)))
            series["loop_reward"].append(float(np.mean(loop_rewards)))
            series["reward_heldout_gap"].append(float(np.mean(loop_rewards)) - float(np.mean(heldout_all)))
            series["ece"].append(cal.expected_calibration_error(conf, correctness))
            series["output_diversity"].append(S.output_diversity(outs))
            series["policy_entropy"].append(float(np.mean(ent_all)) if ent_all else 0.0)
            series["spectral_entropy"].append(float(spec))
            print(f"[{regime}] it{t}: p@1={series['pass_at_1'][-1]:.3f} p@k={series['pass_at_k'][-1]:.3f} "
                  f"rvar={series['reward_variance'][-1]:.4f} ent={series['policy_entropy'][-1]:.3f} "
                  f"spec={series['spectral_entropy'][-1]:.3f} div={series['output_diversity'][-1]:.3f} "
                  f"ece={series['ece'][-1]:.3f} loopR={series['loop_reward'][-1]:.3f}")
        # onset = first iter held-out drops >0.1 below its running max (or None)
        hv = np.array(series["heldout"]); onset = None
        for i in range(3, len(hv)):
            if hv[i] < hv[:i].max() - 0.1:
                onset = i; break
        series["_onset"] = onset
        traj[regime] = {k: (v if k == "_onset" else list(map(float, v))) for k, v in series.items()}
        out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(traj, open(out, "w"), indent=2)   # CHECKPOINT after each regime (Colab-safe)
        print(f"[{regime}] DONE -> wrote {out} ({len(traj)} regime(s) so far)")
        del solver
        if dev == "cuda":
            torch.cuda.empty_cache()

    print(f"all regimes done -> {args.out}")


if __name__ == "__main__":
    run()
