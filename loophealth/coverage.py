"""
coverage.py — Per-prompt coverage as a LEADING early-warning indicator of silent sharpening in code RLVR.

Thesis (the unclaimed angle, see PLAN.md / RCA): a grounded RLVR loop raises pass@1 while pass@k (the set of
problems it can still solve) silently shrinks. The cheap PER-PROMPT diversity signals (answer-entropy,
unique-answer-ratio, within-group spectral/diversity) turn down BEFORE pass@k accuracy degrades and BEFORE the
POOLED versions of those same signals move — i.e. the signal everyone logs (pooled) is the one that hides the
collapse; measured per-prompt it is an early-warning light.

This harness trains a small solver with GRPO on code problems and, every --eval-every steps, evaluates pass@k
(k up to 128, temp 0.6, top-p 0.95 per Yue et al. 2504.13837) on a HELD-OUT split. Each train step logs the
per-prompt (within-group) signals AND their pooled counterparts so the lead can be measured directly.

Literature-grade settings (A100): --full-ft, --lr 1e-6, KL off (default), --steps 300+, --eval-k 128.
Run:
  python -m loophealth.coverage --solver Qwen/Qwen2.5-Coder-1.5B-Instruct --device cuda --dtype bf16 \
      --full-ft --lr 1e-6 --steps 300 --batch-prompts 8 --group-size 8 \
      --n-train 40 --n-eval 20 --eval-k 128 --eval-every 15 --mode grounded
"""
import argparse
import json
import pathlib
import random
import re
import sys

import numpy as np
from collections import Counter

_HERE = pathlib.Path(__file__).resolve().parents[1]  # repo root (loophealth/ sits at the root)
sys.path.insert(0, str(_HERE / "experiments" / "srag_probe"))


def normalize_code(s):
    """Structural answer-key for consensus/diversity: canonicalize via AST (ignores comments/whitespace/formatting
    so cosmetic variation doesn't inflate the cluster count and stall the consensus reward), with a whitespace-
    normalized string fallback for unparseable generations."""
    import ast
    s = str(s)
    try:
        return ast.dump(ast.parse(s), annotate_fields=False)
    except Exception:
        return re.sub(r"\s+", " ", re.sub(r"#.*", "", s)).strip()


def screen_difficulty(model, tok, probs, score_full, dev, k=8, lo=0.2, hi=0.6, temp=0.6):
    """One-time pre-screen: keep problems whose BASE pass@1 (frac of k samples fully correct) is in [lo,hi] — the
    MEDIUM-HARD band where the model's plurality is often WRONG, so no-anchor self-consensus pulls AWAY from
    correct (the condition for a HARMFUL collapse, per SRT 2505.21444). Returns the kept problems."""
    import exp_b_real_verifier as eb
    kept = []
    for prob in probs:
        sols = eb.gen_solutions(model, tok, prob, k, dev, temperature=temp)
        p1 = float(np.mean([1.0 if score_full(prob, s) >= 0.999 else 0.0 for s in sols]))
        if lo <= p1 <= hi:
            kept.append(prob)
    return kept


def eval_passk(model, tok, probs, score_full, K, dev, temp, ks=(1, 2, 4, 8, 16, 32, 64, 128)):
    """Held-out eval: K samples/problem at `temp`; unbiased pass@k for each k<=K + per-prompt answer signals."""
    import exp_b_real_verifier as eb
    from . import signals as S
    ks = sorted({k for k in list(ks) + [K] if k <= K})
    p_at = {k: [] for k in ks}
    ent, uniq = [], []
    for prob in probs:
        sols = eb.gen_solutions(model, tok, prob, K, dev, temperature=temp)
        full = [1.0 if score_full(prob, s) >= 0.999 else 0.0 for s in sols]
        c, n = int(sum(full)), len(sols)
        for k in ks:
            p_at[k].append(S.pass_at_k(n, c, k))
        keys = [normalize_code(s) for s in sols]
        ent.append(S.answer_entropy(keys))
        uniq.append(S.unique_fraction(keys))
    out = {f"pass_at_{k}": float(np.mean(p_at[k])) for k in ks}
    out["eval_answer_entropy"] = float(np.mean(ent))
    out["eval_unique_fraction"] = float(np.mean(uniq))
    return out


def run():
    import torch
    from torch.optim import AdamW
    import _load, model_utils as mu, calibration as cal
    import exp_b_real_verifier as eb
    from . import signals as S
    from .regimes import load_problems

    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--full-ft", action="store_true", help="full fine-tune (literature-grade); else high-rank LoRA")
    ap.add_argument("--lora-rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--clip-grad", type=float, default=1.0, help="max grad norm (stability; prevents RL divergence)")
    ap.add_argument("--kl-coef", type=float, default=0.02, help="KL-to-frozen-reference penalty (trust region; the restoring force that prevents collapse). LoRA only.")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-prompts", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--train-temp", type=float, default=1.0)
    ap.add_argument("--eval-temp", type=float, default=0.6)
    ap.add_argument("--eval-k", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=15)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=20)
    ap.add_argument("--mode", default="grounded", choices=["grounded", "hack", "no_anchor"])
    ap.add_argument("--difficulty-band", type=float, nargs=2, default=[0.0, 1.0],
                    help="keep only problems whose BASE pass@1 is in [lo,hi] (medium-hard band). For no_anchor use: 0.2 0.6")
    ap.add_argument("--screen-k", type=int, default=8, help="samples/problem for the one-time difficulty pre-screen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="loophealth/results/coverage.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def score_full(prob, sol):
        return eb.passrate(prob.get("preamble", "") + "\n" + sol, prob["gt_tests"])

    hp = _load.srag_lora_hparams()
    dev = args.device
    tok, model = mu.load_real_model(args.solver, dev, args.dtype)
    if args.full_ft:
        for p in model.parameters():
            p.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        hp = dict(hp); hp["rank"] = args.lora_rank; hp["alpha"] = 2 * args.lora_rank
        model = mu.attach_lora(model, hp).to(dev)
        trainable = [p for _, p in mu.lora_named(model)]
    opt = AdamW(trainable, lr=args.lr, weight_decay=0.0)
    print(f"trainable tensors: {len(trainable)}")

    # difficulty pre-screen (load extra, keep the medium-hard band) — the high-leverage knob for no_anchor
    lo, hi = args.difficulty_band
    screening = (lo > 0.0 or hi < 1.0)
    n_load = 256 if screening else (args.n_train + args.n_eval)  # screen the FULL MBPP test pool to maximize medium-hard yield
    pool = load_problems("mbpp", n_load, seed=args.seed)
    if screening:
        pool = screen_difficulty(model, tok, pool, score_full, dev, k=args.screen_k, lo=lo, hi=hi, temp=args.eval_temp)
        print(f"difficulty screen [{lo},{hi}] base pass@1: kept {len(pool)} / {n_load}")
    if len(pool) < args.n_train + args.n_eval:
        print(f"WARN: only {len(pool)} problems available (< n_train+n_eval={args.n_train + args.n_eval})")
    train_pool = pool[:args.n_train]
    eval_set = pool[args.n_train:args.n_train + args.n_eval] or pool[:args.n_eval]
    print(f"train={len(train_pool)} eval={len(eval_set)} | mode={args.mode} band=[{lo},{hi}] lr={args.lr} kl={args.kl_coef}")

    series = []
    for step in range(args.steps):
        batch = rng.sample(train_pool, min(args.batch_prompts, len(train_pool)))
        opt.zero_grad()
        pp_div, pp_ans, pp_uniq, pp_spec, pp_rvar = [], [], [], [], []
        pool_outs, pool_reprs = [], []
        loop_r, held, kl_track = [], [], []
        for prob in batch:
            sols = eb.gen_solutions(model, tok, prob, args.group_size, dev, temperature=args.train_temp)
            keys = [normalize_code(s) for s in sols]
            held += [score_full(prob, s) for s in sols]   # held metric ALWAYS ground-truth (diagnostic, not the reward)
            if args.mode == "no_anchor":
                # SOFT self-consensus reward — NO ground truth: reward_i = fraction of the group sharing your code.
                # Rewards "agree with your own plurality" -> mode amplification -> collapse. SOFT (fraction, not
                # binary) keeps advantage variance nonzero until fully unanimous (avoids the no-gradient stall).
                cnt = Counter(keys)
                rewards = [cnt[k] / len(sols) for k in keys]
            elif args.mode == "hack":
                rewards = [eb.passrate(prob.get("preamble", "") + "\n" + s, prob["gt_tests"][:1]) for s in sols]
            else:  # grounded
                rewards = [score_full(prob, s) for s in sols]
            loop_r += rewards
            # ONE batched forward: policy logprob (differentiable) + pooled hidden reprs for the whole group
            logp, hid = mu.batched_probe(model, tok, prob["prompt"], sols, dev, want_hidden=True)
            pp_div.append(S.output_diversity(sols))
            pp_ans.append(S.answer_entropy(keys))
            pp_uniq.append(S.unique_fraction(keys))
            Rg = np.asarray(hid); Rgc = Rg - Rg.mean(0, keepdims=True)
            pp_spec.append(S.spectral_entropy(Rgc @ Rgc.T))
            pp_rvar.append(float(np.var(rewards)))
            pool_outs += sols; pool_reprs += list(Rg)
            # ONE batched reference forward (frozen base = LoRA disabled) for the KL trust region — the restoring force
            ref_logp = None
            if args.kl_coef > 0 and not args.full_ft:
                with torch.no_grad(), model.disable_adapter():
                    ref_logp, _ = mu.batched_probe(model, tok, prob["prompt"], sols, dev, want_hidden=False)
            var = float(np.var(rewards))
            # group-relative advantage; zero on degenerate groups, but the KL term still applies so the
            # policy can ESCAPE the all-identical absorbing state (the v3 collapse trap)
            adv = _load.srag_group_advantages(list(rewards), [1.0] * len(rewards)) if var >= 1e-8 else [0.0] * len(sols)
            adv_t = torch.tensor([float(a) for a in adv], device=dev, dtype=logp.dtype)
            loss = -(adv_t * logp).sum()                                  # group-relative PG (/len(batch) below)
            if ref_logp is not None:
                log_ratio = ref_logp.to(logp.dtype) - logp                # ref - policy
                kl = torch.exp(log_ratio) - log_ratio - 1.0               # k3 estimator (>=0)
                loss = loss + args.kl_coef * kl.sum()
                kl_track.append(float(kl.mean().item()))
            (loss / len(batch)).backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
        opt.step()

        Rp = np.stack(pool_reprs); Rpc = Rp - Rp.mean(0, keepdims=True)
        rec = {
            "step": step,
            "perprompt_answer_entropy": float(np.mean(pp_ans)),
            "perprompt_unique_fraction": float(np.mean(pp_uniq)),
            "perprompt_diversity": float(np.mean(pp_div)),
            "perprompt_spectral": float(np.mean(pp_spec)),
            "reward_variance": float(np.mean(pp_rvar)),
            "pooled_diversity": S.output_diversity(pool_outs),         # the signal that HIDES collapse
            "pooled_spectral": S.spectral_entropy(Rpc @ Rpc.T),
            "loop_reward": float(np.mean(loop_r)),
            "heldout_passrate": float(np.mean(held)),
            "grad_norm": float(gnorm),
            "kl": float(np.mean(kl_track)) if kl_track else 0.0,
        }
        if step % args.eval_every == 0 or step == args.steps - 1:
            rec.update(eval_passk(model, tok, eval_set, score_full, args.eval_k, dev, args.eval_temp))
            print(f"[{step}] EVAL p@1={rec.get('pass_at_1', float('nan')):.3f} "
                  f"p@{min(args.eval_k,128)}={rec.get(f'pass_at_{min(args.eval_k,128)}', float('nan')):.3f} "
                  f"| pp_ans_ent={rec['perprompt_answer_entropy']:.3f} pp_uniq={rec['perprompt_unique_fraction']:.3f} "
                  f"pooled_div={rec['pooled_diversity']:.3f} loopR={rec['loop_reward']:.3f}")
        else:
            print(f"[{step}] pp_ans_ent={rec['perprompt_answer_entropy']:.3f} pp_uniq={rec['perprompt_unique_fraction']:.3f} "
                  f"pp_div={rec['perprompt_diversity']:.3f} pooled_div={rec['pooled_diversity']:.3f} "
                  f"rvar={rec['reward_variance']:.4f} loopR={rec['loop_reward']:.3f} kl={rec['kl']:.3f} g={rec['grad_norm']:.1f}")
        series.append(rec)
        out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"args": vars(args), "series": series}, open(out, "w"), indent=2)  # checkpoint every step
    print(f"done -> {args.out}")


if __name__ == "__main__":
    run()
