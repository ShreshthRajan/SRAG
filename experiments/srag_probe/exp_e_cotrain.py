"""
Exp E — Do the auxiliary verification agents earn their keep? (4-agent loop, done right)

The original SRAG-V run never trained Players 3 & 4 (a wiring bug: they were never in
player_roles; their adapters saved at lora_B=0). This harness runs the loop *correctly* on
the de-risked path: solver AND verifier are both real Qwen models with trainable LoRA, both
updated by GRPO every iteration, with per-role health instrumentation. We compare:

  --frozen-verifier   : verifier fixed (baseline; reward = pass-rate on its tests)
  --cotrain-verifier  : verifier ALSO trained by GRPO (its real VerificationGeneratorReward)

We measure, per role, per iteration: reward variance, effective-sample fraction, lora_B
grad-norm + cumulative Δ (so we can SEE each role actually train — no silent zeros this time),
and the solver's held-out pass-rate on GROUND-TRUTH tests (the real learning signal, independent
of the verifier's own tests). Honest outcomes:
  - co-training sustains/raises held-out pass-rate  -> auxiliary agent earns its keep.
  - reward variance / gradient collapse while held-out flat -> genuine silent stall (instrument catches it).

--smoke validates the two-model GRPO plumbing on CPU (tiny model, fixed sequences, synthetic rewards):
the point is to prove BOTH roles receive nonzero gradient (the thing the original run failed to do).

Reuses exp_b helpers (PROBLEMS, executor, generation) + SRAG-V advantage + reward system.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.optim import AdamW

import _load
import model_utils as mu
import exp_b_real_verifier as eb


def grpo_update(model, tok, prompts_outputs_rewards, hp, device, do_step=True):
    """One GRPO step over a list of (prompt, [outputs], [rewards]) groups for ONE role.
    Returns (grad_norm_B, mean_reward_variance, effective_fraction, mean_abs_adv)."""
    opt = AdamW([p for _, p in mu.lora_named(model)], lr=hp["lr"], weight_decay=0.01)
    opt.zero_grad()
    variances, absadv, used, total = [], [], 0, 0
    for prompt, outs, rewards in prompts_outputs_rewards:
        total += 1
        v = float(np.var(rewards)) if len(rewards) > 1 else 0.0
        variances.append(v)
        if v < 1e-8:
            continue  # zero-variance group contributes no gradient (the gate)
        used += 1
        adv = _load.srag_group_advantages(list(rewards), [1.0] * len(rewards))
        absadv.append(float(np.mean(np.abs(adv))))
        for out, a in zip(outs, adv):
            if isinstance(out, tuple):  # smoke: (ids, prompt_len)
                lp = mu.logprob_from_ids(model, out[0].to(device), out[1])
            else:
                lp = mu.logprob_from_text(model, tok, prompt, "\n" + out, device=device)
            loss = -(torch.tensor(float(a), device=device, dtype=lp.dtype) * lp) / (len(outs) * max(1, len(prompts_outputs_rewards)))
            loss.backward()
    gnB = mu.grad_l2(mu.lora_named(model, "lora_B"))
    if do_step:
        opt.step()
    return gnB, (float(np.mean(variances)) if variances else 0.0), (used / total if total else 0.0), (float(np.mean(absadv)) if absadv else 0.0)


def run_real(args, hp):
    dev = args.device
    tok_s, solver = mu.load_real_model(args.solver, dev, args.dtype)
    solver = mu.attach_lora(solver, hp).to(dev)
    tok_v, verifier = mu.load_real_model(args.verifier, dev, args.dtype)
    verifier = mu.attach_lora(verifier, hp).to(dev)
    initB_s = mu.clone_state(mu.lora_named(solver, "lora_B"))
    initB_v = mu.clone_state(mu.lora_named(verifier, "lora_B"))
    reward_sys = _load.load_reward_system()
    vfn = reward_sys.VerificationGeneratorReward()

    rows = []
    for t in range(args.iters):
        solver_groups, verifier_groups = [], []
        heldout = []
        for prob in eb.PROBLEMS:
            sols = eb.gen_solutions(solver, tok_s, prob, args.group_size, dev)
            # verifier proposes tests (a group of test-suites)
            test_suites = [eb.gen_verifier_tests(verifier, tok_v, prob, dev) for _ in range(args.group_size)]
            # solver reward = pass-rate on the FIRST verifier suite (the deployed verifier)
            vtests = test_suites[0]
            s_rewards = [eb.passrate(s, vtests) for s in sols]
            solver_groups.append((prob["prompt"], sols, s_rewards))
            # held-out: solver pass-rate on GROUND-TRUTH tests (real signal)
            heldout.append(float(np.mean([eb.passrate(s, prob["gt_tests"]) for s in sols])))
            # verifier reward = VerificationGeneratorReward of each proposed suite
            if args.cotrain_verifier:
                ctx = {"problem": {"question": prob["prompt"]}, "solutions": [{"code": s} for s in sols]}
                v_outs, v_rewards = [], []
                for suite in test_suites:
                    txt = "\n".join(suite)
                    try:
                        m = vfn.compute_reward(txt, ctx); r = float(getattr(m, "final_reward", 0.0))
                    except Exception:
                        r = 0.0
                    v_outs.append(txt); v_rewards.append(r)
                verifier_groups.append((f"Write tests for {prob['func']}", v_outs, v_rewards))

        gnB_s, var_s, eff_s, adv_s = grpo_update(solver, tok_s, solver_groups, hp, dev)
        cumB_s = mu.delta_l2(mu.lora_named(solver, "lora_B"), initB_s)
        if args.cotrain_verifier:
            gnB_v, var_v, eff_v, adv_v = grpo_update(verifier, tok_v, verifier_groups, hp, dev)
            cumB_v = mu.delta_l2(mu.lora_named(verifier, "lora_B"), initB_v)
        else:
            gnB_v = var_v = eff_v = adv_v = cumB_v = 0.0

        ho = float(np.mean(heldout))
        rows.append({"iter": t, "heldout_passrate": ho,
                     "solver": {"reward_var": var_s, "eff_frac": round(eff_s, 3), "grad_B": gnB_s, "cumΔ_B": cumB_s},
                     "verifier": {"reward_var": var_v, "eff_frac": round(eff_v, 3), "grad_B": gnB_v, "cumΔ_B": cumB_v}})
        print(f"iter {t}: heldout={ho:.3f} | SOLVER var={var_s:.4f} eff={eff_s:.2f} gradB={gnB_s:.3e} cumΔ={cumB_s:.3e}"
              f" | VERIFIER var={var_v:.4f} eff={eff_v:.2f} gradB={gnB_v:.3e} cumΔ={cumB_v:.3e}")
    return rows


def run_smoke(args, hp):
    print("[smoke] two tiny LoRA models, fixed sequences, synthetic rewards — proving BOTH roles get gradient")
    dev = "cpu"
    solver = mu.attach_lora(mu.build_tiny_model(seed=1, vocab=256), hp)
    verifier = mu.attach_lora(mu.build_tiny_model(seed=2, vocab=256), hp)
    initB_s = mu.clone_state(mu.lora_named(solver, "lora_B"))
    initB_v = mu.clone_state(mu.lora_named(verifier, "lora_B"))
    g = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    rows = []
    for t in range(3):
        sg, vg = [], []
        for _ in range(len(eb.PROBLEMS)):
            outs = [(torch.randint(1, 256, (1, 20), generator=g), 8) for _ in range(args.group_size)]
            sg.append(("p", outs, list(rng.uniform(0, 1, args.group_size))))
            vouts = [(torch.randint(1, 256, (1, 20), generator=g), 8) for _ in range(args.group_size)]
            vg.append(("p", vouts, list(rng.uniform(0, 1, args.group_size))))
        gnB_s, var_s, eff_s, _ = grpo_update(solver, None, sg, hp, dev)
        cumB_s = mu.delta_l2(mu.lora_named(solver, "lora_B"), initB_s)
        gnB_v, var_v, eff_v, _ = grpo_update(verifier, None, vg, hp, dev)
        cumB_v = mu.delta_l2(mu.lora_named(verifier, "lora_B"), initB_v)
        rows.append({"iter": t, "solver_gradB": gnB_s, "solver_cumΔ": cumB_s, "verifier_gradB": gnB_v, "verifier_cumΔ": cumB_v})
        print(f"iter {t}: SOLVER gradB={gnB_s:.3e} cumΔ={cumB_s:.3e} | VERIFIER gradB={gnB_v:.3e} cumΔ={cumB_v:.3e}")
    ok = all(r["solver_gradB"] > 0 and r["verifier_gradB"] > 0 for r in rows)
    print(f"\nBOTH roles received nonzero gradient every iter: {ok}  "
          f"(this is exactly what the original run FAILED to do for verifier/meta)")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--solver", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--verifier", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--group-size", type=int, default=5)
    ap.add_argument("--cotrain-verifier", action="store_true")
    ap.add_argument("--frozen-verifier", action="store_true")
    ap.add_argument("--out", default="experiments/srag_probe/out/exp_e")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    if args.frozen_verifier:
        args.cotrain_verifier = False
    hp = _load.srag_lora_hparams()

    rows = run_smoke(args, hp) if args.smoke else run_real(args, hp)
    tag = "_cotrain" if args.cotrain_verifier else "_frozen"
    out = pathlib.Path(args.out + ("" if args.smoke else tag))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".json", "w") as f:
        json.dump({"cotrain_verifier": args.cotrain_verifier, "rows": rows}, f, indent=2)
    print(f"wrote {out}.json")
    if args.plot and not args.smoke:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            x = [r["iter"] for r in rows]
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].plot(x, [r["heldout_passrate"] for r in rows], "o-")
            ax[0].set_title("Solver held-out pass-rate (real learning)"); ax[0].set_xlabel("iter")
            ax[1].plot(x, [r["solver"]["reward_var"] for r in rows], label="solver reward var")
            ax[1].plot(x, [r["verifier"]["reward_var"] for r in rows], label="verifier reward var")
            ax[1].plot(x, [r["solver"]["grad_B"] for r in rows], "--", label="solver gradB")
            ax[1].plot(x, [r["verifier"]["grad_B"] for r in rows], "--", label="verifier gradB")
            ax[1].set_title("Per-role health (variance / gradient)"); ax[1].set_xlabel("iter"); ax[1].legend()
            fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=150); print(f"wrote {out}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
