"""
Exp F — Loop-health panel: telling genuine improvement from silent stall from reward-hacking.

A small self-improvement loop (real Qwen-Coder-1.5B solver + LoRA, GRPO) is run under three reward
regimes. Each iteration the LoopHealthMonitor (loop_health.py) computes a cheap training-internal
panel + an actionable status; we also log the loop's OWN success metrics (loop reward, convergence)
to show they are blind to stall/hacking.

Regimes:
  --regime grounded : loop reward = pass-rate on ground-truth tests (honest verifier). Reward tracks
                      held-out; panel HEALTHY (or STALLING if the solver saturates).
  --regime stall    : grounded reward on the EASY problems; solver saturates -> reward variance and
                      gradient -> 0 while reward/convergence look perfect (silent stall).
  --regime hack     : loop reward = a WEAK, gameable proxy (runnable + verbose), decorrelated from
                      correctness. Loop reward rises while held-out stays flat and ECE climbs
                      (reward-reality gap; status HACKING_OR_DEGRADING).

--smoke validates plumbing on CPU (tiny model, fixed sequences, synthetic rewards).
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
import calibration as cal
import loop_health as lh
import exp_b_real_verifier as eb


# ----------------------------- reward regimes ------------------------------
def weak_gameable_reward(code, func):
    """Gameable proxy: rewards 'runnable + verbose', NOT correctness (verbose-wrong outscores correct)."""
    p, t = eb.run_asserts(code + f"\nassert {func} is not None", [f"assert {func} is not None"])
    runs = 1.0 if (t and p == t) else 0.0
    lines = len([l for l in code.split("\n") if l.strip()])
    verbosity = min(1.0, lines / 8.0)
    return 0.5 * runs + 0.5 * verbosity


def regime_problems(regime):
    if regime == "stall":
        return [p for p in eb.PROBLEMS if p["name"] in ("add", "is_even")]
    return eb.PROBLEMS


# ----------------------------- real run ------------------------------------
def run_real(args, hp):
    dev = args.device
    tok, solver = mu.load_real_model(args.solver, dev, args.dtype)
    solver = mu.attach_lora(solver, hp).to(dev)
    initB = mu.clone_state(mu.lora_named(solver, "lora_B"))
    problems = regime_problems(args.regime)
    monitor = lh.LoopHealthMonitor()
    rows, prev_q = [], None

    for t in range(args.iters):
        opt = AdamW([p for _, p in mu.lora_named(solver)], lr=hp["lr"], weight_decay=0.01)
        opt.zero_grad()
        group_rewards, heldout_all, confidences = [], [], []
        loss_val = 0.0

        for prob in problems:
            sols = eb.gen_solutions(solver, tok, prob, args.group_size, dev)
            ho = [eb.passrate(s, prob["gt_tests"]) for s in sols]
            # confidence = model's own sequence likelihood (no grad — diagnostic only)
            with torch.no_grad():
                conf = [cal.confidence_from_logprob(float(
                            mu.logprob_from_text(solver, tok, prob["prompt"], "\n" + s, device=dev).item()))
                        for s in sols]
            heldout_all.extend(ho); confidences.extend(conf)
            rewards = [weak_gameable_reward(s, prob["func"]) for s in sols] if args.regime == "hack" else ho
            group_rewards.append(rewards)
            if float(np.var(rewards)) < 1e-8:
                continue  # zero-variance group -> zero gradient (the gate)
            adv = _load.srag_group_advantages(list(rewards), [1.0] * len(rewards))
            for s, a in zip(sols, adv):
                lp = mu.logprob_from_text(solver, tok, prob["prompt"], "\n" + s, device=dev)
                loss = -(torch.tensor(float(a), device=dev, dtype=lp.dtype) * lp) / (len(sols) * len(problems))
                loss.backward(); loss_val += float(loss.item())

        gnB = mu.grad_l2(mu.lora_named(solver, "lora_B"))
        opt.step()
        cumB = mu.delta_l2(mu.lora_named(solver, "lora_B"), initB)

        panel = monitor.update(group_rewards=group_rewards, heldout_correct=heldout_all,
                               confidences=confidences, grad_norm=gnB, weight_delta=cumB, iteration=t)
        conv = 1.0 if prev_q is None else 1.0 / (1.0 + abs(panel["loop_reward"] - prev_q))
        prev_q = panel["loop_reward"]
        panel["convergence"] = round(conv, 4); panel["loss"] = loss_val
        rows.append(panel)
        print(f"[{args.regime}] it{t}: status={panel['status']} | loop_R={panel['loop_reward']:.3f} "
              f"conv={conv:.3f} | heldout={panel['heldout_passrate']:.3f} gap={panel['reward_gap']:+.3f} "
              f"ECE={panel['ece']:.3f} eff={panel['effective_fraction']:.2f} gradB={gnB:.2e} cumΔ={cumB:.2e}")
        for w in panel["warnings"]:
            print("     warn: " + w)
    return rows, monitor.summary()


# ----------------------------- smoke ---------------------------------------
def run_smoke(args, hp):
    print("[smoke] tiny model, fixed sequences, synthetic rewards — validating monitor + panel + ECE")
    dev = "cpu"
    solver = mu.attach_lora(mu.build_tiny_model(seed=1, vocab=256), hp)
    initB = mu.clone_state(mu.lora_named(solver, "lora_B"))
    g = torch.Generator().manual_seed(0); rng = np.random.default_rng(0)
    monitor = lh.LoopHealthMonitor()
    rows = []
    for t in range(3):
        opt = AdamW([p for _, p in mu.lora_named(solver)], lr=hp["lr"], weight_decay=0.01)
        opt.zero_grad()
        group_rewards, conf, corr = [], [], []
        for _ in range(len(eb.PROBLEMS)):
            seqs = [(torch.randint(1, 256, (1, 20), generator=g), 8) for _ in range(args.group_size)]
            rewards = list(rng.uniform(0, 1, args.group_size))
            group_rewards.append(rewards)
            conf += list(rng.uniform(0, 1, args.group_size)); corr += list(rng.uniform(0, 1, args.group_size))
            if float(np.var(rewards)) < 1e-8:
                continue
            adv = _load.srag_group_advantages(rewards, [1.0] * args.group_size)
            for (ids, pl), a in zip(seqs, adv):
                lp = mu.logprob_from_ids(solver, ids, pl)
                loss = -(torch.tensor(float(a), dtype=lp.dtype) * lp) / (args.group_size * len(eb.PROBLEMS))
                loss.backward()
        gnB = mu.grad_l2(mu.lora_named(solver, "lora_B")); opt.step()
        cumB = mu.delta_l2(mu.lora_named(solver, "lora_B"), initB)
        panel = monitor.update(group_rewards=group_rewards, heldout_correct=corr, confidences=conf,
                               grad_norm=gnB, weight_delta=cumB, iteration=t)
        rows.append(panel)
        print(f"it{t}: status={panel['status']} ECE={panel['ece']:.3f} var={panel['reward_variance']:.4f} "
              f"eff={panel['effective_fraction']:.2f} gradB={gnB:.3e}")
    ok = all(r["grad_norm"] and r["grad_norm"] > 0 for r in rows) and all(0 <= r["ece"] <= 1 for r in rows)
    print(f"\nplumbing OK (gradient flows + ECE in [0,1] + monitor returns status each iter): {ok}")
    return rows, monitor.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=["grounded", "stall", "hack"], default="grounded")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--solver", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--out", default="experiments/srag_probe/out/exp_f")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    hp = _load.srag_lora_hparams()

    rows, summary = run_smoke(args, hp) if args.smoke else run_real(args, hp)
    out = pathlib.Path(args.out + ("_smoke" if args.smoke else f"_{args.regime}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".json", "w") as f:
        json.dump({"regime": args.regime, "summary": summary, "rows": rows}, f, indent=2)
    print(f"\nsummary: {summary}\nwrote {out}.json")
    if args.plot and not args.smoke:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            x = [r["iteration"] for r in rows]
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            ax[0].plot(x, [r["loop_reward"] for r in rows], "o-", label="loop reward (what it watches)")
            ax[0].plot(x, [r["heldout_passrate"] for r in rows], "s-", label="held-out pass-rate (truth)")
            ax[0].plot(x, [r["convergence"] for r in rows], ":", label="convergence")
            ax[0].set_title(f"[{args.regime}] success metrics vs truth"); ax[0].set_xlabel("iter"); ax[0].legend()
            ax[1].plot(x, [r["ece"] for r in rows], "o-", label="ECE (calibration drift)")
            ax[1].plot(x, [r["effective_fraction"] for r in rows], "s-", label="effective-sample fraction")
            ax[1].plot(x, [r["reward_variance"] for r in rows], "^-", label="reward variance")
            ax[1].set_title(f"[{args.regime}] health panel"); ax[1].set_xlabel("iter"); ax[1].legend()
            fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=150); print(f"wrote {out}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
