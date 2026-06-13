"""
Exp B — Silent collapse of a co-evolving loop, and why dynamic sampling only delays it.

A solver is trained by GRPO against a verifier that supplies the reward. As the verifier
homogenizes over iterations (it stops disagreeing — emits near-identical rewards across a
group), intra-group reward variance decays. We log two families of metrics:

  UPPER (what people plot, lagging):  loss, mean reward ("quality"), a convergence score.
  LOWER (leading):  intra-group reward variance, effective-sample fraction,
                    mean |advantage|, cumulative lora_B weight movement.

Prediction: the upper metrics stay healthy-looking while the lower metrics decay to zero,
and lora_B stops moving. With --dynamic-sampling (DAPO), degenerate groups are resampled/
dropped, which DELAYS the effective-fraction collapse but cannot prevent it once the verifier
is fully homogenized — you cannot oversample your way out of a verifier that no longer disagrees.

SMOKE (CPU, default): the verifier homogenization is *modeled* by a decaying reward spread,
so the silent pattern is demonstrable in seconds. The Colab/paper version drives SRAG-V's real
verification_generator player + RoleConditionedRewardSystem (see COLAB.md).
"""
import argparse
import json
import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.optim import AdamW

import _load
import model_utils as mu


def verifier_rewards(rng, G, spread, mean=0.6, quant=2):
    """Model a verifier emitting G rewards with a given spread, quantized so that as the
    verifier homogenizes (spread->0) the rewards become *exactly* equal (zero variance)."""
    r = mean + rng.normal(0.0, spread, size=G)
    r = np.clip(np.round(r, quant), 0.0, 1.0)
    return r


def maybe_dynamic_sample(rng, G, spread, mean, enabled, thresh=1e-8, max_tries=8):
    """DAPO-style: if a group has ~zero reward variance, resample up to max_tries seeking
    nonzero variance. Returns (rewards, used) where used=False means the group was dropped."""
    r = verifier_rewards(rng, G, spread, mean)
    if not enabled:
        return r, True
    tries = 0
    while np.var(r) < thresh and tries < max_tries:
        r = verifier_rewards(rng, G, spread, mean)
        tries += 1
    return r, bool(np.var(r) >= thresh)


def run(iters, groups, G, spread0, decay, dynamic, seed, device="cpu"):
    hp = _load.srag_lora_hparams()
    model = mu.attach_lora(mu.build_tiny_model(seed=seed, vocab=256), hp).to(device)
    rng = np.random.default_rng(seed)
    g = torch.Generator().manual_seed(seed)
    seqs = [(torch.randint(1, 256, (1, 20), generator=g), 8) for _ in range(G)]

    init_B = mu.clone_state(mu.lora_named(model, "lora_B"))
    rows = []
    prev_quality = None
    for t in range(iters):
        spread = spread0 * math.exp(-decay * t)
        opt = AdamW([p for _, p in mu.lora_named(model)], lr=hp["lr"], weight_decay=0.01)
        opt.zero_grad()

        var_list, absadv_list, q_list, used = [], [], [], 0
        loss_val = 0.0
        for _ in range(groups):
            rewards, ok = maybe_dynamic_sample(rng, G, spread, 0.6, dynamic)
            var_list.append(float(np.var(rewards)))
            q_list.append(float(np.mean(rewards)))
            if not ok:
                continue  # dropped degenerate group (dynamic sampling)
            used += 1
            adv = _load.srag_group_advantages(list(rewards), [1.0] * G)
            absadv_list.append(float(np.mean(np.abs(adv))))
            for (ids, pl), a in zip(seqs, adv):
                lp = mu.logprob_from_ids(model, ids.to(device), pl)
                loss = -(torch.tensor(float(a), device=device, dtype=lp.dtype) * lp) / (G * groups)
                loss.backward()
                loss_val += float(loss.item())

        gnB = mu.grad_l2(mu.lora_named(model, "lora_B"))
        opt.step()
        cum_dB = mu.delta_l2(mu.lora_named(model, "lora_B"), init_B)

        quality = float(np.mean(q_list))
        convergence = 1.0 if prev_quality is None else 1.0 / (1.0 + abs(quality - prev_quality))
        prev_quality = quality
        rows.append({
            "iter": t,
            "verifier_spread": round(spread, 5),
            # UPPER (lagging, look healthy)
            "loss": loss_val,
            "quality_mean_reward": quality,
            "convergence": round(convergence, 4),
            # LOWER (leading)
            "reward_variance_mean": float(np.mean(var_list)),
            "effective_fraction": round(used / groups, 4),
            "mean_abs_advantage": float(np.mean(absadv_list)) if absadv_list else 0.0,
            "grad_norm_loraB": gnB,
            "cum_weight_delta_loraB": cum_dB,
        })
    return rows, hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--spread0", type=float, default=0.25)
    ap.add_argument("--decay", type=float, default=0.30)
    ap.add_argument("--dynamic-sampling", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/srag_probe/out/exp_b")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    rows, hp = run(args.iters, args.groups, args.group_size, args.spread0, args.decay,
                   args.dynamic_sampling, args.seed)
    tag = "_dapo" if args.dynamic_sampling else ""
    out = pathlib.Path(args.out + tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".json", "w") as f:
        json.dump({"hparams": hp, "dynamic_sampling": args.dynamic_sampling, "rows": rows}, f, indent=2)

    print(f"dynamic_sampling={args.dynamic_sampling}")
    print(f"{'it':>3} {'spread':>7} | {'loss':>8} {'quality':>8} {'conv':>6} | "
          f"{'rvar':>8} {'eff_frac':>8} {'|adv|':>7} {'gradB':>10} {'cumΔB':>10}")
    for r in rows:
        print(f"{r['iter']:>3} {r['verifier_spread']:>7.4f} | {r['loss']:>8.4f} "
              f"{r['quality_mean_reward']:>8.4f} {r['convergence']:>6.3f} | "
              f"{r['reward_variance_mean']:>8.5f} {r['effective_fraction']:>8.3f} "
              f"{r['mean_abs_advantage']:>7.3f} {r['grad_norm_loraB']:>10.3e} {r['cum_weight_delta_loraB']:>10.3e}")
    # headline: when did the gradient die vs when did loss/quality notice?
    dead_iter = next((r["iter"] for r in rows if r["grad_norm_loraB"] == 0.0), None)
    print(f"\nGradient first EXACTLY zero at iter: {dead_iter}")
    print(f"Quality at that iter still: {next((r['quality_mean_reward'] for r in rows if r['iter']==dead_iter), None)}"
          f" | convergence still: {next((r['convergence'] for r in rows if r['iter']==dead_iter), None)}")
    print(f"wrote {out}.json")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            x = [r["iter"] for r in rows]
            fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            ax[0].plot(x, [r["loss"] for r in rows], label="loss")
            ax[0].plot(x, [r["quality_mean_reward"] for r in rows], label="quality (mean reward)")
            ax[0].plot(x, [r["convergence"] for r in rows], label="convergence")
            ax[0].set_title("UPPER: what people plot — looks healthy throughout"); ax[0].legend()
            ax[1].plot(x, [r["reward_variance_mean"] for r in rows], label="reward variance")
            ax[1].plot(x, [r["effective_fraction"] for r in rows], label="effective-sample fraction")
            ax[1].plot(x, [r["grad_norm_loraB"] for r in rows], label="grad-norm lora_B")
            ax[1].set_title("LOWER: leading indicators — decay to zero")
            ax[1].set_xlabel("iteration"); ax[1].legend()
            fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=150)
            print(f"wrote {out}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
