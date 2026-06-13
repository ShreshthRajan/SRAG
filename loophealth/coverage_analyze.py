"""
Analyze coverage.py output. The headline questions:
  (1) Does grounded RLVR SHARPEN — pass@1 up while pass@k (esp. k=128) flat/down?
  (2) Do the PER-PROMPT signals (answer-entropy, unique-fraction, within-group diversity/spectral) drop
      BEFORE pass@k degrades and BEFORE the POOLED versions move? (the leading-indicator claim)

Outputs a printed lead-time table + two figures.
"""
import argparse
import json
import pathlib

import numpy as np


def first_cross(steps, vals, baseline_n=3, warmup=1, frac=0.10, direction=-1):
    """Step where `vals` first crosses `frac` (relative) past its baseline. direction=-1 drop, +1 rise. None if never."""
    if len(vals) < baseline_n + 1:
        return None
    base = float(np.mean(vals[:baseline_n]))
    if base == 0:
        return None
    for s, v in zip(steps, vals):
        if s < warmup:
            continue
        if direction < 0 and v < base * (1 - frac):
            return s
        if direction > 0 and v > base * (1 + frac):
            return s
    return None


def col(series, key):
    rows = [(r["step"], r[key]) for r in series if key in r and r[key] is not None]
    return [s for s, _ in rows], [v for _, v in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="loophealth/results/coverage.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    d = json.load(open(a.inp)); series = d["series"]
    out = pathlib.Path(a.out or a.inp.replace(".json", "_analysis"))
    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if any(f"pass_at_{k}" in r for r in series)]
    passk_key = f"pass_at_{ks[-1]}" if ks else None

    print("=== pass@k over training (coverage) ===")
    for k in ks:
        s, v = col(series, f"pass_at_{k}")
        if v:
            print(f"  pass_at_{k:<3}: {v[0]:.3f} -> {v[-1]:.3f}   (Δ {v[-1]-v[0]:+.3f})")
    if "pass_at_1" in series[0] or any("pass_at_1" in r for r in series):
        _, p1 = col(series, "pass_at_1"); _, pk = col(series, passk_key)
        if p1 and pk:
            sharp = (p1[-1] - p1[0] > 0.02) and (pk[-1] - pk[0] < 0.02)
            print(f"\nSHARPENING (pass@1 up, pass@{ks[-1]} flat/down): {'YES' if sharp else 'no'} "
                  f"[Δp@1={p1[-1]-p1[0]:+.3f}, Δp@{ks[-1]}={pk[-1]-pk[0]:+.3f}]")

    print("\n=== LEAD analysis: step of first 10% drop (signal) vs pass@k degradation ===")
    pk_steps, pk_vals = col(series, passk_key) if passk_key else ([], [])
    pk_drop = first_cross(pk_steps, pk_vals, direction=-1) if pk_vals else None
    print(f"  {passk_key} first 10% drop at step: {pk_drop}")
    rows = []
    for sg in ("perprompt_answer_entropy", "perprompt_unique_fraction", "perprompt_diversity",
               "perprompt_spectral", "pooled_diversity", "pooled_spectral", "reward_variance"):
        s, v = col(series, sg)
        sd = first_cross(s, v, direction=-1) if v else None
        lead = (pk_drop - sd) if (pk_drop is not None and sd is not None) else None
        rows.append((sg, sd, lead))
        leadstr = (f"+{lead}" if (lead is not None and lead >= 0) else (str(lead) if lead is not None else "n/a"))
        print(f"  {sg:>26}: first drop @ step {str(sd):>5}   lead vs {passk_key} = {leadstr}")
    print("\nRead: a PER-PROMPT signal with a POSITIVE lead that the POOLED version does NOT show is the result —"
          "\nthe pooled signal hides the collapse; the per-prompt one is an early warning.")

    # ACTIONABILITY: the monitor isn't just diagnostic — early-stopping at its first warning PRESERVES coverage.
    print("\n=== actionability: early-stop at the monitor's first per-prompt warning ===")
    ws, wv = col(series, "perprompt_answer_entropy")
    warn = first_cross(ws, wv, direction=-1)
    if warn is not None and passk_key and pk_vals:
        at_warn = next((v for s, v in zip(pk_steps, pk_vals) if s >= warn), pk_vals[-1])
        best = max(pk_vals)
        print(f"  monitor warns @ step {warn}: {passk_key}={at_warn:.3f}  |  final (no stop)={pk_vals[-1]:.3f}"
              f"  |  best-seen={best:.3f}")
        print(f"  => early-stopping at the warning PRESERVES {at_warn - pk_vals[-1]:+.3f} {passk_key} "
              f"that the unstopped loop destroys.")
    else:
        print("  (no per-prompt warning fired, or no pass@k series — nothing to act on)")

    if a.plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
            for k in ks:
                s, v = col(series, f"pass_at_{k}"); ax[0].plot(s, v, marker="o", lw=1.8, label=f"pass@{k}")
            ax[0].set_title("Coverage during RLVR (pass@k)"); ax[0].set_xlabel("step"); ax[0].set_ylim(0, 1); ax[0].legend()
            for sg, style in [("perprompt_answer_entropy", "-"), ("perprompt_unique_fraction", "-"),
                              ("pooled_diversity", "--"), ("pooled_spectral", "--")]:
                s, v = col(series, sg)
                if v:
                    ax[1].plot(s, np.array(v) / (v[0] or 1), style, lw=1.8, label=sg + " (norm)")
            if pk_steps:
                ax[1].plot(pk_steps, np.array(pk_vals) / (pk_vals[0] or 1), "k:", lw=2.2, label=passk_key + " (norm)")
            ax[1].set_title("Per-prompt signals (solid) vs pooled (dashed) vs pass@k (dotted), normalized")
            ax[1].set_xlabel("step"); ax[1].legend(fontsize=7)
            fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=150); print(f"\nwrote {out}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")
    json.dump({"pass_at_k_drop_step": pk_drop, "leads": {r[0]: r[2] for r in rows}},
              open(str(out) + ".json", "w"), indent=2)
    print(f"wrote {out}.json")


if __name__ == "__main__":
    main()
