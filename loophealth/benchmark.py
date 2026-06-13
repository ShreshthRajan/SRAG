"""
Head-to-head early-warning benchmark for self-improvement loops.

Question: across failure regimes (silent stall / collapse / reward-hacking), WHICH signal warns earliest
(lead time before the held-out metric degrades), and do classical early-warning signals (rolling variance,
lag-1 autocorrelation) FAIL for the discontinuous collapse regime (per Entropy-Collapse, arXiv:2512.12381)?

--synthetic (default, runs anywhere): regimes whose dynamics are ENCODED FROM THE LITERATURE (DAPO variance
gate; Entropy-Collapse first-order discontinuity with no classical precursors; reward-hacking decoupling +
ECE drift). This VALIDATES the benchmark methodology + reproduces the qualitative structure. It is NOT a
finding — the real quantitative head-to-head comes from real-model regimes (regimes.py, on Colab) fed in via
--trajectories.

Output: a signal x regime LEAD-TIME table + a false-alarm check on the healthy regime + a figure.
"""
import argparse
import json
import pathlib
import numpy as np

from . import signals as S

SIGNALS = ["reward_variance", "effective_sample_fraction", "policy_entropy", "spectral_entropy",
           "output_diversity", "ece", "reward_heldout_gap", "rolling_variance", "lag1_autocorr"]
REGIMES = ["healthy", "stall", "collapse", "hack"]


def synth_regime(name, T=40, seed=0):
    """Per-iteration signal series with literature-grounded dynamics. Returns dict signal->series + onset."""
    rng = np.random.default_rng(seed)
    n = lambda s: rng.normal(0, s, T)
    t = np.arange(T)
    out = {}
    if name == "healthy":
        out["heldout"] = np.clip(0.3 + 0.5 * (1 - np.exp(-t / 8)) + n(0.01), 0, 1)
        out["reward_variance"] = 0.12 + n(0.01)
        out["effective_sample_fraction"] = np.clip(0.9 + n(0.03), 0, 1)
        out["policy_entropy"] = 2.0 + n(0.03)
        out["spectral_entropy"] = 0.8 + n(0.01)
        out["output_diversity"] = 0.6 + n(0.02)
        out["ece"] = 0.05 + np.abs(n(0.01))
        onset = None
    elif name == "stall":  # variance gate: reward variance collapses -> learning stops; held-out plateaus
        plateau = 14
        out["heldout"] = np.clip(np.where(t < plateau, 0.3 + 0.4 * (t / plateau), 0.7) + n(0.008), 0, 1)
        out["reward_variance"] = np.clip(0.12 * np.exp(-t / 6) + n(0.004), 0, None)   # decays early (leads)
        out["effective_sample_fraction"] = np.clip(1.0 - t / 16, 0, 1) + n(0.02)      # decays early (leads)
        out["policy_entropy"] = 2.0 + n(0.03)
        out["spectral_entropy"] = 0.8 + n(0.01)
        out["output_diversity"] = 0.6 + n(0.02)
        out["ece"] = 0.05 + np.abs(n(0.01))
        onset = plateau  # "failure" = improvement stops
    elif name == "collapse":  # first-order: held-out rises then sharp drop at t_c; entropy/spectral/div lead
        tc = 26
        rise = 0.3 + 0.4 * (1 - np.exp(-t / 7))
        out["heldout"] = np.clip(np.where(t < tc, rise, rise[tc] - 0.5 * (t - tc) / (T - tc)) + n(0.01), 0, 1)
        # entropy/spectral/diversity decay GRADUALLY before tc (the lead):
        out["spectral_entropy"] = np.clip(0.8 - 0.5 * np.clip((t - 8) / (tc - 8), 0, 1) + n(0.008), 0, 1)
        out["policy_entropy"] = np.clip(2.0 - 1.2 * np.clip((t - 10) / (tc - 10), 0, 1) + n(0.02), 0, None)
        out["output_diversity"] = np.clip(0.6 - 0.45 * np.clip((t - 9) / (tc - 9), 0, 1) + n(0.01), 0, 1)
        out["reward_variance"] = 0.12 + n(0.01)             # not informative here
        out["effective_sample_fraction"] = np.clip(0.9 + n(0.03), 0, 1)
        out["ece"] = 0.05 + np.abs(n(0.01))
        onset = tc
    elif name == "hack":  # loop reward up, held-out flat-low; ECE + gap rise (lead), others quiet
        out["heldout"] = np.clip(0.35 + n(0.01), 0, 1)
        out["loop_reward"] = np.clip(0.35 + 0.5 * (1 - np.exp(-t / 7)) + n(0.01), 0, 1)
        out["reward_heldout_gap"] = out["loop_reward"] - out["heldout"]
        out["ece"] = np.clip(0.05 + 0.5 * (1 - np.exp(-t / 8)) + np.abs(n(0.01)), 0, 1)  # rises (leads)
        out["reward_variance"] = 0.10 + n(0.01)
        out["effective_sample_fraction"] = np.clip(0.85 + n(0.03), 0, 1)
        out["policy_entropy"] = 2.0 + n(0.03)
        out["spectral_entropy"] = 0.8 + n(0.01)
        out["output_diversity"] = 0.6 + n(0.02)
        onset = 8  # by when the loop "succeeds" falsely (gap clearly opened)
    # classical EWS computed on held-out series (to show they fail on discontinuous collapse)
    hv = out["heldout"]
    out["rolling_variance"] = np.array([S.rolling_variance(hv, i) for i in range(T)])
    out["lag1_autocorr"] = np.array([S.lag1_autocorr(hv, i) for i in range(T)])
    out["_onset"] = onset
    return out


def detect_onset(heldout, warmup=2, drop=0.05, tol=0.02):
    """'Failure onset' = first iter (after warmup) where held-out DROPS below its running max (collapse)
    OR PLATEAUS — stops improving for the rest of the run (stall). None if it keeps improving (healthy)."""
    if heldout is None:
        return None
    h = np.asarray(heldout, float)
    if len(h) < warmup + 2:
        return None
    runmax = np.maximum.accumulate(h)
    late = 0.6 * (len(h) - 1)   # a plateau in the LAST 40% is healthy diminishing-returns, not a stall
    for t in range(warmup, len(h) - 1):
        if h[t] < runmax[t] - drop:          # held-out dropped below its running max => collapse
            return int(t)
        if t <= late and float(h[t + 1:].max()) <= h[t] + tol:   # EARLY plateau => stall
            return int(t)
    return None


def fire_iter(series, direction, baseline_iters=2, drop_frac=0.5, rise_abs=0.15):
    s = np.asarray(series, float)
    if len(s) < baseline_iters + 1:
        return None
    base = float(np.mean(s[:baseline_iters]))
    for t in range(baseline_iters, len(s)):
        if direction < 0 and base > 1e-9 and s[t] < drop_frac * base:
            return t
        if direction > 0 and s[t] > base + rise_abs:
            return t
    return None


def analyze(regime_data):
    """For each signal: fire iteration + lead time (onset - fire). Returns table + false-alarm flags."""
    rows = {}
    for sig in SIGNALS:
        direction = S.SIGNAL_DIRECTION[sig]
        rows[sig] = {}
        for rname, data in regime_data.items():
            if sig not in data:
                rows[sig][rname] = None
                continue
            fi = fire_iter(data[sig], direction)
            onset = data["_onset"]
            if fi is None:
                rows[sig][rname] = ("no-fire", None)
            elif onset is None:  # healthy: any fire is a FALSE ALARM
                rows[sig][rname] = ("FALSE-ALARM", fi)
            else:
                rows[sig][rname] = ("lead", onset - fi)  # +ve => warned that many steps early
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", default=None, help="JSON of real-model regime trajectories (Colab)")
    ap.add_argument("--T", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="loophealth/results/benchmark")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    if args.trajectories:
        raw = json.load(open(args.trajectories))
        regime_data = {}
        for k, v in raw.items():
            d = {kk: np.asarray(vv) for kk, vv in v.items() if kk != "_onset"}
            d["_onset"] = detect_onset(d.get("heldout"))   # recompute from held-out (plateau-or-drop)
            regime_data[k] = d
        mode = "REAL (onset = held-out plateau-or-drop, recomputed)"
    else:
        regime_data = {r: synth_regime(r, T=args.T, seed=args.seed) for r in REGIMES}
        mode = "SYNTHETIC (methodology validation; dynamics encoded from literature — NOT a finding)"

    rows = analyze(regime_data)
    regs = [r for r in REGIMES if r in regime_data] or REGIMES   # only regimes actually present (robust to partial runs)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    print(f"=== LoopHealth head-to-head early-warning benchmark [{mode}] ===")
    print(f"{'signal':<28} " + " ".join(f"{r:>12}" for r in regs))
    for sig, d in rows.items():
        cells = []
        for r in regs:
            v = d.get(r)
            if v is None or v[0] == "no-fire":
                cells.append(f"{'—':>12}")
            elif v[0] == "FALSE-ALARM":
                cells.append(f"{'FALSE':>12}")
            else:
                cells.append(f"{('+' if v[1]>=0 else '')+str(v[1]):>12}")
        print(f"{sig:<28} " + " ".join(cells))
    print("\n(cells = lead time in iters before held-out degrades; '—'=never fired; 'FALSE'=false alarm in healthy)")
    print("Read: best early-warning per regime = largest positive lead with NO false alarm in 'healthy'.")

    # honest headline checks
    if "collapse" in regs:
        classical_collapse = [rows[s].get("collapse") for s in ("rolling_variance", "lag1_autocorr")]
        print(f"\nClassical EWS on collapse (rolling_variance, lag1_autocorr): {classical_collapse}  "
              f"(expect weak/no lead — Entropy-Collapse 2512.12381)")

    # pass@1 vs pass@k divergence — the silent-sharpening check (a loop can raise pass@1 while pass@k collapses)
    print("\n=== coverage: pass@1 (reward proxy) vs pass@k (what the loop can STILL solve) ===")
    for r in regs:
        d = regime_data.get(r, {})
        if "pass_at_1" in d and "pass_at_k" in d:
            p1 = np.asarray(d["pass_at_1"], float); pk = np.asarray(d["pass_at_k"], float)
            d1, dk = float(p1[-1] - p1[0]), float(pk[-1] - pk[0])
            flag = "   <-- SILENT SHARPENING (reward up, coverage down)" if (d1 > 0.02 and dk < -0.02) else ""
            print(f"  {r:>9}: Δpass@1={d1:+.3f}  Δpass@k={dk:+.3f}{flag}")

    json.dump({s: {r: rows[s].get(r) for r in regs} for s in SIGNALS}, open(str(out) + ".json", "w"), indent=2, default=str)
    print(f"wrote {out}.json")

    if args.plot:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, len(regs), figsize=(4 * len(regs), 4), sharex=True)
            ax = np.atleast_1d(ax)
            for j, r in enumerate(regs):
                for sig in ("reward_variance", "effective_sample_fraction", "spectral_entropy",
                            "policy_entropy", "output_diversity", "ece", "heldout"):
                    if sig in regime_data[r]:
                        ax[j].plot(regime_data[r][sig], label=sig, lw=1.4)
                if regime_data[r]["_onset"] is not None:
                    ax[j].axvline(regime_data[r]["_onset"], color="k", ls="--", lw=1, label="failure onset")
                ax[j].set_title(r); ax[j].set_xlabel("iteration")
            ax[0].legend(fontsize=6, loc="best")
            fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=150); print(f"wrote {out}.png")
            # headline coverage figure: pass@1 vs pass@k per regime
            if any("pass_at_1" in regime_data[r] for r in regs):
                fig2, ax2 = plt.subplots(1, len(regs), figsize=(4 * len(regs), 3.5), sharey=True)
                ax2 = np.atleast_1d(ax2)
                for j, r in enumerate(regs):
                    d = regime_data.get(r, {})
                    if "pass_at_1" in d:
                        ax2[j].plot(d["pass_at_1"], label="pass@1", lw=2, color="C0")
                        ax2[j].plot(d["pass_at_k"], label="pass@k", lw=2, color="C3")
                    ax2[j].set_title(r); ax2[j].set_xlabel("iteration"); ax2[j].set_ylim(0, 1)
                ax2[0].set_ylabel("coverage"); ax2[0].legend(fontsize=8)
                fig2.suptitle("Coverage: pass@1 (reward proxy) vs pass@k (what the loop can still solve)")
                fig2.tight_layout(); fig2.savefig(str(out) + "_coverage.png", dpi=150)
                print(f"wrote {out}_coverage.png")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
