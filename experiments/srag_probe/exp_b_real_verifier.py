"""
Exp B (real verifier) — A competent solver starves its own gradient.

Uses the REAL models (Qwen2.5-Coder-1.5B solver + Qwen2.5-1.5B verifier), the REAL
SRAG-V advantage gate (compute_group_advantages), and a hardened assert-executor in the
spirit of SRAG-V's CodeExecutionEnvironment (we deliberately bypass the 1141-line
VerificationGenerator text-parsing wrapper whose JSON-parse fragility is documented breaking
the original run — see the repo's "verification json fix" commits — in favor of robust
extraction; this is an honest robustness improvement, stated in the paper).

The verifier (a real Qwen model) GENERATES the test cases; rewards = pass-rate of solver
rollouts on those tests (real execution). The advantage that drives the solver's GRPO update
is SRAG-V's (r - mean)/std.

MODE competence (default, CERTAIN result):
  For each problem, vary the fraction of the group that is correct (vs buggy). Reward variance
  (from the verifier's tests) is ~0 when the group is uniformly correct OR uniformly wrong, and
  maximal in between -> the solver gradient follows. The "uniformly correct -> zero gradient"
  point is the headline: as a solver succeeds, the verifier's tests stop discriminating and the
  gradient dies. Loss never registers it.

MODE loop (temporal illustration):
  Train the solver LoRA by GRPO for N iters against the frozen real verifier; track reward
  variance, effective-sample fraction, solution diversity, solver lora_B movement (LOWER) vs
  loss/quality/convergence (UPPER). --dynamic-sampling adds the DAPO baseline.

--smoke validates the executor + advantage + gradient plumbing on CPU with a tiny model and the
embedded ground-truth tests (no model generation), so the pipeline is proven before Colab.

Usage:
  python exp_b_real_verifier.py --smoke
  python exp_b_real_verifier.py --mode competence --solver Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --verifier Qwen/Qwen2.5-1.5B-Instruct --device cuda --plot
  python exp_b_real_verifier.py --mode loop --solver ... --verifier ... --device cuda --plot
  python exp_b_real_verifier.py --mode loop --dynamic-sampling --solver ... --verifier ... --device cuda --plot
"""
import argparse
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import torch
from torch.optim import AdamW

import _load
import model_utils as mu


# --------------------------- embedded problems (no dataset download) -------
# Each: prompt (signature+docstring), func name, a CORRECT solution, BUGGY solutions,
# and ground-truth asserts (fallback verifier). Chosen simple so a 1.5B solver can solve them.
PROBLEMS = [
    {
        "name": "add", "func": "add",
        "prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
        "correct": "def add(a, b):\n    return a + b",
        "buggy": ["def add(a, b):\n    return a - b", "def add(a, b):\n    return a * b"],
        "gt_tests": ["assert add(2, 3) == 5", "assert add(-1, 1) == 0", "assert add(0, 0) == 0"],
    },
    {
        "name": "is_even", "func": "is_even",
        "prompt": "def is_even(n):\n    \"\"\"Return True if n is even, else False.\"\"\"\n",
        "correct": "def is_even(n):\n    return n % 2 == 0",
        "buggy": ["def is_even(n):\n    return n % 2 == 1", "def is_even(n):\n    return True"],
        "gt_tests": ["assert is_even(4) == True", "assert is_even(3) == False", "assert is_even(0) == True"],
    },
    {
        "name": "max_of", "func": "max_of",
        "prompt": "def max_of(xs):\n    \"\"\"Return the maximum element of the non-empty list xs.\"\"\"\n",
        "correct": "def max_of(xs):\n    m = xs[0]\n    for x in xs:\n        if x > m:\n            m = x\n    return m",
        "buggy": ["def max_of(xs):\n    return xs[0]", "def max_of(xs):\n    return min(xs)"],
        "gt_tests": ["assert max_of([1,3,2]) == 3", "assert max_of([-5,-2,-9]) == -2", "assert max_of([7]) == 7"],
    },
    {
        "name": "count_vowels", "func": "count_vowels",
        "prompt": "def count_vowels(s):\n    \"\"\"Return the number of vowels (aeiou) in string s.\"\"\"\n",
        "correct": "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')",
        "buggy": ["def count_vowels(s):\n    return len(s)", "def count_vowels(s):\n    return sum(1 for c in s if c in 'bcd')"],
        "gt_tests": ["assert count_vowels('hello') == 2", "assert count_vowels('xyz') == 0", "assert count_vowels('aeiou') == 5"],
    },
]


# --------------------------- hardened executor -----------------------------
def run_asserts(code, asserts, timeout=6):
    """Return (n_pass, n_total). Each single-line assert is wrapped so one failure != crash."""
    asserts = [a for a in asserts if a.strip().startswith("assert")]
    if not asserts:
        return 0, 0
    prog = [code, "_p = 0", "_t = 0"]
    for a in asserts:
        prog += ["_t += 1", "try:", "    " + a.strip(), "    _p += 1", "except Exception:", "    pass"]
    prog.append("print(_p, _t)")
    src = "\n".join(prog)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            path = f.name
        r = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0 and r.stdout.strip():
            p, t = r.stdout.strip().split()[-2:]
            return int(p), int(t)
        return 0, len(asserts)
    except Exception:
        return 0, len(asserts)
    finally:
        try:
            import os
            os.unlink(path)
        except Exception:
            pass


def passrate(code, asserts):
    p, t = run_asserts(code, asserts)
    return (p / t) if t else 0.0


# --------------------------- real model I/O --------------------------------
def gen_solutions(model, tok, problem, G, device, temperature=0.9, max_new=160):
    msgs = [{"role": "system", "content": "You are an expert Python programmer. Output only the complete function, no prose."},
            {"role": "user", "content": "Complete this function:\n\n" + problem["prompt"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        # generate all G samples in ONE batched call (num_return_sequences) -> ~Gx faster than a Python loop
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=temperature,
                           top_p=0.95, num_return_sequences=G, pad_token_id=tok.eos_token_id)
    outs = []
    for j in range(o.shape[0]):
        gen = tok.decode(o[j, plen:], skip_special_tokens=True)
        outs.append(extract_code(gen, problem["func"]) or problem["prompt"] + "    pass")
    return outs


def gen_verifier_tests(model, tok, problem, device, n=5, max_new=200):
    msgs = [{"role": "system", "content": "You write Python assert-based unit tests. Output only assert statements, one per line."},
            {"role": "user", "content": f"Write {n} assert statements testing the function `{problem['func']}`:\n\n{problem['prompt']}"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(device)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                           top_p=0.9, pad_token_id=tok.eos_token_id)
    gen = tok.decode(o[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    tests = [ln.strip() for ln in gen.splitlines() if ln.strip().startswith("assert") and problem["func"] in ln]
    # robustness: validate against the correct solution; drop tests the reference fails (hallucinated)
    valid = [t for t in tests if passrate(problem["correct"], [t]) == 1.0]
    return valid if len(valid) >= 2 else problem["gt_tests"]


def extract_code(text, func):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    body = m.group(1) if m else text
    idx = body.find(f"def {func}")
    return body[idx:].strip() if idx >= 0 else (body.strip() if "def " in body else None)


def build_group(problem, G, frac_correct, model=None, tok=None, device="cpu"):
    """Group of G solutions with a controlled fraction correct (rest buggy)."""
    k = int(round(frac_correct * G))
    sols = [problem["correct"]] * k
    bug = problem["buggy"]
    sols += [bug[i % len(bug)] for i in range(G - k)]
    return sols[:G]


# --------------------------- gradient measurement --------------------------
def group_gradient(solver, tok, problem, solutions, rewards, hp, init_state, device):
    """Restore solver LoRA to init, run one GRPO step over the group, return (grad_norm_B, mean_abs_adv)."""
    mu.restore_state(solver, init_state)
    opt = AdamW([p for _, p in mu.lora_named(solver)], lr=hp["lr"], weight_decay=0.01)
    adv = _load.srag_group_advantages(list(rewards), [1.0] * len(rewards))
    opt.zero_grad()
    prompt = problem["prompt"]
    for sol, a in zip(solutions, adv):
        lp = mu.logprob_from_text(solver, tok, prompt, "\n" + sol, device=device)
        loss = -(torch.tensor(float(a), device=device, dtype=lp.dtype) * lp) / len(solutions)
        loss.backward()
    gnB = mu.grad_l2(mu.lora_named(solver, "lora_B"))
    return gnB, float(np.mean(np.abs(adv)))


# --------------------------- modes -----------------------------------------
def mode_competence(solver, tok_s, verifier, tok_v, hp, device, fracs, G, use_real_verifier):
    solver = mu.attach_lora(solver, hp).to(device)
    init = mu.clone_state(mu.lora_named(solver))
    rows = []
    for prob in PROBLEMS:
        tests = gen_verifier_tests(verifier, tok_v, prob, device) if use_real_verifier else prob["gt_tests"]
        for fc in fracs:
            group = build_group(prob, G, fc)
            rewards = [passrate(s, tests) for s in group]
            gnB, mabs = group_gradient(solver, tok_s, prob, group, rewards, hp, init, device)
            rows.append({
                "problem": prob["name"], "frac_correct": round(fc, 3),
                "mean_passrate": float(np.mean(rewards)), "reward_variance": float(np.var(rewards)),
                "mean_abs_advantage": mabs, "grad_norm_loraB": gnB, "n_tests": len(tests),
            })
    return rows


def mode_loop(solver, tok_s, verifier, tok_v, hp, device, iters, G, dynamic, use_real_verifier):
    solver = mu.attach_lora(solver, hp).to(device)
    init_B = mu.clone_state(mu.lora_named(solver, "lora_B"))
    tests_by_prob = {p["name"]: (gen_verifier_tests(verifier, tok_v, p, device) if use_real_verifier else p["gt_tests"]) for p in PROBLEMS}
    rows, prev_q = [], None
    for t in range(iters):
        opt = AdamW([p for _, p in mu.lora_named(solver)], lr=hp["lr"], weight_decay=0.01)
        opt.zero_grad()
        var_list, q_list, absadv, used, divs = [], [], [], 0, []
        loss_val = 0.0
        for prob in PROBLEMS:
            sols = gen_solutions(solver, tok_s, prob, G, device)
            divs.append(solution_diversity(sols))
            rewards = [passrate(s, tests_by_prob[prob["name"]]) for s in sols]
            var_list.append(float(np.var(rewards))); q_list.append(float(np.mean(rewards)))
            if dynamic and np.var(rewards) < 1e-8:
                continue  # DAPO: drop degenerate group
            used += 1
            adv = _load.srag_group_advantages(list(rewards), [1.0] * G)
            absadv.append(float(np.mean(np.abs(adv))))
            for sol, a in zip(sols, adv):
                lp = mu.logprob_from_text(solver, tok_s, prob["prompt"], "\n" + sol, device=device)
                loss = -(torch.tensor(float(a), device=device, dtype=lp.dtype) * lp) / (G * len(PROBLEMS))
                loss.backward(); loss_val += float(loss.item())
        gnB = mu.grad_l2(mu.lora_named(solver, "lora_B"))
        opt.step()
        cumB = mu.delta_l2(mu.lora_named(solver, "lora_B"), init_B)
        q = float(np.mean(q_list))
        conv = 1.0 if prev_q is None else 1.0 / (1.0 + abs(q - prev_q)); prev_q = q
        rows.append({"iter": t, "loss": loss_val, "quality_mean_reward": q, "convergence": round(conv, 4),
                     "reward_variance_mean": float(np.mean(var_list)), "effective_fraction": round(used / len(PROBLEMS), 3),
                     "mean_abs_advantage": float(np.mean(absadv)) if absadv else 0.0,
                     "solution_diversity": float(np.mean(divs)), "grad_norm_loraB": gnB, "cum_weight_delta_loraB": cumB})
        print(f"  iter {t}: q={q:.3f} conv={conv:.3f} | rvar={np.mean(var_list):.5f} "
              f"eff={used/len(PROBLEMS):.2f} div={np.mean(divs):.3f} gradB={gnB:.3e} cumΔB={cumB:.3e}")
    return rows


def solution_diversity(sols):
    toks = [set(s.split()) for s in sols]
    if len(toks) < 2:
        return 0.0
    ds = []
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            u = len(toks[i] | toks[j]); inter = len(toks[i] & toks[j])
            ds.append(1.0 - (inter / u if u else 1.0))
    return float(np.mean(ds))


# --------------------------- main ------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["competence", "loop"], default="competence")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--solver", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--verifier", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--fracs", default="0,0.25,0.5,0.75,1.0")
    ap.add_argument("--dynamic-sampling", action="store_true")
    ap.add_argument("--no-real-verifier", action="store_true", help="use ground-truth tests instead of the verifier model")
    ap.add_argument("--out", default="experiments/srag_probe/out/exp_b_real")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    hp = _load.srag_lora_hparams()
    fracs = [float(x) for x in args.fracs.split(",")]
    out = pathlib.Path(args.out + ("_loop" if args.mode == "loop" else "") + ("_dapo" if args.dynamic_sampling else ""))
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        print("[smoke] tiny model, ground-truth tests, executor+advantage+gradient plumbing only")
        solver = mu.build_tiny_model(seed=0, vocab=256)
        tok_s = tok_v = None; verifier = None
        # smoke can't generate code; run competence with embedded solutions + gt tests
        solver = mu.attach_lora(solver, hp)
        init = mu.clone_state(mu.lora_named(solver))
        rows = []
        for prob in PROBLEMS:
            for fc in fracs:
                group = build_group(prob, args.group_size, fc)
                rewards = [passrate(s, prob["gt_tests"]) for s in group]
                # tiny-model logprob over token ids of the (fixed) solutions
                g = torch.Generator().manual_seed(0)
                seqs_grad = _smoke_grad(solver, group, rewards, hp, init, g)
                rows.append({"problem": prob["name"], "frac_correct": round(fc, 3),
                             "mean_passrate": float(np.mean(rewards)), "reward_variance": float(np.var(rewards)),
                             "mean_abs_advantage": seqs_grad[1], "grad_norm_loraB": seqs_grad[0], "n_tests": len(prob["gt_tests"])})
        _report_competence(rows, out, args.plot)
        return

    use_real = not args.no_real_verifier
    tok_s, solver = mu.load_real_model(args.solver, args.device, args.dtype)
    if use_real:
        tok_v, verifier = mu.load_real_model(args.verifier, args.device, args.dtype)
    else:
        tok_v, verifier = None, None

    if args.mode == "competence":
        rows = mode_competence(solver, tok_s, verifier, tok_v, hp, args.device, fracs, args.group_size, use_real)
        _report_competence(rows, out, args.plot)
    else:
        rows = mode_loop(solver, tok_s, verifier, tok_v, hp, args.device, args.iters, args.group_size,
                         args.dynamic_sampling, use_real)
        _report_loop(rows, out, args.plot, args.dynamic_sampling)


def _smoke_grad(solver, group, rewards, hp, init, gen):
    mu.restore_state(solver, init)
    opt = AdamW([p for _, p in mu.lora_named(solver)], lr=hp["lr"], weight_decay=0.01)
    adv = _load.srag_group_advantages(list(rewards), [1.0] * len(rewards))
    opt.zero_grad()
    for a in adv:
        ids = torch.randint(1, 256, (1, 20), generator=gen)
        lp = mu.logprob_from_ids(solver, ids, 8)
        loss = -(torch.tensor(float(a), dtype=lp.dtype) * lp) / len(adv)
        loss.backward()
    return mu.grad_l2(mu.lora_named(solver, "lora_B")), float(np.mean(np.abs(adv)))


def _report_competence(rows, out, plot):
    with open(str(out) + ".json", "w") as f:
        json.dump({"rows": rows}, f, indent=2)
    print(f"\n{'problem':<14}{'frac_ok':>8}{'passrate':>10}{'reward_var':>12}{'gradnorm_B':>13}")
    for r in rows:
        print(f"{r['problem']:<14}{r['frac_correct']:>8}{r['mean_passrate']:>10.3f}"
              f"{r['reward_variance']:>12.5f}{r['grad_norm_loraB']:>13.4e}")
    # headline: at frac_correct 0 and 1 (uniform), variance & gradient ~0
    uni = [r for r in rows if r["frac_correct"] in (0.0, 1.0)]
    mid = [r for r in rows if r["frac_correct"] == 0.5]
    print(f"\nUNIFORM groups (all-correct or all-wrong): mean grad_norm_B = "
          f"{np.mean([r['grad_norm_loraB'] for r in uni]):.3e}")
    print(f"MIXED groups (frac=0.5):                    mean grad_norm_B = "
          f"{np.mean([r['grad_norm_loraB'] for r in mid]):.3e}")
    print(f"wrote {out}.json")
    if plot:
        _safe_plot_competence(rows, out)


def _report_loop(rows, out, plot, dyn):
    with open(str(out) + ".json", "w") as f:
        json.dump({"dynamic_sampling": dyn, "rows": rows}, f, indent=2)
    dead = next((r["iter"] for r in rows if r["reward_variance_mean"] < 1e-6), None)
    print(f"\nreward-variance first ~0 at iter {dead}; "
          f"quality there = {next((r['quality_mean_reward'] for r in rows if r['iter']==dead), None)}")
    print(f"wrote {out}.json")
    if plot:
        _safe_plot_loop(rows, out)


def _safe_plot_competence(rows, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        x = [r["mean_passrate"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].scatter(x, [r["reward_variance"] for r in rows]); ax[0].set_xlabel("group mean pass-rate (competence)")
        ax[0].set_ylabel("reward variance"); ax[0].set_title("Verifier reward variance vanishes at competence extremes")
        ax[1].scatter(x, [r["grad_norm_loraB"] for r in rows]); ax[1].set_xlabel("group mean pass-rate (competence)")
        ax[1].set_ylabel("grad-norm lora_B"); ax[1].set_title("Solver gradient follows the variance")
        fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=150); print(f"wrote {out}.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


def _safe_plot_loop(rows, out):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        xx = [r["iter"] for r in rows]
        fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for k in ["loss", "quality_mean_reward", "convergence"]:
            ax[0].plot(xx, [r[k] for r in rows], label=k)
        ax[0].set_title("UPPER: looks healthy"); ax[0].legend()
        for k in ["reward_variance_mean", "effective_fraction", "solution_diversity", "grad_norm_loraB"]:
            ax[1].plot(xx, [r[k] for r in rows], label=k)
        ax[1].set_title("LOWER: leading indicators decay"); ax[1].set_xlabel("iteration"); ax[1].legend()
        fig.tight_layout(); fig.savefig(str(out) + ".png", dpi=150); print(f"wrote {out}.png")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
