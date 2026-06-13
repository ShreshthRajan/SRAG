"""
Figure 1 — the dead verifiers of the real SRAG-V run.

Reads the actual Phase-3 GRPO checkpoints and reports, for every saved LoRA adapter,
the fraction of lora_B tensors that are *exactly* zero. lora_B is initialized to 0;
under the variance gate it can only leave zero by receiving a nonzero gradient. So
"100% lora_B zero" == "this adapter received zero learning signal for the whole run."

No GPU and no torch required: parses safetensors headers + raw bytes directly.
"""
import argparse
import glob
import json
import struct
import pathlib


def lora_b_zero_fraction(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data_start = 8 + n
        lb = [k for k in header if "lora_B" in k]
        if not lb:
            return None
        zero = 0
        for k in lb:
            b, e = header[k]["data_offsets"]
            f.seek(data_start + b)
            raw = f.read(e - b)
            if not any(byte != 0 for byte in raw):
                zero += 1
        return zero, len(lb)


def scan(ckpt_root):
    rows = []
    for path in sorted(glob.glob(str(pathlib.Path(ckpt_root) / "**" / "adapter_model.safetensors"), recursive=True)):
        res = lora_b_zero_fraction(path)
        p = pathlib.Path(path)
        label = "/".join(p.parts[-3:-1])  # iteration/role
        if res is None:
            rows.append({"label": label, "path": path, "lora_B_total": 0, "lora_B_zero": 0, "note": "no lora_B keys"})
        else:
            zero, total = res
            rows.append({
                "label": label, "path": path,
                "lora_B_total": total, "lora_B_zero": zero,
                "zero_fraction": round(zero / total, 4),
                "learned": zero < total,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="checkpoints/phase3_grpo")
    ap.add_argument("--out", default="experiments/srag_probe/out/fig1_dead_adapters")
    ap.add_argument("--plot", action="store_true", help="also render a bar chart (needs matplotlib)")
    args = ap.parse_args()

    rows = scan(args.ckpt_root)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"{'adapter':<48} {'lora_B zero/total':>18}  status")
    print("-" * 80)
    for r in rows:
        frac = f"{r['lora_B_zero']}/{r['lora_B_total']}"
        status = "DEAD (no learning)" if r.get("learned") is False else ("learned" if r.get("learned") else r.get("note", ""))
        print(f"{r['label']:<48} {frac:>18}  {status}")

    with open(str(out) + ".json", "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nwrote {out}.json")

    # also report which roles are completely dead (any iteration)
    dead = sorted({r["label"].split("/")[-1] for r in rows if r.get("learned") is False})
    learned = sorted({r["label"].split("/")[-1] for r in rows if r.get("learned") is True})
    print(f"DEAD roles (100% lora_B zero): {dead}")
    print(f"Learned roles (some lora_B nonzero): {learned}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            labels = [r["label"] for r in rows if r["lora_B_total"]]
            fracs = [r["zero_fraction"] for r in rows if r["lora_B_total"]]
            colors = ["#c0392b" if fr == 1.0 else "#27ae60" for fr in fracs]
            fig, ax = plt.subplots(figsize=(8, 4.2))
            ax.barh(labels, fracs, color=colors)
            ax.set_xlabel("fraction of lora_B tensors exactly == 0")
            ax.set_xlim(0, 1.0)
            ax.set_title("SRAG-V Phase-3: which adapters received any learning signal")
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(str(out) + ".png", dpi=150)
            print(f"wrote {out}.png")
        except Exception as e:
            print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
