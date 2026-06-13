"""
Robust loader for SRAG-V source modules.

We load grpo_trainer.py and reward_system.py *by file path* (not via `import sragv`),
because (a) it avoids executing the package __init__ chain (heavy / fragile on Colab),
and (b) both files were verified to have NO intra-package imports, so path-loading is safe.

This keeps the experiments genuinely built on SRAG-V's own code:
  - compute_group_advantages  (the advantage gate under study)
  - RoleConditionedRewardSystem / SolutionGeneratorReward (the execution-grounded reward)
  - GRPOConfig (LoRA r=64, alpha=128, the 7 target modules used in the real run)
"""
import importlib.util
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
# experiments/srag_probe -> experiments -> <repo root>
_REPO = _HERE.parents[1]
SRC = pathlib.Path(os.environ.get("SRAG_SRC", _REPO / "sragv_4agent" / "src")).resolve()


def load_module(modname: str, relpath: str):
    path = SRC / relpath
    if not path.exists():
        raise FileNotFoundError(
            f"SRAG-V source not found at {path}.\n"
            f"Run from inside the srag repo, or set SRAG_SRC to the repo's src/ dir."
        )
    spec = importlib.util.spec_from_file_location(modname, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- SRAG-V advantage (the object of study) -------------------------------
_grpo = load_module("srag_grpo_trainer", "sragv/training/grpo_trainer.py")
GRPOTrainer = _grpo.GRPOTrainer
GRPOConfig = _grpo.GRPOConfig
_BARE = GRPOTrainer.__new__(GRPOTrainer)  # bypass __init__; compute_group_advantages uses no self-state


def srag_group_advantages(rewards, role_weights=None):
    """SRAG-V's exact group-relative advantage:  A_i = (r_i - mean)/std  (std floored to 1 if <1e-8)."""
    return _BARE.compute_group_advantages(list(rewards), role_weights)


def srag_lora_hparams():
    """The LoRA hyperparameters from the real Phase-3 GRPO run."""
    cfg = GRPOConfig()
    return dict(
        rank=cfg.lora_rank,                 # 64
        alpha=cfg.lora_alpha,               # 128
        dropout=cfg.lora_dropout,           # 0.1
        target_modules=list(cfg.target_modules),  # 7 modules
        lr=cfg.learning_rate,               # 1e-5
    )


# ---- SRAG-V reward system (loaded lazily; only needed for real/grounded runs) ----
def load_reward_system():
    rs = load_module("srag_reward_system", "sragv/training/reward_system.py")
    return rs


if __name__ == "__main__":
    # quick self-check
    import numpy as np
    print("SRC:", SRC)
    print("LoRA hparams:", srag_lora_hparams())
    eq = srag_group_advantages([1.0, 1.0, 1.0, 1.0])
    spread = srag_group_advantages([0.0, 0.5, 0.5, 1.0])
    print("zero-variance advantages:", eq, "-> all zero:", bool(np.allclose(eq, 0.0)))
    print("nonzero-variance advantages:", [round(x, 4) for x in spread])
