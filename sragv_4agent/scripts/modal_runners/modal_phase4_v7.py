"""
Phase 4 V7: GRPO HumanEval Evaluation with merge_and_unload()
Final fix - bakes GRPO weights into model.
"""

import modal
import os

app = modal.App("srag-phase4-v7-final")

local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.25.0",
        "peft>=0.16.0",
        "bitsandbytes>=0.41.0",
        "datasets==2.14.6",
        "pyarrow==14.0.1",
        "numpy>=1.24.0,<2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "human-eval>=1.0.0",
    )
    .add_local_dir(local_dir, remote_path="/workspace/srag")
)

volume = modal.Volume.from_name("srag-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=18000,
    volumes={"/results": volume},
    memory=32768
)
def run_phase4_v7():
    """Run Phase 4 V7 with merge_and_unload()."""
    import subprocess
    import sys
    import os
    import shutil
    from pathlib import Path

    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("PHASE 4 V7: GRPO HUMANEVAL (merge_and_unload)", flush=True)
    print("=" * 80, flush=True)
    print("Fix: Bake GRPO weights into model permanently", flush=True)
    print("Method: merge_and_unload() - guaranteed adapter application", flush=True)
    print("=" * 80, flush=True)

    # Download checkpoint
    print("\n📦 Downloading GRPO checkpoint...", flush=True)
    volume_checkpoint = Path("/results/phase3_grpo/grpo_iteration_16")
    local_checkpoint = Path("checkpoints/phase3_grpo/grpo_iteration_16")

    if volume_checkpoint.exists():
        local_checkpoint.mkdir(parents=True, exist_ok=True)

        if (volume_checkpoint / "solution_generator_adapter").exists():
            shutil.copytree(
                volume_checkpoint / "solution_generator_adapter",
                local_checkpoint / "solution_generator_adapter",
                dirs_exist_ok=True
            )
            print("  ✅ GRPO adapter downloaded", flush=True)
        else:
            raise FileNotFoundError("Adapter not found")

    print("\n🚀 Running evaluation...\n", flush=True)

    process = subprocess.Popen(
        [sys.executable, "-u", "scripts/phase4_runners/run_phase4_grpo_humaneval_v7.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)

    return_code = process.wait()

    print("\n" + "=" * 80, flush=True)
    print(f"Completed with return code: {return_code}", flush=True)
    print("=" * 80, flush=True)

    # Save results
    results_local = Path("results/phase4_humaneval_v7")
    results_volume = Path("/results/phase4_humaneval_v7")

    if results_local.exists():
        results_volume.mkdir(parents=True, exist_ok=True)
        for item in results_local.glob("*"):
            if item.is_file():
                shutil.copy2(item, results_volume / item.name)
                print(f"  Saved: {item.name}", flush=True)
        volume.commit()
        print("✅ Results committed", flush=True)

    return {"success": return_code == 0}


@app.local_entrypoint()
def main():
    print("=" * 80)
    print("PHASE 4 V7: GRPO HUMANEVAL (FINAL FIX)")
    print("=" * 80)
    print("\nMethod: merge_and_unload()")
    print("  - Bakes GRPO weights into model")
    print("  - No PEFT complexity")
    print("  - Guaranteed adapter application")
    print("\nExpected:")
    print("  Baseline:  ~60%")
    print("  Trained:   62-65%")
    print("  Improvement: +2-5%")
    print("\nRuntime: 40-50 min | Cost: $6-10")
    print("=" * 80)
    print()

    print("🚀 Deploying to Modal...\n")
    result = run_phase4_v7.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("🎉 PHASE 4 V7 COMPLETED!")
        print("\nDownload results:")
        print("  /usr/bin/python3 -m modal volume get srag-results phase4_humaneval_v7 results/phase4_humaneval_v7")
    else:
        print("❌ Evaluation failed")
    print("=" * 80)
