"""
Phase 4: GRPO HumanEval Evaluation on Modal
Evaluates baseline vs GRPO-trained Qwen2.5-Coder-1.5B-Instruct.

Cost: ~$10-15 | Runtime: ~3-5 hours | GPU: 1×A100-80GB
"""

import modal
import os

app = modal.App("srag-phase4-grpo-humaneval-v6")

# Get project root
local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Define image with human-eval
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.25.0",
        "peft>=0.16.0",
        "bitsandbytes>=0.41.0",
        "datasets==2.14.6",  # APPS compatibility
        "pyarrow==14.0.1",
        "numpy>=1.24.0,<2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "human-eval>=1.0.0",  # HumanEval official package
    )
    .add_local_dir(local_dir, remote_path="/workspace/srag")
)

# Persistent volume for checkpoints and results
volume = modal.Volume.from_name("srag-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB",  # Single GPU sufficient for evaluation
    timeout=18000,  # 5 hours max
    volumes={"/results": volume},
    memory=32768  # 32GB RAM
)
def run_phase4_evaluation():
    """Run Phase 4 GRPO HumanEval evaluation."""
    import subprocess
    import sys
    import os
    import shutil
    from pathlib import Path

    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("SRAG-V PHASE 4: GRPO HUMANEVAL EVALUATION", flush=True)
    print("=" * 80, flush=True)
    print("Experiment: Baseline vs GRPO-Trained (iteration 20)", flush=True)
    print("Model: Qwen2.5-Coder-1.5B-Instruct", flush=True)
    print("Benchmark: HumanEval (164 problems)", flush=True)
    print("Training: 20 iterations, 1,340 samples, $100 compute", flush=True)
    print("Expected runtime: 3-5 hours", flush=True)
    print("=" * 80, flush=True)
    print()

    # Download GRPO checkpoint from volume
    print("📦 Stage 1: Downloading GRPO checkpoint from volume...", flush=True)

    volume_checkpoint_path = Path("/results/phase3_grpo/grpo_iteration_16")
    local_checkpoint_path = Path("checkpoints/phase3_grpo/grpo_iteration_16")

    if volume_checkpoint_path.exists():
        print("  Found GRPO checkpoint in volume", flush=True)

        # Copy checkpoint directory
        local_checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Copy solution_generator_adapter (most important)
        if (volume_checkpoint_path / "solution_generator_adapter").exists():
            shutil.copytree(
                volume_checkpoint_path / "solution_generator_adapter",
                local_checkpoint_path / "solution_generator_adapter",
                dirs_exist_ok=True
            )
            print("  ✅ solution_generator_adapter copied", flush=True)
        else:
            raise FileNotFoundError("solution_generator_adapter not found in volume")

        # Copy other adapters (for completeness)
        for adapter_name in ["problem_generator_adapter", "verification_generator_adapter", "meta_verifier_adapter"]:
            if (volume_checkpoint_path / adapter_name).exists():
                shutil.copytree(
                    volume_checkpoint_path / adapter_name,
                    local_checkpoint_path / adapter_name,
                    dirs_exist_ok=True
                )
                print(f"  ✅ {adapter_name} copied", flush=True)

        # Copy training state
        for state_file in ["training_state.json", "optimizers.pt"]:
            if (volume_checkpoint_path / state_file).exists():
                shutil.copy2(
                    volume_checkpoint_path / state_file,
                    local_checkpoint_path / state_file
                )
                print(f"  ✅ {state_file} copied", flush=True)

        print("✅ GRPO checkpoint ready", flush=True)
    else:
        raise FileNotFoundError(f"GRPO checkpoint not found in volume: {volume_checkpoint_path}")

    print()

    # Run Phase 4 evaluation
    print("🚀 Stage 2: Running HumanEval evaluation...", flush=True)
    print()

    process = subprocess.Popen(
        [sys.executable, "-u", "scripts/phase4_runners/run_phase4_grpo_humaneval.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    # Stream output
    for line in process.stdout:
        print(line, end='', flush=True)

    return_code = process.wait()

    print("\n" + "=" * 80, flush=True)
    print(f"Phase 4 completed with return code: {return_code}", flush=True)
    print("=" * 80, flush=True)

    # Save results to volume
    print("💾 Stage 3: Saving results to volume...", flush=True)

    results_local = Path("results/phase4_humaneval_v6")
    results_volume = Path("/results/phase4_humaneval_v6")

    if results_local.exists():
        results_volume.mkdir(parents=True, exist_ok=True)

        # Copy all result files
        for item in results_local.glob("*"):
            if item.is_file():
                shutil.copy2(item, results_volume / item.name)
                print(f"  Saved: {item.name}", flush=True)

        volume.commit()
        print("✅ Results committed to volume", flush=True)

    return {
        "success": return_code == 0,
        "return_code": return_code
    }


@app.local_entrypoint()
def main():
    """Launch Phase 4 GRPO HumanEval evaluation."""
    print("=" * 80)
    print("SRAG-V PHASE 4: GRPO HUMANEVAL EVALUATION")
    print("=" * 80)
    print()
    print("Experiment: Baseline vs GRPO-Trained (iteration 20)")
    print("Model: Qwen2.5-Coder-1.5B-Instruct")
    print("Benchmark: HumanEval (164 problems)")
    print()
    print("Training completed:")
    print("  - 20 iterations, 1,340 samples")
    print("  - Quality: 0.825 → 0.872 (peak) → 0.845 (final)")
    print("  - Loss: 0.011 → -0.043 (4x improvement)")
    print()
    print("Expected:")
    print("  ⏱️  Runtime: 3-5 hours")
    print("  💰 Cost: $10-15 (1×A100-80GB)")
    print("  📊 Improvement: +0.8% Pass@1 (median prediction)")
    print("  🎯 Confidence: 7/10")
    print()
    print("=" * 80)
    print()

    print("🚀 Deploying to Modal...")
    result = run_phase4_evaluation.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("🎉 PHASE 4 EVALUATION COMPLETED SUCCESSFULLY!")
        print()
        print("Results saved to Modal volume: srag-results/phase4_humaneval/")
        print()
        print("Download results with:")
        print("  python3 -m modal volume get srag-results phase4_humaneval results/phase4_humaneval")
        print()
        print("Analyze results with:")
        print("  python3 scripts/utilities/analyze_phase4_results.py")
    else:
        print("❌ Phase 4 evaluation failed")
        print(f"Return code: {result['return_code']}")

    print("=" * 80)
