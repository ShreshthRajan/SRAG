"""
SRAG-V Phase 1: Calibration Training on Modal
Model: Qwen2.5-Coder-1.5B-Instruct (70% HumanEval baseline)
Objective: Achieve ECE < 0.05 with proper confidence calibration
"""

import modal
import os

app = modal.App("srag-phase1-calibration")

local_dir = os.path.dirname(os.path.abspath(__file__))

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.25.0",
        "peft>=0.16.0",
        "bitsandbytes>=0.41.0",
        "datasets>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "psutil>=5.9.0",
    )
    .add_local_dir(local_dir, remote_path="/workspace/srag")
)

# Create volume for persistent results
results_volume = modal.Volume.from_name("srag-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=14400,  # 4 hours
    volumes={"/results": results_volume}
)
def run_phase1_calibration():
    """Run Phase 1 calibration training."""
    import subprocess
    import sys
    import os
    import shutil

    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("SRAG-V PHASE 1: CALIBRATION TRAINING", flush=True)
    print("=" * 80, flush=True)
    print("Model: Qwen2.5-Coder-1.5B-Instruct", flush=True)
    print("Target: ECE < 0.05", flush=True)
    print("Expected time: 2-3 hours", flush=True)
    print("=" * 80, flush=True)
    print()

    # Run Phase 1 training
    process = subprocess.Popen(
        [sys.executable, "-u", "run_phase1_star_training.py"],
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
    print(f"Phase 1 completed with return code: {return_code}", flush=True)
    print("=" * 80, flush=True)

    # Copy results to persistent volume
    if os.path.exists("phase1_results"):
        print("Copying results to persistent volume...", flush=True)
        os.makedirs("/results/phase1_1.5B", exist_ok=True)

        # Copy all result files
        for item in os.listdir("phase1_results"):
            src = os.path.join("phase1_results", item)
            dst = os.path.join("/results/phase1_1.5B", item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Copied: {item}", flush=True)

        # Copy checkpoint
        if os.path.exists("checkpoints"):
            checkpoint_files = [f for f in os.listdir("checkpoints") if "phase1" in f.lower()]
            for ckpt in checkpoint_files:
                src = os.path.join("checkpoints", ckpt)
                dst = os.path.join("/results/phase1_1.5B", ckpt)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"  Copied checkpoint: {ckpt}", flush=True)

        # Commit volume to persist
        results_volume.commit()
        print("✅ Results saved to persistent volume", flush=True)

    return {
        "success": return_code == 0,
        "return_code": return_code
    }

@app.local_entrypoint()
def main():
    """Run Phase 1 calibration."""
    print("=" * 80)
    print("SRAG-V PHASE 1: CALIBRATION TRAINING")
    print("=" * 80)
    print("Configuration:")
    print("  Model: Qwen2.5-Coder-1.5B-Instruct")
    print("  Baseline: ~70% HumanEval")
    print("  Target: ECE < 0.05")
    print("  Problems: ~200 diverse")
    print("  Solutions per problem: 6")
    print()
    print("Expected:")
    print("  Time: 2-3 hours")
    print("  Cost: ~$15-20")
    print()
    print("Results will be saved to:")
    print("  - Local: phase1_results/phase1_final_report.json")
    print("  - Volume: srag-results/phase1_1.5B/")
    print("=" * 80)
    print()

    result = run_phase1_calibration.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ PHASE 1 CALIBRATION COMPLETED SUCCESSFULLY")
        print()
        print("Next step: Run Phase 3 GRPO training")
        print("Results saved in persistent Modal volume")
    else:
        print(f"❌ PHASE 1 FAILED (exit code {result['return_code']})")
        print()
        print("Check logs above for errors")
    print("=" * 80)
