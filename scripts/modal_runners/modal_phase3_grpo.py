"""
SRAG-V Phase 3: GRPO Training on Modal
Runs 4-player self-play training with reinforcement learning.

Expected: 15-20 hours, $150-200
"""

import modal
import os

app = modal.App("srag-phase3-grpo")

# Get project root
local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Define image
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

# Persistent volume for checkpoints and results
volume = modal.Volume.from_name("srag-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=28800,  # 8 hours (pilot run)
    volumes={"/results": volume}
)
def run_phase3_training():
    """Run Phase 3 GRPO training."""
    import subprocess
    import sys
    import os
    import shutil

    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("SRAG-V PHASE 3: GRPO SELF-PLAY TRAINING", flush=True)
    print("=" * 80, flush=True)
    print("Configuration:", flush=True)
    print("  Model: Qwen2.5-Coder-1.5B-Instruct (calibrated)", flush=True)
    print("  Method: 4-player GRPO", flush=True)
    print("  Iterations: 6", flush=True)
    print("  Problems/iteration: 32", flush=True)
    print("  Expected: 15-20 hours", flush=True)
    print("=" * 80, flush=True)
    print()

    # Download Phase 1 checkpoint from volume
    print("Downloading Phase 1 checkpoint from volume...", flush=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Copy checkpoint if available
    if os.path.exists("/results/phase1_1.5B"):
        for item in os.listdir("/results/phase1_1.5B"):
            if item.endswith(".pt"):
                src = os.path.join("/results/phase1_1.5B", item)
                dst = os.path.join("checkpoints", item)
                shutil.copy2(src, dst)
                print(f"  Downloaded: {item}", flush=True)
        print("✅ Phase 1 checkpoint ready", flush=True)
    else:
        print("⚠️ Phase 1 checkpoint not found in volume", flush=True)

    print()

    # Run Phase 3 training
    process = subprocess.Popen(
        [sys.executable, "-u", "scripts/phase3_runners/run_phase3_grpo_training.py"],
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
    print(f"Phase 3 completed with return code: {return_code}", flush=True)
    print("=" * 80, flush=True)

    # Save results to volume
    if os.path.exists("results/phase3_grpo"):
        print("Saving results to persistent volume...", flush=True)
        os.makedirs("/results/phase3_grpo", exist_ok=True)

        for item in os.listdir("results/phase3_grpo"):
            src = os.path.join("results/phase3_grpo", item)
            dst = os.path.join("/results/phase3_grpo", item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Saved: {item}", flush=True)

        # Save checkpoints
        if os.path.exists("checkpoints/phase3_grpo"):
            for item in os.listdir("checkpoints/phase3_grpo"):
                src = os.path.join("checkpoints/phase3_grpo", item)
                dst = os.path.join("/results/phase3_grpo", item)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    print(f"  Saved checkpoint: {item}", flush=True)

        volume.commit()
        print("✅ Results committed to volume", flush=True)

    return {
        "success": return_code == 0,
        "return_code": return_code
    }


@app.local_entrypoint()
def main():
    """Launch Phase 3 GRPO training."""
    print("=" * 80)
    print("SRAG-V PHASE 3: GRPO TRAINING")
    print("=" * 80)
    print()
    print("PILOT RUN - Validate 4-player architecture")
    print()
    print("This will:")
    print("  1. Load Phase 1 calibrated checkpoint (ECE 0.00395)")
    print("  2. Initialize 4-player architecture")
    print("  3. Run 5 iterations of GRPO self-play (PILOT)")
    print("  4. Train with execution-based rewards")
    print("  5. Save checkpoint for Phase 4 evaluation")
    print()
    print("Configuration:")
    print("  - Iterations: 5 (pilot)")
    print("  - Problems/iteration: 20")
    print("  - Solutions/problem: 8")
    print("  - Total episodes: 100")
    print("  - Method: 4-player GRPO + MAP-Elites")
    print()
    print("Expected:")
    print("  - Time: 5-6 hours")
    print("  - Cost: $40-50")
    print("  - Goal: Detect if architecture works (+0.5-1% improvement)")
    print()
    print("After pilot completes:")
    print("  → Run Phase 4 eval to measure improvement")
    print("  → If positive: scale to 20 iterations ($400 budget)")
    print("  → If zero/negative: analyze and adjust")
    print()
    print("=" * 80)
    print()

    input("Press Enter to start Phase 3 training (or Ctrl+C to cancel)...")
    print()

    result = run_phase3_training.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ PHASE 3 GRPO TRAINING COMPLETED")
        print()
        print("Next step: Run Phase 4 HumanEval evaluation")
        print("Results saved in Modal volume: srag-results/phase3_grpo/")
    else:
        print(f"❌ PHASE 3 FAILED (code {result['return_code']})")
        print()
        print("Check logs above for errors")
    print("=" * 80)
