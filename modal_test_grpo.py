"""
Modal deployment for GRPO pipeline test.
Tests SelfPlayTrainer + base model on small dataset (~30 min, $10).
"""

import modal

# Create Modal app
app = modal.App("srag-grpo-test")

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")  # Need git for cloning
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
        "human-eval",
    )
)

# Define GPU requirements - single A100 for test
@app.function(
    image=image,
    gpu="A100",  # Single A100 (updated syntax)
    timeout=3600,  # 1 hour timeout for test
)
def run_grpo_test():
    """Run GRPO pipeline test on Modal."""
    import subprocess
    import sys

    # Clone repo
    subprocess.run([
        "git", "clone",
        "https://github.com/ShreshthRajan/SRAG.git",
        "/workspace/srag"
    ], check=True)

    # Change to repo directory
    import os
    os.chdir("/workspace/srag")

    # Run test
    result = subprocess.run(
        [sys.executable, "test_grpo_pipeline.py"],
        capture_output=True,
        text=True
    )

    print("=" * 80)
    print("STDOUT:")
    print(result.stdout)
    print("=" * 80)
    print("STDERR:")
    print(result.stderr)
    print("=" * 80)
    print(f"Return code: {result.returncode}")

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.local_entrypoint()
def main():
    """Run test from local machine."""
    print("Launching GRPO pipeline test on Modal...")
    print("This will:")
    print("  1. Test base model loading")
    print("  2. Test SelfPlayTrainer initialization")
    print("  3. Test GRPO training step")
    print("  4. Verify rewards and model updates")
    print()
    print("Expected time: 10-30 minutes")
    print("Expected cost: ~$10")
    print()

    result = run_grpo_test.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ GRPO PIPELINE TEST PASSED!")
        print()
        print("Your system is ready for:")
        print("  - Full Phase 3 with GRPO")
        print("  - Base model training")
        print("  - Production deployment")
    else:
        print("❌ GRPO PIPELINE TEST FAILED")
        print()
        print("Review errors above and fix before proceeding")

    print("=" * 80)
