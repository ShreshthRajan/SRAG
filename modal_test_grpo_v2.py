"""
Modal deployment for GRPO pipeline test - Version 2.
Uses local code mount instead of git clone (much faster).
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("srag-grpo-test-v2")

# Define image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
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

# Mount local source code
srag_mount = modal.Mount.from_local_dir(
    str(Path(__file__).parent),
    remote_path="/workspace/srag"
)

# Define GPU requirements
@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=3600,
    mounts=[srag_mount]
)
def run_grpo_test():
    """Run GRPO pipeline test on Modal."""
    import subprocess
    import sys
    import os

    # Already in mounted directory
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
    print("Launching GRPO pipeline test on Modal (using local code)...")
    print("This will:")
    print("  1. Upload your local SRAG code")
    print("  2. Test base model loading")
    print("  3. Test SelfPlayTrainer + GRPO")
    print("  4. Return results")
    print()
    print("Expected time: 10-30 minutes")
    print("Expected cost: ~$10")
    print()

    result = run_grpo_test.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ GRPO PIPELINE TEST PASSED!")
        print()
        print("Your system is ready for full Phase 3")
    else:
        print("❌ GRPO PIPELINE TEST FAILED")
        print()
        print("Review errors above")

    print("=" * 80)
