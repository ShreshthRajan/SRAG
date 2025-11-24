"""
Modal deployment for GRPO pipeline test - Version 3 (Verbose).
Uses test_grpo_verbose.py with immediate output flushing.
"""

import modal
import os

app = modal.App("srag-grpo-test-v3")

local_dir = os.path.dirname(os.path.abspath(__file__))

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
    )
    .add_local_dir(local_dir, remote_path="/workspace/srag")
)

@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=7200  # 2 hours
)
def run_grpo_test():
    """Run verbose GRPO pipeline test."""
    import subprocess
    import sys
    import os

    os.chdir("/workspace/srag")

    # Run verbose test
    result = subprocess.run(
        [sys.executable, "test_grpo_verbose.py"],
        capture_output=True,
        text=True
    )

    print("=" * 80, flush=True)
    print("STDOUT:", flush=True)
    print(result.stdout, flush=True)
    print("=" * 80, flush=True)
    print("STDERR:", flush=True)
    print(result.stderr, flush=True)
    print("=" * 80, flush=True)
    print(f"Return code: {result.returncode}", flush=True)

    return {
        "success": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr
    }

@app.local_entrypoint()
def main():
    """Run test from local machine."""
    print("=" * 80)
    print("GRPO PIPELINE TEST V3 - VERBOSE MODE")
    print("=" * 80)
    print("This will:")
    print("  1. Load 4 models (1.5B + 7B + 1.5B + 0.5B)")
    print("  2. Run 2 GRPO training iterations on 5 problems")
    print("  3. Print progress updates in real-time")
    print()
    print("Expected time: 20-30 minutes")
    print("Expected cost: ~$15")
    print("=" * 80)
    print()

    result = run_grpo_test.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ GRPO PIPELINE TEST PASSED!")
        print()
        print("Next step: Modify main config and run full Phase 3")
    else:
        print("❌ GRPO PIPELINE TEST FAILED")
        print()
        print("Review errors above")
    print("=" * 80)
