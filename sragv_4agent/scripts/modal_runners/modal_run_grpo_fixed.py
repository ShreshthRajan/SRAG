"""
Modal deployment for GRPO pipeline test - Version 4 (Real-time streaming).
Tests Qwen2.5-Coder-1.5B-Instruct (70% HumanEval baseline) with GRPO training.
Streams output in real-time instead of buffering.
"""

import modal
import os

app = modal.App("srag-grpo-fixed-v1")

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
    """Run verbose GRPO pipeline test with real-time output."""
    import subprocess
    import sys
    import os

    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("STARTING GRPO PIPELINE TEST", flush=True)
    print("=" * 80, flush=True)

    # Run with unbuffered output and stream in real-time
    process = subprocess.Popen(
        [sys.executable, "-u", "test_grpo_verbose.py"],  # -u for unbuffered
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1  # Line buffered
    )

    # Stream output line by line
    for line in process.stdout:
        print(line, end='', flush=True)

    # Wait for completion
    return_code = process.wait()

    print("\n" + "=" * 80, flush=True)
    print(f"Process completed with return code: {return_code}", flush=True)
    print("=" * 80, flush=True)

    return {
        "success": return_code == 0,
        "return_code": return_code
    }

@app.local_entrypoint()
def main():
    """Run test from local machine."""
    print("=" * 80)
    print("GRPO PIPELINE TEST V4 - REAL-TIME STREAMING")
    print("=" * 80)
    print("This will:")
    print("  1. Load 4 models (1.5B + 7B + 1.5B + 0.5B)")
    print("  2. Run 2 GRPO training iterations on 5 problems")
    print("  3. Stream progress updates in REAL-TIME")
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
        print(f"❌ GRPO PIPELINE TEST FAILED (exit code {result['return_code']})")
        print()
        print("Review errors above")
    print("=" * 80)
