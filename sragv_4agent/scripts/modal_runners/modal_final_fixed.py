"""
Final fixed GRPO test - all bugs resolved.
Tests Qwen2.5-Coder-1.5B-Instruct with proper reward extraction and brace escaping.
"""

import modal
import os

app = modal.App("srag-grpo-final-fixed")

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
    timeout=7200
)
def run_grpo_test():
    """Run GRPO pipeline test with all fixes applied."""
    import subprocess
    import sys
    import os

    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("GRPO FINAL TEST - ALL BUGS FIXED", flush=True)
    print("=" * 80, flush=True)

    # Run with unbuffered output
    process = subprocess.Popen(
        [sys.executable, "-u", "test_grpo_verbose.py"],
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
    print(f"Return code: {return_code}", flush=True)
    print("=" * 80, flush=True)

    return {"success": return_code == 0}

@app.local_entrypoint()
def main():
    """Run test."""
    print("=" * 80)
    print("SRAG-V GRPO PIPELINE TEST - FINAL FIXED VERSION")
    print("=" * 80)
    print("Fixes applied:")
    print("  1. Escaped JSON braces in verification generator")
    print("  2. Fixed GRPO reward extraction (isinstance check first)")
    print("  3. Proper type handling for float/numpy types")
    print()
    print("Expected: ZERO errors, clean 2-iteration completion")
    print("=" * 80)
    print()

    result = run_grpo_test.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ ALL TESTS PASSED - GRPO PIPELINE FULLY OPERATIONAL")
    else:
        print("❌ TEST FAILED - Review errors above")
    print("=" * 80)
