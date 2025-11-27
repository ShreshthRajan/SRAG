"""
Production GRPO Test - Final Version
All bugs fixed: JSON brace escaping + reward extraction
"""

import modal
import os

app = modal.App("srag-production-test")

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
def run_production_test():
    """Run production GRPO test."""
    import subprocess
    import sys
    import os

    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("SRAG-V GRPO PRODUCTION TEST", flush=True)
    print("=" * 80, flush=True)

    process = subprocess.Popen(
        [sys.executable, "-u", "test_grpo_verbose.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)

    return_code = process.wait()

    print("\n" + "=" * 80, flush=True)
    print(f"Return code: {return_code}", flush=True)
    print("=" * 80, flush=True)

    return {"success": return_code == 0, "return_code": return_code}

@app.local_entrypoint()
def main():
    """Run production test."""
    print("=" * 80)
    print("SRAG-V GRPO PRODUCTION TEST")
    print("=" * 80)
    print("All critical bugs fixed:")
    print("  ✅ JSON brace escaping in verification generator")
    print("  ✅ Reward extraction from RewardMetrics (single-layer)")
    print("  ✅ Removed redundant wrapper that was destroying rewards")
    print()
    print("Expected: Clean execution with actual reward values")
    print("=" * 80)
    print()

    result = run_production_test.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ PRODUCTION TEST PASSED")
    else:
        print(f"❌ TEST FAILED (exit code {result['return_code']})")
    print("=" * 80)
