"""
Final Production GRPO Test - All Bugs Fixed
Version: 1.0 - Production Ready
"""

import modal
import os

app = modal.App("srag-grpo-final")

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
def run_final_test():
    """Run final production GRPO test."""
    import subprocess
    import sys
    import os

    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("SRAG-V GRPO - FINAL PRODUCTION TEST", flush=True)
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
    print(f"Process exit code: {return_code}", flush=True)
    print("=" * 80, flush=True)

    return {"success": return_code == 0, "return_code": return_code}

@app.local_entrypoint()
def main():
    """Run final test."""
    print("=" * 80)
    print("SRAG-V GRPO PIPELINE - FINAL PRODUCTION TEST")
    print("=" * 80)
    print("All bugs fixed:")
    print("  ✅ JSON brace escaping (verification generator)")
    print("  ✅ Reward extraction (GRPO trainer)")
    print("  ✅ Tensor device matching (CUDA compatibility)")
    print()
    print("Configuration:")
    print("  - Model: Qwen2.5-Coder-1.5B-Instruct")
    print("  - Iterations: 2")
    print("  - Problems: 5")
    print("  - GPU: A100-80GB")
    print()
    print("Expected: Clean 2-iteration completion with GRPO training")
    print("=" * 80)
    print()

    result = run_final_test.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ GRPO PIPELINE TEST PASSED")
        print()
        print("Next step: Run full Phase 1-3-4 training pipeline")
    else:
        print(f"❌ TEST FAILED (exit code {result['return_code']})")
    print("=" * 80)
