"""
SRAG-V: VERIFY 4-PLAYER TRAINING FIX
Quick 2-iteration run on Modal to verify all 4 players are training.

Usage:
    modal run scripts/modal_runners/modal_verify_fix.py

Expected output:
    ✅ "GRPO training data: {...verification_generator..., ...meta_verifier...}"
    ✅ "✅ All 4 players trained - losses: ..."

If you see "⚠️ MISSING TRAINING FOR ROLES", the fix didn't work.

Expected cost: ~$5-8
Expected time: ~15-30 minutes
"""

import modal
import os

app = modal.App("srag-verify-4player-fix")

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
        "datasets==2.14.6",
        "pyarrow==14.0.1",
        "numpy>=1.24.0,<2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "psutil>=5.9.0",
    )
    .add_local_dir(local_dir, remote_path="/workspace/srag")
)

# Persistent volume for results
volume = modal.Volume.from_name("srag-results", create_if_missing=True)


@app.function(
    image=image,
    gpu="A100-80GB:2",  # 2×A100-80GB to match production (avoids OOM during GRPO step)
    timeout=7200,  # 2 hours (2 iterations take ~70 min, plus buffer)
    volumes={"/results": volume}
)
def run_verification():
    """Run 2-iteration verification of 4-player fix."""
    import subprocess
    import sys
    import os

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace/srag")

    print("=" * 80, flush=True)
    print("SRAG-V: VERIFY 4-PLAYER TRAINING FIX", flush=True)
    print("=" * 80, flush=True)
    print("Purpose: Verify all 4 players receive GRPO training", flush=True)
    print("Iterations: 2", flush=True)
    print("", flush=True)
    print("WHAT TO LOOK FOR:", flush=True)
    print("  ✅ 'GRPO training data: {...verification_generator..., ...meta_verifier...}'", flush=True)
    print("  ✅ '✅ All 4 players trained - losses: ...'", flush=True)
    print("  ❌ '⚠️ MISSING TRAINING FOR ROLES' = FIX FAILED", flush=True)
    print("=" * 80, flush=True)
    print("", flush=True)

    # Run verification script
    process = subprocess.Popen(
        [sys.executable, "-u", "scripts/phase3_runners/run_phase3_verify_fix.py"],
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
    if return_code == 0:
        print("✅ VERIFICATION PASSED - All 4 players are training!", flush=True)

        # Verify checkpoint files were saved correctly
        print("\n📁 CHECKPOINT VERIFICATION:", flush=True)
        import glob
        checkpoint_dirs = glob.glob("checkpoints/phase3_verify_*/grpo_iteration_*")
        if checkpoint_dirs:
            for ckpt_dir in checkpoint_dirs:
                print(f"  Found: {ckpt_dir}", flush=True)
                # Check for all 4 adapter directories
                adapters = ["problem_generator", "solution_generator", "verification_generator", "meta_verifier"]
                for adapter in adapters:
                    adapter_path = os.path.join(ckpt_dir, f"{adapter}_adapter", "adapter_model.safetensors")
                    if os.path.exists(adapter_path):
                        size_mb = os.path.getsize(adapter_path) / (1024 * 1024)
                        print(f"    ✅ {adapter}_adapter: {size_mb:.1f} MB", flush=True)
                    else:
                        print(f"    ❌ {adapter}_adapter: MISSING!", flush=True)
        else:
            print("  ⚠️ No checkpoint directories found", flush=True)
    else:
        print("❌ VERIFICATION FAILED - Check logs above", flush=True)
    print("=" * 80, flush=True)

    return {"success": return_code == 0, "return_code": return_code}


@app.local_entrypoint()
def main():
    """Launch verification run."""
    print("=" * 80)
    print("SRAG-V: VERIFY 4-PLAYER TRAINING FIX")
    print("=" * 80)
    print()
    print("This will run 2 iterations to verify:")
    print("  - All 4 players receive training data")
    print("  - All 4 players have loss values in metrics")
    print("  - Checkpoints save correctly")
    print()
    print("Expected cost: ~$5-8")
    print("Expected time: ~15-30 minutes")
    print()
    print("Starting Modal job...")
    print("=" * 80)

    result = run_verification.remote()

    print()
    print("=" * 80)
    if result["success"]:
        print("✅ VERIFICATION PASSED!")
        print("You can now run full training with:")
        print("  modal run scripts/modal_runners/modal_phase3_grpo.py")
    else:
        print("❌ VERIFICATION FAILED")
        print("Check the logs above for errors.")
    print("=" * 80)
