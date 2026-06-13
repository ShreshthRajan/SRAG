"""
Test resume functionality on Modal with 2+2 iterations.

Run this FIRST to verify resume works ($6 cost).
Then run full phase3_grpo with confidence.
"""

import modal
import os

app = modal.App("srag-phase3-test-resume")

# Get project root
local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Use same image as production
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

# Use same volume as production
volume = modal.Volume.from_name("srag-results", create_if_missing=True)

@app.function(
    image=image,
    gpu="A100-80GB:2",
    timeout=10800,  # 3 hours (test only)
    volumes={"/results": volume}
)
def run_test():
    """Run resume test."""
    import subprocess
    import sys
    import os

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    os.chdir("/workspace/srag")

    print("=" * 80)
    print("RESUME FUNCTIONALITY TEST")
    print("=" * 80)
    print("This runs 4 iterations total:")
    print("  - If fresh: Runs 1-4, saves checkpoint at 2")
    print("  - If resuming: Detects checkpoint at 2, runs 3-4")
    print()
    print("Expected:")
    print("  - First run: Completes iterations 1-4, checkpoint at 2")
    print("  - Second run: Resumes from 2, runs 3-4 again")
    print("  - Cost: ~$6")
    print("=" * 80)
    print()

    # Run test
    process = subprocess.Popen(
        [sys.executable, "-u", "scripts/phase3_runners/run_phase3_test_resume.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)

    return_code = process.wait()

    print("\n" + "=" * 80)
    print(f"Test completed with return code: {return_code}")
    print("=" * 80)

    # Copy any results to volume
    import shutil
    if os.path.exists("checkpoints/phase3_grpo"):
        print("Copying test checkpoints to volume...")
        os.makedirs("/results/phase3_grpo_test", exist_ok=True)
        for item in os.listdir("checkpoints/phase3_grpo"):
            src = os.path.join("checkpoints/phase3_grpo", item)
            dst = os.path.join("/results/phase3_grpo_test", item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"  Copied: {item}")

        volume.commit()
        print("✅ Test results saved to volume")

    if return_code == 0:
        print("\n✅ TEST PASSED - Resume functionality working")
        print("Ready to run full 10+4 training")
    else:
        print("\n❌ TEST FAILED - Check logs above")

    return {"success": return_code == 0, "return_code": return_code}


@app.local_entrypoint()
def main():
    """Run resume test."""
    print("=" * 80)
    print("SRAG-V PHASE 3: RESUME FUNCTIONALITY TEST")
    print("=" * 80)
    print()
    print("This test verifies checkpoint resume works:")
    print("  1. Runs 4 iterations (saves checkpoint at 2)")
    print("  2. On second run, resumes from 2 and continues")
    print()
    print("Cost: ~$6")
    print("Time: ~45 minutes")
    print()
    print("If successful, you can run full 10+4 with confidence")
    print("=" * 80)
    print()

    input("Press Enter to start resume test (or Ctrl+C to cancel)...")
    print()

    result = run_test.remote()

    print("\n" + "=" * 80)
    if result["success"]:
        print("✅ RESUME TEST PASSED")
        print()
        print("Next steps:")
        print("  1. Run full training: modal run scripts/modal_runners/modal_phase3_grpo.py")
        print("  2. After 10 iterations complete, checkpoints saved to volume")
        print("  3. Run again to resume from iteration 10 and complete 11-14")
    else:
        print(f"❌ TEST FAILED (code {result['return_code']})")
        print()
        print("Do not run full training until resume is fixed")
    print("=" * 80)
