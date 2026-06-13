#!/usr/bin/env python3
"""
Verify Phase 4 is ready to run.
Checks all prerequisites before Modal deployment.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent


def check_prerequisites():
    """Check all prerequisites for Phase 4."""

    print("=" * 80)
    print("PHASE 4 READINESS CHECK")
    print("=" * 80)
    print()

    all_ready = True

    # Check 1: GRPO checkpoint exists
    print("[1/6] Checking GRPO checkpoint...")

    grpo_checkpoint = project_root / "checkpoints" / "phase3_grpo" / "grpo_iteration_16"
    solution_adapter = grpo_checkpoint / "solution_generator_adapter"
    adapter_model = solution_adapter / "adapter_model.safetensors"

    if not grpo_checkpoint.exists():
        print("  ❌ GRPO checkpoint directory not found")
        print(f"     Path: {grpo_checkpoint}")
        print("     Download with:")
        print("       python3 -m modal volume get srag-results phase3_grpo/grpo_iteration_16 checkpoints/phase3_grpo/grpo_iteration_16")
        all_ready = False
    elif not solution_adapter.exists():
        print("  ❌ Solution generator adapter not found")
        all_ready = False
    elif not adapter_model.exists():
        print("  ❌ Adapter model file not found")
        all_ready = False
    else:
        adapter_size = adapter_model.stat().st_size / (1024 * 1024)
        print(f"  ✅ GRPO checkpoint ready ({adapter_size:.1f} MB)")

    # Check 2: Config file
    print("[2/6] Checking config file...")

    config_file = project_root / "config" / "config.yaml"

    if not config_file.exists():
        print("  ❌ Config file not found")
        all_ready = False
    else:
        print(f"  ✅ Config file exists")

    # Check 3: Source code
    print("[3/6] Checking source code...")

    required_files = [
        "src/sragv/orchestrator.py",
        "src/sragv/models/solution_generator.py",
        "src/sragv/models/base_player.py"
    ]

    missing_files = []
    for file_path in required_files:
        if not (project_root / file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"  ❌ Missing source files: {missing_files}")
        all_ready = False
    else:
        print(f"  ✅ All source files present")

    # Check 4: Evaluation scripts
    print("[4/6] Checking evaluation scripts...")

    eval_script = project_root / "scripts" / "phase4_runners" / "run_phase4_grpo_humaneval.py"
    modal_script = project_root / "scripts" / "modal_runners" / "modal_phase4_grpo_humaneval.py"

    if not eval_script.exists():
        print("  ❌ Evaluation script not found")
        all_ready = False
    elif not modal_script.exists():
        print("  ❌ Modal deployment script not found")
        all_ready = False
    else:
        print(f"  ✅ Evaluation scripts ready")

    # Check 5: Python dependencies (for Modal deployment)
    print("[5/6] Checking Modal connection...")

    try:
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "modal", "token", "show"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            print("  ✅ Modal authenticated")
        else:
            print("  ⚠️  Modal not authenticated")
            print("     Run: modal token set --token-id xxx --token-secret yyy")
            all_ready = False
    except Exception:
        print("  ⚠️  Could not check Modal authentication")

    # Check 6: Volume checkpoint exists
    print("[6/6] Checking Modal volume checkpoint...")

    try:
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "modal", "volume", "ls", "srag-results/phase3_grpo"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if "grpo_iteration_16" in result.stdout:
            print("  ✅ GRPO checkpoint exists in Modal volume")
        else:
            print("  ⚠️  GRPO checkpoint not found in Modal volume")
            print("     This is OK if checkpoint was just downloaded locally")
    except Exception as e:
        print(f"  ⚠️  Could not check Modal volume: {e}")

    print()
    print("=" * 80)

    if all_ready:
        print("✅ ALL CHECKS PASSED - READY FOR PHASE 4")
        print()
        print("Run deployment with:")
        print("  python3 scripts/modal_runners/modal_phase4_grpo_humaneval.py")
        print()
        print("Expected:")
        print("  ⏱️  Runtime: 3-5 hours")
        print("  💰 Cost: $10-15")
        print("  📊 Improvement: +0.5-1.5% (predicted)")
        print()
    else:
        print("❌ PREREQUISITES MISSING")
        print()
        print("Fix issues above before deploying to Modal")
        print()

    print("=" * 80)

    return all_ready


if __name__ == "__main__":
    ready = check_prerequisites()
    sys.exit(0 if ready else 1)
