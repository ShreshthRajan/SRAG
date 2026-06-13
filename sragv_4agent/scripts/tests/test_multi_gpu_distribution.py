#!/usr/bin/env python3
"""
Test multi-GPU distribution fix.
Ensures models are placed on separate GPUs as expected.
"""

import torch
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

def test_gpu_distribution():
    """Test that models distribute across GPUs correctly."""

    print("=" * 70)
    print("MULTI-GPU DISTRIBUTION TEST (v4 - Integer Device IDs)")
    print("=" * 70)

    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    print(f"\n1. GPU Detection:")
    print(f"   Available GPUs: {num_gpus}")

    if num_gpus < 2:
        print(f"   ⚠️  WARNING: Only {num_gpus} GPU(s) available")
        print(f"   Multi-GPU distribution requires 2+ GPUs")
        print(f"   Test will proceed but all models will be on same GPU")
    else:
        print(f"   ✅ Sufficient GPUs for distribution")

    # Initialize orchestrator
    print(f"\n2. Initializing orchestrator...")
    try:
        from sragv.orchestrator import SRAGVOrchestrator
        orchestrator = SRAGVOrchestrator("config/config.yaml")
        print(f"   ✅ Orchestrator created")
    except Exception as e:
        print(f"   ❌ Failed to create orchestrator: {e}")
        return 1

    # Initialize players with GPU distribution
    print(f"\n3. Initializing 4-player architecture...")
    try:
        orchestrator.initialize_players()
        print(f"   ✅ All players initialized")
    except Exception as e:
        print(f"   ❌ Failed to initialize players: {e}")
        import traceback
        print(traceback.format_exc())
        return 1

    # Check device placement
    print(f"\n4. Verifying device placement:")

    players = [
        ("Problem Generator", orchestrator.problem_generator),
        ("Solution Generator", orchestrator.solution_generator),
        ("Verification Generator", orchestrator.verification_generator),
        ("Meta-Verifier", orchestrator.meta_verifier)
    ]

    device_map = {}
    all_correct = True

    for name, player in players:
        model = player.get_training_model()
        device = next(model.parameters()).device
        device_map[name] = device
        print(f"   {name}: {device}")

    # Verify expected distribution
    print(f"\n5. Validating distribution strategy:")

    if num_gpus >= 2:
        # Expected: Problem + Verification on GPU 0, Solution + Meta on GPU 1
        expected = {
            "Problem Generator": "cuda:0",
            "Solution Generator": "cuda:1",
            "Verification Generator": "cuda:0",
            "Meta-Verifier": "cuda:1"
        }

        for name, expected_device in expected.items():
            actual_device = str(device_map[name])
            if actual_device == expected_device:
                print(f"   ✅ {name}: {actual_device} (expected {expected_device})")
            else:
                print(f"   ❌ {name}: {actual_device} (expected {expected_device})")
                all_correct = False
    else:
        # Single GPU mode - all should be on cuda:0
        for name, device in device_map.items():
            if str(device) == "cuda:0":
                print(f"   ✅ {name}: {device} (single GPU mode)")
            else:
                print(f"   ❌ {name}: {device} (expected cuda:0 for single GPU)")
                all_correct = False

    # Test memory distribution
    print(f"\n6. GPU memory usage:")
    for i in range(num_gpus):
        allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
        reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
        print(f"   GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

    # Verify balanced memory distribution for multi-GPU
    if num_gpus >= 2:
        gpu0_mem = torch.cuda.memory_allocated(0) / 1024**3
        gpu1_mem = torch.cuda.memory_allocated(1) / 1024**3
        print(f"\n7. Memory balance check:")
        print(f"   GPU 0: {gpu0_mem:.2f} GB (Problem + Verification = 3B params)")
        print(f"   GPU 1: {gpu1_mem:.2f} GB (Solution + Meta = 2B params)")

        if gpu0_mem > 0 and gpu1_mem > 0:
            print(f"   ✅ Both GPUs have models loaded")
        else:
            print(f"   ❌ One or more GPUs have no models")
            all_correct = False

    print(f"\n" + "=" * 70)
    if all_correct:
        print("✅ ALL TESTS PASSED - Multi-GPU distribution is correct!")
        if num_gpus >= 2:
            print("Models are properly distributed across GPUs using integer device IDs.")
            print("GRPO training should no longer hit OOM errors.")
        print("=" * 70)
        return 0
    else:
        print("❌ TESTS FAILED - Device placement is incorrect")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(test_gpu_distribution())
