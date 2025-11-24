#!/usr/bin/env python3
"""
Verbose GRPO Pipeline Test - with immediate output flushing.
"""

import os
import sys
from pathlib import Path

# Add src to path FIRST
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

def log(msg):
    """Print with immediate flush."""
    print(msg, flush=True)

def test_grpo_pipeline():
    """Test GRPO pipeline with verbose output."""

    log("=" * 80)
    log("GRPO PIPELINE TEST - VERBOSE MODE")
    log("=" * 80)

    # Step 1: Create config
    log("\n[1/5] Creating test config...")
    import yaml

    test_config = {
        'sragv': {
            'dataset': {
                'cache_dir': '.cache',
                'max_problems': 10
            },
            'models': {
                'problem_generator': {
                    'name': 'Qwen/Qwen2.5-1.5B-Instruct',
                    'max_length': 2048,
                    'temperature': 0.8,
                    'top_p': 0.95
                },
                'solution_generator': {
                    'name': 'Qwen/Qwen2.5-Coder-1.5B-Instruct',  # Use 1.5B-Instruct (70% baseline, room for improvement)
                    'max_length': 4096,
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'quantization': '4bit',
                    'lora_rank': 64
                },
                'verification_generator': {
                    'name': 'Qwen/Qwen2.5-1.5B-Instruct',
                    'max_length': 2048,
                    'temperature': 0.7,
                    'top_p': 0.9
                },
                'meta_verifier': {
                    'name': 'Qwen/Qwen2.5-0.5B-Instruct',
                    'max_length': 1024,
                    'temperature': 0.1,
                    'top_p': 0.95
                }
            }
        }
    }

    with open('test_config.yaml', 'w') as f:
        yaml.dump(test_config, f)
    log("✅ Config created")

    # Step 2: Import modules
    log("\n[2/5] Importing SRAG modules...")
    try:
        from sragv.orchestrator import SRAGVOrchestrator
        from sragv.training.self_play_trainer import SelfPlayTrainer, SelfPlayConfig
        log("✅ Imports successful")
    except Exception as e:
        log(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Initialize orchestrator (loads all 4 models)
    log("\n[3/5] Initializing orchestrator (loading 4 models)...")
    log("   This will take 5-10 minutes...")
    try:
        orchestrator = SRAGVOrchestrator('test_config.yaml')
        log("   Config loaded, initializing players...")
        orchestrator.initialize_players()
        log("✅ Orchestrator initialized (all 4 models loaded)")
    except Exception as e:
        log(f"❌ Orchestrator init failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Initialize trainer
    log("\n[4/5] Initializing SelfPlayTrainer with GRPO...")
    try:
        self_play_config = SelfPlayConfig(
            num_iterations=2,  # Just 2 iterations
            problems_per_iteration=5,  # Small test
            solutions_per_problem=4,
        )

        trainer = SelfPlayTrainer(
            orchestrator=orchestrator,
            config=self_play_config
        )
        log("✅ Trainer initialized")
    except Exception as e:
        log(f"❌ Trainer init failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Run training
    log("\n[5/5] Running GRPO training (2 iterations, 5 problems)...")
    log("   This will take 10-20 minutes...")
    try:
        training_results = trainer.run_self_play_training()
        log("✅ Training completed!")

        # Verify training actually succeeded
        iterations_completed = len(training_results.get('iterations', []))
        expected_iterations = 2

        log(f"   Iterations completed: {iterations_completed}/{expected_iterations}")

        if iterations_completed < expected_iterations:
            log(f"❌ Training incomplete - only {iterations_completed} iterations completed")
            return False

        # Check for errors
        if 'error' in training_results:
            log(f"❌ Training had errors: {training_results['error']}")
            return False

        log(f"   Status: SUCCESS")
        log(f"   Training time: {training_results.get('training_time', 0):.1f}s")

    except Exception as e:
        log(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    log("\n" + "=" * 80)
    log("✅ ALL TESTS PASSED - GRPO PIPELINE WORKS!")
    log("=" * 80)

    return True

if __name__ == "__main__":
    success = test_grpo_pipeline()
    sys.exit(0 if success else 1)
