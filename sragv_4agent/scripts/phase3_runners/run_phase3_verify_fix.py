#!/usr/bin/env python3
"""
SRAG-V Phase 3: VERIFY 4-PLAYER FIX
Quick 2-iteration run to verify all 4 players are training.

Look for these log messages:
- "GRPO training data: {'problem_generator': N, 'solution_generator': N, 'verification_generator': N, 'meta_verifier': N}"
- "✅ All 4 players trained - losses: meta_verifier=X.XX, problem_generator=X.XX, solution_generator=X.XX, verification_generator=X.XX"

If you see "⚠️ MISSING TRAINING FOR ROLES", the fix didn't work.
"""

import os
import sys
import time
import json
import logging
import torch
from datetime import datetime
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Create directories
for dir_name in ["logs", "checkpoints/phase3_grpo", "results/phase3_grpo"]:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

# Logging
log_file = f"logs/phase3_verify_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)


def run_verification():
    """Run 2-iteration verification of 4-player training fix."""
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("SRAG-V PHASE 3: VERIFY 4-PLAYER TRAINING FIX")
    logger.info("=" * 80)
    logger.info("Purpose: Verify all 4 players receive GRPO training")
    logger.info("Iterations: 2 (minimum for checkpoint test)")
    logger.info("")
    logger.info("WHAT TO LOOK FOR:")
    logger.info("  ✅ 'GRPO training data: {...verification_generator..., ...meta_verifier...}'")
    logger.info("  ✅ '✅ All 4 players trained - losses: ...'")
    logger.info("  ❌ '⚠️ MISSING TRAINING FOR ROLES' = FIX FAILED")
    logger.info("=" * 80)

    try:
        from sragv.orchestrator import SRAGVOrchestrator
        from sragv.training.self_play_trainer import SelfPlayTrainer, SelfPlayConfig

        logger.info("✅ Imports successful")

        # Initialize orchestrator
        logger.info("Initializing 4-player architecture...")
        orchestrator = SRAGVOrchestrator("config/config.yaml")
        orchestrator.initialize_players()
        logger.info("✅ Players initialized")

        # Configure for quick verification (2 iterations)
        self_play_config = SelfPlayConfig(
            num_iterations=2,  # Just 2 to verify fix
            bootstrap_iterations=1,
            competitive_iterations=1,
            league_iterations=0,

            problems_per_iteration=4,  # Small batch for speed
            solutions_per_problem=4,   # Reduced for memory
            test_cases_per_problem=4,  # Reduced for memory

            gradient_accumulation_steps=2,
            mixed_precision=True,
            checkpoint_every_iterations=1,  # Checkpoint after each iteration

            min_solution_accuracy=0.3,  # Lower threshold for quick test
            min_test_validity=0.3,
            convergence_threshold=0.01,

            skip_volume_resume=True  # Don't resume from old 2-GPU checkpoints
        )

        # Use a fresh checkpoint dir to avoid resuming from old 2-GPU checkpoints
        import uuid
        fresh_checkpoint_dir = f"checkpoints/phase3_verify_{uuid.uuid4().hex[:8]}"

        trainer = SelfPlayTrainer(
            orchestrator=orchestrator,
            config=self_play_config,
            checkpoint_dir=fresh_checkpoint_dir
        )

        logger.info("✅ Trainer configured")
        logger.info(f"  Iterations: {self_play_config.num_iterations}")
        logger.info(f"  Problems/iteration: {self_play_config.problems_per_iteration}")

        # Run training
        logger.info("")
        logger.info("=" * 80)
        logger.info("STARTING 2-ITERATION VERIFICATION RUN")
        logger.info("=" * 80)

        results = trainer.run_self_play_training()

        # Check results
        logger.info("")
        logger.info("=" * 80)
        logger.info("VERIFICATION RESULTS")
        logger.info("=" * 80)

        # Check training metrics for all 4 players
        iterations = results.get("iterations", [])
        all_4_trained = True

        for i, iteration in enumerate(iterations):
            metrics = iteration.get("training_metrics", {})
            logger.info(f"Iteration {i+1} metrics:")

            expected_losses = [
                "problem_generator_loss",
                "solution_generator_loss",
                "verification_generator_loss",
                "meta_verifier_loss"
            ]

            for loss_name in expected_losses:
                loss_val = metrics.get(loss_name, "MISSING")
                if loss_val == "MISSING":
                    logger.error(f"  ❌ {loss_name}: MISSING")
                    all_4_trained = False
                else:
                    logger.info(f"  ✅ {loss_name}: {loss_val:.4f}")

        logger.info("")
        if all_4_trained:
            logger.info("=" * 80)
            logger.info("✅ SUCCESS: ALL 4 PLAYERS ARE TRAINING!")
            logger.info("=" * 80)
            logger.info("The fix is working. You can now run full training.")
        else:
            logger.error("=" * 80)
            logger.error("❌ FAILURE: NOT ALL 4 PLAYERS ARE TRAINING")
            logger.error("=" * 80)
            logger.error("The fix did not work. Check the code.")

        # Save verification results
        results_file = f"results/phase3_verify_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                "verification": "4-player training fix",
                "all_4_trained": all_4_trained,
                "duration": time.time() - start_time,
                "iterations": len(iterations),
                "results": results
            }, f, indent=2, default=str)

        logger.info(f"Results saved to: {results_file}")

        return all_4_trained

    except Exception as e:
        import traceback
        logger.error(f"❌ Verification failed: {e}")
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
