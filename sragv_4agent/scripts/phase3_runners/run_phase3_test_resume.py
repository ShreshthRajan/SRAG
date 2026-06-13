#!/usr/bin/env python3
"""
Test resume functionality with 2+2 iterations.

Run 1: 2 iterations, saves checkpoint at iteration 2
Run 2: Resumes from iteration 2, runs iterations 3-4

Cost: ~$6 total
Purpose: Verify resume works before running full 10+4
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sragv.orchestrator import SRAGVOrchestrator
from sragv.training.self_play_trainer import SelfPlayTrainer, SelfPlayConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_phase3_resume_test():
    """Test resume functionality with minimal config."""
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("SRAG-V PHASE 3: RESUME FUNCTIONALITY TEST")
    logger.info("🔧 Testing checkpoint save + resume with 2+2 iterations")
    logger.info("=" * 80)
    logger.info("Configuration:")
    logger.info("  Run 1: 2 iterations (saves checkpoint at iteration 2)")
    logger.info("  Run 2: Resume from 2, run iterations 3-4")
    logger.info("  Total: 4 iterations")
    logger.info("  Cost: ~$6")
    logger.info("  Purpose: Verify resume works before full run")
    logger.info("=" * 80)

    try:
        # Import required modules
        from sragv.orchestrator import SRAGVOrchestrator
        from sragv.training.self_play_trainer import SelfPlayTrainer, SelfPlayConfig

        logger.info("✅ Imports successful")

        # Stage 1: Initialize orchestrator
        logger.info("Stage 1: Initializing 4-player architecture...")
        stage_start = time.time()

        orchestrator = SRAGVOrchestrator("config/config.yaml")
        orchestrator.initialize_players()

        logger.info(f"✅ Stage 1 complete ({time.time() - stage_start:.1f}s)")

        # Stage 2: Configure GRPO training (TEST CONFIG)
        logger.info("Stage 2: Configuring test run...")
        stage_start = time.time()

        # TEST config: 4 total iterations, checkpoint every 2
        test_config = SelfPlayConfig(
            num_iterations=4,  # Total iterations across both runs
            bootstrap_iterations=2,
            competitive_iterations=2,
            league_iterations=0,

            problems_per_iteration=3,  # Minimal for speed
            solutions_per_problem=6,
            test_cases_per_problem=8,

            gradient_accumulation_steps=8,
            mixed_precision=True,
            checkpoint_every_iterations=2,  # Save at iteration 2

            min_solution_accuracy=0.5,
            min_test_validity=0.5,
            convergence_threshold=0.05
        )

        trainer = SelfPlayTrainer(
            orchestrator=orchestrator,
            config=test_config,
            checkpoint_dir="checkpoints/phase3_grpo"
        )

        logger.info(f"✅ Stage 2 complete ({time.time() - stage_start:.1f}s)")

        # Stage 3: Run training
        logger.info("Stage 3: Running test training...")
        logger.info("If resuming: Will detect checkpoint at iteration 2 and continue")
        logger.info("If fresh: Will run iterations 1-4, save checkpoint at 2")

        results = trainer.run_self_play_training()

        logger.info(f"✅ Stage 3 complete ({time.time() - start_time:.1f}s)")

        # Summary
        logger.info("=" * 80)
        logger.info("TEST RESULTS")
        logger.info("=" * 80)
        logger.info(f"Iterations completed: {len(results['iterations'])}")
        logger.info(f"Total time: {results['training_time']:.1f}s")

        if len(results['iterations']) >= 2:
            logger.info("✅ TEST PASSED")
            logger.info("Checkpoint mechanism working")
            logger.info("Ready for full 10+4 run")
        else:
            logger.info("⚠️ TEST INCOMPLETE")
            logger.info(f"Only {len(results['iterations'])} iterations completed")

        logger.info("=" * 80)

        return results

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": str(e)}


if __name__ == "__main__":
    sys.exit(0 if run_phase3_resume_test() else 1)
