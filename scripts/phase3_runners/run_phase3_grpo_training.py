#!/usr/bin/env python3
"""
SRAG-V Phase 3: GRPO Self-Play Training
Builds on Phase 1 calibrated checkpoint (ECE 0.00395).

Implements:
- 4-player self-play architecture
- GRPO (Group Relative Policy Optimization)
- Execution-based rewards
- MAP-Elites diversity archive
- End-to-end reinforcement learning

Model: Qwen2.5-Coder-1.5B-Instruct (calibrated)
Target: 70% → 80-85% on HumanEval
Approach: Multi-agent RL with execution feedback

Author: Shreshth Rajan
Date: November 2025
"""

import os
import sys
import time
import json
import logging
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

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
log_file = f"logs/phase3_grpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)


def load_phase1_checkpoint(orchestrator):
    """Load Phase 1 calibrated checkpoint into solution generator."""
    logger.info("Loading Phase 1 calibrated checkpoint...")

    # Try multiple checkpoint locations
    checkpoint_paths = [
        "checkpoints/phase1_star_calibrator_1764099746.pt",
        "checkpoints/phase1_star_calibrator_latest.pt",
        "checkpoints/phase1_star_calibrator_success.pt",
    ]

    for ckpt_path in checkpoint_paths:
        if Path(ckpt_path).exists():
            try:
                orchestrator.solution_generator.load_calibrator(ckpt_path)
                logger.info(f"✅ Loaded calibrator from {ckpt_path}")
                return True
            except Exception as e:
                logger.warning(f"Failed to load {ckpt_path}: {e}")
                continue

    logger.warning("⚠️ No Phase 1 checkpoint found - proceeding with uncalibrated model")
    return False


def run_phase3_grpo_training():
    """Run Phase 3 GRPO self-play training."""
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("SRAG-V PHASE 3: GRPO SELF-PLAY TRAINING (PILOT)")
    logger.info("🔧 CODE VERSION: FIXED - Gradients enabled, KeyError resolved")
    logger.info("=" * 80)
    logger.info("Model: Qwen2.5-Coder-1.5B-Instruct (calibrated)")
    logger.info("Method: 4-player GRPO with execution feedback")
    logger.info("Iterations: 5 (pilot to validate architecture)")
    logger.info("Problems per iteration: 20")
    logger.info("Expected duration: 5-6 hours")
    logger.info("Expected cost: $40-50")
    logger.info("Fixes applied:")
    logger.info("  ✅ Gradient bug fixed (removed no_grad from log_prob computation)")
    logger.info("  ✅ KeyError fixed (handles question/description fields)")
    logger.info("  ✅ JSON cleaning improved (verification generator)")
    logger.info("=" * 80)

    results = {
        "phase": "phase3_grpo_training",
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "config": {
            "model": "Qwen2.5-Coder-1.5B-Instruct",
            "num_iterations": 6,
            "problems_per_iteration": 32,
            "method": "GRPO"
        }
    }

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

        # Load Phase 1 calibrated checkpoint
        load_phase1_checkpoint(orchestrator)

        results["stages"] = {
            "initialization": {
                "duration": time.time() - stage_start,
                "status": "completed"
            }
        }
        logger.info(f"✅ Stage 1 complete ({results['stages']['initialization']['duration']:.1f}s)")

        # Stage 2: Configure GRPO training
        logger.info("Stage 2: Configuring GRPO self-play...")
        stage_start = time.time()

        # Pilot config: 5 iterations to validate architecture
        self_play_config = SelfPlayConfig(
            num_iterations=5,
            bootstrap_iterations=2,
            competitive_iterations=2,
            league_iterations=1,

            problems_per_iteration=20,  # Reduced for pilot
            solutions_per_problem=8,
            test_cases_per_problem=8,

            gradient_accumulation_steps=4,
            mixed_precision=True,
            checkpoint_every_iterations=2,

            min_solution_accuracy=0.5,  # Lower threshold for 1.5B
            min_test_validity=0.5,
            convergence_threshold=0.05
        )

        trainer = SelfPlayTrainer(
            orchestrator=orchestrator,
            config=self_play_config,
            checkpoint_dir="checkpoints/phase3_grpo"
        )

        results["stages"]["configuration"] = {
            "duration": time.time() - stage_start,
            "status": "completed",
            "config": {
                "iterations": self_play_config.num_iterations,
                "problems_per_iteration": self_play_config.problems_per_iteration,
                "solutions_per_problem": self_play_config.solutions_per_problem
            }
        }
        logger.info(f"✅ Stage 2 complete ({results['stages']['configuration']['duration']:.1f}s)")

        # Stage 3: Run GRPO training
        logger.info("Stage 3: Running GRPO self-play training...")
        logger.info("This will take 15-20 hours...")
        stage_start = time.time()

        training_results = trainer.run_self_play_training()

        results["stages"]["grpo_training"] = {
            "duration": time.time() - stage_start,
            "status": "completed",
            "iterations_completed": len(training_results.get("iterations", [])),
            "training_time": training_results.get("training_time", 0),
            "breakthrough_score": training_results.get("breakthrough_indicators", {}).get("breakthrough_score", 0)
        }
        logger.info(f"✅ Stage 3 complete ({results['stages']['grpo_training']['duration']:.1f}s)")

        # Final results
        total_time = time.time() - start_time
        results.update({
            "status": "completed",
            "end_time": datetime.now().isoformat(),
            "total_duration": total_time,
            "training_results": training_results,
            "checkpoint_dir": "checkpoints/phase3_grpo",
            "pilot_run": True,
            "total_episodes": 100
        })

        logger.info("=" * 80)
        logger.info("🎉 PHASE 3 PILOT TRAINING COMPLETED")
        logger.info(f"Total time: {total_time/3600:.1f} hours")
        logger.info(f"Iterations: {results['stages']['grpo_training']['iterations_completed']}/5")
        logger.info(f"Total episodes: 100")
        logger.info(f"Checkpoint saved to: checkpoints/phase3_grpo/")
        logger.info("=" * 80)
        logger.info("")
        logger.info("NEXT STEP: Run Phase 4 HumanEval evaluation to measure improvement")
        logger.info("Expected: +0.5% to +1.5% improvement if architecture is effective")
        logger.info("=" * 80)

        # Save results
        results_path = f"results/phase3_grpo/phase3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_path}")

        return 0

    except Exception as e:
        logger.error(f"❌ Phase 3 training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        results.update({
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

        # Save partial results
        results_path = f"results/phase3_grpo/phase3_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        return 1


if __name__ == "__main__":
    sys.exit(run_phase3_grpo_training())
