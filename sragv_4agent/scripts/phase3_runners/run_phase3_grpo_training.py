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
    logger.info("SRAG-V PHASE 3: FULL GRPO SELF-PLAY TRAINING")
    logger.info("🔧 CODE VERSION: v25 - Per-sample backward (8x memory reduction)")
    logger.info("=" * 80)
    logger.info("Model: Qwen2.5-Coder-1.5B-Instruct (calibrated)")
    logger.info("Method: 4-player GRPO with execution feedback")
    logger.info("Training scope: 16 iterations (single run)")
    logger.info("Group size: 4 (research-valid 4-8 range, memory-optimized)")
    logger.info("Data: Real APPS coding problems (in-distribution for HumanEval)")
    logger.info("Total samples: 16 × 16 × 4 = 1,024 (meets 1,000 threshold)")
    logger.info("Expected duration: ~12-15 hours")
    logger.info("Expected cost: ~$137 training + ~$15 evals = ~$152 total")
    logger.info("")
    logger.info("🔧 MEMORY FIX v25 (addresses OOM on 80GB A100):")
    logger.info("  ✅ Per-SAMPLE processing: backward() after EACH forward (not batched)")
    logger.info("  ✅ Sequence length capped: max_length=1024 (vs unbounded 2000+)")
    logger.info("  ✅ Aggressive memory cleanup: del tensors + empty_cache() after each sample")
    logger.info("  ✅ Peak memory: ~10GB per GPU (was 79GB, 8x reduction)")
    logger.info("")
    logger.info("Memory architecture:")
    logger.info("  OLD v24: Per role: 8 log_probs → backward → still OOM (80GB)")
    logger.info("  NEW v25: Per sample: 1 forward → backward → delete → repeat (10GB safe)")
    logger.info("")
    logger.info("Additional optimizations:")
    logger.info("  ✅ Gradient checkpointing enabled for all 4 players")
    logger.info("  ✅ fp16 precision for all models")
    logger.info("  ✅ GPU cache clearing between chunks and iterations")
    logger.info("Multi-GPU distribution:")
    logger.info("  ✅ GPU 0: Problem + Verification Generators")
    logger.info("  ✅ GPU 1: Solution + Meta-Verifier")
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

        # CRITICAL: Validate APPS dataset loads successfully
        logger.info("=" * 80)
        logger.info("VALIDATING APPS DATASET (CRITICAL CHECK)")
        logger.info("=" * 80)

        orchestrator.initialize_data_loaders()
        bootstrap_problems, categorized = orchestrator.load_bootstrap_data()

        # Check if we got real APPS data
        if bootstrap_problems:
            first_problem_source = bootstrap_problems[0].get('source', 'unknown')
            problem_count = len(bootstrap_problems)

            logger.info(f"Loaded {problem_count} problems")
            logger.info(f"First problem source: {first_problem_source}")
            logger.info(f"First problem ID: {bootstrap_problems[0].get('problem_id', 'unknown')}")

            if first_problem_source == 'synthetic':
                logger.error("=" * 80)
                logger.error("❌ CRITICAL ERROR: Using SYNTHETIC data instead of APPS!")
                logger.error("This will NOT transfer to HumanEval and Phase 4 will fail")
                logger.error("APPS dataset failed to load - check datasets version")
                logger.error("=" * 80)
                raise RuntimeError("APPS dataset not loaded - using synthetic data. ABORT training.")
            else:
                logger.info("✅ VERIFIED: Using real APPS dataset")
                logger.info(f"   Problem source: {first_problem_source}")
                logger.info(f"   Total problems: {problem_count}")
                logger.info("   In-distribution for HumanEval transfer ✅")
        else:
            logger.error("❌ No bootstrap problems loaded!")
            raise RuntimeError("Failed to load any bootstrap data")

        logger.info("=" * 80)

        # Stage 2: Configure GRPO training
        logger.info("Stage 2: Configuring GRPO self-play...")
        stage_start = time.time()

        # Phase 3 config: 16 iterations with memory-optimized 4-player training
        # MEMORY FIX APPLIED: Per-role processing + max_seq_length=1024
        # Total: 16 × 16 × 4 = 1,024 APPS samples for in-distribution training
        #
        # Memory calculation with fix:
        #   - Per-role processing: ~4 forward passes per role per chunk (vs 36 all roles)
        #   - Sequence length capped: 1024 tokens (vs unbounded 2000+)
        #   - Peak memory: ~13 GB per GPU (vs 79+ GB before)
        self_play_config = SelfPlayConfig(
            num_iterations=16,  # Conservative budget: 16 iters = $137 training + $15 evals = $152 (leaves $34 buffer)
            bootstrap_iterations=4,
            competitive_iterations=8,
            league_iterations=4,

            problems_per_iteration=16,  # Increased from 12 (safe with memory fix)
            solutions_per_problem=4,    # Reduced from 6 (group_size=4 is memory-safe, research-valid range 4-8)
            test_cases_per_problem=4,   # Reduced from 8 (matches verification config that worked)

            gradient_accumulation_steps=8,  # 8 chunks for smooth gradient accumulation
            mixed_precision=True,
            checkpoint_every_iterations=4,  # Save every 4 iterations (4, 8, 12, 16)

            min_solution_accuracy=0.5,  # Lower threshold for 1.5B
            min_test_validity=0.5,
            convergence_threshold=0.05,

            skip_volume_resume=True  # CRITICAL: Skip old checkpoint, start fresh with memory fix
        )

        # Use fresh checkpoint directory to avoid contamination from old 2-player training
        fresh_checkpoint_dir = "checkpoints/phase3_grpo_4player"

        trainer = SelfPlayTrainer(
            orchestrator=orchestrator,
            config=self_play_config,
            checkpoint_dir=fresh_checkpoint_dir
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
