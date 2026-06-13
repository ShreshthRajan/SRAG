#!/usr/bin/env python3
"""
Quick test of Phase 4 evaluation on 3 HumanEval problems.
Verifies baseline and GRPO-trained models work before full deployment.
"""

import os
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import evaluation components
try:
    from human_eval.data import read_problems
    from human_eval.execution import check_correctness
except ImportError:
    logger.error("human-eval not installed. Install with: pip install human-eval")
    sys.exit(1)

try:
    from sragv.orchestrator import SRAGVOrchestrator
except Exception as e:
    logger.error(f"Failed to import SRAG-V: {e}")
    sys.exit(1)


def test_model(model_type: str, checkpoint_path: str = None):
    """Test model on 3 HumanEval problems."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {model_type} model")
    logger.info(f"{'='*60}")

    # Load model
    logger.info("Loading model...")
    config_path = project_root / "config" / "config.yaml"
    orchestrator = SRAGVOrchestrator(str(config_path))
    orchestrator.initialize_players()

    # Load adapter if specified
    if checkpoint_path:
        logger.info(f"Loading GRPO adapter: {checkpoint_path}")
        orchestrator.solution_generator.load_adapter(checkpoint_path)
        logger.info("✅ GRPO adapter loaded")
    else:
        logger.info("✅ Baseline model loaded (no adapter)")

    # Load HumanEval
    problems = read_problems()
    problem_list = list(problems.items())[:3]  # First 3 problems

    logger.info(f"Testing on {len(problem_list)} HumanEval problems\n")

    # Evaluate each problem
    passed = 0
    for i, (task_id, problem) in enumerate(problem_list, 1):
        logger.info(f"[{i}/3] {task_id}")

        try:
            # Generate solution
            sragv_problem = {"question": problem["prompt"]}
            solutions = orchestrator.solution_generator.generate(
                sragv_problem, num_solutions=1
            )

            if not solutions:
                logger.warning("  ❌ No solution generated")
                continue

            code = solutions[0].get("code", "")

            if not code:
                logger.warning("  ❌ Empty code")
                continue

            # Check syntax
            try:
                compile(code, '<solution>', 'exec')
                logger.info("  ✅ Syntax valid")
            except SyntaxError as e:
                logger.warning(f"  ❌ Syntax error: {e}")
                continue

            # Execute with HumanEval
            result = check_correctness(problem, code, timeout=15.0)

            if result["passed"]:
                passed += 1
                logger.info("  ✅ PASSED")
            else:
                logger.info(f"  ❌ Failed: {str(result.get('result', ''))[:100]}")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")

    logger.info(f"\n{'='*60}")
    logger.info(f"{model_type} Results: {passed}/3 passed ({passed/3*100:.1f}%)")
    logger.info(f"{'='*60}\n")

    return passed


def main():
    """Run quick test on both models."""

    logger.info("🧪 PHASE 4 QUICK TEST")
    logger.info("Testing baseline and GRPO-trained models on 3 HumanEval problems")
    logger.info()

    # Test baseline
    baseline_passed = test_model("BASELINE")

    # Test GRPO-trained
    grpo_checkpoint = project_root / "checkpoints" / "phase3_grpo_4player" / "grpo_iteration_16" / "solution_generator_adapter"

    if not grpo_checkpoint.exists():
        logger.error(f"GRPO checkpoint not found: {grpo_checkpoint}")
        logger.error("Download with:")
        logger.error("  python3 -m modal volume get srag-results phase3_grpo/grpo_iteration_16 checkpoints/phase3_grpo/grpo_iteration_16")
        sys.exit(1)

    grpo_passed = test_model("GRPO-TRAINED", str(grpo_checkpoint))

    # Summary
    logger.info("=" * 60)
    logger.info("📊 QUICK TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline:      {baseline_passed}/3 ({baseline_passed/3*100:.1f}%)")
    logger.info(f"GRPO-Trained:  {grpo_passed}/3 ({grpo_passed/3*100:.1f}%)")
    logger.info(f"Improvement:   {(grpo_passed - baseline_passed)/3*100:+.1f}%")
    logger.info("=" * 60)

    if grpo_passed > baseline_passed:
        logger.info("✅ GRPO model shows improvement - proceed to full evaluation")
    elif grpo_passed == baseline_passed:
        logger.info("⚠️ No improvement detected - full evaluation still recommended")
    else:
        logger.info("❌ GRPO model worse than baseline - investigate before full evaluation")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n🛑 Test interrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
