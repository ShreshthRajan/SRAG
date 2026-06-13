#!/usr/bin/env python3
"""
Test the HumanEval generation fix on 5 problems.
Verifies stop sequences and direct prompting work correctly.
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from human_eval.data import read_problems
    from human_eval.execution import check_correctness
except ImportError:
    logger.error("human-eval not installed")
    sys.exit(1)

try:
    from sragv.orchestrator import SRAGVOrchestrator
except Exception as e:
    logger.error(f"Failed to import SRAG-V: {e}")
    sys.exit(1)


def test_humaneval_fix():
    """Test HumanEval generation with stop sequences."""

    logger.info("=" * 80)
    logger.info("TESTING HUMANEVAL GENERATION FIX")
    logger.info("=" * 80)

    # Load orchestrator
    config_path = project_root / "config" / "config.yaml"
    orchestrator = SRAGVOrchestrator(str(config_path))
    orchestrator.initialize_players()

    # Load HumanEval
    problems = read_problems()
    problem_list = list(problems.items())[:5]  # First 5 problems

    logger.info(f"\nTesting on {len(problem_list)} problems with FIXED evaluation\n")

    passed = 0
    for i, (task_id, problem) in enumerate(problem_list, 1):
        logger.info(f"[{i}/5] {task_id}")

        try:
            # Generate using NEW humaneval_generate method
            code = orchestrator.solution_generator.generate_humaneval(
                prompt=problem["prompt"],
                temperature=0.0
            )

            if not code:
                logger.warning("  ❌ No code generated")
                continue

            logger.info(f"  Generated {len(code)} chars")

            # Check for stop sequences in output (should be trimmed)
            stop_seqs = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']
            found_stops = [s for s in stop_seqs if s in code]
            if found_stops:
                logger.warning(f"  ⚠️  Found stop sequences: {found_stops} (trimming failed?)")

            # Execute with HumanEval
            result = check_correctness(problem, code, timeout=15.0)

            if result["passed"]:
                passed += 1
                logger.info("  ✅ PASSED")
            else:
                error_msg = str(result.get('result', ''))[:100]
                logger.info(f"  ❌ Failed: {error_msg}")

        except Exception as e:
            logger.error(f"  ❌ Error: {e}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Results: {passed}/5 passed ({passed/5*100:.1f}%)")
    logger.info(f"{'='*80}")

    if passed >= 3:
        logger.info("✅ Fix appears to be working - proceed to full evaluation")
    else:
        logger.info("⚠️  Fix may have issues - investigate further")

    return passed


if __name__ == "__main__":
    try:
        passed = test_humaneval_fix()
        sys.exit(0 if passed >= 3 else 1)
    except Exception as e:
        logger.error(f"💥 Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
