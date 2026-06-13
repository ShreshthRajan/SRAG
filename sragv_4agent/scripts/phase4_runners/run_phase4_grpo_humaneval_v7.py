#!/usr/bin/env python3
"""
Phase 4 V7: GRPO HumanEval Evaluation with merge_and_unload()
Final fix using adapter weight merging to guarantee GRPO weights are applied.

Research-grade evaluation for ICML submission.
Author: Claude & Shreshth
Date: December 2025
"""

import os
import sys
import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import numpy as np
import torch

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Environment setup
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Setup logging
log_dir = project_root / "logs"
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"phase4_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

# Import HumanEval
try:
    from human_eval.data import read_problems, write_jsonl
    from human_eval.evaluation import evaluate_functional_correctness
except ImportError as e:
    logger.error(f"human-eval not installed: {e}")
    sys.exit(1)

# Import SRAG-V components
try:
    from sragv.models.solution_generator import SolutionGenerator
    from sragv.models.base_player import PlayerConfig
except Exception as e:
    logger.error(f"Failed to import SRAG-V: {e}")
    sys.exit(1)


@dataclass
class EvaluationResult:
    """Results from evaluating a single model."""
    model_name: str
    total_problems: int
    completed_problems: int
    pass_at_1: float
    avg_generation_time: float
    syntax_errors: int
    runtime_errors: int
    timeouts: int
    correct_solutions: int
    problem_results: List[Dict]


class Phase4V7Evaluator:
    """
    Phase 4 V7: Clean evaluation with merge_and_unload().

    Key fix: Bakes GRPO adapter weights into model using merge_and_unload(),
    guaranteeing trained weights are actually used during inference.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, temperature: float = 0.0):
        self.checkpoint_path = checkpoint_path
        self.temperature = temperature
        self.solution_generator = None

        # Results directory
        self.results_dir = project_root / "results" / "phase4_humaneval_v7"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        model_type = "trained" if checkpoint_path else "baseline"
        self.checkpoint_file = self.results_dir / f"{model_type}_checkpoint.json"

        logger.info(f"Phase 4 V7 Evaluator initialized")
        logger.info(f"Model: {'GRPO-trained (merged)' if checkpoint_path else 'Baseline'}")
        logger.info(f"Temperature: {temperature}")

    def load_model(self):
        """
        Load model with merge_and_unload() for guaranteed adapter application.

        CRITICAL FIX:
        - Load at fp16 (not 4bit) for reliable merge
        - Use lora_rank=0 (no fresh adapter)
        - For trained: load_adapter() + merge_and_unload()
        - Bakes GRPO weights into model permanently
        """
        logger.info("=" * 80)
        logger.info("Loading model with merge_and_unload() approach...")
        logger.info("=" * 80)

        # Create config with NO quantization (fp16 for reliable merge)
        eval_config = PlayerConfig(
            model_name="Qwen/Qwen2.5-Coder-1.5B-Instruct",
            max_length=4096,
            temperature=0.0,
            top_p=0.95,
            quantization=None,  # CRITICAL: No quantization for reliable merge
            lora_rank=0,  # No fresh adapter
            device=0
        ).to_dict()

        # Create solution generator
        self.solution_generator = SolutionGenerator(eval_config)
        self.solution_generator.load_model()

        # Get initial parameter sum for verification
        initial_param_sum = sum(p.sum().item() for p in self.solution_generator.model.parameters())
        logger.info(f"Initial model parameter sum: {initial_param_sum:.2f}")

        if self.checkpoint_path:
            # TRAINED: Load and merge GRPO adapter
            logger.info(f"Loading GRPO adapter from: {self.checkpoint_path}")

            try:
                # Load adapter onto clean base model
                from peft import PeftModel
                peft_model = PeftModel.from_pretrained(
                    self.solution_generator.model,
                    self.checkpoint_path
                )

                logger.info("Merging adapter weights into base model...")

                # Merge adapter weights (bakes them in permanently)
                merged_model = peft_model.merge_and_unload()

                # Replace model with merged version
                self.solution_generator.model = merged_model
                self.solution_generator.peft_model = None  # No PEFT anymore

                # Verify merge worked
                merged_param_sum = sum(p.sum().item() for p in merged_model.parameters())
                param_diff = abs(merged_param_sum - initial_param_sum)

                logger.info(f"Merged model parameter sum: {merged_param_sum:.2f}")
                logger.info(f"Parameter difference: {param_diff:.2f}")

                if param_diff < 1.0:
                    logger.warning("⚠️ Merge may have failed - parameters barely changed")
                else:
                    logger.info(f"✅ Merge verified - parameters changed by {param_diff:.2f}")

                logger.info("✅ GRPO weights merged into model (no PEFT overhead)")

            except Exception as e:
                logger.error(f"❌ Adapter merge failed: {e}")
                raise

        else:
            logger.info("✅ Baseline: Using pure base model")

        # Final verification
        test_problem = {"question": "Write a function that returns 'hello'"}
        try:
            test_solutions = self.solution_generator.generate(
                test_problem, num_solutions=1
            )
            if test_solutions:
                logger.info("✅ Model verification successful")
            else:
                raise Exception("Model generated no solutions")
        except Exception as e:
            logger.error(f"❌ Model verification failed: {e}")
            raise

    def evaluate_on_humaneval(self) -> EvaluationResult:
        """Evaluate model on full HumanEval benchmark."""
        logger.info("=" * 80)
        logger.info("Starting HumanEval evaluation...")
        logger.info("=" * 80)

        # Load HumanEval
        problems = read_problems()
        problem_list = list(problems.items())
        total_problems = len(problem_list)

        logger.info(f"Loaded {total_problems} HumanEval problems")

        # Check for checkpoint
        start_idx = 0
        previous_results = []

        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            previous_results = checkpoint_data.get("problem_results", [])
            start_idx = len(previous_results)
            logger.info(f"Resuming from problem {start_idx + 1}/{total_problems}")

        problem_results = previous_results.copy()
        syntax_errors = 0
        runtime_errors = 0
        timeouts = 0
        correct_solutions = 0
        generation_times = []

        model_name = "grpo_trained" if self.checkpoint_path else "baseline"
        samples_file = self.results_dir / f"{model_name}_samples.jsonl"
        samples = []

        # Evaluate each problem
        for idx in range(start_idx, total_problems):
            task_id, problem = problem_list[idx]

            logger.info(f"[{idx + 1}/{total_problems}] Evaluating {task_id}")

            problem_start = time.time()

            try:
                # Convert HumanEval to SRAG-V format
                sragv_problem = {
                    "question": problem["prompt"],
                    "entry_point": problem["entry_point"],
                    "problem_id": task_id
                }

                # Generate solution
                try:
                    solutions = self.solution_generator.generate(
                        problem=sragv_problem,
                        num_solutions=1
                    )
                except Exception as gen_error:
                    logger.error(f"  ❌ Generation failed: {gen_error}")
                    solutions = []

                generation_time = time.time() - problem_start
                generation_times.append(generation_time)

                if not solutions:
                    logger.warning(f"  ⚠️ No solution generated")
                    code = ""
                    generated = False
                else:
                    code = solutions[0].get("code", "")
                    generated = True
                    logger.info(f"  ✅ Generated solution ({generation_time:.1f}s)")

                # Syntax check
                syntax_valid = False
                if code:
                    try:
                        compile(code, '<solution>', 'exec')
                        syntax_valid = True
                    except SyntaxError as e:
                        syntax_errors += 1
                        logger.warning(f"  ⚠️ Syntax error: {str(e)[:100]}")

                # Store for official evaluation
                samples.append({
                    "task_id": task_id,
                    "completion": code
                })

                # Record result
                problem_result = {
                    "task_id": task_id,
                    "generated": generated,
                    "syntax_valid": syntax_valid,
                    "generation_time": generation_time,
                    "code_length": len(code)
                }

                problem_results.append(problem_result)

                # Checkpoint every 10 problems
                if (idx + 1) % 10 == 0:
                    self._save_checkpoint(problem_results, idx + 1, total_problems)
                    logger.info(f"  💾 Checkpoint saved ({idx + 1}/{total_problems})")

                # Memory cleanup every 20 problems
                if (idx + 1) % 20 == 0:
                    torch.cuda.empty_cache()
                    logger.info(f"  🧹 GPU memory cleared")

            except Exception as e:
                logger.error(f"  ❌ Error evaluating {task_id}: {e}")
                continue

        # Save samples
        logger.info(f"Saving samples to {samples_file}")
        write_jsonl(str(samples_file), samples)

        # Run official evaluation
        logger.info("Running official HumanEval evaluation...")
        results = evaluate_functional_correctness(
            sample_file=str(samples_file),
            k=[1],
            n_workers=4,
            timeout=15.0
        )

        pass_at_1 = results.get("pass@1", 0.0)

        # Count correct from results file
        results_file = str(samples_file).replace(".jsonl", "_results.jsonl")
        if Path(results_file).exists():
            with open(results_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    if result.get("passed", False):
                        correct_solutions += 1

        # Create evaluation result
        eval_result = EvaluationResult(
            model_name=model_name,
            total_problems=total_problems,
            completed_problems=len(problem_results),
            pass_at_1=pass_at_1,
            avg_generation_time=np.mean(generation_times) if generation_times else 0.0,
            syntax_errors=syntax_errors,
            runtime_errors=runtime_errors,
            timeouts=timeouts,
            correct_solutions=correct_solutions,
            problem_results=problem_results
        )

        logger.info("=" * 80)
        logger.info(f"✅ Evaluation complete: {model_name}")
        logger.info(f"Pass@1: {pass_at_1:.4f} ({correct_solutions}/{total_problems})")
        logger.info(f"Syntax errors: {syntax_errors}")
        logger.info(f"Avg generation time: {eval_result.avg_generation_time:.2f}s")
        logger.info("=" * 80)

        return eval_result

    def _save_checkpoint(self, problem_results: List[Dict], current: int, total: int):
        """Save evaluation checkpoint."""
        checkpoint_data = {
            "problem_results": problem_results,
            "progress": f"{current}/{total}",
            "timestamp": datetime.now().isoformat()
        }
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'solution_generator') and self.solution_generator:
            if hasattr(self.solution_generator, 'unload_model'):
                self.solution_generator.unload_model()
        torch.cuda.empty_cache()
        logger.info("✅ Resources cleaned up")


def compute_statistical_significance(baseline_results: List[bool], trained_results: List[bool]) -> Dict:
    """Compute statistical significance using binomial test."""
    from scipy import stats

    n = len(baseline_results)
    baseline_pass = sum(baseline_results)
    trained_pass = sum(trained_results)

    baseline_rate = baseline_pass / n
    trained_rate = trained_pass / n
    improvement = trained_rate - baseline_rate

    # Binomial test
    p_value = stats.binom_test(
        trained_pass,
        n,
        baseline_rate,
        alternative='greater'
    )

    # Confidence interval
    z = 1.96
    p = trained_rate
    center = (p + z**2 / (2*n)) / (1 + z**2 / n)
    margin = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / (1 + z**2/n)
    ci_lower = center - margin
    ci_upper = center + margin

    return {
        "n_problems": n,
        "baseline_pass": baseline_pass,
        "trained_pass": trained_pass,
        "baseline_rate": baseline_rate,
        "trained_rate": trained_rate,
        "improvement": improvement,
        "improvement_percent": improvement * 100,
        "p_value": p_value,
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "confidence_interval_95": [ci_lower, ci_upper],
        "relative_improvement": (improvement / baseline_rate * 100) if baseline_rate > 0 else 0
    }


def run_full_evaluation():
    """Run complete Phase 4 V7 evaluation."""

    start_time = time.time()

    logger.info("🚀 PHASE 4 V7: GRPO HumanEval Evaluation (merge_and_unload)")
    logger.info("=" * 80)
    logger.info("Fix: Bake GRPO weights into model using merge_and_unload()")
    logger.info("=" * 80)

    results = {
        "experiment": "phase4_v7_grpo_humaneval_merged",
        "start_time": datetime.now().isoformat(),
        "checkpoint": "grpo_iteration_16",
        "method": "merge_and_unload",
        "quantization": "fp16"
    }

    try:
        # STAGE 1: BASELINE
        logger.info("\n📊 STAGE 1: Baseline Model Evaluation")
        stage_start = time.time()

        baseline_evaluator = Phase4V7Evaluator(
            checkpoint_path=None,
            temperature=0.0
        )
        baseline_evaluator.load_model()
        baseline_result = baseline_evaluator.evaluate_on_humaneval()
        baseline_evaluator.cleanup()

        stage_duration = time.time() - stage_start
        results["baseline"] = {
            **asdict(baseline_result),
            "duration": stage_duration
        }

        logger.info(f"✅ Baseline complete ({stage_duration/60:.1f} min)")
        logger.info(f"Baseline Pass@1: {baseline_result.pass_at_1:.4f}")

        # STAGE 2: GRPO-TRAINED (MERGED)
        logger.info("\n🌟 STAGE 2: GRPO-Trained Model Evaluation (Merged)")
        stage_start = time.time()

        grpo_checkpoint = project_root / "checkpoints" / "phase3_grpo_4player" / "grpo_iteration_16" / "solution_generator_adapter"

        if not grpo_checkpoint.exists():
            raise FileNotFoundError(f"GRPO checkpoint not found: {grpo_checkpoint}")

        trained_evaluator = Phase4V7Evaluator(
            checkpoint_path=str(grpo_checkpoint),
            temperature=0.0
        )
        trained_evaluator.load_model()
        trained_result = trained_evaluator.evaluate_on_humaneval()
        trained_evaluator.cleanup()

        stage_duration = time.time() - stage_start
        results["grpo_trained"] = {
            **asdict(trained_result),
            "duration": stage_duration
        }

        logger.info(f"✅ GRPO-trained complete ({stage_duration/60:.1f} min)")
        logger.info(f"Trained Pass@1: {trained_result.pass_at_1:.4f}")

        # STAGE 3: STATISTICAL ANALYSIS
        logger.info("\n📈 STAGE 3: Statistical Analysis")

        baseline_results_file = str(baseline_evaluator.results_dir / "baseline_samples_results.jsonl")
        trained_results_file = str(trained_evaluator.results_dir / "grpo_trained_samples_results.jsonl")

        baseline_passes = []
        trained_passes = []

        if Path(baseline_results_file).exists():
            with open(baseline_results_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    baseline_passes.append(result.get("passed", False))

        if Path(trained_results_file).exists():
            with open(trained_results_file, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    trained_passes.append(result.get("passed", False))

        if baseline_passes and trained_passes:
            stats_analysis = compute_statistical_significance(baseline_passes, trained_passes)
            results["statistical_analysis"] = stats_analysis

            logger.info("=" * 80)
            logger.info("STATISTICAL ANALYSIS:")
            logger.info(f"  Baseline: {stats_analysis['baseline_rate']:.4f} ({stats_analysis['baseline_pass']}/164)")
            logger.info(f"  Trained:  {stats_analysis['trained_rate']:.4f} ({stats_analysis['trained_pass']}/164)")
            logger.info(f"  Improvement: {stats_analysis['improvement_percent']:.2f}%")
            logger.info(f"  P-value: {stats_analysis['p_value']:.6f}")
            logger.info(f"  Significant: {stats_analysis['significant_at_0.05']}")
            logger.info("=" * 80)

        # FINAL SUMMARY
        total_time = time.time() - start_time
        improvement = trained_result.pass_at_1 - baseline_result.pass_at_1

        results.update({
            "end_time": datetime.now().isoformat(),
            "total_duration": total_time,
            "improvement": improvement,
            "improvement_percent": improvement * 100,
            "success": improvement > 0.005,
            "publishable": improvement > 0.005 and stats_analysis.get("significant_at_0.05", False)
        })

        # Save results
        results_dir = project_root / "results" / "phase4_humaneval_v7"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"phase4_v7_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n" + "=" * 80)
        logger.info("🎉 PHASE 4 V7 EVALUATION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Baseline:    {baseline_result.pass_at_1:.4f}")
        logger.info(f"Trained:     {trained_result.pass_at_1:.4f}")
        logger.info(f"Improvement: +{improvement*100:.2f}%")
        logger.info(f"Success:     {results['success']}")
        logger.info(f"Publishable: {results['publishable']}")
        logger.info(f"Results:     {results_file}")
        logger.info("=" * 80)

        return results

    except Exception as e:
        logger.error(f"💥 Evaluation failed: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    try:
        results = run_full_evaluation()
        sys.exit(0 if results["success"] else 1)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
