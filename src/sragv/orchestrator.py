"""
SRAG-V Orchestrator - Coordinates the 4-player self-play architecture.
Implements the complete learning verification through self-play system.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
from dataclasses import dataclass

import yaml
import torch

from .models import (
    ProblemGenerator, 
    SolutionGenerator, 
    VerificationGenerator, 
    MetaVerifier,
    PlayerConfig
)
from .utils.apps_loader import APPSDataLoader
from .utils.simple_data_loader import SimpleAPPSDataLoader

logger = logging.getLogger(__name__)


@dataclass
class IterationMetrics:
    """Metrics for a single self-play iteration."""
    iteration: int
    problems_generated: int
    solutions_generated: int
    tests_generated: int
    valid_tests: int
    meta_accuracy: float
    avg_solution_quality: float
    test_coverage_score: float
    processing_time: float


class SRAGVOrchestrator:
    """
    Main orchestrator for SRAG-V 4-player architecture.
    
    Coordinates:
    - Problem generation with diversity
    - Solution generation (16 per problem)  
    - Test case generation (8 per problem)
    - Meta-verification of test cases
    - Self-play training loop
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the orchestrator with configuration."""
        self.config_path = config_path
        self.config = self.load_config()

        
        # Initialize players (will be loaded on demand)
        self.problem_generator = None
        self.solution_generator = None
        self.verification_generator = None
        self.meta_verifier = None
        
        # Data loaders
        self.data_loader = None
        self.simple_data_loader = None
        
        # Iteration tracking
        self.current_iteration = 0
        self.iteration_metrics = []
        
        # MAP-Elites archive for diversity
        self.problem_archive = []
        self.max_archive_size = 1000
        
        logger.info("SRAG-V Orchestrator initialized")
    
    def load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config['sragv']
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def initialize_players(self):
        """Initialize all 4 players with their configurations."""
        logger.info("Initializing 4-player architecture...")
        
        # Player 1: Problem Generator (1.5B)
        problem_config = PlayerConfig(
            model_name=self.config['models']['problem_generator']['name'],
            max_length=self.config['models']['problem_generator']['max_length'],
            temperature=self.config['models']['problem_generator']['temperature'],
            top_p=self.config['models']['problem_generator']['top_p'],
            lora_rank=32,
            lora_alpha=64
        )
        self.problem_generator = ProblemGenerator(problem_config.to_dict())
        
        # Player 2: Solution Generator (7B with QLoRA)
        solution_config = PlayerConfig(
            model_name=self.config['models']['solution_generator']['name'],
            max_length=self.config['models']['solution_generator']['max_length'],
            temperature=self.config['models']['solution_generator']['temperature'],
            top_p=self.config['models']['solution_generator']['top_p'],
            quantization=self.config['models']['solution_generator']['quantization'],
            lora_rank=self.config['models']['solution_generator']['lora_rank']
        )
        self.solution_generator = SolutionGenerator(solution_config.to_dict())
        
        # Player 3: Verification Generator (1.5B)
        verification_config = PlayerConfig(
            model_name=self.config['models']['verification_generator']['name'],
            max_length=self.config['models']['verification_generator']['max_length'],
            temperature=self.config['models']['verification_generator']['temperature'],
            top_p=self.config['models']['verification_generator']['top_p'],
            lora_rank=32,
            lora_alpha=64
        )
        self.verification_generator = VerificationGenerator(verification_config.to_dict())
        
        # Player 4: Meta-Verifier (0.5B)
        meta_config = PlayerConfig(
            model_name=self.config['models']['meta_verifier']['name'],
            max_length=self.config['models']['meta_verifier']['max_length'],
            temperature=self.config['models']['meta_verifier']['temperature'],
            top_p=self.config['models']['meta_verifier']['top_p'],
            lora_rank=16,  # Smaller for 0.5B model
            lora_alpha=32
        )
        self.meta_verifier = MetaVerifier(meta_config.to_dict())
        
        # Load all models (sequential to avoid meta tensor issues)
        logger.info("Loading models...")
        self.problem_generator.load_model()
        self.solution_generator.load_model()
        self.verification_generator.load_model()
        self.meta_verifier.load_model()
        
        logger.info("All 4 players initialized successfully")
    
    def initialize_data_loaders(self):
        """Initialize data loaders for APPS dataset."""
        logger.info("Initializing data loaders...")
        
        # Try real APPS loader first
        try:
            self.data_loader = APPSDataLoader(
                cache_dir=self.config['dataset']['cache_dir'],
                max_problems=self.config['dataset']['max_problems']
            )
            logger.info("Real APPS data loader initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize APPS loader: {e}")
            self.data_loader = None
        
        # Always have simple loader as fallback
        self.simple_data_loader = SimpleAPPSDataLoader(
            cache_dir=self.config['dataset']['cache_dir'],
            max_problems=min(100, self.config['dataset']['max_problems'])
        )
        logger.info("Simple data loader initialized")
    
    def load_bootstrap_data(self) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """Load initial bootstrap data."""
        logger.info("Loading bootstrap data...")
        
        # Try real APPS data first
        if self.data_loader:
            try:
                problems, categorized = self.data_loader.prepare_bootstrap_data()
                if problems:
                    logger.info(f"Loaded {len(problems)} real APPS problems")
                    return problems, categorized
            except Exception as e:
                logger.warning(f"Failed to load real APPS data: {e}")
        
        # Fall back to simple data
        logger.info("Using simple synthetic data for bootstrap")
        problems, categorized = self.simple_data_loader.prepare_bootstrap_data()
        return problems, categorized
    
    def run_single_iteration(
        self,
        iteration: int,
        bootstrap_problems: Optional[List[Dict]] = None
    ) -> IterationMetrics:
        """Run a single self-play iteration."""
        logger.info(f"Starting iteration {iteration}")
        start_time = time.time()
        
        # Initialize players if needed
        if self.problem_generator is None:
            self.initialize_players()
        
        metrics = IterationMetrics(
            iteration=iteration,
            problems_generated=0,
            solutions_generated=0,
            tests_generated=0,
            valid_tests=0,
            meta_accuracy=0.0,
            avg_solution_quality=0.0,
            test_coverage_score=0.0,
            processing_time=0.0
        )
        
        try:
            # Step 1: Problem Generation (15 minutes budget)
            logger.info("Step 1: Generating problems...")
            if bootstrap_problems and iteration == 1:
                # Use bootstrap problems for first iteration
                problems = bootstrap_problems[:32]  # Limit to 32 for testing
                logger.info(f"Using {len(problems)} bootstrap problems")
            else:
                # Generate new problems based on archive diversity
                success_rates = self.compute_success_rates()
                type_distribution = self.compute_type_distribution()
                
                problems = self.problem_generator.generate_diverse_batch(
                    success_rate_histogram=success_rates,
                    problem_type_distribution=type_distribution,
                    archive=self.problem_archive,
                    batch_size=32
                )
            
            metrics.problems_generated = len(problems)
            
            # Step 2: Solution Generation (45 minutes budget)
            logger.info("Step 2: Generating solutions...")
            all_solutions = []
            
            for i, problem in enumerate(problems):
                logger.debug(f"Generating solutions for problem {i+1}/{len(problems)}")
                
                # Generate 16 diverse solutions per problem
                solutions = self.solution_generator.generate(
                    problem=problem,
                    num_solutions=16,
                    max_attempts_per_solution=2  # Limit attempts for time
                )
                
                # Add problem reference to solutions
                for solution in solutions:
                    solution['problem_id'] = problem.get('problem_id', f'prob_{i}')
                
                all_solutions.extend(solutions)
            
            metrics.solutions_generated = len(all_solutions)
            
            # Step 3: Verification Generation (30 minutes budget)
            logger.info("Step 3: Generating test cases...")
            all_test_cases = []
            
            for i, problem in enumerate(problems):
                # Get solutions for this problem
                problem_solutions = [s for s in all_solutions 
                                   if s.get('problem_id') == problem.get('problem_id', f'prob_{i}')]
                
                if problem_solutions:
                    logger.debug(f"Generating tests for problem {i+1}/{len(problems)}")
                    
                    # Generate 8 test cases per problem
                    test_cases = self.verification_generator.generate(
                        problem=problem,
                        solutions=problem_solutions,
                        num_test_cases=8
                    )
                    
                    all_test_cases.extend(test_cases)
            
            metrics.tests_generated = len(all_test_cases)
            
            # Step 4: Meta-Verification (10 minutes budget)
            logger.info("Step 4: Meta-verification...")
            valid_test_cases = []
            total_confidence = 0.0
            
            # Group test cases by problem for validation
            problem_test_groups = {}
            for test_case in all_test_cases:
                problem_id = test_case.get('problem_id', 'unknown')
                if problem_id not in problem_test_groups:
                    problem_test_groups[problem_id] = []
                problem_test_groups[problem_id].append(test_case)
            
            for problem_id, test_cases in problem_test_groups.items():
                # Find corresponding problem and solutions
                problem = next((p for p in problems if p.get('problem_id') == problem_id), None)
                problem_solutions = [s for s in all_solutions if s.get('problem_id') == problem_id]
                
                if problem and problem_solutions:
                    # Validate test cases
                    validated_tests = self.meta_verifier.validate_test_cases(
                        problem=problem,
                        solutions=problem_solutions,
                        test_cases=test_cases
                    )
                    
                    # Filter valid tests
                    valid_tests = self.meta_verifier.filter_valid_tests(validated_tests)
                    valid_test_cases.extend(valid_tests)
                    
                    # Accumulate confidence scores
                    for test in validated_tests:
                        total_confidence += test.get('confidence', 0.5)
            
            metrics.valid_tests = len(valid_test_cases)
            metrics.meta_accuracy = total_confidence / max(len(all_test_cases), 1)
            
            # Step 5: Compute additional metrics
            if all_solutions:
                avg_score = sum(s.get('score', 0.0) for s in all_solutions) / len(all_solutions)
                metrics.avg_solution_quality = avg_score
            
            if valid_test_cases:
                # Simple coverage score based on test categories
                categories = set(tc.get('category', 'basic') for tc in valid_test_cases)
                metrics.test_coverage_score = len(categories) / 8.0  # Normalize by max categories
            
            # Update archive with new problems
            self.update_problem_archive(problems, all_solutions, valid_test_cases)
            
            # Update metrics
            metrics.processing_time = time.time() - start_time
            self.iteration_metrics.append(metrics)
            
            logger.info(f"Iteration {iteration} complete: "
                       f"{metrics.problems_generated} problems, "
                       f"{metrics.solutions_generated} solutions, "
                       f"{metrics.valid_tests}/{metrics.tests_generated} valid tests, "
                       f"meta accuracy: {metrics.meta_accuracy:.3f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in iteration {iteration}: {e}")
            metrics.processing_time = time.time() - start_time
            return metrics
    
    def compute_success_rates(self) -> Dict[str, float]:
        """Compute success rates by difficulty from archive."""
        if not self.problem_archive:
            return {"easy": 0.8, "medium": 0.5, "hard": 0.2}
        
        success_rates = {}
        difficulty_counts = {}
        
        for entry in self.problem_archive:
            problem = entry.get('problem', {})
            solutions = entry.get('solutions', [])
            
            difficulty = problem.get('difficulty', 'medium')
            if solutions:
                # Compute success rate based on solution scores
                avg_score = sum(s.get('score', 0.0) for s in solutions) / len(solutions)
                success_rates[difficulty] = success_rates.get(difficulty, 0.0) + avg_score
                difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
        
        # Average the success rates
        for difficulty in success_rates:
            if difficulty_counts[difficulty] > 0:
                success_rates[difficulty] /= difficulty_counts[difficulty]
        
        # Ensure all difficulties are present
        default_rates = {"easy": 0.8, "medium": 0.5, "hard": 0.2}
        for difficulty, default_rate in default_rates.items():
            if difficulty not in success_rates:
                success_rates[difficulty] = default_rate
        
        return success_rates
    
    def compute_type_distribution(self) -> Dict[str, float]:
        """Compute problem type distribution from archive."""
        if not self.problem_archive:
            return {"arithmetic": 0.25, "string_manipulation": 0.25, "array_processing": 0.25, "graph_algorithms": 0.25}
        
        type_counts = {}
        total_count = 0
        
        for entry in self.problem_archive:
            problem = entry.get('problem', {})
            problem_type = problem.get('problem_type', 'arithmetic')
            type_counts[problem_type] = type_counts.get(problem_type, 0) + 1
            total_count += 1
        
        # Convert to probabilities
        if total_count > 0:
            type_distribution = {ptype: count/total_count for ptype, count in type_counts.items()}
        else:
            type_distribution = {"arithmetic": 1.0}
        
        return type_distribution
    
    def update_problem_archive(
        self,
        problems: List[Dict],
        solutions: List[Dict],
        test_cases: List[Dict]
    ):
        """Update MAP-Elites archive with new problems."""
        for problem in problems:
            problem_id = problem.get('problem_id', '')
            
            # Get associated solutions and tests
            problem_solutions = [s for s in solutions if s.get('problem_id') == problem_id]
            problem_tests = [t for t in test_cases if t.get('problem_id') == problem_id]
            
            # Create archive entry
            entry = {
                'problem': problem,
                'solutions': problem_solutions,
                'test_cases': problem_tests,
                'quality_score': sum(s.get('score', 0.0) for s in problem_solutions) / max(len(problem_solutions), 1),
                'iteration': self.current_iteration
            }
            
            self.problem_archive.append(entry)
        
        # Trim archive if too large
        if len(self.problem_archive) > self.max_archive_size:
            # Sort by quality score and keep the best
            self.problem_archive.sort(key=lambda x: x.get('quality_score', 0.0), reverse=True)
            self.problem_archive = self.problem_archive[:self.max_archive_size]
        
        logger.debug(f"Archive updated: {len(self.problem_archive)} entries")
    
    def run_bootstrap_phase(self) -> List[Dict]:
        """Run the 3-stage bootstrapping protocol."""
        logger.info("Starting 3-stage bootstrapping protocol...")
        
        # Initialize data loaders
        self.initialize_data_loaders()
        
        # Load bootstrap data
        bootstrap_problems, categorized = self.load_bootstrap_data()
        
        # Stage 1: Ground Truth Imitation (100 problems)
        stage1_problems = bootstrap_problems[:self.config['bootstrap']['stage1_problems']]
        logger.info(f"Stage 1: Ground truth imitation with {len(stage1_problems)} problems")
        
        # Stage 2: Consistency Learning (500 problems) 
        stage2_problems = bootstrap_problems[:self.config['bootstrap']['stage2_problems']]
        logger.info(f"Stage 2: Consistency learning with {len(stage2_problems)} problems")
        
        # Stage 3: Early Self-Play (500 problems)
        stage3_problems = bootstrap_problems[:self.config['bootstrap']['stage3_problems']]
        logger.info(f"Stage 3: Early self-play with {len(stage3_problems)} problems")
        
        # For now, return stage 3 problems for iteration 1
        return stage3_problems
    
    def save_checkpoint(self, checkpoint_dir: str = "checkpoints"):
        """Save current state as checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        checkpoint_data = {
            'current_iteration': self.current_iteration,
            'iteration_metrics': [
                {
                    'iteration': m.iteration,
                    'problems_generated': m.problems_generated,
                    'solutions_generated': m.solutions_generated,
                    'tests_generated': m.tests_generated,
                    'valid_tests': m.valid_tests,
                    'meta_accuracy': m.meta_accuracy,
                    'avg_solution_quality': m.avg_solution_quality,
                    'test_coverage_score': m.test_coverage_score,
                    'processing_time': m.processing_time
                }
                for m in self.iteration_metrics
            ],
            'archive_size': len(self.problem_archive)
        }
        
        # Save metrics
        with open(checkpoint_path / f"iteration_{self.current_iteration}_metrics.json", 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Save archive (subset for space)
        archive_subset = self.problem_archive[-100:]  # Keep last 100 entries
        with open(checkpoint_path / f"iteration_{self.current_iteration}_archive.json", 'w') as f:
            json.dump(archive_subset, f, indent=2)
        
        logger.info(f"Checkpoint saved for iteration {self.current_iteration}")
    
    def run_self_play_training(self, num_iterations: int = 5) -> List[IterationMetrics]:
        """Run the complete self-play training loop."""
        logger.info(f"Starting self-play training for {num_iterations} iterations")
        
        # Bootstrap phase
        bootstrap_problems = self.run_bootstrap_phase()
        
        # Run iterations
        all_metrics = []
        
        for iteration in range(1, num_iterations + 1):
            self.current_iteration = iteration
            
            # Use bootstrap problems only for first iteration
            problems_to_use = bootstrap_problems if iteration == 1 else None
            
            # Run iteration
            metrics = self.run_single_iteration(iteration, problems_to_use)
            all_metrics.append(metrics)
            
            # Save checkpoint
            self.save_checkpoint()
            
            # Log progress
            logger.info(f"Completed iteration {iteration}/{num_iterations}")
        
        logger.info("Self-play training complete!")
        return all_metrics