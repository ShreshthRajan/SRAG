"""
Integrated Self-Play Training System for SRAG-V.
Combines GRPO, MAP-Elites, and role-conditioned rewards for 4-player self-play training.
Implements the complete Step 2 training loop with emergent verification learning.
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch

from .grpo_trainer import GRPOTrainer, GRPOConfig
from .map_elites import MAPElitesArchive, MAPElitesConfig
from .reward_system import RoleConditionedRewardSystem
from ..orchestrator import SRAGVOrchestrator, IterationMetrics

logger = logging.getLogger(__name__)


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training system."""
    
    # Training schedule
    num_iterations: int = 12  # Based on July 2025 research
    bootstrap_iterations: int = 3  # Initial learning phase
    competitive_iterations: int = 6  # Main competition phase  
    league_iterations: int = 3  # Historical opponent phase
    
    # Problem generation
    problems_per_iteration: int = 32  # Batch size for training
    dynamic_problem_generation: bool = True  # Generate new problems each iteration
    problem_difficulty_scaling: bool = True  # Increase difficulty over time
    
    # Self-play dynamics
    solutions_per_problem: int = 16  # Multiple solutions for diversity
    test_cases_per_problem: int = 8  # Test cases for verification
    archive_sampling_probability: float = 0.3  # Use archived problems
    
    # Training optimization
    gradient_accumulation_steps: int = 4  # For memory efficiency
    mixed_precision: bool = True  # Use fp16 for speed
    checkpoint_every_iterations: int = 2  # Frequent checkpointing
    
    # Emergent behavior tracking
    track_emergent_behaviors: bool = True
    behavior_analysis_frequency: int = 5  # Every N iterations
    novel_pattern_threshold: float = 0.8  # Threshold for detecting novel behaviors
    
    # Quality gates
    min_solution_accuracy: float = 0.7  # Minimum accuracy to proceed
    min_test_validity: float = 0.6  # Minimum test case validity
    convergence_threshold: float = 0.05  # Convergence detection


class EmergentBehaviorTracker:
    """Tracks and analyzes emergent verification behaviors during self-play."""
    
    def __init__(self):
        self.behavior_history: List[Dict[str, Any]] = []
        self.novel_patterns: List[Dict[str, Any]] = []
        self.pattern_library: Dict[str, List[str]] = {
            "verification_strategies": [],
            "problem_types": [],
            "solution_approaches": [],
            "test_patterns": []
        }
    
    def analyze_iteration_behaviors(self, iteration_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze behaviors from a single iteration."""
        analysis = {
            "iteration": iteration_data.get("iteration", 0),
            "timestamp": time.time(),
            "verification_patterns": [],
            "novel_discoveries": [],
            "complexity_evolution": {},
            "emergent_strategies": []
        }
        
        # Extract patterns from solutions and test cases
        problems = iteration_data.get("problems", [])
        solutions = iteration_data.get("solutions", [])
        test_cases = iteration_data.get("test_cases", [])
        
        # Analyze verification strategies
        verification_strategies = self._extract_verification_strategies(test_cases, solutions)
        analysis["verification_patterns"] = verification_strategies
        
        # Detect novel patterns
        novel_patterns = self._detect_novel_patterns(verification_strategies)
        analysis["novel_discoveries"] = novel_patterns
        
        # Track complexity evolution
        complexity_metrics = self._compute_complexity_metrics(problems, solutions, test_cases)
        analysis["complexity_evolution"] = complexity_metrics
        
        # Identify emergent strategies
        emergent_strategies = self._identify_emergent_strategies(iteration_data)
        analysis["emergent_strategies"] = emergent_strategies
        
        self.behavior_history.append(analysis)
        
        if novel_patterns:
            self.novel_patterns.extend(novel_patterns)
            logger.info(f"Discovered {len(novel_patterns)} novel verification patterns in iteration {analysis['iteration']}")
        
        return analysis
    
    def _extract_verification_strategies(self, test_cases: List[Dict], solutions: List[Dict]) -> List[Dict[str, Any]]:
        """Extract verification strategies from test cases and solutions."""
        strategies = []
        
        for test_case in test_cases:
            strategy = {
                "type": "test_case_strategy",
                "approach": self._classify_test_approach(test_case),
                "coverage_type": self._classify_coverage_type(test_case),
                "complexity": self._compute_test_complexity(test_case),
                "pattern_signature": self._compute_pattern_signature(test_case)
            }
            strategies.append(strategy)
        
        return strategies
    
    def _classify_test_approach(self, test_case: Dict[str, Any]) -> str:
        """Classify the approach used in a test case."""
        content = str(test_case.get("content", "")).lower()
        
        if "edge" in content or "boundary" in content:
            return "boundary_testing"
        elif "random" in content or "stress" in content:
            return "stress_testing"
        elif "equivalence" in content or "partition" in content:
            return "equivalence_partitioning"
        elif "negative" in content or "invalid" in content:
            return "negative_testing"
        elif "assert" in content and "==" in content:
            return "assertion_based"
        else:
            return "standard_testing"
    
    def _classify_coverage_type(self, test_case: Dict[str, Any]) -> str:
        """Classify the type of coverage this test case provides."""
        inputs = test_case.get("inputs", [])
        
        if not inputs:
            return "unknown"
        
        # Check for specific coverage patterns
        for inp in inputs:
            if inp is None or (isinstance(inp, str) and inp == ""):
                return "null_coverage"
            elif isinstance(inp, (int, float)) and inp == 0:
                return "zero_coverage"
            elif isinstance(inp, (int, float)) and inp < 0:
                return "negative_coverage"
            elif isinstance(inp, (list, tuple)) and len(inp) == 0:
                return "empty_collection_coverage"
            elif isinstance(inp, (list, tuple)) and len(inp) == 1:
                return "single_element_coverage"
        
        return "standard_coverage"
    
    def _compute_test_complexity(self, test_case: Dict[str, Any]) -> float:
        """Compute complexity score for a test case."""
        complexity = 0.0
        
        # Content complexity
        content = str(test_case.get("content", ""))
        complexity += min(len(content) / 100.0, 1.0) * 0.3
        
        # Input complexity
        inputs = test_case.get("inputs", [])
        if inputs:
            input_complexity = sum(self._compute_input_complexity(inp) for inp in inputs) / len(inputs)
            complexity += input_complexity * 0.4
        
        # Logical complexity (number of assertions, conditions)
        logical_elements = content.count("assert") + content.count("if") + content.count("for")
        complexity += min(logical_elements / 5.0, 1.0) * 0.3
        
        return complexity
    
    def _compute_input_complexity(self, inp: Any) -> float:
        """Compute complexity score for a single input."""
        if inp is None:
            return 0.1
        elif isinstance(inp, (int, float)):
            return min(abs(inp) / 1000.0, 1.0)
        elif isinstance(inp, str):
            return min(len(inp) / 100.0, 1.0)
        elif isinstance(inp, (list, tuple)):
            return min(len(inp) / 20.0, 1.0)
        else:
            return 0.5
    
    def _compute_pattern_signature(self, test_case: Dict[str, Any]) -> str:
        """Compute unique signature for test pattern."""
        import hashlib
        
        # Create signature from key characteristics
        signature_data = {
            "approach": self._classify_test_approach(test_case),
            "coverage": self._classify_coverage_type(test_case),
            "complexity_bin": int(self._compute_test_complexity(test_case) * 10),
            "input_types": [type(inp).__name__ for inp in test_case.get("inputs", [])]
        }
        
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()[:16]
    
    def _detect_novel_patterns(self, verification_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect novel verification patterns not seen before."""
        novel_patterns = []
        
        for strategy in verification_strategies:
            signature = strategy.get("pattern_signature", "")
            
            # Check if this pattern signature is novel
            if signature not in [p.get("pattern_signature", "") for p in self.novel_patterns]:
                # Additional novelty checks
                if self._is_truly_novel(strategy):
                    novel_pattern = {
                        "pattern_signature": signature,
                        "discovery_time": time.time(),
                        "strategy": strategy,
                        "novelty_score": self._compute_novelty_score(strategy)
                    }
                    novel_patterns.append(novel_pattern)
        
        return novel_patterns
    
    def _is_truly_novel(self, strategy: Dict[str, Any]) -> bool:
        """Check if strategy represents truly novel behavior."""
        # Check against historical patterns
        approach = strategy.get("approach", "")
        coverage = strategy.get("coverage_type", "")
        complexity = strategy.get("complexity", 0.0)
        
        # Novel if high complexity with uncommon approach
        if complexity > 0.7 and approach in ["boundary_testing", "stress_testing"]:
            return True
        
        # Novel if unique combination of approach and coverage
        combination = f"{approach}_{coverage}"
        historical_combinations = [
            f"{s.get('approach', '')}_{s.get('coverage_type', '')}"
            for hist in self.behavior_history
            for s in hist.get("verification_patterns", [])
        ]
        
        return combination not in historical_combinations
    
    def _compute_novelty_score(self, strategy: Dict[str, Any]) -> float:
        """Compute novelty score for a strategy."""
        base_score = 0.5
        
        # Bonus for high complexity
        base_score += strategy.get("complexity", 0.0) * 0.3
        
        # Bonus for rare approaches
        rare_approaches = ["boundary_testing", "stress_testing", "equivalence_partitioning"]
        if strategy.get("approach", "") in rare_approaches:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def _compute_complexity_metrics(self, problems: List[Dict], solutions: List[Dict], test_cases: List[Dict]) -> Dict[str, float]:
        """Compute complexity evolution metrics."""
        metrics = {}
        
        # Problem complexity
        if problems:
            problem_complexities = [self._compute_problem_complexity(p) for p in problems]
            metrics["avg_problem_complexity"] = np.mean(problem_complexities)
            metrics["max_problem_complexity"] = np.max(problem_complexities)
        
        # Solution complexity
        if solutions:
            solution_complexities = [self._compute_solution_complexity(s) for s in solutions]
            metrics["avg_solution_complexity"] = np.mean(solution_complexities)
            metrics["solution_complexity_variance"] = np.var(solution_complexities)
        
        # Test case complexity
        if test_cases:
            test_complexities = [self._compute_test_complexity(tc) for tc in test_cases]
            metrics["avg_test_complexity"] = np.mean(test_complexities)
            metrics["test_complexity_trend"] = self._compute_complexity_trend(test_complexities)
        
        return metrics
    
    def _compute_problem_complexity(self, problem: Dict[str, Any]) -> float:
        """Compute complexity score for a problem."""
        question = problem.get("question", "")
        return min(len(question.split()) / 100.0, 1.0)
    
    def _compute_solution_complexity(self, solution: Dict[str, Any]) -> float:
        """Compute complexity score for a solution."""
        code = solution.get("code", "")
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        return min(len(lines) / 20.0, 1.0)
    
    def _compute_complexity_trend(self, complexities: List[float]) -> float:
        """Compute trend in complexity over time."""
        if len(complexities) < 2:
            return 0.0
        
        # Simple linear trend
        x = np.arange(len(complexities))
        y = np.array(complexities)
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _identify_emergent_strategies(self, iteration_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify emergent verification strategies."""
        strategies = []
        
        # Look for patterns that combine multiple verification approaches
        test_cases = iteration_data.get("test_cases", [])
        
        # Group test cases by problem
        problem_test_groups = {}
        for tc in test_cases:
            problem_id = tc.get("problem_id", "unknown")
            if problem_id not in problem_test_groups:
                problem_test_groups[problem_id] = []
            problem_test_groups[problem_id].append(tc)
        
        # Analyze each problem's test suite for emergent strategies
        for problem_id, test_group in problem_test_groups.items():
            if len(test_group) >= 3:  # Need multiple tests to identify strategy
                strategy = self._analyze_test_suite_strategy(test_group)
                if strategy:
                    strategy["problem_id"] = problem_id
                    strategies.append(strategy)
        
        return strategies
    
    def _analyze_test_suite_strategy(self, test_cases: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Analyze a test suite to identify emergent strategies."""
        approaches = [self._classify_test_approach(tc) for tc in test_cases]
        coverage_types = [self._classify_coverage_type(tc) for tc in test_cases]
        
        # Check for comprehensive coverage strategy
        unique_approaches = set(approaches)
        unique_coverage = set(coverage_types)
        
        if len(unique_approaches) >= 3 and len(unique_coverage) >= 3:
            return {
                "type": "comprehensive_coverage_strategy",
                "approaches": list(unique_approaches),
                "coverage_types": list(unique_coverage),
                "test_count": len(test_cases),
                "strategy_score": len(unique_approaches) * len(unique_coverage) / (len(test_cases) + 1)
            }
        
        # Check for progressive complexity strategy
        complexities = [self._compute_test_complexity(tc) for tc in test_cases]
        if len(complexities) >= 3:
            complexity_trend = self._compute_complexity_trend(complexities)
            if abs(complexity_trend) > 0.1:
                return {
                    "type": "progressive_complexity_strategy",
                    "complexity_trend": complexity_trend,
                    "complexity_range": max(complexities) - min(complexities),
                    "strategy_score": abs(complexity_trend) * (max(complexities) - min(complexities))
                }
        
        return None
    
    def get_behavior_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of observed behaviors."""
        if not self.behavior_history:
            return {"status": "no_data"}
        
        summary = {
            "total_iterations_analyzed": len(self.behavior_history),
            "novel_patterns_discovered": len(self.novel_patterns),
            "complexity_evolution": self._analyze_complexity_evolution(),
            "strategy_diversity": self._analyze_strategy_diversity(),
            "emergent_behavior_trends": self._analyze_emergent_trends(),
            "breakthrough_indicators": self._identify_breakthrough_indicators()
        }
        
        return summary
    
    def _analyze_complexity_evolution(self) -> Dict[str, Any]:
        """Analyze how complexity has evolved over iterations."""
        if len(self.behavior_history) < 2:
            return {"status": "insufficient_data"}
        
        iterations = []
        avg_complexities = []
        
        for hist in self.behavior_history:
            complexity_metrics = hist.get("complexity_evolution", {})
            if "avg_test_complexity" in complexity_metrics:
                iterations.append(hist.get("iteration", 0))
                avg_complexities.append(complexity_metrics["avg_test_complexity"])
        
        if len(avg_complexities) >= 2:
            overall_trend = self._compute_complexity_trend(avg_complexities)
            return {
                "overall_trend": overall_trend,
                "complexity_increase": overall_trend > 0.05,
                "final_complexity": avg_complexities[-1],
                "complexity_range": max(avg_complexities) - min(avg_complexities)
            }
        
        return {"status": "insufficient_complexity_data"}
    
    def _analyze_strategy_diversity(self) -> Dict[str, Any]:
        """Analyze diversity of verification strategies."""
        all_approaches = []
        all_coverage_types = []
        
        for hist in self.behavior_history:
            patterns = hist.get("verification_patterns", [])
            for pattern in patterns:
                all_approaches.append(pattern.get("approach", "unknown"))
                all_coverage_types.append(pattern.get("coverage_type", "unknown"))
        
        return {
            "unique_approaches": len(set(all_approaches)),
            "unique_coverage_types": len(set(all_coverage_types)),
            "approach_distribution": {approach: all_approaches.count(approach) for approach in set(all_approaches)},
            "coverage_distribution": {coverage: all_coverage_types.count(coverage) for coverage in set(all_coverage_types)},
            "diversity_score": len(set(all_approaches)) * len(set(all_coverage_types))
        }
    
    def _analyze_emergent_trends(self) -> Dict[str, Any]:
        """Analyze trends in emergent behaviors."""
        recent_behaviors = self.behavior_history[-5:] if len(self.behavior_history) >= 5 else self.behavior_history
        
        total_emergent_strategies = sum(len(hist.get("emergent_strategies", [])) for hist in recent_behaviors)
        novel_pattern_rate = len([p for p in self.novel_patterns if p.get("discovery_time", 0) > time.time() - 3600]) / max(len(recent_behaviors), 1)
        
        return {
            "recent_emergent_strategies": total_emergent_strategies,
            "novel_pattern_rate_per_iteration": novel_pattern_rate,
            "breakthrough_potential": total_emergent_strategies > 3 and novel_pattern_rate > 0.5
        }
    
    def _identify_breakthrough_indicators(self) -> Dict[str, Any]:
        """Identify indicators of breakthrough-level emergent behavior."""
        indicators = {
            "high_complexity_evolution": False,
            "novel_strategy_combinations": False,
            "consistent_pattern_discovery": False,
            "emergent_meta_strategies": False,
            "breakthrough_score": 0.0
        }
        
        # Check complexity evolution
        complexity_analysis = self._analyze_complexity_evolution()
        if complexity_analysis.get("complexity_increase", False) and complexity_analysis.get("overall_trend", 0) > 0.1:
            indicators["high_complexity_evolution"] = True
            indicators["breakthrough_score"] += 0.25
        
        # Check for novel strategy combinations
        strategy_diversity = self._analyze_strategy_diversity()
        if strategy_diversity.get("diversity_score", 0) > 9:  # 3+ approaches × 3+ coverage types
            indicators["novel_strategy_combinations"] = True
            indicators["breakthrough_score"] += 0.25
        
        # Check pattern discovery consistency
        if len(self.novel_patterns) > len(self.behavior_history) * 0.5:  # More than 0.5 novel patterns per iteration
            indicators["consistent_pattern_discovery"] = True
            indicators["breakthrough_score"] += 0.25
        
        # Check for emergent meta-strategies
        recent_emergent = self._analyze_emergent_trends()
        if recent_emergent.get("breakthrough_potential", False):
            indicators["emergent_meta_strategies"] = True
            indicators["breakthrough_score"] += 0.25
        
        return indicators


class SelfPlayTrainer:
    """
    Integrated self-play trainer for SRAG-V 4-player system.
    
    Combines:
    - GRPO training for policy optimization
    - MAP-Elites for behavioral diversity
    - Role-conditioned rewards for multi-agent learning
    - Emergent behavior tracking for breakthrough detection
    """
    
    def __init__(
        self,
        orchestrator: SRAGVOrchestrator,
        config: SelfPlayConfig,
        checkpoint_dir: Optional[str] = None
    ):
        self.orchestrator = orchestrator
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints/self_play")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize training components
        self.grpo_config = self._create_grpo_config()
        self.map_elites_config = self._create_map_elites_config()
        self.reward_system = RoleConditionedRewardSystem()
        self.behavior_tracker = EmergentBehaviorTracker()
        
        # Training state
        self.current_iteration = 0
        self.training_history: List[Dict[str, Any]] = []
        self.convergence_metrics: List[float] = []
        
        # Initialize components after orchestrator players are loaded
        self.grpo_trainer = None
        self.map_elites_archive = None
        
        logger.info(f"Self-play trainer initialized with {config.num_iterations} iterations planned")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _create_grpo_config(self) -> GRPOConfig:
        """Create GRPO configuration based on self-play config."""
        return GRPOConfig(
            group_size=8,
            batch_size=self.config.problems_per_iteration,
            ppo_epochs=4,
            learning_rate=1e-5,
            use_gradient_checkpointing=True,
            use_8bit_optimizer=True,
            lora_rank=64,
            lora_alpha=128,
            warmup_steps=100,
            max_steps=self.config.num_iterations * 1000,
            save_steps=self.config.checkpoint_every_iterations * 1000
        )
    
    def _create_map_elites_config(self) -> MAPElitesConfig:
        """Create MAP-Elites configuration."""
        return MAPElitesConfig(
            complexity_levels=5,
            approach_categories=16,
            verification_levels=10,
            archive_size=8000,  # Research-recommended size
            redundancy_factor=10,
            quality_weight=0.7,
            diversity_weight=0.3
        )
    
    def initialize_training_components(self):
        """Initialize GRPO trainer and MAP-Elites archive with loaded players."""
        if not self.orchestrator.problem_generator:
            raise ValueError("Orchestrator players must be initialized before training components")
        
        # Get player models
        players = {
            'problem_generator': self.orchestrator.problem_generator,
            'solution_generator': self.orchestrator.solution_generator,
            'verification_generator': self.orchestrator.verification_generator,
            'meta_verifier': self.orchestrator.meta_verifier
        }
        
        # Create reward functions
        reward_functions = {
            'problem_generator': self.reward_system.get_reward_function('problem_generator'),
            'solution_generator': self.reward_system.get_reward_function('solution_generator'),
            'verification_generator': self.reward_system.get_reward_function('verification_generator'),
            'meta_verifier': self.reward_system.get_reward_function('meta_verifier')
        }
        
        # Initialize GRPO trainer
        self.grpo_trainer = GRPOTrainer(
            config=self.grpo_config,
            players=players,
            reward_functions=reward_functions
        )
        
        # Initialize MAP-Elites archive
        self.map_elites_archive = MAPElitesArchive(self.map_elites_config)
        
        logger.info("Training components initialized successfully")
    
    def run_self_play_training(self) -> Dict[str, Any]:
        """
        Run the complete self-play training process.
        
        Returns comprehensive training results and analysis.
        """
        logger.info(f"🚀 Starting SRAG-V Self-Play Training - {self.config.num_iterations} iterations")
        logger.info("=" * 80)
        
        # Initialize training if not already done
        if not self.grpo_trainer:
            self.orchestrator.initialize_players()
            self.initialize_training_components()
        
        # Load bootstrap data
        bootstrap_problems = self._prepare_bootstrap_data()
        
        training_results = {
            "iterations": [],
            "convergence_analysis": {},
            "emergent_behaviors": {},
            "final_archive_stats": {},
            "breakthrough_indicators": {},
            "training_time": 0.0
        }
        
        start_time = time.time()
        
        try:
            # Run training iterations
            for iteration in range(1, self.config.num_iterations + 1):
                self.current_iteration = iteration
                
                logger.info(f"\n{'='*20} ITERATION {iteration}/{self.config.num_iterations} {'='*20}")
                
                # Determine iteration type
                if iteration <= self.config.bootstrap_iterations:
                    iteration_type = "bootstrap"
                    problems_to_use = bootstrap_problems
                elif iteration <= self.config.bootstrap_iterations + self.config.competitive_iterations:
                    iteration_type = "competitive"
                    problems_to_use = self._generate_competitive_problems(iteration)
                else:
                    iteration_type = "league"
                    problems_to_use = self._generate_league_problems(iteration)
                
                # Run single iteration
                iteration_result = self._run_single_iteration(
                    iteration=iteration,
                    iteration_type=iteration_type,
                    problems=problems_to_use
                )
                
                training_results["iterations"].append(iteration_result)
                
                # Analyze emergent behaviors
                if self.config.track_emergent_behaviors and iteration % self.config.behavior_analysis_frequency == 0:
                    behavior_analysis = self.behavior_tracker.analyze_iteration_behaviors(iteration_result)
                    logger.info(f"Behavior analysis: {len(behavior_analysis.get('novel_discoveries', []))} novel patterns discovered")
                
                # Check convergence
                convergence_score = self._compute_convergence_score(iteration_result)
                self.convergence_metrics.append(convergence_score)
                
                if len(self.convergence_metrics) >= 3:
                    recent_convergence = np.mean(self.convergence_metrics[-3:])
                    if recent_convergence < self.config.convergence_threshold:
                        logger.info(f"Training converged at iteration {iteration} (score: {recent_convergence:.4f})")
                        break
                
                # Quality gates
                if not self._check_quality_gates(iteration_result):
                    logger.warning(f"Quality gates failed at iteration {iteration}, adjusting parameters")
                    self._adjust_training_parameters()
                
                # Save checkpoint
                if iteration % self.config.checkpoint_every_iterations == 0:
                    self._save_checkpoint(iteration)

                # Clear GPU cache to prevent memory accumulation across iterations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("GPU cache cleared after iteration")

                logger.info(f"Iteration {iteration} complete - Quality: {iteration_result.get('quality_score', 0):.3f}")
            
            # Final analysis
            training_time = time.time() - start_time
            training_results["training_time"] = training_time
            
            # Convergence analysis
            training_results["convergence_analysis"] = self._analyze_convergence()
            
            # Emergent behavior analysis
            training_results["emergent_behaviors"] = self.behavior_tracker.get_behavior_summary()
            
            # Final archive statistics
            training_results["final_archive_stats"] = self.map_elites_archive.get_archive_statistics()
            
            # Breakthrough indicators
            training_results["breakthrough_indicators"] = training_results["emergent_behaviors"].get("breakthrough_indicators", {})
            
            # Save final results
            self._save_final_results(training_results)
            
            logger.info("🎉 Self-play training completed successfully!")
            logger.info(f"Training time: {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
            logger.info(f"Breakthrough score: {training_results['breakthrough_indicators'].get('breakthrough_score', 0):.3f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Save partial results
            training_results["error"] = str(e)
            training_results["training_time"] = time.time() - start_time
            self._save_final_results(training_results)
            raise
    
    def _prepare_bootstrap_data(self) -> List[Dict[str, Any]]:
        """Prepare bootstrap problems for initial training."""
        logger.info("Preparing bootstrap data...")
        
        # Initialize data loaders
        self.orchestrator.initialize_data_loaders()
        
        # Load expanded dataset
        problems, categorized = self.orchestrator.load_bootstrap_data()
        
        # Select problems for bootstrap phase
        bootstrap_problems = problems[:self.config.problems_per_iteration * self.config.bootstrap_iterations]
        
        logger.info(f"Bootstrap data prepared: {len(bootstrap_problems)} problems")
        return bootstrap_problems
    
    def _generate_competitive_problems(self, iteration: int) -> List[Dict[str, Any]]:
        """Generate problems for competitive phase."""
        problems = []
        
        # Use archive sampling + dynamic generation
        archive_samples = int(self.config.problems_per_iteration * self.config.archive_sampling_probability)
        new_problems = self.config.problems_per_iteration - archive_samples
        
        # Sample from archive
        if self.map_elites_archive.archive:
            diverse_elites = self.map_elites_archive.sample_diverse_elites(archive_samples)
            for elite in diverse_elites:
                problems.append(elite.solution.get('problem', {}))
        
        # Generate new problems dynamically
        if new_problems > 0:
            success_rates = self.orchestrator.compute_success_rates()
            type_distribution = self.orchestrator.compute_type_distribution()
            
            # Increase difficulty over iterations
            difficulty_multiplier = 1.0 + (iteration - self.config.bootstrap_iterations) * 0.1
            
            new_generated = self.orchestrator.problem_generator.generate_diverse_batch(
                success_rate_histogram=success_rates,
                problem_type_distribution=type_distribution,
                archive=self.orchestrator.problem_archive,
                batch_size=new_problems,
                difficulty_scaling=difficulty_multiplier if self.config.problem_difficulty_scaling else 1.0
            )
            problems.extend(new_generated)
        
        logger.info(f"Generated {len(problems)} competitive problems (archive: {archive_samples}, new: {new_problems})")
        return problems
    
    def _generate_league_problems(self, iteration: int) -> List[Dict[str, Any]]:
        """Generate problems for league phase."""
        # Use high-quality problems from archive + challenging new problems
        problems = []
        
        # Get top performers from archive
        if self.map_elites_archive.archive:
            elite_problems = self.map_elites_archive.sample_diverse_elites(self.config.problems_per_iteration // 2)
            for elite in elite_problems:
                problems.append(elite.solution.get('problem', {}))
        
        # Generate challenging new problems
        remaining_problems = self.config.problems_per_iteration - len(problems)
        if remaining_problems > 0:
            # Use highest difficulty settings
            challenging_problems = self.orchestrator.problem_generator.generate_diverse_batch(
                success_rate_histogram={"easy": 0.1, "medium": 0.3, "hard": 0.6},
                problem_type_distribution=self.orchestrator.compute_type_distribution(),
                archive=self.orchestrator.problem_archive,
                batch_size=remaining_problems,
                difficulty_scaling=2.0  # Maximum difficulty
            )
            problems.extend(challenging_problems)
        
        logger.info(f"Generated {len(problems)} league problems")
        return problems
    
    def _run_single_iteration(
        self, 
        iteration: int, 
        iteration_type: str, 
        problems: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run a single self-play iteration with GRPO training."""
        iteration_start = time.time()
        
        iteration_result = {
            "iteration": iteration,
            "type": iteration_type,
            "problems": problems,
            "solutions": [],
            "test_cases": [],
            "validation_results": [],
            "training_metrics": {},
            "quality_score": 0.0,
            "processing_time": 0.0
        }
        
        try:
            # Generate solutions using solution generator
            logger.info(f"Generating {self.config.solutions_per_problem} solutions per problem...")
            all_solutions = []
            
            for i, problem in enumerate(problems):
                solutions = self.orchestrator.solution_generator.generate(
                    problem=problem,
                    num_solutions=self.config.solutions_per_problem,
                    max_attempts_per_solution=2
                )
                
                for solution in solutions:
                    solution['problem_id'] = problem.get('problem_id', f'iter_{iteration}_prob_{i}')
                
                all_solutions.extend(solutions)
            
            iteration_result["solutions"] = all_solutions
            logger.info(f"Generated {len(all_solutions)} solutions")
            
            # Generate test cases using verification generator
            logger.info(f"Generating {self.config.test_cases_per_problem} test cases per problem...")
            all_test_cases = []
            
            for i, problem in enumerate(problems):
                problem_id = problem.get('problem_id', f'iter_{iteration}_prob_{i}')
                problem_solutions = [s for s in all_solutions if s.get('problem_id') == problem_id]
                
                if problem_solutions:
                    test_cases = self.orchestrator.verification_generator.generate(
                        problem=problem,
                        solutions=problem_solutions,
                        num_test_cases=self.config.test_cases_per_problem
                    )
                    
                    for test_case in test_cases:
                        test_case['problem_id'] = problem_id
                    
                    all_test_cases.extend(test_cases)
            
            iteration_result["test_cases"] = all_test_cases
            logger.info(f"Generated {len(all_test_cases)} test cases")
            
            # Meta-verification
            logger.info("Running meta-verification...")
            all_validation_results = []
            
            # Group by problem for validation
            problem_groups = {}
            for problem in problems:
                problem_id = problem.get('problem_id', f'iter_{iteration}_prob_{problems.index(problem)}')
                problem_solutions = [s for s in all_solutions if s.get('problem_id') == problem_id]
                problem_tests = [t for t in all_test_cases if t.get('problem_id') == problem_id]
                
                if problem_solutions and problem_tests:
                    validation_results = self.orchestrator.meta_verifier.validate_test_cases(
                        problem=problem,
                        solutions=problem_solutions,
                        test_cases=problem_tests
                    )
                    # Handle both dict (with strategic_feedback) and list returns
                    if isinstance(validation_results, dict):
                        validation_results = validation_results.get('validated_test_cases', [])
                    all_validation_results.extend(validation_results)
            
            iteration_result["validation_results"] = all_validation_results
            
            # GRPO Training Step
            logger.info("Running GRPO training step...")
            
            # Prepare training data
            training_prompts = []
            training_roles = []
            training_contexts = []
            
            # Add problem generation training data
            for problem in problems:
                training_prompts.append(f"Generate a coding problem: {problem.get('question', '')}")
                training_roles.append('problem_generator')
                training_contexts.append({'problem': problem, 'target_difficulty': 'medium'})
            
            # Add solution generation training data
            for i, solution in enumerate(all_solutions[:self.config.problems_per_iteration]):  # Limit for memory
                problem_id = solution.get('problem_id', '')
                problem = next((p for p in problems if p.get('problem_id') == problem_id), {})
                
                training_prompts.append(f"Solve this problem: {problem.get('question', '')}")
                training_roles.append('solution_generator')
                training_contexts.append({
                    'problem': problem,
                    'test_cases': [tc for tc in all_test_cases if tc.get('problem_id') == problem_id]
                })
            
            # Run GRPO training
            training_metrics = self.grpo_trainer.train_step(
                prompts=training_prompts,
                player_roles=training_roles,
                contexts=training_contexts
            )
            
            iteration_result["training_metrics"] = training_metrics
            
            # Update MAP-Elites archive
            logger.info("Updating MAP-Elites archive...")
            archive_additions = 0
            
            for i, problem in enumerate(problems):
                problem_id = problem.get('problem_id', f'iter_{iteration}_prob_{i}')
                problem_solutions = [s for s in all_solutions if s.get('problem_id') == problem_id]
                problem_tests = [t for t in all_test_cases if t.get('problem_id') == problem_id]
                
                if problem_solutions and problem_tests:
                    # Create solution package for archive
                    solution_package = {
                        'problem': problem,
                        'code': problem_solutions[0].get('code', ''),  # Use best solution
                        'test_cases': problem_tests,
                        'validation': [v for v in all_validation_results if v.get('problem_id') == problem_id]
                    }
                    
                    # Compute quality score
                    quality_score = self._compute_solution_quality(solution_package)
                    
                    # Add to archive
                    was_added = self.map_elites_archive.add_solution(
                        solution=solution_package,
                        quality=quality_score,
                        player_role='integrated_solution',
                        metadata={'iteration': iteration, 'type': iteration_type}
                    )
                    
                    if was_added:
                        archive_additions += 1
            
            logger.info(f"Added {archive_additions} solutions to MAP-Elites archive")
            
            # Advance archive generation
            self.map_elites_archive.advance_generation()
            
            # Compute iteration quality score
            iteration_result["quality_score"] = self._compute_iteration_quality(iteration_result)
            
            # Update orchestrator state
            self.orchestrator.current_iteration = iteration
            self.orchestrator.update_problem_archive(problems, all_solutions, all_test_cases)
            
            iteration_result["processing_time"] = time.time() - iteration_start
            
            logger.info(f"Iteration {iteration} completed in {iteration_result['processing_time']:.1f}s")
            logger.info(f"Quality score: {iteration_result['quality_score']:.3f}")
            
            return iteration_result
            
        except Exception as e:
            import traceback
            logger.error(f"Error in iteration {iteration}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            iteration_result["error"] = str(e)
            iteration_result["error_type"] = type(e).__name__
            iteration_result["error_traceback"] = traceback.format_exc()
            iteration_result["processing_time"] = time.time() - iteration_start
            return iteration_result
    
    def _compute_solution_quality(self, solution_package: Dict[str, Any]) -> float:
        """Compute quality score for a solution package."""
        quality_components = []
        
        # Problem quality
        problem = solution_package.get('problem', {})
        problem_quality = len(problem.get('question', '').split()) / 50.0  # Normalized by word count
        quality_components.append(min(problem_quality, 1.0))
        
        # Code quality
        code = solution_package.get('code', '')
        code_quality = 0.5  # Default
        if 'def ' in code and 'return' in code:
            code_quality = 0.8
        if len(code.split('\n')) > 3 and 'if' in code:
            code_quality = 1.0
        quality_components.append(code_quality)
        
        # Test case quality
        test_cases = solution_package.get('test_cases', [])
        test_quality = min(len(test_cases) / 8.0, 1.0)  # Normalized by target number
        quality_components.append(test_quality)
        
        # Validation quality
        validations = solution_package.get('validation', [])
        if validations:
            valid_count = sum(1 for v in validations if v.get('is_valid', False))
            validation_quality = valid_count / len(validations)
        else:
            validation_quality = 0.5
        quality_components.append(validation_quality)
        
        return np.mean(quality_components)
    
    def _compute_iteration_quality(self, iteration_result: Dict[str, Any]) -> float:
        """Compute overall quality score for an iteration."""
        quality_factors = []
        
        # Solution generation success
        solutions = iteration_result.get("solutions", [])
        problems = iteration_result.get("problems", [])
        if problems:
            solution_ratio = len(solutions) / (len(problems) * self.config.solutions_per_problem)
            quality_factors.append(solution_ratio)
        
        # Test case generation success
        test_cases = iteration_result.get("test_cases", [])
        if problems:
            test_ratio = len(test_cases) / (len(problems) * self.config.test_cases_per_problem)
            quality_factors.append(test_ratio)
        
        # Validation success
        validations = iteration_result.get("validation_results", [])
        if validations:
            valid_ratio = sum(1 for v in validations if v.get('is_valid', False)) / len(validations)
            quality_factors.append(valid_ratio)
        
        # Training metrics
        training_metrics = iteration_result.get("training_metrics", {})
        if training_metrics:
            mean_reward = training_metrics.get("mean_reward", 0.0)
            normalized_reward = max(0, min(1, (mean_reward + 1) / 2))  # Normalize to [0,1]
            quality_factors.append(normalized_reward)
        
        return np.mean(quality_factors) if quality_factors else 0.0
    
    def _compute_convergence_score(self, iteration_result: Dict[str, Any]) -> float:
        """Compute convergence score based on improvement rate."""
        if len(self.training_history) < 2:
            return 1.0  # Not enough history
        
        current_quality = iteration_result.get("quality_score", 0.0)
        previous_quality = self.training_history[-1].get("quality_score", 0.0)
        
        improvement = abs(current_quality - previous_quality)
        return improvement
    
    def _check_quality_gates(self, iteration_result: Dict[str, Any]) -> bool:
        """Check if iteration meets minimum quality requirements."""
        quality_score = iteration_result.get("quality_score", 0.0)
        
        # Check solution accuracy
        solutions = iteration_result.get("solutions", [])
        if solutions:
            # Simplified accuracy check
            successful_solutions = sum(1 for s in solutions if s.get('score', 0) > 0.5)
            solution_accuracy = successful_solutions / len(solutions)
            if solution_accuracy < self.config.min_solution_accuracy:
                logger.warning(f"Solution accuracy {solution_accuracy:.3f} below threshold {self.config.min_solution_accuracy}")
                return False
        
        # Check test validity
        validations = iteration_result.get("validation_results", [])
        if validations:
            valid_tests = sum(1 for v in validations if v.get('is_valid', False))
            test_validity = valid_tests / len(validations)
            if test_validity < self.config.min_test_validity:
                logger.warning(f"Test validity {test_validity:.3f} below threshold {self.config.min_test_validity}")
                return False
        
        return True
    
    def _adjust_training_parameters(self):
        """Adjust training parameters when quality gates fail."""
        # Reduce learning rate
        for scheduler in self.grpo_trainer.schedulers.values():
            for param_group in scheduler.optimizer.param_groups:
                param_group['lr'] *= 0.8
        
        logger.info("Reduced learning rates by 20% due to quality gate failure")
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze convergence patterns."""
        if len(self.convergence_metrics) < 3:
            return {"status": "insufficient_data"}
        
        recent_metrics = self.convergence_metrics[-5:]
        overall_trend = np.mean(np.diff(recent_metrics)) if len(recent_metrics) > 1 else 0
        
        return {
            "converged": np.mean(recent_metrics) < self.config.convergence_threshold,
            "convergence_trend": overall_trend,
            "final_convergence_score": self.convergence_metrics[-1],
            "iterations_to_convergence": len(self.convergence_metrics)
        }
    
    def _save_checkpoint(self, iteration: int):
        """Save training checkpoint."""
        checkpoint_data = {
            "iteration": iteration,
            "config": self.config.__dict__,
            "training_history": self.training_history[-10:],  # Keep last 10 iterations
            "convergence_metrics": self.convergence_metrics,
            "behavior_summary": self.behavior_tracker.get_behavior_summary()
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iteration_{iteration}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        # Save GRPO trainer checkpoint
        grpo_checkpoint_dir = self.checkpoint_dir / f"grpo_iteration_{iteration}"
        self.grpo_trainer.save_checkpoint(str(grpo_checkpoint_dir))
        
        # Save MAP-Elites archive
        archive_path = self.checkpoint_dir / f"archive_iteration_{iteration}.json"
        self.map_elites_archive.save_archive(str(archive_path))
        
        logger.info(f"Checkpoint saved for iteration {iteration}")
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Save final training results."""
        results_path = self.checkpoint_dir / "final_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save behavior analysis
        behavior_summary = self.behavior_tracker.get_behavior_summary()
        behavior_path = self.checkpoint_dir / "emergent_behaviors.json"
        with open(behavior_path, 'w') as f:
            json.dump(behavior_summary, f, indent=2, default=str)
        
        logger.info(f"Final results saved to {results_path}")
        logger.info(f"Emergent behavior analysis saved to {behavior_path}")