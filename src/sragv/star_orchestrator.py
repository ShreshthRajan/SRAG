"""
STAR Enhanced Orchestrator - Phase 2 Implementation
Integrates the Strategic Oracle with the existing 4-player architecture.

This orchestrator enables strategic problem selection instead of random selection,
dramatically improving learning efficiency for Phase 3 self-amplification.
"""

import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import time

from .orchestrator import Orchestrator
from .strategic_oracle import StrategicOracle
from .models.verification_generator import TestQualityPredictor

logger = logging.getLogger(__name__)


class STAROrchestrator(Orchestrator):
    """
    STAR Enhanced Orchestrator with Strategic Oracle Integration.
    
    Extends the base orchestrator with strategic problem selection capabilities:
    1. Uses Strategic Oracle for intelligent problem selection
    2. Integrates test quality prediction from enhanced VerificationGenerator
    3. Maintains all existing 4-player functionality
    4. Provides analytics and performance tracking
    """
    
    def __init__(self, config: Dict):
        # Initialize base orchestrator first
        super().__init__(config)
        
        # STAR Phase 2 enhancements
        self.strategic_oracle = None
        self.enable_strategic_selection = config.get("enable_strategic_selection", True)
        self.selection_analytics = {
            'total_selections': 0,
            'strategic_selections': 0,
            'random_selections': 0,
            'selection_history': []
        }
        
        logger.info("STAR Orchestrator initialized with strategic selection capability")
    
    def initialize_strategic_oracle(
        self,
        selection_strategy: str = "uncertainty_diversity_balanced",
        weights: Optional[Dict[str, float]] = None
    ):
        """Initialize the Strategic Oracle with the Phase 1 confidence calibrator."""
        
        if not self.enable_strategic_selection:
            logger.info("Strategic selection disabled - using random selection")
            return
        
        try:
            # Get confidence calibrator from solution generator
            confidence_calibrator = None
            if hasattr(self.solution_generator, 'confidence_calibrator'):
                confidence_calibrator = self.solution_generator.confidence_calibrator
                logger.info("✅ Using Phase 1 confidence calibrator for strategic oracle")
            else:
                logger.warning("⚠️ No confidence calibrator found - strategic selection may be limited")
            
            # Default weights optimized for code generation learning
            default_weights = {
                'uncertainty_weight': 0.4,  # Focus on hard problems
                'diversity_weight': 0.4,    # Ensure broad coverage
                'quality_weight': 0.2       # Prefer reliable test signals
            }
            
            if weights:
                default_weights.update(weights)
            
            # Initialize strategic oracle
            self.strategic_oracle = StrategicOracle(
                confidence_calibrator=confidence_calibrator,
                selection_strategy=selection_strategy,
                **default_weights
            )
            
            logger.info(f"🎯 Strategic Oracle initialized with strategy: {selection_strategy}")
            logger.info(f"   Weights: {default_weights}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Strategic Oracle: {e}")
            logger.info("Falling back to random selection")
            self.enable_strategic_selection = False
    
    def select_problems_strategically(
        self,
        candidate_problems: List[Dict[str, Any]],
        num_select: int,
        solution_history: Optional[List[Dict]] = None,
        selection_context: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Strategic problem selection using the STAR oracle.
        
        This is the core Phase 2 functionality that replaces random selection
        with intelligent uncertainty + diversity + quality based selection.
        """
        
        if not self.enable_strategic_selection or not self.strategic_oracle:
            logger.info("Using random problem selection (Strategic Oracle not available)")
            selected = np.random.choice(
                candidate_problems, 
                size=min(num_select, len(candidate_problems)), 
                replace=False
            ).tolist()
            self.selection_analytics['random_selections'] += len(selected)
            return selected
        
        logger.info(f"🎯 STAR Strategic Problem Selection")
        logger.info(f"   Candidates: {len(candidate_problems)}")
        logger.info(f"   Selecting: {num_select}")
        
        # Prepare selection context
        selected_problems = selection_context.get('selected_problems', []) if selection_context else []
        
        # Strategic selection using oracle
        start_time = time.time()
        
        try:
            selected_problems = self.strategic_oracle.strategic_problem_selection(
                candidate_problems=candidate_problems,
                num_select=num_select,
                selected_problems=selected_problems,
                solution_history=solution_history or []
            )
            
            selection_time = time.time() - start_time
            
            logger.info(f"✅ Strategic selection completed in {selection_time:.2f}s")
            logger.info(f"   Selected {len(selected_problems)} problems")
            
            # Update analytics
            self.selection_analytics['strategic_selections'] += len(selected_problems)
            self.selection_analytics['total_selections'] += len(selected_problems)
            self.selection_analytics['selection_history'].append({
                'timestamp': time.time(),
                'num_candidates': len(candidate_problems),
                'num_selected': len(selected_problems),
                'selection_time': selection_time,
                'strategy': 'strategic'
            })
            
            return selected_problems
            
        except Exception as e:
            logger.error(f"Strategic selection failed: {e}")
            logger.info("Falling back to random selection")
            
            # Fallback to random selection
            selected = np.random.choice(
                candidate_problems,
                size=min(num_select, len(candidate_problems)),
                replace=False
            ).tolist()
            
            self.selection_analytics['random_selections'] += len(selected)
            return selected
    
    def process_with_strategic_enhancements(
        self,
        problems: List[Dict[str, Any]],
        num_problems_to_process: Optional[int] = None,
        enable_test_quality_prediction: bool = True
    ) -> Dict[str, Any]:
        """
        Process problems with STAR Phase 2 enhancements.
        
        Integrates strategic selection with enhanced test quality prediction
        to create a complete strategic learning pipeline.
        """
        
        logger.info("🚀 STAR Enhanced Processing Pipeline")
        logger.info("=" * 60)
        
        # Determine how many problems to process
        if num_problems_to_process is None:
            num_problems_to_process = min(len(problems), 20)  # Default batch size
        
        total_problems = len(problems)
        logger.info(f"📊 Processing {num_problems_to_process} problems from {total_problems} available")
        
        # Strategic problem selection
        selected_problems = self.select_problems_strategically(
            candidate_problems=problems,
            num_select=num_problems_to_process
        )
        
        if not selected_problems:
            logger.error("No problems selected - cannot proceed")
            return {'success': False, 'error': 'No problems selected'}
        
        logger.info(f"✅ Selected {len(selected_problems)} problems strategically")
        
        # Process each selected problem with full SRAG-V pipeline
        results = {
            'problems_processed': [],
            'strategic_analytics': {},
            'performance_metrics': {},
            'success': True
        }
        
        for i, problem in enumerate(selected_problems):
            logger.info(f"🔧 Processing problem {i+1}/{len(selected_problems)}: {problem.get('problem_id', 'unknown')}")
            
            try:
                # Step 1: Generate solutions (P2 - Solution Generator)
                solutions = self.solution_generator.generate(
                    problem=problem,
                    num_solutions=6  # Smaller batch for efficiency
                )
                
                if not solutions:
                    logger.warning(f"No solutions generated for problem {i+1}")
                    continue
                
                logger.info(f"  ✅ Generated {len(solutions)} solutions")
                
                # Step 2: Generate tests with quality prediction (P3 - Enhanced VerificationGenerator)
                test_generation_result = {'test_cases': [], 'quality_metrics': {'overall_quality': 0.5}}
                
                if enable_test_quality_prediction and hasattr(self.verification_generator, 'generate_with_quality_prediction'):
                    test_generation_result = self.verification_generator.generate_with_quality_prediction(
                        problem=problem,
                        solutions=solutions,
                        num_test_cases=6,  # Smaller batch for efficiency
                        quality_threshold=0.6
                    )
                    logger.info(f"  🧪 Generated {test_generation_result['num_generated']} test cases")
                    logger.info(f"     Quality Score: {test_generation_result['quality_metrics']['overall_quality']:.3f}")
                else:
                    # Fallback to regular test generation
                    test_cases = self.verification_generator.generate(problem, solutions, 6)
                    test_generation_result = {
                        'test_cases': test_cases,
                        'quality_metrics': {'overall_quality': 0.5},
                        'quality_sufficient': True,
                        'num_generated': len(test_cases) if test_cases else 0
                    }
                    logger.info(f"  🧪 Generated {len(test_cases)} test cases (no quality prediction)")
                
                # Step 3: Meta-verification (P4 - MetaVerifier as strategic feedback)
                verification_scores = []
                if hasattr(self.meta_verifier, 'verify'):
                    for solution in solutions:
                        score = self.meta_verifier.verify(
                            problem=problem,
                            solution=solution,
                            test_cases=test_generation_result['test_cases']
                        )
                        verification_scores.append(score)
                
                # Compile results for this problem
                problem_result = {
                    'problem_id': problem.get('problem_id', f'problem_{i}'),
                    'problem': problem,
                    'solutions': solutions,
                    'test_generation': test_generation_result,
                    'verification_scores': verification_scores,
                    'processing_successful': True
                }
                
                results['problems_processed'].append(problem_result)
                logger.info(f"  ✅ Problem {i+1} processed successfully")
                
            except Exception as e:
                logger.error(f"  ❌ Error processing problem {i+1}: {e}")
                problem_result = {
                    'problem_id': problem.get('problem_id', f'problem_{i}'),
                    'problem': problem,
                    'error': str(e),
                    'processing_successful': False
                }
                results['problems_processed'].append(problem_result)
        
        # Compile strategic analytics
        if self.strategic_oracle:
            results['strategic_analytics'] = self.strategic_oracle.get_selection_analytics()
        
        results['strategic_analytics'].update(self.selection_analytics)
        
        # Compute performance metrics
        successful_problems = [p for p in results['problems_processed'] if p.get('processing_successful', False)]
        results['performance_metrics'] = {
            'total_problems_attempted': len(selected_problems),
            'successful_problems': len(successful_problems),
            'success_rate': len(successful_problems) / len(selected_problems) if selected_problems else 0,
            'average_solutions_per_problem': np.mean([len(p.get('solutions', [])) for p in successful_problems]) if successful_problems else 0,
            'average_test_quality': np.mean([
                p.get('test_generation', {}).get('quality_metrics', {}).get('overall_quality', 0.5) 
                for p in successful_problems
            ]) if successful_problems else 0.5
        }
        
        logger.info("🎯 STAR Enhanced Processing Complete")
        logger.info(f"   Success Rate: {results['performance_metrics']['success_rate']:.2%}")
        logger.info(f"   Avg Solutions/Problem: {results['performance_metrics']['average_solutions_per_problem']:.1f}")
        logger.info(f"   Avg Test Quality: {results['performance_metrics']['average_test_quality']:.3f}")
        logger.info("=" * 60)
        
        return results
    
    def get_star_analytics(self) -> Dict[str, Any]:
        """Get comprehensive STAR Phase 2 analytics."""
        
        analytics = {
            'orchestrator_type': 'STAR_Enhanced',
            'strategic_selection': {
                'enabled': self.enable_strategic_selection,
                'oracle_initialized': self.strategic_oracle is not None
            },
            'selection_analytics': self.selection_analytics
        }
        
        if self.strategic_oracle:
            analytics['oracle_analytics'] = self.strategic_oracle.get_selection_analytics()
        
        return analytics
    
    def update_strategic_performance(
        self,
        selected_problems: List[Dict[str, Any]],
        performance_results: Dict[str, Any]
    ):
        """Update strategic oracle with performance feedback for adaptive learning."""
        
        if not self.strategic_oracle:
            return
        
        try:
            # Extract performance metrics
            performance_metrics = {
                'success_rate': performance_results.get('success_rate', 0.0),
                'avg_solution_quality': performance_results.get('avg_solution_quality', 0.5),
                'avg_test_quality': performance_results.get('avg_test_quality', 0.5)
            }
            
            # Update oracle
            self.strategic_oracle.update_performance_feedback(
                selected_problems=selected_problems,
                performance_metrics=performance_metrics
            )
            
            logger.info("📈 Strategic oracle updated with performance feedback")
            
        except Exception as e:
            logger.warning(f"Failed to update strategic performance: {e}")
    
    def save_star_state(self, path: str):
        """Save STAR orchestrator state including strategic oracle."""
        
        state = {
            'orchestrator_type': 'STAR_Enhanced',
            'enable_strategic_selection': self.enable_strategic_selection,
            'selection_analytics': self.selection_analytics
        }
        
        if self.strategic_oracle:
            state['oracle_analytics'] = self.strategic_oracle.get_selection_analytics()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"STAR orchestrator state saved to {path}")
    
    def load_star_state(self, path: str):
        """Load STAR orchestrator state."""
        
        try:
            with open(path, 'r') as f:
                state = json.load(f)
            
            self.enable_strategic_selection = state.get('enable_strategic_selection', True)
            self.selection_analytics = state.get('selection_analytics', self.selection_analytics)
            
            logger.info(f"STAR orchestrator state loaded from {path}")
            
        except Exception as e:
            logger.warning(f"Failed to load STAR state: {e}")
            logger.info("Using default STAR configuration")