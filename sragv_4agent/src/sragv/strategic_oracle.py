"""
STAR Strategic Oracle - Phase 2 Implementation
Transforms the MetaVerifier into an intelligent problem selection system.

The Strategic Oracle combines:
1. Uncertainty scoring from confidence variance (Phase 1 calibration)
2. Diversity scoring from problem embedding distances  
3. Quality scoring from test/verification assessment
4. Strategic sampling for maximum learning efficiency

Based on 2024-2025 research on uncertainty-guided self-training and strategic selection.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
import json
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class StrategicOracle:
    """
    STAR Strategic Oracle - Core of Phase 2 architecture.
    
    Transforms problem selection from random to strategic based on:
    - Uncertainty: Problems where models are most uncertain (high learning potential)
    - Diversity: Problems covering different algorithmic domains
    - Quality: Problems with reliable test cases and clear specifications
    
    This enables data-efficient self-amplification in Phase 3.
    """
    
    def __init__(
        self,
        confidence_calibrator,
        embedding_model: Optional[str] = None,
        selection_strategy: str = "uncertainty_diversity_balanced",
        diversity_weight: float = 0.4,
        uncertainty_weight: float = 0.4,
        quality_weight: float = 0.2
    ):
        self.confidence_calibrator = confidence_calibrator
        self.selection_strategy = selection_strategy
        
        # Strategic weights based on STAR methodology
        self.diversity_weight = diversity_weight
        self.uncertainty_weight = uncertainty_weight  
        self.quality_weight = quality_weight
        
        # Problem embeddings for diversity calculation
        self.problem_embeddings = {}
        self.problem_clusters = None
        
        # Selection history for adaptive learning
        self.selection_history = []
        self.performance_history = []
        
        logger.info(f"Strategic Oracle initialized with strategy: {selection_strategy}")
        logger.info(f"Weights - Uncertainty: {uncertainty_weight}, Diversity: {diversity_weight}, Quality: {quality_weight}")
    
    def compute_uncertainty_score(
        self, 
        problem: Dict[str, Any], 
        solutions: List[Dict[str, Any]]
    ) -> float:
        """
        Compute uncertainty score based on confidence variance across solutions.
        Higher uncertainty = higher learning potential.
        
        Uses Phase 1 calibrated confidence to assess model uncertainty.
        """
        try:
            if not solutions or len(solutions) < 2:
                return 0.5  # Medium uncertainty for single solutions
            
            # Extract confidence scores from all solutions
            confidences = []
            for solution in solutions:
                confidence = solution.get('score', 0.5)
                
                # Apply calibration if available and trained
                if (hasattr(self.confidence_calibrator, 'use_calibration') and 
                    self.confidence_calibrator.use_calibration):
                    
                    # Use calibrated confidence scoring
                    code = solution.get('code', '')
                    if code:
                        try:
                            # Get calibrated confidence
                            features = self.confidence_calibrator.extract_solution_features(code, problem)
                            calibrated_conf, _ = self.confidence_calibrator.calibrate_confidence(
                                base_confidence=confidence,
                                temperature=1.0,
                                features=features
                            )
                            confidences.append(calibrated_conf)
                        except Exception as e:
                            logger.debug(f"Calibration failed, using base confidence: {e}")
                            confidences.append(confidence)
                    else:
                        confidences.append(confidence)
                else:
                    confidences.append(confidence)
            
            # Calculate uncertainty metrics
            conf_array = np.array(confidences)
            
            # Primary uncertainty: Standard deviation of confidences
            uncertainty = np.std(conf_array)
            
            # Secondary uncertainty: Mean distance from high confidence
            mean_conf = np.mean(conf_array)
            confidence_gap = max(0, 0.9 - mean_conf)  # Gap from high confidence
            
            # Combined uncertainty score
            combined_uncertainty = 0.7 * uncertainty + 0.3 * confidence_gap
            
            # Normalize to [0, 1] range
            normalized_uncertainty = min(1.0, combined_uncertainty * 2.0)
            
            logger.debug(f"Problem uncertainty: {normalized_uncertainty:.3f} (std: {uncertainty:.3f}, gap: {confidence_gap:.3f})")
            return normalized_uncertainty
            
        except Exception as e:
            logger.warning(f"Error computing uncertainty score: {e}")
            return 0.5
    
    def compute_diversity_score(
        self, 
        problem: Dict[str, Any], 
        selected_problems: List[Dict[str, Any]]
    ) -> float:
        """
        Compute diversity score based on problem embedding distances.
        Higher diversity = better coverage of problem space.
        
        Uses simple but effective text-based embedding for computational efficiency.
        """
        try:
            problem_id = problem.get('problem_id', str(hash(str(problem))))
            
            # Get or compute problem embedding
            if problem_id not in self.problem_embeddings:
                embedding = self._compute_problem_embedding(problem)
                self.problem_embeddings[problem_id] = embedding
            
            current_embedding = self.problem_embeddings[problem_id]
            
            if not selected_problems:
                return 1.0  # Maximum diversity for first problem
            
            # Compute distances to already selected problems
            distances = []
            for selected_prob in selected_problems:
                selected_id = selected_prob.get('problem_id', str(hash(str(selected_prob))))
                
                if selected_id not in self.problem_embeddings:
                    selected_embedding = self._compute_problem_embedding(selected_prob)
                    self.problem_embeddings[selected_id] = selected_embedding
                else:
                    selected_embedding = self.problem_embeddings[selected_id]
                
                # Compute cosine distance
                distance = cosine_distances([current_embedding], [selected_embedding])[0][0]
                distances.append(distance)
            
            # Diversity score is minimum distance to existing problems
            # (we want problems far from what we've already selected)
            diversity_score = min(distances) if distances else 1.0
            
            logger.debug(f"Problem diversity: {diversity_score:.3f} (min distance to selected)")
            return diversity_score
            
        except Exception as e:
            logger.warning(f"Error computing diversity score: {e}")
            return 0.5
    
    def compute_quality_score(
        self, 
        problem: Dict[str, Any],
        test_results: Optional[List[Dict]] = None
    ) -> float:
        """
        Compute quality score based on problem specification and test case reliability.
        Higher quality = better learning signal.
        
        Assesses problem clarity, test coverage, and specification completeness.
        """
        try:
            quality_score = 0.0
            
            # 1. Problem specification quality (40% weight)
            problem_text = problem.get('question', problem.get('description', ''))
            spec_quality = self._assess_specification_quality(problem_text)
            quality_score += 0.4 * spec_quality
            
            # 2. Test case quality (35% weight)  
            test_quality = self._assess_test_quality(problem)
            quality_score += 0.35 * test_quality
            
            # 3. Problem complexity appropriateness (25% weight)
            complexity_score = self._assess_complexity_appropriateness(problem_text)
            quality_score += 0.25 * complexity_score
            
            logger.debug(f"Problem quality: {quality_score:.3f} (spec: {spec_quality:.2f}, test: {test_quality:.2f}, complex: {complexity_score:.2f})")
            return quality_score
            
        except Exception as e:
            logger.warning(f"Error computing quality score: {e}")
            return 0.5
    
    def strategic_problem_selection(
        self,
        candidate_problems: List[Dict[str, Any]],
        num_select: int,
        selected_problems: Optional[List[Dict[str, Any]]] = None,
        solution_history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Core STAR strategic selection algorithm.
        
        Selects problems that maximize learning efficiency by balancing:
        - Uncertainty (where models struggle most)
        - Diversity (broad problem coverage)  
        - Quality (reliable learning signal)
        """
        logger.info(f"🎯 Strategic selection: {num_select} problems from {len(candidate_problems)} candidates")
        
        if not candidate_problems:
            return []
        
        if num_select >= len(candidate_problems):
            logger.info("Selecting all available problems")
            return candidate_problems
        
        selected_problems = selected_problems or []
        solution_history = solution_history or []
        
        # Score all candidate problems
        problem_scores = []
        
        for problem in candidate_problems:
            # Get solutions for this problem if available
            problem_solutions = [
                sol for sol in solution_history 
                if sol.get('problem_id') == problem.get('problem_id')
            ]
            
            # Compute strategic scores
            uncertainty = self.compute_uncertainty_score(problem, problem_solutions)
            diversity = self.compute_diversity_score(problem, selected_problems)
            quality = self.compute_quality_score(problem)
            
            # Combined strategic score
            strategic_score = (
                self.uncertainty_weight * uncertainty +
                self.diversity_weight * diversity +
                self.quality_weight * quality
            )
            
            problem_scores.append({
                'problem': problem,
                'strategic_score': strategic_score,
                'uncertainty': uncertainty,
                'diversity': diversity,
                'quality': quality
            })
            
            logger.debug(f"Problem {problem.get('problem_id', 'unknown')}: "
                        f"strategic={strategic_score:.3f} "
                        f"(U:{uncertainty:.2f}, D:{diversity:.2f}, Q:{quality:.2f})")
        
        # Sort by strategic score (highest first)
        problem_scores.sort(key=lambda x: x['strategic_score'], reverse=True)
        
        # Select top problems
        selected = problem_scores[:num_select]
        
        # Log selection results
        logger.info("🎯 Strategic Selection Results:")
        for i, item in enumerate(selected):
            logger.info(f"  #{i+1}: Score {item['strategic_score']:.3f} "
                       f"(U:{item['uncertainty']:.2f}, D:{item['diversity']:.2f}, Q:{item['quality']:.2f})")
        
        # Update selection history
        self.selection_history.extend([item['problem'] for item in selected])
        
        return [item['problem'] for item in selected]
    
    def _compute_problem_embedding(self, problem: Dict[str, Any]) -> np.ndarray:
        """Compute simple but effective problem embedding for diversity calculation."""
        
        problem_text = problem.get('question', problem.get('description', ''))
        
        # Extract key features for embedding
        features = []
        
        # 1. Text length features
        words = problem_text.lower().split()
        features.extend([
            len(words),
            len(problem_text),
            np.mean([len(word) for word in words]) if words else 0
        ])
        
        # 2. Algorithmic keywords
        algo_keywords = [
            'sort', 'search', 'tree', 'graph', 'dynamic', 'recursive', 
            'optimize', 'minimize', 'maximize', 'count', 'find', 'calculate'
        ]
        algo_features = [1 if kw in problem_text.lower() else 0 for kw in algo_keywords]
        features.extend(algo_features)
        
        # 3. Data structure keywords
        ds_keywords = ['array', 'list', 'string', 'dict', 'set', 'queue', 'stack', 'heap']
        ds_features = [1 if kw in problem_text.lower() else 0 for kw in ds_keywords]
        features.extend(ds_features)
        
        # 4. Problem type indicators
        type_keywords = ['implement', 'design', 'simulate', 'parse', 'validate']
        type_features = [1 if kw in problem_text.lower() else 0 for kw in type_keywords]
        features.extend(type_features)
        
        # 5. Complexity indicators
        complexity_words = ['nested', 'multiple', 'all', 'any', 'optimal', 'efficient']
        complexity_features = [1 if kw in problem_text.lower() else 0 for kw in complexity_words]
        features.extend(complexity_features)
        
        # Normalize embedding
        embedding = np.array(features, dtype=float)
        if embedding.sum() > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def _assess_specification_quality(self, problem_text: str) -> float:
        """Assess problem specification clarity and completeness."""
        
        score = 0.0
        
        # Length appropriateness
        word_count = len(problem_text.split())
        if 20 <= word_count <= 200:
            score += 0.3
        elif 10 <= word_count <= 300:
            score += 0.2
        else:
            score += 0.1
        
        # Clarity indicators
        clarity_indicators = ['input:', 'output:', 'return', 'example', 'constraint']
        clarity_score = sum(1 for indicator in clarity_indicators if indicator in problem_text.lower())
        score += min(0.3, clarity_score * 0.1)
        
        # Specification completeness
        if 'function' in problem_text.lower() or 'implement' in problem_text.lower():
            score += 0.2
        
        # Avoid ambiguous problems
        ambiguous_words = ['maybe', 'might', 'could', 'unclear', 'ambiguous']
        if any(word in problem_text.lower() for word in ambiguous_words):
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _assess_test_quality(self, problem: Dict[str, Any]) -> float:
        """Assess test case coverage and reliability."""
        
        input_output = problem.get('input_output', {})
        
        if not input_output:
            return 0.3  # Low score for no test cases
        
        inputs = input_output.get('inputs', [])
        outputs = input_output.get('outputs', [])
        
        if len(inputs) != len(outputs):
            return 0.2  # Mismatched test cases
        
        test_count = len(inputs)
        
        # Score based on test count
        if test_count >= 5:
            count_score = 0.5
        elif test_count >= 3:
            count_score = 0.4
        elif test_count >= 1:
            count_score = 0.3
        else:
            return 0.1
        
        # Check test variety (simple heuristic)
        variety_score = 0.0
        if test_count > 1:
            # Check if inputs have different lengths/types
            input_lengths = [len(str(inp)) for inp in inputs]
            if len(set(input_lengths)) > 1:
                variety_score += 0.2
            
            # Check if outputs are varied
            output_lengths = [len(str(out)) for out in outputs]
            if len(set(output_lengths)) > 1:
                variety_score += 0.2
        
        return min(1.0, count_score + variety_score)
    
    def _assess_complexity_appropriateness(self, problem_text: str) -> float:
        """Assess if problem complexity is appropriate for learning."""
        
        # Too simple indicators
        simple_indicators = ['print', 'hello', 'add two numbers', 'basic']
        if any(indicator in problem_text.lower() for indicator in simple_indicators):
            return 0.3
        
        # Too complex indicators  
        complex_indicators = ['NP-hard', 'exponential', 'intractable', 'research-level']
        if any(indicator in problem_text.lower() for indicator in complex_indicators):
            return 0.4
        
        # Appropriate complexity indicators
        good_indicators = ['algorithm', 'data structure', 'optimize', 'efficient', 'implement']
        good_score = sum(1 for indicator in good_indicators if indicator in problem_text.lower())
        
        return min(1.0, 0.5 + good_score * 0.1)
    
    def get_selection_analytics(self) -> Dict[str, Any]:
        """Get analytics on strategic selection performance."""
        
        return {
            'total_selections': len(self.selection_history),
            'strategy': self.selection_strategy,
            'weights': {
                'uncertainty': self.uncertainty_weight,
                'diversity': self.diversity_weight,
                'quality': self.quality_weight
            },
            'embedding_cache_size': len(self.problem_embeddings),
            'performance_history': self.performance_history
        }
    
    def update_performance_feedback(
        self,
        selected_problems: List[Dict[str, Any]],
        performance_metrics: Dict[str, float]
    ):
        """Update oracle with performance feedback for adaptive learning."""
        
        feedback = {
            'problems': [p.get('problem_id') for p in selected_problems],
            'metrics': performance_metrics,
            'timestamp': len(self.performance_history)
        }
        
        self.performance_history.append(feedback)
        
        # Adaptive weight adjustment based on performance
        if len(self.performance_history) > 5:
            self._adjust_selection_weights()
    
    def _adjust_selection_weights(self):
        """Adaptively adjust selection weights based on performance history."""
        
        # Simple adaptive strategy: if recent selections underperform,
        # increase uncertainty weight (focus on harder problems)
        recent_performance = self.performance_history[-3:]
        avg_performance = np.mean([p['metrics'].get('success_rate', 0.5) for p in recent_performance])
        
        if avg_performance < 0.6:  # Underperforming
            self.uncertainty_weight = min(0.6, self.uncertainty_weight + 0.05)
            self.diversity_weight = max(0.2, self.diversity_weight - 0.025)
            self.quality_weight = max(0.2, self.quality_weight - 0.025)
            logger.info(f"Adaptive adjustment: Increased uncertainty focus to {self.uncertainty_weight:.2f}")
        elif avg_performance > 0.8:  # Overperforming
            self.diversity_weight = min(0.6, self.diversity_weight + 0.05)
            self.uncertainty_weight = max(0.2, self.uncertainty_weight - 0.025)
            logger.info(f"Adaptive adjustment: Increased diversity focus to {self.diversity_weight:.2f}")