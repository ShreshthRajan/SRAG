#!/usr/bin/env python3
"""
STAR Phase 3.2: Bayesian Pseudo-Labeling with Uncertainty Quantification
Implements state-of-the-art Bayesian pseudo-labeling with adaptive thresholds.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class BayesianPseudoLabeler:
    """
    Advanced Bayesian pseudo-labeling with uncertainty quantification and adaptive thresholds.
    
    Based on current SOTA (August 2025) research in:
    - Bayesian deep learning for uncertainty quantification
    - Adaptive thresholding for pseudo-labeling
    - Quality-aware pseudo-label generation
    - Confidence calibration integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = self._create_default_config()
        if config:
            # Merge provided config with defaults
            default_config.update(config)
        self.config = default_config
        
        # Bayesian parameters
        self.confidence_history = []
        self.accuracy_history = []
        self.threshold_history = []
        
        # Quality tracking
        self.pseudo_label_quality_tracker = defaultdict(list)
        self.uncertainty_estimates = []
        
        logger.info("BayesianPseudoLabeler initialized with adaptive thresholding")
        logger.info(f"Base confidence threshold: {self.config['base_confidence_threshold']}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration for Bayesian pseudo-labeling."""
        return {
            # Core thresholds
            'base_confidence_threshold': 0.85,
            'min_confidence_threshold': 0.75,
            'max_confidence_threshold': 0.95,
            
            # Bayesian parameters
            'uncertainty_weight': 0.3,
            'quality_weight': 0.4,
            'confidence_weight': 0.3,
            
            # Adaptive thresholding
            'adaptive_threshold_enabled': True,
            'threshold_adaptation_rate': 0.1,
            'min_samples_for_adaptation': 50,
            
            # Quality gates
            'min_quality_score': 0.8,
            'max_uncertainty': 0.2,
            'syntax_validity_required': True,
            
            # Bayesian inference
            'prior_alpha': 2.0,  # Beta distribution prior
            'prior_beta': 2.0,
            'credible_interval': 0.95,
            
            # Performance tracking
            'track_quality_metrics': True,
            'quality_window_size': 100
        }
    
    def create_pseudo_labels(
        self,
        solutions: List[Dict[str, Any]],
        iteration: int = 0
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        Create high-quality pseudo-labels using Bayesian approach with uncertainty quantification.
        
        Args:
            solutions: List of solutions with confidence predictions
            iteration: Current training iteration for threshold adaptation
            
        Returns:
            Tuple of (pseudo_labeled_solutions, labeling_metrics)
        """
        logger.info(f"🧠 Creating Bayesian pseudo-labels for {len(solutions)} solutions")
        
        if not solutions:
            return [], {'pseudo_label_rate': 0.0, 'avg_confidence': 0.0}
        
        # Step 1: Compute adaptive threshold
        adaptive_threshold = self._compute_adaptive_threshold(solutions, iteration)
        
        # Step 2: Bayesian quality assessment
        quality_assessments = self._batch_quality_assessment(solutions)
        
        # Step 3: Uncertainty quantification
        uncertainty_estimates = self._quantify_uncertainties(solutions)
        
        # Step 4: Bayesian pseudo-label selection
        pseudo_labels = []
        confidence_scores = []
        quality_scores = []
        
        for i, solution in enumerate(solutions):
            confidence = solution.get('score', 0.0)  # Use STAR calibrated confidence from 'score' field
            quality = quality_assessments[i]
            uncertainty = uncertainty_estimates[i]
            
            confidence_scores.append(confidence)
            quality_scores.append(quality)
            
            # Bayesian selection criteria
            if self._meets_bayesian_criteria(
                confidence, quality, uncertainty, adaptive_threshold
            ):
                pseudo_label = self._create_pseudo_label(
                    solution, confidence, quality, uncertainty, iteration
                )
                pseudo_labels.append(pseudo_label)
        
        # Step 5: Update tracking and metrics
        labeling_metrics = self._update_tracking_metrics(
            confidence_scores, quality_scores, uncertainty_estimates,
            len(pseudo_labels), adaptive_threshold
        )
        
        logger.info(f"✅ Created {len(pseudo_labels)} pseudo-labels "
                   f"(rate: {labeling_metrics['pseudo_label_rate']:.3f})")
        logger.info(f"Adaptive threshold: {adaptive_threshold:.3f}, "
                   f"Avg quality: {labeling_metrics['avg_quality']:.3f}")
        
        return pseudo_labels, labeling_metrics
    
    def _compute_adaptive_threshold(
        self, 
        solutions: List[Dict[str, Any]], 
        iteration: int
    ) -> float:
        """Compute adaptive confidence threshold using Bayesian approach."""
        if not self.config['adaptive_threshold_enabled']:
            return self.config['base_confidence_threshold']
        
        confidences = [s.get('score', 0.0) for s in solutions]  # Use STAR calibrated confidence
        
        if len(self.confidence_history) < self.config['min_samples_for_adaptation']:
            # Not enough data for adaptation, use base threshold
            return self.config['base_confidence_threshold']
        
        # Bayesian threshold adaptation based on historical performance
        historical_confidences = np.array(self.confidence_history[-self.config['quality_window_size']:])
        historical_accuracies = np.array(self.accuracy_history[-self.config['quality_window_size']:])
        
        if len(historical_confidences) == 0:
            return self.config['base_confidence_threshold']
        
        # Find optimal threshold using Bayesian optimization
        # Use the threshold that maximizes expected pseudo-label quality
        candidate_thresholds = np.linspace(
            self.config['min_confidence_threshold'],
            self.config['max_confidence_threshold'],
            20
        )
        
        best_threshold = self.config['base_confidence_threshold']
        best_expected_quality = 0.0
        
        for threshold in candidate_thresholds:
            # Estimate expected quality for this threshold
            mask = historical_confidences >= threshold
            if np.sum(mask) > 5:  # Need minimum samples
                expected_quality = np.mean(historical_accuracies[mask])
                
                # Bayesian adjustment with uncertainty
                n_samples = np.sum(mask)
                uncertainty_penalty = 1.0 / np.sqrt(n_samples)  # Higher penalty for fewer samples
                adjusted_quality = expected_quality - uncertainty_penalty
                
                if adjusted_quality > best_expected_quality:
                    best_expected_quality = adjusted_quality
                    best_threshold = threshold
        
        # Smooth adaptation to prevent oscillation
        if self.threshold_history:
            last_threshold = self.threshold_history[-1]
            adaptation_rate = self.config['threshold_adaptation_rate']
            adapted_threshold = (1 - adaptation_rate) * last_threshold + adaptation_rate * best_threshold
        else:
            adapted_threshold = best_threshold
        
        # Ensure bounds
        adapted_threshold = np.clip(
            adapted_threshold,
            self.config['min_confidence_threshold'],
            self.config['max_confidence_threshold']
        )
        
        self.threshold_history.append(adapted_threshold)
        return adapted_threshold
    
    def _batch_quality_assessment(self, solutions: List[Dict[str, Any]]) -> List[float]:
        """Batch quality assessment using multiple quality indicators."""
        quality_scores = []
        
        for solution in solutions:
            quality_factors = []
            
            # Syntactic correctness (high weight)
            if solution.get('syntax_valid', True):
                quality_factors.append(('syntax', 0.95, 0.3))
            else:
                quality_factors.append(('syntax', 0.1, 0.3))
            
            # Execution success
            if solution.get('execution_success', False):
                quality_factors.append(('execution', 0.9, 0.25))
            else:
                quality_factors.append(('execution', 0.3, 0.25))
            
            # Test pass rate
            pass_rate = solution.get('pass_rate', 0.0)
            quality_factors.append(('pass_rate', pass_rate, 0.25))
            
            # Code quality heuristics
            code_quality = solution.get('code_quality_score', 0.5)
            quality_factors.append(('code_quality', code_quality, 0.2))
            
            # Bayesian quality combination with uncertainty
            total_quality = 0.0
            total_weight = 0.0
            
            for factor_name, score, weight in quality_factors:
                # Add uncertainty based on factor reliability
                if factor_name == 'syntax':
                    uncertainty = 0.05  # High confidence in syntax checking
                elif factor_name == 'execution':
                    uncertainty = 0.1   # Medium confidence
                else:
                    uncertainty = 0.15  # Lower confidence in heuristics
                
                # Bayesian adjustment
                adjusted_score = score * (1 - uncertainty)
                total_quality += adjusted_score * weight
                total_weight += weight
            
            final_quality = total_quality / total_weight if total_weight > 0 else 0.5
            quality_scores.append(final_quality)
        
        return quality_scores
    
    def _quantify_uncertainties(self, solutions: List[Dict[str, Any]]) -> List[float]:
        """Quantify epistemic and aleatoric uncertainty for each solution."""
        uncertainties = []
        
        for solution in solutions:
            uncertainty_factors = []
            
            # Confidence-based uncertainty (epistemic)
            confidence = solution.get('score', 0.5)  # Use STAR calibrated confidence
            confidence_uncertainty = 1.0 - confidence
            uncertainty_factors.append(confidence_uncertainty)
            
            # Prediction variance (aleatoric) - if available from multiple samples
            prediction_variance = solution.get('prediction_variance', 0.1)
            uncertainty_factors.append(prediction_variance)
            
            # Code complexity uncertainty
            code_length = len(solution.get('code', ''))
            complexity_uncertainty = min(0.3, code_length / 1000)  # Higher uncertainty for longer code
            uncertainty_factors.append(complexity_uncertainty)
            
            # Test coverage uncertainty
            test_coverage = solution.get('test_coverage', 0.5)
            coverage_uncertainty = 1.0 - test_coverage
            uncertainty_factors.append(coverage_uncertainty * 0.5)  # Moderate weight
            
            # Combined uncertainty (geometric mean for conservative estimate)
            if uncertainty_factors:
                total_uncertainty = np.prod(uncertainty_factors) ** (1.0 / len(uncertainty_factors))
            else:
                total_uncertainty = 0.5
            
            uncertainties.append(total_uncertainty)
        
        return uncertainties
    
    def _meets_bayesian_criteria(
        self,
        confidence: float,
        quality: float,
        uncertainty: float,
        threshold: float
    ) -> bool:
        """Check if solution meets Bayesian pseudo-labeling criteria."""
        # Basic confidence threshold
        if confidence < threshold:
            return False
        
        # Quality threshold
        if quality < self.config['min_quality_score']:
            return False
        
        # Uncertainty threshold
        if uncertainty > self.config['max_uncertainty']:
            return False
        
        # Bayesian combined score
        bayesian_score = (
            confidence * self.config['confidence_weight'] +
            quality * self.config['quality_weight'] +
            (1 - uncertainty) * self.config['uncertainty_weight']
        )
        
        # Use Bayesian credible interval for final decision
        # Assume Beta distribution for the score
        alpha = self.config['prior_alpha'] + bayesian_score * 10
        beta = self.config['prior_beta'] + (1 - bayesian_score) * 10
        
        # Check if lower bound of credible interval is above minimum threshold
        credible_lower = stats.beta.ppf(
            (1 - self.config['credible_interval']) / 2, alpha, beta
        )
        
        return credible_lower > 0.5  # Reasonable threshold for credible interval
    
    def _create_pseudo_label(
        self,
        solution: Dict[str, Any],
        confidence: float,
        quality: float,
        uncertainty: float,
        iteration: int
    ) -> Dict[str, Any]:
        """Create a pseudo-label with comprehensive metadata."""
        # Compute final pseudo-label confidence
        pseudo_confidence = self._compute_pseudo_label_confidence(
            confidence, quality, uncertainty
        )
        
        pseudo_label = {
            'solution': solution,
            'original_confidence': confidence,
            'quality_score': quality,
            'uncertainty': uncertainty,
            'pseudo_label_confidence': pseudo_confidence,
            
            # Bayesian metadata
            'bayesian_score': (
                confidence * self.config['confidence_weight'] +
                quality * self.config['quality_weight'] +
                (1 - uncertainty) * self.config['uncertainty_weight']
            ),
            
            # Tracking metadata
            'creation_iteration': iteration,
            'creation_time': datetime.now().isoformat(),
            'labeler_version': 'bayesian_v1.0',
            
            # Quality indicators
            'syntax_valid': solution.get('syntax_valid', True),
            'execution_success': solution.get('execution_success', False),
            'pass_rate': solution.get('pass_rate', 0.0)
        }
        
        return pseudo_label
    
    def _compute_pseudo_label_confidence(
        self,
        confidence: float,
        quality: float,
        uncertainty: float
    ) -> float:
        """Compute final pseudo-label confidence using Bayesian combination."""
        # Bayesian combination with uncertainty discounting
        base_confidence = confidence * quality
        uncertainty_discount = 1.0 - uncertainty
        
        # Apply Bayesian updating
        pseudo_confidence = base_confidence * uncertainty_discount
        
        # Ensure reasonable bounds
        return np.clip(pseudo_confidence, 0.1, 0.99)
    
    def _update_tracking_metrics(
        self,
        confidences: List[float],
        qualities: List[float],
        uncertainties: List[float],
        num_pseudo_labels: int,
        threshold: float
    ) -> Dict[str, float]:
        """Update tracking metrics and return labeling statistics."""
        # Update history for adaptive thresholding
        self.confidence_history.extend(confidences)
        
        # Estimate accuracies based on qualities (proxy)
        estimated_accuracies = [min(1.0, q * 1.2) for q in qualities]  # Conservative estimate
        self.accuracy_history.extend(estimated_accuracies)
        
        # Maintain window size
        window_size = self.config['quality_window_size']
        if len(self.confidence_history) > window_size:
            self.confidence_history = self.confidence_history[-window_size:]
            self.accuracy_history = self.accuracy_history[-window_size:]
        
        # Compute metrics
        metrics = {
            'pseudo_label_rate': num_pseudo_labels / len(confidences) if confidences else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'avg_quality': np.mean(qualities) if qualities else 0.0,
            'avg_uncertainty': np.mean(uncertainties) if uncertainties else 0.0,
            'adaptive_threshold': threshold,
            'total_solutions_processed': len(confidences),
            'high_confidence_solutions': sum(1 for c in confidences if c > 0.9),
            'quality_std': np.std(qualities) if qualities else 0.0
        }
        
        return metrics
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status and statistics."""
        return {
            'threshold_history': self.threshold_history[-10:],  # Last 10 thresholds
            'current_threshold': self.threshold_history[-1] if self.threshold_history else None,
            'confidence_history_size': len(self.confidence_history),
            'accuracy_history_size': len(self.accuracy_history),
            'avg_recent_accuracy': np.mean(self.accuracy_history[-20:]) if len(self.accuracy_history) >= 20 else None,
            'adaptation_enabled': self.config['adaptive_threshold_enabled']
        }
    
    def reset_adaptation(self) -> None:
        """Reset adaptation history (useful for new training phases)."""
        self.confidence_history.clear()
        self.accuracy_history.clear()
        self.threshold_history.clear()
        self.uncertainty_estimates.clear()
        
        logger.info("🔄 Bayesian adaptation history reset")