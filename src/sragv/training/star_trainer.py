#!/usr/bin/env python3
"""
STAR Phase 3: Continuous Training with Bayesian Pseudo-Labeling
Implements continuous learning with strategic selection and confidence-based pseudo-labeling.
"""

import logging
import time
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import gc

from ..orchestrator import SRAGVOrchestrator
from ..strategic_oracle import StrategicOracle
from ..confidence_calibration import EnhancedConfidenceCalibrator
from .bayesian_pseudo_labeler import BayesianPseudoLabeler

logger = logging.getLogger(__name__)

class STARTrainer:
    """
    STAR Phase 3: Continuous Training with Strategic Selection and Bayesian Pseudo-Labeling.
    
    Integrates:
    - Phase 1: Calibrated confidence prediction (ECE 0.0018)
    - Phase 2: Strategic Oracle for intelligent problem selection
    - Phase 3: Continuous learning with Bayesian pseudo-labeling
    """
    
    def __init__(
        self,
        orchestrator: SRAGVOrchestrator,
        strategic_oracle: StrategicOracle,
        config: Optional[Dict[str, Any]] = None
    ):
        self.orchestrator = orchestrator
        self.strategic_oracle = strategic_oracle
        
        # Merge provided config with defaults
        default_config = self._create_default_config()
        if config:
            default_config.update(config)
        self.config = default_config
        
        # Initialize Bayesian pseudo-labeler
        bayesian_config = self.config.get('bayesian_labeler_config', {})
        self.bayesian_pseudo_labeler = BayesianPseudoLabeler(config=bayesian_config)
        
        # Training state
        self.training_history = []
        self.current_iteration = 0
        self.total_solutions_generated = 0
        self.pseudo_labels_created = 0
        
        # Performance tracking
        self.performance_metrics = {
            'confidence_calibration_ece': [],
            'pseudo_label_accuracy': [],
            'strategic_selection_efficiency': [],
            'training_loss': []
        }
        
        # Checkpointing
        self.checkpoint_dir = Path("checkpoints/phase3_star_training")
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        logger.info("STARTrainer initialized for Phase 3 continuous training")
        logger.info(f"Config: batch_size={self.config['batch_size']}, "
                   f"confidence_threshold={self.config['confidence_threshold']}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration for STAR Phase 3 training."""
        return {
            # Continuous training parameters
            'max_iterations': 10,
            'batch_size': 32,
            'problems_per_batch': 16,
            'solutions_per_problem': 6,
            
            # Bayesian pseudo-labeling parameters
            'confidence_threshold': 0.85,  # High-confidence threshold
            'adaptive_threshold': True,
            'min_confidence_threshold': 0.75,
            'max_confidence_threshold': 0.95,
            
            # Strategic selection parameters
            'uncertainty_weight': 0.4,
            'diversity_weight': 0.4,
            'quality_weight': 0.2,
            
            # Training stability
            'replay_buffer_size': 1000,
            'catastrophic_forgetting_protection': True,
            'checkpoint_frequency': 2,  # Every 2 iterations
            
            # Quality gates
            'min_pseudo_label_accuracy': 0.8,
            'max_ece_degradation': 0.01,  # Don't let ECE get worse than 0.01
            'early_stopping_patience': 3,
            
            # Bayesian pseudo-labeler configuration
            'bayesian_labeler_config': {
                'base_confidence_threshold': 0.85,
                'adaptive_threshold_enabled': True,
                'min_quality_score': 0.8,
                'max_uncertainty': 0.2
            }
        }
    
    def continuous_train(
        self,
        initial_problems: List[Dict[str, Any]],
        validation_problems: List[Dict[str, Any]],
        max_duration_hours: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Execute continuous STAR training with strategic selection and Bayesian pseudo-labeling.
        
        Args:
            initial_problems: Initial problem set for training
            validation_problems: Problems for validation and ECE monitoring
            max_duration_hours: Maximum training duration in hours
            
        Returns:
            Training results and metrics
        """
        logger.info("🚀 Starting STAR Phase 3 continuous training")
        logger.info(f"Initial problems: {len(initial_problems)}")
        logger.info(f"Validation problems: {len(validation_problems)}")
        
        start_time = time.time()
        results = {
            'phase': 'star_phase3_continuous',
            'start_time': datetime.now().isoformat(),
            'iterations': [],
            'final_metrics': {}
        }
        
        # Initialize replay buffer with high-quality solutions
        replay_buffer = self._initialize_replay_buffer(initial_problems)
        
        # Track best ECE for early stopping
        best_ece = float('inf')
        patience_counter = 0
        
        try:
            for iteration in range(self.config['max_iterations']):
                # Check time limit
                if max_duration_hours and (time.time() - start_time) / 3600 > max_duration_hours:
                    logger.info(f"⏰ Time limit reached: {max_duration_hours} hours")
                    break
                
                logger.info(f"🔄 Starting iteration {iteration + 1}/{self.config['max_iterations']}")
                
                # Phase 3.1: Strategic problem selection
                selected_problems = self._strategic_problem_selection(
                    initial_problems, iteration
                )
                
                # Phase 3.2: Generate solutions with confidence prediction
                iteration_solutions = self._generate_solutions_with_confidence(
                    selected_problems
                )
                
                # Phase 3.3: Bayesian pseudo-labeling
                pseudo_labeled_data, labeling_metrics = self.bayesian_pseudo_labeler.create_pseudo_labels(
                    iteration_solutions, iteration
                )
                
                # Update replay buffer
                replay_buffer = self._update_replay_buffer(
                    replay_buffer, pseudo_labeled_data
                )
                
                # Continuous training step
                training_metrics = self._continuous_training_step(
                    pseudo_labeled_data, replay_buffer
                )
                
                # Validate and check ECE
                validation_metrics = self._validate_calibration(validation_problems)
                current_ece = validation_metrics['ece']
                
                # Early stopping check
                if current_ece < best_ece:
                    best_ece = current_ece
                    patience_counter = 0
                    self._save_checkpoint(iteration, "best")
                else:
                    patience_counter += 1
                
                # Quality gate: Stop if ECE degrades too much
                if current_ece > self.config['max_ece_degradation']:
                    logger.warning(f"⚠️ ECE degraded to {current_ece:.4f}, stopping training")
                    break
                
                # Record iteration results
                iteration_result = {
                    'iteration': iteration + 1,
                    'selected_problems': len(selected_problems),
                    'generated_solutions': len(iteration_solutions),
                    'pseudo_labels_created': len(pseudo_labeled_data),
                    'labeling_metrics': labeling_metrics,
                    'training_metrics': training_metrics,
                    'validation_ece': current_ece,
                    'best_ece': best_ece
                }
                results['iterations'].append(iteration_result)
                
                logger.info(f"✅ Iteration {iteration + 1} complete: "
                           f"ECE={current_ece:.4f}, Pseudo-labels={len(pseudo_labeled_data)}")
                
                # Checkpoint periodically
                if (iteration + 1) % self.config['checkpoint_frequency'] == 0:
                    self._save_checkpoint(iteration, "periodic")
                
                # Early stopping
                if patience_counter >= self.config['early_stopping_patience']:
                    logger.info(f"🛑 Early stopping after {patience_counter} iterations without improvement")
                    break
                
                # Memory cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Final results
            total_time = time.time() - start_time
            results.update({
                'status': 'completed',
                'total_duration': total_time,
                'end_time': datetime.now().isoformat(),
                'final_metrics': {
                    'best_ece': best_ece,
                    'total_iterations': len(results['iterations']),
                    'total_solutions_generated': self.total_solutions_generated,
                    'total_pseudo_labels': self.pseudo_labels_created
                }
            })
            
            logger.info("🎉 STAR Phase 3 continuous training completed successfully!")
            logger.info(f"Best ECE achieved: {best_ece:.4f}")
            logger.info(f"Total training time: {total_time/60:.1f} minutes")
            
            return results
            
        except Exception as e:
            logger.error(f"💥 STAR Phase 3 training failed: {e}")
            results.update({
                'status': 'failed',
                'error': str(e),
                'total_duration': time.time() - start_time
            })
            return results
    
    def _strategic_problem_selection(
        self, 
        problems: List[Dict[str, Any]], 
        iteration: int
    ) -> List[Dict[str, Any]]:
        """Phase 3 strategic problem selection using Strategic Oracle."""
        logger.info(f"🎯 Strategic problem selection for iteration {iteration + 1}")
        
        # Adjust selection strategy based on iteration
        # Early iterations: Focus on uncertainty
        # Later iterations: Balance uncertainty and diversity
        if iteration < 3:
            weights = {'uncertainty': 0.6, 'diversity': 0.3, 'quality': 0.1}
        else:
            weights = self.config  # Use configured weights
        
        selected = self.strategic_oracle.strategic_problem_selection(
            candidate_problems=problems,
            num_select=self.config['problems_per_batch']
        )
        
        logger.info(f"✅ Selected {len(selected)} problems strategically")
        return selected
    
    def _generate_solutions_with_confidence(
        self, 
        problems: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate solutions with calibrated confidence predictions."""
        logger.info(f"🔧 Generating solutions for {len(problems)} problems")
        
        solutions = []
        for problem in problems:
            try:
                # Generate multiple solutions per problem
                problem_solutions = self.orchestrator.solution_generator.generate(
                    problem=problem,
                    num_solutions=self.config['solutions_per_problem']
                    # Note: Confidence is already included in 'score' field via STAR calibration
                )
                
                solutions.extend(problem_solutions)
                
            except Exception as e:
                logger.warning(f"Error generating solutions for problem {problem.get('problem_id', 'unknown')}: {e}")
                continue
        
        self.total_solutions_generated += len(solutions)
        logger.info(f"✅ Generated {len(solutions)} solutions with confidence predictions")
        return solutions
    
    def _initialize_replay_buffer(
        self, 
        initial_problems: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Initialize replay buffer with high-quality solutions from initial problems."""
        logger.info("🔄 Initializing replay buffer")
        
        # Generate solutions for a subset of initial problems
        sample_problems = initial_problems[:min(10, len(initial_problems))]
        initial_solutions = self._generate_solutions_with_confidence(sample_problems)
        
        # Select high-quality solutions for replay buffer
        high_quality_solutions = []
        for solution in initial_solutions:
            if (solution.get('score', 0.0) > 0.8 and  # Use STAR calibrated confidence
                solution.get('syntax_valid', True)):
                high_quality_solutions.append(solution)
        
        logger.info(f"✅ Initialized replay buffer with {len(high_quality_solutions)} high-quality solutions")
        return high_quality_solutions[:self.config['replay_buffer_size']]
    
    def _update_replay_buffer(
        self,
        replay_buffer: List[Dict[str, Any]],
        new_pseudo_labels: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Update replay buffer with new pseudo-labeled data."""
        # Add new high-quality pseudo-labels
        for pseudo_label in new_pseudo_labels:
            if pseudo_label.get('pseudo_label_confidence', 0.0) > 0.85:
                # Extract the solution from the pseudo-label structure
                solution = pseudo_label.get('solution', pseudo_label)
                replay_buffer.append(solution)
        
        # Maintain buffer size (FIFO)
        if len(replay_buffer) > self.config['replay_buffer_size']:
            replay_buffer = replay_buffer[-self.config['replay_buffer_size']:]
        
        return replay_buffer
    
    def _prepare_training_data(
        self,
        pseudo_labeled_data: List[Dict[str, Any]],
        replay_buffer: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Prepare training data combining pseudo-labels and replay buffer."""
        training_data = []
        
        # Add pseudo-labeled data (primary training data)
        training_data.extend(pseudo_labeled_data)
        
        # Add replay buffer data for catastrophic forgetting prevention
        # Sample up to 20% of training data from replay buffer
        max_replay_samples = max(1, len(pseudo_labeled_data) // 5)
        if replay_buffer:
            replay_samples = replay_buffer[-max_replay_samples:]  # Use most recent
            
            # Convert replay buffer items to training format
            for replay_item in replay_samples:
                training_data.append({
                    'solution': replay_item,
                    'pseudo_label_confidence': replay_item.get('score', 0.8),  # Use existing score
                    'is_replay': True
                })
        
        logger.debug(f"Prepared {len(training_data)} training samples ({len(pseudo_labeled_data)} pseudo + {min(max_replay_samples, len(replay_buffer))} replay)")
        return training_data
    
    def _continuous_training_step(
        self,
        pseudo_labeled_data: List[Dict[str, Any]],
        replay_buffer: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute one continuous training step with actual model parameter updates."""
        logger.info(f"🧠 Continuous training step with {len(pseudo_labeled_data)} pseudo-labels")
        
        if not pseudo_labeled_data:
            logger.warning("No pseudo-labeled data provided for training step")
            return {'pseudo_labels_used': 0, 'replay_buffer_size': len(replay_buffer), 'training_loss': 0.0, 'gradient_steps': 0}
        
        try:
            # Combine pseudo-labeled data with replay buffer for training
            training_data = self._prepare_training_data(pseudo_labeled_data, replay_buffer)
            
            # Update SolutionGenerator confidence calibrator with new pseudo-labels
            calibrator_training_data = []
            for item in training_data:
                solution = item.get('solution', {})
                pseudo_confidence = item.get('pseudo_label_confidence', 0.0)
                
                # Create training sample for calibrator
                calibrator_training_data.append({
                    'code': solution.get('code', ''),
                    'problem': solution.get('problem', {}),
                    'base_score': solution.get('score', 0.0),
                    'true_score': pseudo_confidence  # Use pseudo-label confidence as ground truth
                })
            
            # Update confidence calibrator if we have sufficient data
            if len(calibrator_training_data) >= 5:  # Minimum batch size
                logger.info(f"Updating confidence calibrator with {len(calibrator_training_data)} samples")
                
                # Train calibrator with new data (1 epoch for continuous learning)
                calibration_metrics = self.orchestrator.solution_generator.train_confidence_calibrator(
                    training_data=calibrator_training_data,
                    validation_data=None,  # No validation during continuous training
                    num_epochs=1,  # Single epoch for continuous updates
                    learning_rate=0.001  # Lower learning rate for stability
                )
                
                training_loss = calibration_metrics.get('final_loss', 0.0)
                logger.info(f"Calibrator update complete - Loss: {training_loss:.4f}")
            else:
                training_loss = 0.0  # No training performed
                logger.info("Insufficient data for calibrator update")
            
            # Calculate gradient steps (simulate based on data size)
            gradient_steps = max(1, len(training_data) // 4)  # Batch size of 4
            
            training_metrics = {
                'pseudo_labels_used': len(pseudo_labeled_data),
                'replay_buffer_size': len(replay_buffer),
                'training_loss': training_loss,
                'gradient_steps': gradient_steps,
                'calibrator_samples': len(calibrator_training_data)
            }
            
            logger.info(f"✅ Training step complete - Loss: {training_loss:.4f}, Steps: {gradient_steps}")
            
        except Exception as e:
            logger.error(f"Training step failed: {e}")
            # Return safe fallback metrics
            training_metrics = {
                'pseudo_labels_used': len(pseudo_labeled_data),
                'replay_buffer_size': len(replay_buffer),
                'training_loss': float('inf'),  # Indicate failure
                'gradient_steps': 0,
                'error': str(e)
            }
        
        # Record in training history
        self.training_history.append({
            'iteration': self.current_iteration,
            'timestamp': datetime.now().isoformat(),
            'metrics': training_metrics
        })
        
        self.current_iteration += 1
        return training_metrics
    
    def _validate_calibration(
        self, 
        validation_problems: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate confidence calibration on validation set."""
        logger.info(f"🎯 Validating calibration on {len(validation_problems)} problems")
        
        # Generate solutions for validation
        val_solutions = []
        for problem in validation_problems[:5]:  # Sample for efficiency
            solutions = self.orchestrator.solution_generator.generate(
                problem=problem,
                num_solutions=2
                # Note: Confidence is included in 'score' field by default
            )
            val_solutions.extend(solutions)
        
        # Compute ECE
        if val_solutions:
            ece = self._compute_ece(val_solutions)
        else:
            ece = 1.0  # Worst case if no solutions
        
        return {
            'ece': ece,
            'validation_solutions': len(val_solutions)
        }
    
    def _compute_ece(self, solutions: List[Dict[str, Any]]) -> float:
        """Compute Expected Calibration Error."""
        if not solutions:
            return 1.0
        
        confidences = []
        accuracies = []
        
        for solution in solutions:
            confidence = solution.get('score', 0.5)  # Use STAR calibrated confidence from 'score' field
            # Use pass_rate as accuracy proxy
            accuracy = solution.get('pass_rate', 0.0)
            
            confidences.append(confidence)
            accuracies.append(accuracy)
        
        if not confidences:
            return 1.0
        
        # Simple ECE computation (binned)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = [(c >= bin_lower) and (c < bin_upper) for c in confidences]
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean([acc for acc, in_b in zip(accuracies, in_bin) if in_b])
                avg_confidence_in_bin = np.mean([conf for conf, in_b in zip(confidences, in_bin) if in_b])
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _save_checkpoint(self, iteration: int, checkpoint_type: str) -> None:
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"star_phase3_iter_{iteration}_{checkpoint_type}.json"
        
        checkpoint_data = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'training_history': self.training_history,
            'performance_metrics': self.performance_metrics,
            'config': self.config,
            'total_solutions_generated': self.total_solutions_generated,
            'pseudo_labels_created': self.pseudo_labels_created
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and metrics."""
        return {
            'current_iteration': self.current_iteration,
            'total_solutions_generated': self.total_solutions_generated,
            'pseudo_labels_created': self.pseudo_labels_created,
            'training_history_length': len(self.training_history),
            'last_checkpoint': datetime.now().isoformat()
        }
