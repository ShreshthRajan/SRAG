"""
Enhanced Confidence Calibration for STAR Training
Temperature scaling and reliability calibration for solution quality scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EnhancedConfidenceCalibrator(nn.Module):
    """
    Enhanced confidence calibrator with temperature scaling for STAR training.
    
    Provides calibrated confidence scores for solution quality assessment
    with temperature-aware reliability and comprehensive calibration metrics.
    """
    
    def __init__(
        self,
        num_classes: int = 1,
        temperature_schedule: str = "adaptive",
        calibration_method: str = "temperature_scaling",
        feature_dim: int = 16,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature_schedule = temperature_schedule
        self.calibration_method = calibration_method
        self.feature_dim = feature_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Temperature scaling parameter (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # Enhanced calibration network for complex confidence mapping
        if calibration_method == "neural_calibration":
            self.calibration_network = nn.Sequential(
                nn.Linear(feature_dim + 1, 32),  # features + base confidence
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:
            self.calibration_network = None
        
        # Calibration history for adaptive learning
        self.calibration_history = []
        self.is_trained = False
        
        # Reliability tracking
        self.reliability_bins = 15
        self.min_bin_count = 10
        
        self.to(self.device)
        logger.info(f"EnhancedConfidenceCalibrator initialized with {calibration_method}")
    
    def extract_solution_features(self, code: str, problem: Dict[str, Any]) -> torch.Tensor:
        """Extract feature vector from solution for calibration."""
        features = []
        
        # Code structure features
        features.append(len(code.split('\n')))  # Line count
        features.append(len(code.split()))  # Token count
        features.append(code.count('def '))  # Function definitions
        features.append(code.count('return'))  # Return statements
        features.append(code.count('if '))  # Conditional statements
        features.append(code.count('for ') + code.count('while '))  # Loops
        features.append(code.count('import ') + code.count('from '))  # Imports
        
        # Code quality indicators
        try:
            compile(code, '<solution>', 'exec')
            features.append(1.0)  # Syntactically valid
        except:
            features.append(0.0)  # Syntax error
        
        # Problem-specific features
        problem_text = problem.get('question', '')
        features.append(len(problem_text.split()))  # Problem complexity
        
        # Code-problem relevance (simple keyword matching)
        problem_keywords = set(problem_text.lower().split())
        code_keywords = set(code.lower().split())
        relevance = len(problem_keywords & code_keywords) / max(len(problem_keywords), 1)
        features.append(relevance)
        
        # Diversity features (placeholder - would use embeddings in full version)
        features.extend([0.5] * 6)  # Pad to feature_dim
        
        # Convert to tensor and normalize
        feature_tensor = torch.tensor(features[:self.feature_dim], dtype=torch.float32, device=self.device)
        return F.normalize(feature_tensor.unsqueeze(0), dim=1)
    
    def calibrate_confidence(
        self,
        logits: Optional[torch.Tensor] = None,
        base_confidence: float = 0.5,
        temperature: float = 1.0,
        features: Optional[torch.Tensor] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply confidence calibration with temperature scaling.
        
        Returns:
            Tuple of (calibrated_confidence, calibration_info)
        """
        with torch.no_grad():
            # Convert base confidence to tensor
            base_conf_tensor = torch.tensor([base_confidence], device=self.device)
            
            if self.calibration_method == "temperature_scaling":
                # Apply temperature scaling
                if self.is_trained:
                    scaled_logits = base_conf_tensor / self.temperature
                    calibrated_conf = torch.sigmoid(scaled_logits).item()
                else:
                    # Use base confidence with temperature adjustment if not trained
                    calibrated_conf = base_confidence / temperature if temperature > 0 else base_confidence
                
            elif self.calibration_method == "neural_calibration" and self.calibration_network is not None:
                # Use neural calibration network
                if features is not None:
                    network_input = torch.cat([features.flatten(), base_conf_tensor])
                    calibrated_conf = self.calibration_network(network_input.unsqueeze(0)).item()
                else:
                    calibrated_conf = base_confidence
            else:
                calibrated_conf = base_confidence
            
            # Compute confidence interval (simple approximation)
            confidence_std = 0.1 * (1 - calibrated_conf) * calibrated_conf  # Beta distribution approximation
            confidence_interval = {
                'lower': max(0.0, calibrated_conf - 1.96 * confidence_std),
                'upper': min(1.0, calibrated_conf + 1.96 * confidence_std)
            }
            
            calibration_info = {
                'base_confidence': base_confidence,
                'calibrated_confidence': calibrated_conf,
                'temperature': self.temperature.item() if self.is_trained else temperature,
                'confidence_std': confidence_std,
                'confidence_interval': confidence_interval,
                'calibration_method': self.calibration_method
            }
            
            return calibrated_conf, calibration_info
    
    def train_calibration(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        num_epochs: int = 50,
        learning_rate: float = 0.01,
        patience: int = 10
    ) -> Dict[str, float]:
        """
        Train the calibration parameters using collected data.
        
        Args:
            training_data: List of {features, base_confidence, true_score}
            validation_data: Optional validation set
            num_epochs: Training epochs
            learning_rate: Learning rate
            patience: Early stopping patience
        
        Returns:
            Training metrics dictionary
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience//2)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Prepare training data
        train_features = []
        train_base_conf = []
        train_true_scores = []
        
        for item in training_data:
            train_features.append(item['features'])
            train_base_conf.append(item['base_confidence'])
            train_true_scores.append(item['true_score'])
        
        train_features = torch.stack(train_features).to(self.device)
        train_base_conf = torch.tensor(train_base_conf, device=self.device)
        train_true_scores = torch.tensor(train_true_scores, device=self.device)
        
        logger.info(f"Training calibrator on {len(training_data)} samples...")
        
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            if self.calibration_method == "temperature_scaling":
                # Temperature scaling loss
                scaled_logits = train_base_conf / self.temperature
                calibrated_probs = torch.sigmoid(scaled_logits)
                
                # Use MSE loss for regression or BCE for classification
                loss = F.mse_loss(calibrated_probs, train_true_scores)
                
            elif self.calibration_method == "neural_calibration":
                # Neural calibration loss
                network_inputs = torch.cat([train_features, train_base_conf.unsqueeze(1)], dim=1)
                calibrated_probs = self.calibration_network(network_inputs).squeeze()
                loss = F.mse_loss(calibrated_probs, train_true_scores)
            
            loss.backward()
            optimizer.step()
            
            # Validation
            if validation_data and epoch % 5 == 0:
                val_loss = self._validate_calibration(validation_data)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        self.is_trained = True
        self.eval()
        
        # Final evaluation
        final_metrics = self.evaluate_calibration(training_data)
        logger.info(f"Calibration training complete. Final ECE: {final_metrics.get('ece', 0):.4f}")
        
        return final_metrics
    
    def _validate_calibration(self, validation_data: List[Dict[str, Any]]) -> float:
        """Validate calibration on held-out data."""
        self.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for item in validation_data:
                features = item['features'].to(self.device)
                base_conf = item['base_confidence']
                true_score = item['true_score']
                
                calibrated_conf, _ = self.calibrate_confidence(
                    base_confidence=base_conf,
                    features=features
                )
                
                val_loss += (calibrated_conf - true_score) ** 2
        
        return val_loss / len(validation_data)
    
    def evaluate_calibration(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate calibration quality using various metrics.
        
        Returns:
            Dictionary with ECE, MCE, Brier score, and reliability diagram data
        """
        self.eval()
        calibrated_confidences = []
        true_scores = []
        
        with torch.no_grad():
            for item in evaluation_data:
                features = item.get('features')
                if features is not None:
                    features = features.to(self.device)
                
                calibrated_conf, _ = self.calibrate_confidence(
                    base_confidence=item['base_confidence'],
                    features=features
                )
                
                calibrated_confidences.append(calibrated_conf)
                true_scores.append(item['true_score'])
        
        # Convert to numpy for sklearn metrics
        confidences = np.array(calibrated_confidences)
        true_labels = np.array(true_scores)
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(confidences, true_labels)
        
        # Maximum Calibration Error (MCE)
        mce = self._compute_mce(confidences, true_labels)
        
        # Brier Score - fix for continuous targets
        try:
            # Convert continuous scores to binary for brier calculation
            binary_labels = (np.array(true_labels) > 0.5).astype(int)
            brier_score = brier_score_loss(binary_labels, confidences)
        except Exception as e:
            logger.warning(f"Brier score calculation failed: {e}, using MSE approximation")
            brier_score = np.mean((np.array(true_labels) - np.array(confidences)) ** 2)
        
        # Reliability diagram data
        bin_boundaries, bin_lowers, bin_uppers, bin_accuracies, bin_confidences, bin_counts = \
            self._reliability_diagram_data(confidences, true_labels)
        
        metrics = {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'temperature': self.temperature.item() if self.is_trained else 1.0,
            'num_samples': len(evaluation_data),
            'reliability_data': {
                'bin_boundaries': bin_boundaries.tolist(),
                'bin_accuracies': bin_accuracies.tolist(),
                'bin_confidences': bin_confidences.tolist(),
                'bin_counts': bin_counts.tolist()
            }
        }
        
        return metrics
    
    def _compute_ece(self, confidences: np.ndarray, true_labels: np.ndarray) -> float:
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.reliability_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_mce(self, confidences: np.ndarray, true_labels: np.ndarray) -> float:
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.reliability_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _reliability_diagram_data(self, confidences: np.ndarray, true_labels: np.ndarray):
        """Generate reliability diagram data."""
        bin_boundaries = np.linspace(0, 1, self.reliability_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.sum()
            
            if prop_in_bin > self.min_bin_count:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
            else:
                accuracy_in_bin = 0.0
                avg_confidence_in_bin = (bin_lower + bin_upper) / 2
            
            bin_accuracies.append(accuracy_in_bin)
            bin_confidences.append(avg_confidence_in_bin)
            bin_counts.append(int(prop_in_bin))
        
        return (bin_boundaries,
                np.array(bin_lowers),
                np.array(bin_uppers),
                np.array(bin_accuracies),
                np.array(bin_confidences),
                np.array(bin_counts))
    
    def save_calibrator(self, path: str):
        """Save calibrator state."""
        calibrator_state = {
            'state_dict': self.state_dict(),
            'config': {
                'num_classes': self.num_classes,
                'temperature_schedule': self.temperature_schedule,
                'calibration_method': self.calibration_method,
                'feature_dim': self.feature_dim
            },
            'is_trained': self.is_trained,
            'calibration_history': self.calibration_history
        }
        
        torch.save(calibrator_state, path)
        logger.info(f"Calibrator saved to {path}")
    
    def load_calibrator(self, path: str):
        """Load calibrator state."""
        calibrator_state = torch.load(path, map_location=self.device)
        self.load_state_dict(calibrator_state['state_dict'])
        
        self.is_trained = calibrator_state.get('is_trained', False)
        self.calibration_history = calibrator_state.get('calibration_history', [])
        
        logger.info(f"Calibrator loaded from {path}")
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration state."""
        return {
            'is_trained': self.is_trained,
            'temperature': self.temperature.item() if self.is_trained else 1.0,
            'calibration_method': self.calibration_method,
            'feature_dim': self.feature_dim,
            'device': str(self.device),
            'num_history_points': len(self.calibration_history)
        }