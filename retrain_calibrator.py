#!/usr/bin/env python3
"""
SURGICAL CALIBRATOR RETRAINING
Retrains only the calibrator using enhanced diverse target scores.
Preserves all existing solution generation work - just fixes confidence mapping.
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragv.confidence_calibration import EnhancedConfidenceCalibrator
from sragv.models.solution_generator import SolutionGenerator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retrain_calibrator_with_diversity(
    enhanced_solutions_path: str,
    calibrator_config: Dict[str, Any] = None,
    output_path: str = "checkpoints/phase1_star_calibrator_enhanced.pt"
):
    """
    Retrain ONLY the calibrator using enhanced diverse target scores.
    This preserves all the expensive solution generation work.
    """
    logger.info("🔧 SURGICAL CALIBRATOR RETRAINING - Enhanced Confidence Diversity")
    logger.info("=" * 80)
    
    # Default config
    config = calibrator_config or {
        "num_classes": 1,
        "temperature_schedule": "adaptive", 
        "calibration_method": "temperature_scaling",
        "feature_dim": 16
    }
    
    # Load enhanced solution data
    logger.info(f"📂 Loading enhanced solutions from: {enhanced_solutions_path}")
    with open(enhanced_solutions_path, 'r') as f:
        solutions = json.load(f)
    
    logger.info(f"📊 Loaded {len(solutions)} solutions with enhanced scores")
    
    # Initialize new calibrator
    calibrator = EnhancedConfidenceCalibrator(**config)
    logger.info("✅ Enhanced calibrator initialized")
    
    # Prepare training data with diverse targets
    logger.info("🧮 Preparing diverse calibration training data...")
    
    calibration_data = []
    score_stats = {'enhanced': [], 'original': []}
    
    for i, solution in enumerate(solutions):
        try:
            code = solution.get('code', '')
            problem = solution.get('problem', {})
            
            # Get enhanced score (diverse target)
            enhanced_score = solution.get('enhanced_score', 0.5)
            original_score = solution.get('original_score', 0.8)
            
            # Extract features using calibrator
            features = calibrator.extract_solution_features(code, problem)
            
            # Prepare training sample
            training_sample = {
                'features': features.squeeze(0) if features.dim() > 1 else features,
                'base_confidence': float(original_score),  # Original heuristic score
                'true_score': float(enhanced_score)  # Enhanced diverse target
            }
            
            calibration_data.append(training_sample)
            score_stats['enhanced'].append(enhanced_score)
            score_stats['original'].append(original_score)
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing solution {i}: {e}")
            continue
    
    # Analyze score improvement
    enhanced_scores = np.array(score_stats['enhanced'])
    original_scores = np.array(score_stats['original'])
    
    logger.info("📊 SCORE DIVERSITY ANALYSIS:")
    logger.info(f"  📈 Original Std Dev: {original_scores.std():.4f}")
    logger.info(f"  🎯 Enhanced Std Dev: {enhanced_scores.std():.4f}")
    logger.info(f"  📊 Diversity Improvement: {enhanced_scores.std() / max(original_scores.std(), 0.001):.2f}x")
    logger.info(f"  🎯 Enhanced Range: {enhanced_scores.min():.3f} - {enhanced_scores.max():.3f}")
    
    if enhanced_scores.std() < 0.05:
        logger.error("❌ Insufficient score diversity - calibration may still collapse")
        return None
    
    logger.info(f"✅ Prepared {len(calibration_data)} diverse training samples")
    
    # Split for training/validation
    split_idx = int(0.8 * len(calibration_data))
    train_data = calibration_data[:split_idx]
    val_data = calibration_data[split_idx:]
    
    logger.info(f"📊 Training split: {len(train_data)} train, {len(val_data)} validation")
    
    # Train the calibrator
    logger.info("🚀 Training enhanced calibrator...")
    
    training_metrics = calibrator.train_calibration(
        training_data=train_data,
        validation_data=val_data,
        num_epochs=50,  # More epochs for better diversity learning
        learning_rate=0.01
    )
    
    logger.info("✅ Calibrator training complete!")
    logger.info(f"  🎯 Final ECE: {training_metrics.get('ece', 0):.4f}")
    logger.info(f"  📊 Brier Score: {training_metrics.get('brier_score', 0):.4f}")
    
    # Save enhanced calibrator
    calibrator.save_calibrator(output_path)
    logger.info(f"💾 Enhanced calibrator saved to: {output_path}")
    
    # Validation: Test confidence diversity
    logger.info("🔍 VALIDATING CONFIDENCE DIVERSITY...")
    
    test_confidences = []
    for sample in val_data[:50]:  # Test on subset
        features = sample['features'].unsqueeze(0) if sample['features'].dim() == 1 else sample['features']
        base_conf = sample['base_confidence']
        
        # Get calibrated confidence
        calibrated_conf, _ = calibrator.calibrate_confidence(
            base_confidence=base_conf,
            temperature=1.0,
            features=features
        )
        test_confidences.append(calibrated_conf)
    
    test_conf_array = np.array(test_confidences)
    logger.info("🎯 CALIBRATED CONFIDENCE VALIDATION:")
    logger.info(f"  📊 Calibrated Std Dev: {test_conf_array.std():.4f}")
    logger.info(f"  🎯 Calibrated Range: {test_conf_array.min():.3f} - {test_conf_array.max():.3f}")
    
    if test_conf_array.std() > 0.05:
        logger.info("✅ SUCCESS: Confidence diversity achieved!")
        logger.info("🎉 STAR strategic oracle can now select based on uncertainty!")
    else:
        logger.warning("⚠️ Confidence diversity still low - may need further enhancement")
    
    # Create summary report
    summary = {
        "status": "enhanced_calibrator_training_complete",
        "training_samples": len(train_data),
        "validation_samples": len(val_data),
        "score_diversity": {
            "original_std": float(original_scores.std()),
            "enhanced_std": float(enhanced_scores.std()),
            "improvement_factor": float(enhanced_scores.std() / max(original_scores.std(), 0.001)),
            "enhanced_range": [float(enhanced_scores.min()), float(enhanced_scores.max())]
        },
        "calibration_metrics": training_metrics,
        "confidence_diversity": {
            "calibrated_std": float(test_conf_array.std()),
            "calibrated_range": [float(test_conf_array.min()), float(test_conf_array.max())],
            "diversity_achieved": test_conf_array.std() > 0.05
        },
        "calibrator_path": output_path,
        "ready_for_phase2": test_conf_array.std() > 0.05
    }
    
    # Save summary
    summary_path = "phase1_results/enhanced_calibration_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"📋 Summary saved to: {summary_path}")
    logger.info("=" * 80)
    
    if summary["ready_for_phase2"]:
        logger.info("🎉 PHASE 1 ENHANCED - READY FOR PHASE 2!")
        logger.info("✅ Confidence diversity achieved")
        logger.info("✅ Strategic oracle can now function properly")
        logger.info("✅ STAR architecture ready for self-amplification")
    else:
        logger.warning("⚠️ Phase 2 readiness uncertain - check confidence diversity")
    
    return summary


if __name__ == "__main__":
    # Run the enhanced calibrator retraining
    enhanced_solutions_path = "phase1_results/solution_data_enhanced.json"
    
    if not Path(enhanced_solutions_path).exists():
        logger.error(f"❌ Enhanced solutions not found: {enhanced_solutions_path}")
        logger.info("💡 Run fix_confidence_diversity.py first!")
        sys.exit(1)
    
    summary = retrain_calibrator_with_diversity(enhanced_solutions_path)
    
    if summary and summary["ready_for_phase2"]:
        print("\n" + "=" * 80)
        print("🎉 SURGICAL FIX SUCCESSFUL!")
        print("✅ Confidence diversity restored")
        print("✅ Phase 1 architecturally sound")
        print("✅ Ready to proceed to Phase 2")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠️ SURGICAL FIX NEEDS REVIEW")
        print("❌ Confidence diversity may still be insufficient")
        print("💡 Consider additional enhancement strategies")
        print("=" * 80)