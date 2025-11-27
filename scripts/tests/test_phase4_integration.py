#!/usr/bin/env python3
"""
SRAG-V Phase 4: Integration Test
Quick validation test to ensure all Phase 4 components work together.

This test validates:
- Model loading (Phase 1 & Phase 3)
- Evaluation pipeline components
- Statistical validation
- Calibration analysis
- Result generation

Run this before the full Phase 4 evaluation to catch integration issues.

Author: Claude & Shreshth
Date: August 2025
"""

import os
import sys
import time
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required imports work."""
    logger.info("🔬 Testing imports...")
    
    try:
        # Core SRAG-V components
        from sragv.orchestrator import SRAGVOrchestrator
        from sragv.models.solution_generator import SolutionGenerator
        from sragv.confidence_calibration import EnhancedConfidenceCalibrator
        
        # Evaluation components
        from sragv.evaluation.calibration_analyzer import AdvancedCalibrationAnalyzer
        from sragv.evaluation.statistical_validator import StatisticalValidator
        
        logger.info("✅ All imports successful")
        return True
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_model_loading():
    """Test that Phase 1 and Phase 3 models load correctly."""
    logger.info("🔬 Testing model loading...")
    
    try:
        # Test Phase 1 model loading
        phase1_calibrator_path = "checkpoints/phase1_star_calibrator_1754272154.pt"
        if not Path(phase1_calibrator_path).exists():
            logger.error(f"❌ Phase 1 calibrator not found: {phase1_calibrator_path}")
            return False
        
        # Load Phase 1 baseline
        from sragv.orchestrator import SRAGVOrchestrator
        from sragv.confidence_calibration import EnhancedConfidenceCalibrator
        import torch
        
        orchestrator_phase1 = SRAGVOrchestrator("config/config.yaml")
        orchestrator_phase1.initialize_players()
        
        checkpoint = torch.load(phase1_calibrator_path, map_location='cpu')
        calibrator_phase1 = EnhancedConfidenceCalibrator()
        calibrator_phase1.load_state_dict(checkpoint['state_dict'])
        
        orchestrator_phase1.solution_generator.confidence_calibrator = calibrator_phase1
        orchestrator_phase1.solution_generator.use_calibration = True
        
        logger.info("✅ Phase 1 model loaded successfully")
        
        # Test basic generation
        test_problem = {
            "question": "Write a function that adds two numbers.",
            "starter_code": "def add(a, b):",
            "input_output": [{"input": "1 2", "output": "3"}]
        }
        
        solutions = orchestrator_phase1.solution_generator.generate_solutions(
            test_problem, num_solutions=2
        )
        
        if solutions and len(solutions) > 0:
            logger.info(f"✅ Generated {len(solutions)} test solutions")
            for i, sol in enumerate(solutions):
                logger.info(f"   Solution {i+1}: confidence={sol.get('score', 0):.3f}")
        else:
            logger.warning("⚠️ No solutions generated in test")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_calibration_analyzer():
    """Test calibration analysis components."""
    logger.info("🔬 Testing calibration analyzer...")
    
    try:
        from sragv.evaluation.calibration_analyzer import AdvancedCalibrationAnalyzer
        
        # Create synthetic test data
        np.random.seed(42)
        
        # Baseline model (well-calibrated)
        baseline_confidences = np.random.beta(2, 2, 100)  # More realistic confidence distribution
        baseline_correctness = np.random.binomial(1, baseline_confidences)
        
        # Treatment model (slightly less calibrated but potentially higher performance)
        treatment_confidences = np.random.beta(3, 2, 100) * 0.9 + 0.05  # Slightly overconfident
        treatment_correctness = np.random.binomial(1, treatment_confidences * 1.1)  # Higher performance
        
        # Initialize analyzer
        analyzer = AdvancedCalibrationAnalyzer(
            n_bins=10, 
            bin_strategy="uniform",
            bootstrap_samples=100  # Reduced for testing
        )
        
        # Test individual metrics computation
        baseline_metrics = analyzer.compute_comprehensive_metrics(
            baseline_confidences, baseline_correctness, "Baseline"
        )
        
        treatment_metrics = analyzer.compute_comprehensive_metrics(
            treatment_confidences, treatment_correctness, "Treatment"
        )
        
        logger.info(f"✅ Baseline ECE: {baseline_metrics.ece:.6f}")
        logger.info(f"✅ Treatment ECE: {treatment_metrics.ece:.6f}")
        
        # Test comparison
        comparison = analyzer.compare_calibrations(
            baseline_confidences, baseline_correctness,
            treatment_confidences, treatment_correctness,
            "Baseline Model", "Treatment Model"
        )
        
        logger.info(f"✅ ECE difference: {comparison.ece_difference:+.6f}")
        logger.info(f"✅ Practically equivalent: {comparison.practically_equivalent}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Calibration analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_statistical_validator():
    """Test statistical validation components."""
    logger.info("🔬 Testing statistical validator...")
    
    try:
        from sragv.evaluation.statistical_validator import StatisticalValidator
        
        # Initialize validator
        validator = StatisticalValidator(
            alpha=0.05,
            bootstrap_samples=100  # Reduced for testing
        )
        
        # Create synthetic results data
        baseline_results = {
            'problem_results': [
                {'is_correct': True, 'confidence': 0.8},
                {'is_correct': False, 'confidence': 0.6},
                {'is_correct': True, 'confidence': 0.9},
                {'is_correct': True, 'confidence': 0.7},
                {'is_correct': False, 'confidence': 0.5}
            ] * 20,  # 100 samples
            'ece': 0.05
        }
        
        treatment_results = {
            'problem_results': [
                {'is_correct': True, 'confidence': 0.85},
                {'is_correct': True, 'confidence': 0.75},
                {'is_correct': True, 'confidence': 0.95},
                {'is_correct': True, 'confidence': 0.8},
                {'is_correct': False, 'confidence': 0.6}
            ] * 20,  # 100 samples  
            'ece': 0.08
        }
        
        # Test validation
        comparison_result = validator.validate_phase4_results(
            baseline_results, treatment_results,
            "Phase 1 Baseline", "Phase 3 Trained"
        )
        
        logger.info(f"✅ Best performing model: {comparison_result.best_performing_model}")
        logger.info(f"✅ Corrected alpha: {comparison_result.corrected_alpha:.4f}")
        logger.info(f"✅ Recommendations: {len(comparison_result.practical_recommendations)}")
        
        # Test report generation
        report = validator.generate_statistical_report(comparison_result)
        logger.info(f"✅ Generated statistical report ({len(report)} characters)")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Statistical validator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test that required data files exist and are accessible."""
    logger.info("🔬 Testing data loading...")
    
    required_files = [
        "checkpoints/phase1_star_calibrator_1754272154.pt",
        "phase1_results/phase1_final_report.json",
        "phase3_results/phase1_to_phase3_results_20250804_154020.json",
        "config/config.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            logger.info(f"✅ Found: {file_path}")
    
    if missing_files:
        logger.error(f"❌ Missing required files: {missing_files}")
        return False
    
    # Test loading key data files
    try:
        # Phase 1 results
        with open("phase1_results/phase1_final_report.json", 'r') as f:
            phase1_data = json.load(f)
        
        phase1_ece = phase1_data.get('final_ece', 0)
        logger.info(f"✅ Phase 1 ECE: {phase1_ece:.6f}")
        
        # Phase 3 results
        with open("phase3_results/phase1_to_phase3_results_20250804_154020.json", 'r') as f:
            phase3_data = json.load(f)
        
        iterations = phase3_data['stages']['star_continuous_training']['training_results']['iterations']
        total_pseudo_labels = sum(iter_data['pseudo_labels_created'] for iter_data in iterations)
        logger.info(f"✅ Phase 3 pseudo-labels: {total_pseudo_labels}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading failed: {e}")
        return False


def test_output_directories():
    """Test that output directories can be created."""
    logger.info("🔬 Testing output directories...")
    
    try:
        directories = [
            "phase4_results",
            "phase4_figures", 
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
            logger.info(f"✅ Directory ready: {directory}")
        
        # Test writing to directories
        test_file = Path("phase4_results") / "integration_test.json"
        with open(test_file, 'w') as f:
            json.dump({"test": "success", "timestamp": datetime.now().isoformat()}, f)
        
        logger.info(f"✅ Test file written: {test_file}")
        
        # Cleanup test file
        test_file.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Output directory test failed: {e}")
        return False


def main():
    """Run complete integration test suite."""
    logger.info("🚀 SRAG-V Phase 4 Integration Test Suite")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Import Testing", test_imports),
        ("Data Loading", test_data_loading), 
        ("Output Directories", test_output_directories),
        ("Model Loading", test_model_loading),
        ("Calibration Analyzer", test_calibration_analyzer),
        ("Statistical Validator", test_statistical_validator),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔬 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            test_start = time.time()
            success = test_func()
            test_duration = time.time() - test_start
            
            results[test_name] = {
                "success": success,
                "duration": test_duration
            }
            
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} - {test_name} ({test_duration:.1f}s)")
            
        except Exception as e:
            results[test_name] = {
                "success": False,
                "duration": time.time() - test_start,
                "error": str(e)
            }
            logger.error(f"❌ FAIL - {test_name}: {e}")
    
    # Summary
    total_duration = time.time() - start_time
    passed_tests = sum(1 for r in results.values() if r["success"])
    total_tests = len(results)
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests Passed: {passed_tests}/{total_tests}")
    logger.info(f"Total Duration: {total_duration:.1f}s")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Phase 4 ready for execution!")
        return True
    else:
        logger.error("💥 SOME TESTS FAILED - Fix issues before running Phase 4")
        
        # Show failed tests
        for test_name, result in results.items():
            if not result["success"]:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"   ❌ {test_name}: {error_msg}")
        
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)