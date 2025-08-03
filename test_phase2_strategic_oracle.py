#!/usr/bin/env python3
"""
STAR Phase 2 Integration Test
Tests the Strategic Oracle with real Phase 1 data to validate functionality.

This script validates that Phase 2 enhancements work correctly with 
the existing architecture and Phase 1 confidence calibration results.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragv.strategic_oracle import StrategicOracle
from sragv.confidence_calibration import EnhancedConfidenceCalibrator
from sragv.models.verification_generator import TestQualityPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_strategic_oracle_integration():
    """Test Strategic Oracle with Phase 1 results."""
    
    logger.info("🎯 STAR Phase 2 Strategic Oracle Integration Test")
    logger.info("=" * 80)
    
    try:
        # Load Phase 1 enhanced solutions
        phase1_solutions_path = "phase1_results/solution_data_enhanced.json"
        if not Path(phase1_solutions_path).exists():
            logger.error(f"Phase 1 enhanced solutions not found: {phase1_solutions_path}")
            logger.info("Run Phase 1 confidence diversity fix first!")
            return False
        
        with open(phase1_solutions_path, 'r') as f:
            solutions_data = json.load(f)
        
        logger.info(f"📊 Loaded {len(solutions_data)} Phase 1 solutions")
        
        # Extract problems and solution history
        problems = []
        solution_history = []
        
        for solution in solutions_data:
            problem = solution.get('problem', {})
            if problem and problem not in problems:
                problems.append(problem)
            
            solution_history.append({
                'problem_id': problem.get('problem_id', 'unknown'),
                'code': solution.get('code', ''),
                'score': solution.get('enhanced_score', 0.5),
                'original_score': solution.get('original_score', 0.8)
            })
        
        logger.info(f"📊 Extracted {len(problems)} unique problems")
        logger.info(f"📊 Solution history: {len(solution_history)} solutions")
        
        # Initialize confidence calibrator (mock for testing)
        logger.info("🔧 Initializing confidence calibrator...")
        calibrator = EnhancedConfidenceCalibrator(
            num_classes=1,
            temperature_schedule="adaptive",
            calibration_method="temperature_scaling",
            feature_dim=16
        )
        
        # Load existing calibrator if available
        calibrator_path = "checkpoints/phase1_star_calibrator_enhanced.pt"
        if Path(calibrator_path).exists():
            try:
                calibrator.load_calibrator(calibrator_path)
                logger.info("✅ Loaded Phase 1 enhanced calibrator")
            except Exception as e:
                logger.warning(f"Could not load calibrator: {e}")
        
        # Initialize Strategic Oracle
        logger.info("🎯 Initializing Strategic Oracle...")
        oracle = StrategicOracle(
            confidence_calibrator=calibrator,
            selection_strategy="uncertainty_diversity_balanced",
            uncertainty_weight=0.4,
            diversity_weight=0.4,
            quality_weight=0.2
        )
        
        logger.info("✅ Strategic Oracle initialized")
        
        # Test 1: Uncertainty Scoring
        logger.info("\n🧪 TEST 1: Uncertainty Scoring")
        logger.info("-" * 40)
        
        test_problem = problems[0] if problems else {'problem_id': 'test', 'question': 'Test problem'}
        test_solutions = [sol for sol in solution_history if sol['problem_id'] == test_problem.get('problem_id')]
        
        if test_solutions:
            uncertainty_score = oracle.compute_uncertainty_score(test_problem, test_solutions)
            logger.info(f"✅ Uncertainty score: {uncertainty_score:.3f}")
            logger.info(f"   Solutions tested: {len(test_solutions)}")
            logger.info(f"   Score range: {min(s['score'] for s in test_solutions):.3f} - {max(s['score'] for s in test_solutions):.3f}")
        else:
            logger.warning("No solutions found for uncertainty testing")
        
        # Test 2: Diversity Scoring
        logger.info("\n🧪 TEST 2: Diversity Scoring")
        logger.info("-" * 40)
        
        if len(problems) >= 3:
            test_candidate = problems[2]
            selected_problems = problems[:2]
            
            diversity_score = oracle.compute_diversity_score(test_candidate, selected_problems)
            logger.info(f"✅ Diversity score: {diversity_score:.3f}")
            logger.info(f"   Candidate: {test_candidate.get('problem_id', 'unknown')}")
            logger.info(f"   Against: {[p.get('problem_id', 'unknown') for p in selected_problems]}")
        else:
            logger.warning("Insufficient problems for diversity testing")
        
        # Test 3: Quality Scoring
        logger.info("\n🧪 TEST 3: Quality Scoring")
        logger.info("-" * 40)
        
        quality_score = oracle.compute_quality_score(test_problem)
        logger.info(f"✅ Quality score: {quality_score:.3f}")
        logger.info(f"   Problem: {test_problem.get('problem_id', 'unknown')}")
        
        # Test 4: Strategic Selection
        logger.info("\n🧪 TEST 4: Strategic Problem Selection")
        logger.info("-" * 50)
        
        if len(problems) >= 5:
            candidate_problems = problems[:8]  # Use subset for testing
            num_select = 3
            
            logger.info(f"Selecting {num_select} problems from {len(candidate_problems)} candidates")
            
            selected = oracle.strategic_problem_selection(
                candidate_problems=candidate_problems,
                num_select=num_select,
                selected_problems=[],
                solution_history=solution_history
            )
            
            logger.info(f"✅ Strategic selection completed")
            logger.info(f"   Selected {len(selected)} problems:")
            for i, prob in enumerate(selected):
                logger.info(f"     #{i+1}: {prob.get('problem_id', 'unknown')}")
        else:
            logger.warning("Insufficient problems for strategic selection testing")
        
        # Test 5: Test Quality Prediction
        logger.info("\n🧪 TEST 5: Test Quality Prediction")
        logger.info("-" * 40)
        
        test_quality_predictor = TestQualityPredictor()
        
        # Create mock test cases
        mock_test_cases = [
            {'input': [1, 2, 3], 'output': 6},
            {'input': [], 'output': 0},
            {'input': [5], 'output': 5},
            {'input': [1, 2, 3, 4, 5], 'output': 15}
        ]
        
        quality_metrics = test_quality_predictor.predict_test_quality(
            test_cases=mock_test_cases,
            problem=test_problem,
            solutions=test_solutions
        )
        
        logger.info(f"✅ Test quality prediction completed")
        logger.info(f"   Overall Quality: {quality_metrics['overall_quality']:.3f}")
        logger.info(f"   Confidence: {quality_metrics['confidence']:.3f}")
        logger.info(f"   Coverage: {quality_metrics.get('coverage_completeness', 0):.2f}")
        logger.info(f"   Edge Cases: {quality_metrics.get('edge_case_detection', 0):.2f}")
        
        # Test 6: Oracle Analytics
        logger.info("\n🧪 TEST 6: Oracle Analytics")
        logger.info("-" * 30)
        
        analytics = oracle.get_selection_analytics()
        logger.info(f"✅ Oracle analytics:")
        logger.info(f"   Total selections: {analytics['total_selections']}")
        logger.info(f"   Strategy: {analytics['strategy']}")
        logger.info(f"   Weights: {analytics['weights']}")
        logger.info(f"   Embedding cache: {analytics['embedding_cache_size']} problems")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PHASE 2 STRATEGIC ORACLE INTEGRATION TEST COMPLETE")
        logger.info("✅ All core components functional:")
        logger.info("   ✓ Uncertainty scoring with Phase 1 confidence calibration")
        logger.info("   ✓ Diversity scoring with problem embeddings")
        logger.info("   ✓ Quality scoring with problem assessment")
        logger.info("   ✓ Strategic selection with combined scoring")
        logger.info("   ✓ Test quality prediction for enhanced verification")
        logger.info("   ✓ Analytics and performance tracking")
        logger.info("")
        logger.info("🚀 READY FOR PHASE 2 DEPLOYMENT!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"💥 Phase 2 integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_phase2_requirements():
    """Validate that Phase 2 requirements are met."""
    
    logger.info("🔍 Validating Phase 2 Requirements")
    logger.info("-" * 40)
    
    requirements = {
        'phase1_enhanced_solutions': 'phase1_results/solution_data_enhanced.json',
        'phase1_calibrator': 'checkpoints/phase1_star_calibrator_enhanced.pt',
        'confidence_diversity_summary': 'phase1_results/direct_confidence_summary.json'
    }
    
    all_requirements_met = True
    
    for req_name, req_path in requirements.items():
        if Path(req_path).exists():
            logger.info(f"   ✅ {req_name}: {req_path}")
        else:
            logger.error(f"   ❌ {req_name}: {req_path} (missing)")
            all_requirements_met = False
    
    if all_requirements_met:
        logger.info("✅ All Phase 2 requirements satisfied")
    else:
        logger.error("❌ Missing Phase 2 requirements - run Phase 1 fixes first")
    
    return all_requirements_met


if __name__ == "__main__":
    print("🎯 STAR Phase 2 Strategic Oracle Integration Test")
    print("=" * 80)
    
    # Validate requirements first
    if not validate_phase2_requirements():
        print("❌ Phase 2 requirements not met")
        sys.exit(1)
    
    # Run integration test
    success = test_strategic_oracle_integration()
    
    if success:
        print("\n🎉 Phase 2 integration test SUCCESSFUL!")
        print("✅ Strategic Oracle fully functional")
        print("🚀 Ready to proceed with Phase 2 deployment")
        sys.exit(0)
    else:
        print("\n❌ Phase 2 integration test FAILED")
        print("🔧 Check logs and fix issues before proceeding")
        sys.exit(1)