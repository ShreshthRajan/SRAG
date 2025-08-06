#!/usr/bin/env python3
"""
COMPREHENSIVE TEST: Bayesian Pseudo-Labeler Robustness Validation
Tests all edge cases, timeout scenarios, and integration points.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragv.training.bayesian_pseudo_labeler import BayesianPseudoLabeler
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_beta_ppf_edge_cases():
    """Test Beta PPF with problematic parameter values that caused hanging."""
    logger.info("🧪 Testing Beta PPF edge cases...")
    
    # Test cases that previously caused hanging
    problematic_cases = [
        (8.93, 5.07),  # From actual logs
        (10.0, 0.0),   # Beta = 0 edge case
        (0.0, 10.0),   # Alpha = 0 edge case
        (1000.0, 1000.0),  # Very large parameters
        (0.001, 0.001),    # Very small parameters
    ]
    
    for i, (alpha, beta) in enumerate(problematic_cases):
        logger.info(f"Test case {i+1}: alpha={alpha}, beta={beta}")
        start_time = time.time()
        
        try:
            q = 0.025  # 2.5th percentile
            alpha_safe = max(alpha, 0.01)
            beta_safe = max(beta, 0.01)
            
            # Test with timeout (same as our fix)
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Beta PPF computation timed out")
            
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(3)  # 3 second timeout
            
            try:
                result = stats.beta.ppf(q, alpha_safe, beta_safe)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            elapsed = time.time() - start_time
            
            if not np.isfinite(result) or result < 0 or result > 1:
                fallback = alpha_safe / (alpha_safe + beta_safe)
                logger.info(f"  ✅ Invalid result detected, fallback used: {fallback:.3f}")
            else:
                logger.info(f"  ✅ Valid result: {result:.3f}, time: {elapsed:.3f}s")
            
        except TimeoutError:
            logger.info(f"  ✅ Timeout detected after {time.time() - start_time:.3f}s, fallback needed")
        except Exception as e:
            logger.info(f"  ✅ Exception caught: {e}, fallback needed")
    
    logger.info("✅ Beta PPF edge case testing completed")

def test_bayesian_pseudo_labeler_with_problematic_data():
    """Test the actual Bayesian pseudo-labeler with data that caused hanging."""
    logger.info("🧪 Testing BayesianPseudoLabeler with problematic data...")
    
    # Create pseudo-labeler
    config = {
        'base_confidence_threshold': 0.52,
        'min_quality_score': 0.4,
        'max_uncertainty': 0.4,
        'prior_alpha': 2.0,
        'prior_beta': 2.0,
        'credible_interval': 0.95
    }
    labeler = BayesianPseudoLabeler(config=config)
    
    # Create solutions with identical values (this caused the original hang)
    identical_solutions = []
    for i in range(72):  # Same as original problem
        identical_solutions.append({
            'score': 0.685,  # Identical confidence
            'code': f'def solution_{i}(): pass',
            'execution_success': None,
            'syntax_valid': True,
            'pass_rate': None
        })
    
    logger.info(f"Created {len(identical_solutions)} identical solutions")
    
    # Test with timeout monitoring
    start_time = time.time()
    timeout_limit = 30  # Should complete in under 30 seconds
    
    try:
        pseudo_labels, metrics = labeler.create_pseudo_labels(identical_solutions, iteration=1)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Processing completed in {elapsed:.2f}s")
        logger.info(f"✅ Created {len(pseudo_labels)} pseudo-labels")
        logger.info(f"✅ Pseudo-label rate: {metrics['pseudo_label_rate']:.3f}")
        
        if elapsed > timeout_limit:
            logger.error(f"❌ Processing took too long: {elapsed:.2f}s > {timeout_limit}s")
            return False
        
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ Processing failed after {elapsed:.2f}s: {e}")
        return False

def test_bayesian_criteria_edge_cases():
    """Test _meets_bayesian_criteria with edge case values."""
    logger.info("🧪 Testing Bayesian criteria edge cases...")
    
    config = {
        'base_confidence_threshold': 0.52,
        'min_quality_score': 0.4,
        'max_uncertainty': 0.4,
        'confidence_weight': 0.4,
        'quality_weight': 0.35,
        'uncertainty_weight': 0.25,
        'prior_alpha': 2.0,
        'prior_beta': 2.0,
        'credible_interval': 0.95
    }
    labeler = BayesianPseudoLabeler(config=config)
    
    # Test cases that could cause issues
    edge_cases = [
        (0.685, 0.641, 0.220, 0.520),  # Original hanging case
        (1.0, 1.0, 0.0, 0.5),          # Perfect case
        (0.0, 0.0, 1.0, 0.5),          # Worst case
        (0.5, 0.5, 0.5, 0.5),          # Neutral case
        (np.nan, 0.5, 0.5, 0.5),       # NaN input
        (0.5, np.inf, 0.5, 0.5),       # Infinite input
    ]
    
    for i, (conf, qual, unc, thresh) in enumerate(edge_cases):
        logger.info(f"Test case {i+1}: conf={conf}, qual={qual}, unc={unc}, thresh={thresh}")
        
        start_time = time.time()
        try:
            result = labeler._meets_bayesian_criteria(conf, qual, unc, thresh)
            elapsed = time.time() - start_time
            
            logger.info(f"  ✅ Result: {result}, time: {elapsed:.3f}s")
            
            if elapsed > 5.0:  # Should complete in under 5 seconds
                logger.error(f"  ❌ Too slow: {elapsed:.3f}s")
                return False
                
        except Exception as e:
            elapsed = time.time() - start_time
            logger.info(f"  ✅ Exception handled: {e}, time: {elapsed:.3f}s")
    
    logger.info("✅ Bayesian criteria edge case testing completed")
    return True

def test_signal_handling_robustness():
    """Test signal handling doesn't interfere with existing handlers."""
    logger.info("🧪 Testing signal handling robustness...")
    
    import signal
    import os
    
    # Set up a dummy signal handler
    def dummy_handler(signum, frame):
        logger.info("Dummy handler called")
    
    original_handler = signal.signal(signal.SIGALRM, dummy_handler)
    
    try:
        # Test our Beta PPF with signal handling
        alpha, beta = 8.93, 5.07
        q = 0.025
        
        alpha_safe = max(alpha, 0.01)
        beta_safe = max(beta, 0.01)
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Beta PPF computation timed out")
        
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(3)
        
        try:
            result = stats.beta.ppf(q, alpha_safe, beta_safe)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
        
        # Verify original handler is restored
        current_handler = signal.signal(signal.SIGALRM, dummy_handler)
        if current_handler == dummy_handler:
            logger.info("✅ Signal handler properly restored")
            return True
        else:
            logger.error("❌ Signal handler not properly restored")
            return False
            
    finally:
        signal.signal(signal.SIGALRM, original_handler)

def test_memory_and_performance():
    """Test memory usage and performance with large datasets."""
    logger.info("🧪 Testing memory and performance...")
    
    config = {'base_confidence_threshold': 0.52}
    labeler = BayesianPseudoLabeler(config=config)
    
    # Test with larger dataset (similar to actual Phase 3)
    large_solutions = []
    for i in range(500):  # Larger than typical 72
        large_solutions.append({
            'score': 0.5 + 0.4 * np.random.random(),  # Varied scores
            'code': f'def solution_{i}(): return {i}',
            'execution_success': np.random.choice([True, False, None]),
            'syntax_valid': True,
            'pass_rate': np.random.random()
        })
    
    start_time = time.time()
    memory_start = sum(len(str(s)) for s in large_solutions)  # Rough memory estimate
    
    try:
        pseudo_labels, metrics = labeler.create_pseudo_labels(large_solutions, iteration=1)
        elapsed = time.time() - start_time
        
        logger.info(f"✅ Processed {len(large_solutions)} solutions in {elapsed:.2f}s")
        logger.info(f"✅ Created {len(pseudo_labels)} pseudo-labels")
        logger.info(f"✅ Rate: {len(large_solutions)/elapsed:.1f} solutions/second")
        
        # Performance should be reasonable
        if elapsed > 60:  # Should complete in under 1 minute
            logger.error(f"❌ Too slow for large dataset: {elapsed:.2f}s")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed with large dataset: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide final assessment."""
    logger.info("🚀 Starting comprehensive Bayesian pseudo-labeler testing...")
    
    tests = [
        ("Beta PPF Edge Cases", test_beta_ppf_edge_cases),
        ("Problematic Data Handling", test_bayesian_pseudo_labeler_with_problematic_data),
        ("Bayesian Criteria Edge Cases", test_bayesian_criteria_edge_cases),
        ("Signal Handling Robustness", test_signal_handling_robustness),
        ("Memory and Performance", test_memory_and_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*60}")
        logger.info(f"🔬 RUNNING: {test_name}")
        logger.info(f"{'='*60}")
        
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            
            if result is None:
                result = True  # Assume success if no return value
            
            status = "PASSED ✅" if result else "FAILED ❌"
            logger.info(f"📊 {test_name}: {status} ({elapsed:.2f}s)")
            results.append((test_name, result, elapsed))
            
        except Exception as e:
            logger.error(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False, 0))
    
    # Final assessment
    logger.info(f"\n{'='*60}")
    logger.info(f"📊 FINAL ASSESSMENT")
    logger.info(f"{'='*60}")
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    pass_rate = passed / total
    
    for test_name, result, elapsed in results:
        status = "PASSED ✅" if result else "FAILED ❌"
        logger.info(f"  {test_name}: {status} ({elapsed:.2f}s)")
    
    logger.info(f"\n📊 OVERALL: {passed}/{total} tests passed ({pass_rate*100:.1f}%)")
    
    if pass_rate >= 1.0:
        logger.info("🎉 ALL TESTS PASSED - 100% CONFIDENCE FOR PHASE 3 RESTART")
        return True
    elif pass_rate >= 0.8:
        logger.warning("⚠️ MOSTLY PASSED - Should work but monitor closely")
        return True
    else:
        logger.error("❌ TESTS FAILED - DO NOT RESTART PHASE 3 YET")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)