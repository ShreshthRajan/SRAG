#!/usr/bin/env python3
"""
Quick Phase 4 Verification - Essential Checks Only
Verifies readiness without downloading large models.
"""

import os
import sys
import json
import logging
import torch
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("🚀 Quick Phase 4 Verification")
    
    checks_passed = 0
    total_checks = 0
    
    # Check 1: HumanEval availability
    total_checks += 1
    try:
        import human_eval
        from human_eval.data import read_problems
        problems = read_problems()
        logger.info(f"✅ HumanEval: {len(problems)} problems available")
        checks_passed += 1
    except Exception as e:
        logger.error(f"❌ HumanEval: {e}")
    
    # Check 2: Phase 1 calibrator exists
    total_checks += 1
    phase1_path = "checkpoints/phase1_star_calibrator_1754272154.pt"
    if Path(phase1_path).exists():
        try:
            checkpoint = torch.load(phase1_path, map_location='cpu', weights_only=False)
            logger.info(f"✅ Phase 1 calibrator: Found with is_trained={checkpoint.get('is_trained', False)}")
            checks_passed += 1
        except Exception as e:
            logger.error(f"❌ Phase 1 calibrator: Load error - {e}")
    else:
        logger.error(f"❌ Phase 1 calibrator: File not found - {phase1_path}")
    
    # Check 3: Phase 3 calibrator exists  
    total_checks += 1
    phase3_path = "checkpoints/phase3_star_training/phase3_final_calibrator_1754491530.pt"
    if Path(phase3_path).exists():
        try:
            checkpoint = torch.load(phase3_path, map_location='cpu', weights_only=False)
            pseudo_labels = checkpoint.get('pseudo_labels_used', 0)
            logger.info(f"✅ Phase 3 calibrator: Found with {pseudo_labels} pseudo-labels")
            checks_passed += 1
        except Exception as e:
            logger.error(f"❌ Phase 3 calibrator: Load error - {e}")
    else:
        logger.error(f"❌ Phase 3 calibrator: File not found - {phase3_path}")
    
    # Check 4: Pass@k calculation
    total_checks += 1
    try:
        import math
        def calculate_pass_at_k(n, c, k):
            if n - c < k: return 1.0
            def comb(n, k):
                if k > n or k < 0: return 0
                return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
            return 1.0 - (comb(n - c, k) / comb(n, k))
        
        # Test case: 1 correct out of 5 samples, pass@1 should be 0.2
        result = calculate_pass_at_k(5, 1, 1)
        expected = 0.2
        if abs(result - expected) < 0.01:
            logger.info(f"✅ Pass@k calculation: {result} ≈ {expected}")
            checks_passed += 1
        else:
            logger.error(f"❌ Pass@k calculation: {result} != {expected}")
    except Exception as e:
        logger.error(f"❌ Pass@k calculation: {e}")
    
    # Check 5: Budget estimation
    total_checks += 1
    try:
        problems = 164
        models = 2  
        solutions_per_problem = 5
        total_generations = problems * models * solutions_per_problem
        time_per_generation = 10  # seconds
        total_hours = (total_generations * time_per_generation) / 3600
        cost = total_hours * 30  # $30/hour ML Foundry
        
        logger.info(f"✅ Budget estimate: {total_generations} generations = {total_hours:.1f}h = ${cost:.2f}")
        if cost <= 300:
            logger.info("✅ Budget sufficient")
            checks_passed += 1
        else:
            logger.error("❌ Budget insufficient")
    except Exception as e:
        logger.error(f"❌ Budget calculation: {e}")
    
    # Final assessment
    success_rate = (checks_passed / total_checks) * 100
    logger.info("=" * 50)
    logger.info(f"VERIFICATION COMPLETE: {checks_passed}/{total_checks} checks passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        logger.info("🎉 READY FOR PHASE 4 IMPLEMENTATION!")
        return True
    else:
        logger.info("⚠️ Issues need to be resolved first")
        return False

if __name__ == "__main__":
    ready = main()
    sys.exit(0 if ready else 1)