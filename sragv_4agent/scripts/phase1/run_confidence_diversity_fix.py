#!/usr/bin/env python3
"""
SURGICAL CONFIDENCE DIVERSITY FIX - COMPLETE PIPELINE
Executes the complete fix: enhance scores + retrain calibrator.
Designed to be run once to fix the confidence collapse issue.
"""

import sys
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the complete confidence diversity fix pipeline."""
    
    logger.info("🚀 LAUNCHING SURGICAL CONFIDENCE DIVERSITY FIX")
    logger.info("=" * 80)
    logger.info("This fix addresses the confidence collapse issue identified in Phase 1")
    logger.info("- Enhances target score computation for meaningful variance")
    logger.info("- Retrains calibrator with diverse confidence targets")
    logger.info("- Preserves all existing solution generation work")
    logger.info("=" * 80)
    
    try:
        # Step 1: Fix confidence diversity in existing data
        logger.info("📊 STEP 1: Enhancing score diversity...")
        
        from fix_confidence_diversity import fix_confidence_diversity
        
        solution_path = "phase1_results/solution_data.json"
        if not Path(solution_path).exists():
            logger.error(f"❌ Solution data not found: {solution_path}")
            logger.info("💡 Ensure Phase 1 training completed successfully")
            return False
        
        enhanced_solutions, stats = fix_confidence_diversity(solution_path)
        
        logger.info("✅ Step 1 complete - Score diversity enhanced")
        logger.info(f"  📊 Score variance: {stats['std']:.4f}")
        logger.info(f"  🎯 Score range: {stats['min']:.3f} - {stats['max']:.3f}")
        
        # Step 2: Retrain calibrator with enhanced scores
        logger.info("🔧 STEP 2: Retraining calibrator...")
        
        from retrain_calibrator import retrain_calibrator_with_diversity
        
        enhanced_path = solution_path.replace('.json', '_enhanced.json')
        summary = retrain_calibrator_with_diversity(enhanced_path)
        
        if not summary:
            logger.error("❌ Calibrator retraining failed")
            return False
        
        logger.info("✅ Step 2 complete - Calibrator retrained")
        
        # Step 3: Validation and results
        logger.info("🔍 STEP 3: Validating fix success...")
        
        diversity_achieved = summary.get("confidence_diversity", {}).get("diversity_achieved", False)
        ready_for_phase2 = summary.get("ready_for_phase2", False)
        
        logger.info("📊 VALIDATION RESULTS:")
        logger.info(f"  ✅ Score diversity improvement: {summary['score_diversity']['improvement_factor']:.2f}x")
        logger.info(f"  🎯 Calibrated confidence std: {summary['confidence_diversity']['calibrated_std']:.4f}")
        logger.info(f"  📈 Confidence range: {summary['confidence_diversity']['calibrated_range']}")
        logger.info(f"  🎉 Diversity achieved: {'YES' if diversity_achieved else 'NO'}")
        logger.info(f"  🚀 Phase 2 ready: {'YES' if ready_for_phase2 else 'NO'}")
        
        # Final status
        logger.info("=" * 80)
        if ready_for_phase2:
            logger.info("🎉 SURGICAL FIX SUCCESSFUL!")
            logger.info("✅ Confidence collapse resolved")
            logger.info("✅ Strategic oracle functionality restored")
            logger.info("✅ STAR architecture ready for Phase 2")
            logger.info("")
            logger.info("NEXT STEPS:")
            logger.info("1. Proceed to Phase 2 - Strategic Oracle implementation")
            logger.info("2. Verify uncertainty-based problem selection works")
            logger.info("3. Begin self-amplification training")
        else:
            logger.warning("⚠️ SURGICAL FIX NEEDS ADDITIONAL WORK")
            logger.warning("❌ Confidence diversity still insufficient")
            logger.warning("💡 May need more aggressive score differentiation")
        
        logger.info("=" * 80)
        
        return ready_for_phase2
        
    except Exception as e:
        logger.error(f"💥 SURGICAL FIX FAILED: {e}")
        logger.error("🔍 Check logs for detailed error information")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)