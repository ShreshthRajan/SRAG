#!/usr/bin/env python3
"""
Quick test to verify calibrator saving works before rerunning full Phase 3.
Tests the critical save functionality in 30 seconds.
"""

import torch
import logging
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragv.orchestrator import SRAGVOrchestrator
from sragv.confidence_calibration import EnhancedConfidenceCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_calibrator_save():
    """Test that calibrator can be loaded, modified, and saved properly."""
    logger.info("🧪 Testing calibrator save functionality...")
    
    try:
        # 1. Load existing Phase 1 calibrator
        logger.info("📦 Loading Phase 1 calibrator...")
        orchestrator = SRAGVOrchestrator("config/config.yaml")
        orchestrator.initialize_players()
        
        calibrator_path = "checkpoints/phase1_star_calibrator_1754272154.pt"
        if not Path(calibrator_path).exists():
            logger.error(f"❌ Phase 1 calibrator not found: {calibrator_path}")
            return False
            
        checkpoint = torch.load(calibrator_path, map_location='cpu')
        calibrator = EnhancedConfidenceCalibrator()
        calibrator.load_state_dict(checkpoint['state_dict'])
        
        orchestrator.solution_generator.confidence_calibrator = calibrator
        orchestrator.solution_generator.use_calibration = True
        
        logger.info("✅ Phase 1 calibrator loaded successfully")
        
        # 2. Test saving calibrator (the critical functionality)
        test_save_path = "checkpoints/TEST_calibrator_save.pt"
        
        logger.info("💾 Testing calibrator save...")
        torch.save({
            'state_dict': calibrator.state_dict(),
            'test_run': True,
            'timestamp': 'test'
        }, test_save_path)
        
        # 3. Verify file was created
        if not Path(test_save_path).exists():
            logger.error("❌ Calibrator save failed - file not created")
            return False
            
        # 4. Test loading the saved calibrator
        logger.info("📥 Testing calibrator load...")
        test_checkpoint = torch.load(test_save_path, map_location='cpu')
        test_calibrator = EnhancedConfidenceCalibrator()
        test_calibrator.load_state_dict(test_checkpoint['state_dict'])
        
        logger.info("✅ Calibrator save/load test successful!")
        
        # Cleanup
        Path(test_save_path).unlink()
        logger.info("🗑️ Test file cleaned up")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_calibrator_save()
    if success:
        print("✅ CALIBRATOR SAVE TEST PASSED - Ready to rerun Phase 3")
        exit(0)
    else:
        print("❌ CALIBRATOR SAVE TEST FAILED - Fix required before Phase 3")
        exit(1)