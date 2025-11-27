#!/usr/bin/env python3
"""
SRAG-V Phase 3: TUNED STAR Continuous Training
Research-backed configuration with progressive thresholds for robust pseudo-labeling.

Key improvements:
1. Progressive thresholds (0.75 → 0.9 adaptive)
2. Permissive initial quality gates (0.6 min quality)
3. Enhanced logging for pseudo-label debugging
4. Balanced confidence/quality/uncertainty weighting

Author: Claude & Shreshth
Date: August 2025
"""

import os
import sys
import time
import json
import logging
import signal
import traceback
import torch
import gc
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables for stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Import components
try:
    from sragv.orchestrator import SRAGVOrchestrator
    from sragv.models.solution_generator import SolutionGenerator
    from sragv.confidence_calibration import EnhancedConfidenceCalibrator
    from sragv.strategic_oracle import StrategicOracle
    from sragv.training.star_trainer import STARTrainer
    logger_import_success = True
except Exception as e:
    print(f"💥 Critical import failure: {e}")
    logger_import_success = False
    sys.exit(1)

# Logging setup with enhanced pseudo-label tracking
log_filename = f"logs/phase3_tuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
Path("logs").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)
Path("phase3_results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename),
        logging.FileHandler('logs/phase3_tuned_latest.log')
    ]
)
logger = logging.getLogger(__name__)


def run_tuned_phase3_training():
    """Tuned Phase 3 training with research-backed progressive thresholds."""
    
    start_time = time.time()
    results = {
        "phase": "phase3_tuned_star_training",
        "status": "running", 
        "start_time": datetime.now().isoformat(),
        "tuning_applied": [
            "Progressive thresholds: 0.75→0.90",
            "Permissive quality gates: 0.6 min_quality",
            "Balanced weighting: 40%/35%/25% confidence/quality/uncertainty",
            "Enhanced pseudo-label tracking",
            "Increased ECE tolerance: 0.02 degradation allowed"
        ],
        "stages": {}
    }
    
    try:
        logger.info("🚀 PHASE 3 TUNED: Research-Backed STAR Training")
        logger.info("🎯 Fixes: Progressive thresholds + permissive quality gates")
        
        # =====================================================
        # STAGE 1: SYSTEM INITIALIZATION  
        # =====================================================
        logger.info("🔧 Stage 1: System Initialization")
        stage_start = time.time()
        
        orchestrator = SRAGVOrchestrator("config/config.yaml")
        orchestrator.initialize_players()
        solution_generator = orchestrator.solution_generator
        
        if solution_generator is None:
            raise Exception("Solution generator failed to initialize")
        
        logger.info("✅ SRAG-V orchestrator initialized")
        results["stages"]["initialization"] = {
            "status": "completed",
            "duration": time.time() - stage_start
        }
        
        # =====================================================
        # STAGE 2: LOAD ECE 0.0003 CALIBRATOR
        # =====================================================
        logger.info("🎯 Stage 2: Loading Phase 1 Calibrator (ECE 0.0003)")
        stage_start = time.time()
        
        calibrator_paths = [
            "checkpoints/phase1_star_calibrator_1754272154.pt",
            "checkpoints/phase1_star_calibrator_latest.pt",
            "checkpoints/phase1_star_calibrator_enhanced.pt"
        ]
        
        calibrator_loaded = False
        for path in calibrator_paths:
            if Path(path).exists():
                try:
                    checkpoint = torch.load(path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        calibrator = EnhancedConfidenceCalibrator()
                        calibrator.load_state_dict(checkpoint['state_dict'])
                        solution_generator.confidence_calibrator = calibrator
                        solution_generator.use_calibration = True
                        
                        logger.info(f"✅ Calibrator loaded: {path}")
                        logger.info(f"✅ Temperature: {float(calibrator.temperature):.3f}")
                        calibrator_loaded = True
                        break
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")
                    continue
        
        if not calibrator_loaded:
            raise Exception("Could not load Phase 1 calibrator")
        
        results["stages"]["calibrator_loading"] = {
            "status": "completed",
            "duration": time.time() - stage_start,
            "baseline_ece": 0.0003
        }
        
        # =====================================================
        # STAGE 3: LOAD TRAINING DATA
        # =====================================================
        logger.info("📚 Stage 3: Loading Training Data")
        stage_start = time.time()
        
        with open('data/phase1_diverse_problems.json', 'r') as f:
            all_problems = json.load(f)
        
        split_point = int(len(all_problems) * 0.8)
        training_problems = all_problems[:split_point]
        validation_problems = all_problems[split_point:]
        
        logger.info(f"✅ Training: {len(training_problems)}, Validation: {len(validation_problems)}")
        
        results["stages"]["data_loading"] = {
            "status": "completed",
            "duration": time.time() - stage_start,
            "training_problems": len(training_problems),
            "validation_problems": len(validation_problems)
        }
        
        # =====================================================
        # STAGE 4: INITIALIZE TUNED PHASE 3 COMPONENTS
        # =====================================================
        logger.info("🧠 Stage 4: Initializing TUNED Phase 3 Components")
        stage_start = time.time()
        
        strategic_oracle = StrategicOracle(solution_generator.confidence_calibrator)
        
        # RESEARCH-BACKED PROGRESSIVE CONFIGURATION
        tuned_star_config = {
            'max_iterations': 6,
            'batch_size': 32,
            'problems_per_batch': 12,
            'solutions_per_problem': 6,
            'confidence_threshold': 0.75,    # START LOWER
            'max_ece_degradation': 0.02,     # MORE TOLERANT
            'early_stopping_patience': 4,
            'checkpoint_frequency': 2,
            'bayesian_labeler_config': {
                'base_confidence_threshold': 0.75,    # PROGRESSIVE START
                'adaptive_threshold_enabled': True,
                'min_quality_score': 0.6,            # PERMISSIVE (was 0.8)
                'max_uncertainty': 0.3,              # MORE TOLERANT
                'confidence_weight': 0.4,            # BALANCED
                'quality_weight': 0.35,              # REDUCED DOMINANCE
                'uncertainty_weight': 0.25,
                'min_confidence_threshold': 0.7,     # PROGRESSIVE BOUNDS
                'max_confidence_threshold': 0.9,
                'threshold_adaptation_rate': 0.15,   # FASTER ADAPTATION
                'min_samples_for_adaptation': 30,    # LOWER REQUIREMENT
            }
        }
        
        star_trainer = STARTrainer(
            orchestrator=orchestrator,
            strategic_oracle=strategic_oracle,
            config=tuned_star_config
        )
        
        logger.info("✅ TUNED STARTrainer initialized")
        logger.info(f"✅ Key changes: confidence_threshold={tuned_star_config['confidence_threshold']}, min_quality={tuned_star_config['bayesian_labeler_config']['min_quality_score']}")
        
        results["stages"]["tuned_phase3_init"] = {
            "status": "completed",
            "duration": time.time() - stage_start,
            "tuned_config": tuned_star_config
        }
        
        # =====================================================
        # STAGE 5: TUNED STAR CONTINUOUS TRAINING
        # =====================================================
        logger.info("🌟 Stage 5: TUNED STAR Continuous Training")
        logger.info("🎯 Progressive thresholds should enable pseudo-labeling")
        stage_start = time.time()
        
        training_results = star_trainer.continuous_train(
            initial_problems=training_problems,
            validation_problems=validation_problems,
            max_duration_hours=3.0
        )
        
        stage_duration = time.time() - stage_start
        final_metrics = training_results.get('final_metrics', {})
        best_ece = final_metrics.get('best_ece', float('inf'))
        total_iterations = final_metrics.get('total_iterations', 0)
        total_solutions = final_metrics.get('total_solutions_generated', 0)
        total_pseudo_labels = final_metrics.get('total_pseudo_labels', 0)
        
        # Success criteria: iterations > 0 AND pseudo-labels > 0
        training_successful = (total_iterations > 0 and total_pseudo_labels > 0)
        phase1_baseline = 0.0003
        
        results["stages"]["tuned_star_training"] = {
            "status": "completed" if training_successful else "failed",
            "duration": stage_duration,
            "best_ece": best_ece,
            "phase1_baseline": phase1_baseline,
            "ece_ratio": best_ece / phase1_baseline if best_ece != float('inf') else float('inf'),
            "total_iterations": total_iterations,
            "solutions_generated": total_solutions,
            "pseudo_labels_created": total_pseudo_labels,
            "pseudo_labeling_success": total_pseudo_labels > 0
        }
        
        # =====================================================
        # STAGE 6: RESULTS ANALYSIS
        # =====================================================
        logger.info("📊 Stage 6: Results Analysis")
        
        total_time = time.time() - start_time
        results.update({
            "status": "successful" if training_successful else "failed_pseudo_labeling",
            "end_time": datetime.now().isoformat(), 
            "total_duration": total_time,
            "key_metrics": {
                "pseudo_labels_created": total_pseudo_labels,
                "iterations_completed": total_iterations,
                "solutions_generated": total_solutions,
                "final_ece": best_ece,
                "ece_degradation": best_ece / phase1_baseline if best_ece != float('inf') else float('inf')
            },
            "success_criteria": {
                "pseudo_labeling_active": total_pseudo_labels > 0,
                "continuous_training_executed": total_iterations > 0,
                "architecture_functional": total_solutions > 0
            }
        })
        
        # Save results
        results_path = f"phase3_results/phase3_tuned_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Comprehensive logging
        logger.info("🎉 TUNED PHASE 3 TRAINING COMPLETED!")
        logger.info("=" * 70)
        logger.info(f"Status: {results['status'].upper()}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"🏷️ Pseudo-labels created: {total_pseudo_labels}")
        logger.info(f"🔄 STAR iterations: {total_iterations}")
        logger.info(f"💡 Solutions generated: {total_solutions}")
        logger.info(f"📊 Final ECE: {best_ece:.4f}")
        logger.info(f"📈 ECE ratio: {best_ece/phase1_baseline:.1f}x baseline")
        logger.info(f"✅ Pseudo-labeling {'SUCCESS' if total_pseudo_labels > 0 else 'FAILED'}")
        logger.info(f"✅ Continuous training {'SUCCESS' if total_iterations > 0 else 'FAILED'}")
        logger.info(f"Results: {results_path}")
        logger.info("=" * 70)
        
        return results
        
    except Exception as e:
        logger.error(f"💥 Tuned Phase 3 training failed: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main execution."""
    logger.info("🚀 Starting TUNED Phase 3 STAR Training")
    
    try:
        results = run_tuned_phase3_training()
        
        if results["status"] == "successful":
            logger.info("🎉 TUNED Phase 3 completed successfully!")
        else:
            logger.warning("⚠️ Phase 3 completed but pseudo-labeling may need further tuning")
            
        return results
        
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()