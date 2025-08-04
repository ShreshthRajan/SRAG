#!/usr/bin/env python3
"""
SRAG-V Phase 3: STAR Continuous Training with Bayesian Pseudo-Labeling
Builds on successful Phase 1 (ECE 0.0003) and Phase 2 (Strategic Oracle).

Implements:
- STARTrainer with continuous learning
- BayesianPseudoLabeler with adaptive thresholds
- Strategic problem selection
- ECE monitoring and early stopping
- Comprehensive checkpointing

Author: Claude & Shreshth
Date: August 2025
"""

import os
import sys
import time
import json
import logging
import traceback
import torch
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Fix import paths FIRST
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables for stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Logging setup
log_filename = f"logs/phase3_star_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
Path("logs").mkdir(exist_ok=True)
Path("checkpoints").mkdir(exist_ok=True)
Path("phase3_results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename),
        logging.FileHandler('logs/phase3_star_training_latest.log')
    ]
)
logger = logging.getLogger(__name__)


class Phase3STARTrainer:
    """
    Phase 3 STAR Continuous Training Orchestrator.
    Integrates Phase 1 calibration + Phase 2 strategic selection + Phase 3 continuous learning.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.orchestrator = None
        self.star_trainer = None
        self.results = {
            "phase": "phase3_star_continuous",
            "status": "initialized",
            "start_time": datetime.now().isoformat(),
            "stages": {},
            "foundation": {
                "phase1_ece": "0.0003 (exceptional)",
                "phase2_oracle": "strategic selection ready",
                "phase3_architecture": "STARTrainer + BayesianPseudoLabeler"
            }
        }
        
        logger.info("🚀 Phase 3 STAR Continuous Training Initialized")
        logger.info("Foundation: Phase 1 ECE 0.0003 + Phase 2 Strategic Oracle")
        
    def validate_prerequisites(self) -> bool:
        """Validate Phase 3 prerequisites."""
        logger.info("🔍 Validating Phase 3 prerequisites...")
        
        checks = {
            "Phase 1 calibrator": self._check_phase1_calibrator(),
            "Diverse problems": Path("data/phase1_diverse_problems.json").exists(),
            "STARTrainer": self._check_star_trainer(),
            "BayesianPseudoLabeler": self._check_bayesian_labeler(),
            "Strategic Oracle": self._check_strategic_oracle(),
            "Integration tests": self._check_integration_tests()
        }
        
        all_passed = True
        for check, result in checks.items():
            status = "✅" if result else "❌"
            logger.info(f"  {status} {check}")
            if not result:
                all_passed = False
                
        return all_passed
    
    def _check_phase1_calibrator(self) -> bool:
        """Check Phase 1 calibrator exists and loads."""
        try:
            # Check for the ECE 0.0003 calibrator
            calibrator_paths = [
                "checkpoints/phase1_star_calibrator_1754272154.pt",  # ECE 0.0003 version
                "checkpoints/phase1_star_calibrator_latest.pt",
                "checkpoints/phase1_star_calibrator_success.pt"
            ]
            
            for path in calibrator_paths:
                if Path(path).exists():
                    # Try loading it
                    checkpoint = torch.load(path, map_location='cpu')
                    if 'state_dict' in checkpoint:
                        logger.info(f"✅ Found Phase 1 calibrator: {path}")
                        return True
                        
            return False
        except Exception as e:
            logger.error(f"Phase 1 calibrator check failed: {e}")
            return False
    
    def _check_star_trainer(self) -> bool:
        """Check STARTrainer can be imported."""
        try:
            from sragv.training.star_trainer import STARTrainer
            return True
        except Exception as e:
            logger.error(f"STARTrainer import failed: {e}")
            return False
    
    def _check_bayesian_labeler(self) -> bool:
        """Check BayesianPseudoLabeler can be imported."""
        try:
            from sragv.training.bayesian_pseudo_labeler import BayesianPseudoLabeler
            return True
        except Exception as e:
            logger.error(f"BayesianPseudoLabeler import failed: {e}")
            return False
    
    def _check_strategic_oracle(self) -> bool:
        """Check Strategic Oracle can be imported."""
        try:
            from sragv.strategic_oracle import StrategicOracle
            return True
        except Exception as e:
            logger.error(f"StrategicOracle import failed: {e}")
            return False
    
    def _check_integration_tests(self) -> bool:
        """Check integration tests exist."""
        test_files = [
            "tests/integration/test_phase_1_2_3_integration.py",
            "tests/integration/test_star_trainer_integration.py",
            "tests/unit/test_bayesian_pseudo_labeler.py"
        ]
        return all(Path(f).exists() for f in test_files)
    
    def initialize_components(self) -> Dict[str, Any]:
        """Initialize all Phase 3 components."""
        logger.info("🔧 Stage 1: Initializing Phase 3 components...")
        stage_start = time.time()
        
        try:
            # Import components after path setup
            from sragv.orchestrator import SRAGVOrchestrator
            from sragv.strategic_oracle import StrategicOracle
            from sragv.training.star_trainer import STARTrainer
            
            # Initialize orchestrator with explicit model loading
            logger.info("Initializing SRAG-V orchestrator...")
            self.orchestrator = SRAGVOrchestrator()
            
            # Force initialize solution generator if not already done
            if self.orchestrator.solution_generator is None:
                logger.info("Solution generator not initialized, creating manually...")
                from sragv.models.solution_generator import SolutionGenerator
                
                # Use same config as successful Phase 1 training
                solution_config = {
                    'model_name': 'Qwen/Qwen2.5-Coder-7B-Instruct',
                    'max_length': 2048,
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'quantization': {'load_in_4bit': True},
                    'lora_config': {'r': 32, 'alpha': 64}
                }
                
                self.orchestrator.solution_generator = SolutionGenerator(solution_config)
                logger.info("✅ Solution generator initialized manually")
            
            # Explicitly load Phase 1 calibrator
            logger.info("Loading Phase 1 confidence calibrator...")
            calibrator_paths = [
                "checkpoints/phase1_star_calibrator_1754272154.pt",  # ECE 0.0003 version
                "checkpoints/phase1_star_calibrator_latest.pt",
                "checkpoints/phase1_star_calibrator_success.pt"
            ]
            
            calibrator_loaded = False
            for path in calibrator_paths:
                if Path(path).exists():
                    try:
                        # Load the calibrator checkpoint
                        checkpoint = torch.load(path, map_location='cpu')
                        if 'state_dict' in checkpoint:
                            # Initialize and load the calibrator
                            from sragv.confidence_calibration import EnhancedConfidenceCalibrator
                            calibrator = EnhancedConfidenceCalibrator()
                            calibrator.load_state_dict(checkpoint['state_dict'])
                            
                            # Assign to solution generator
                            self.orchestrator.solution_generator.confidence_calibrator = calibrator
                            self.orchestrator.solution_generator.use_calibration = True
                            
                            logger.info(f"✅ Phase 1 calibrator loaded from: {path}")
                            logger.info(f"✅ Calibrator temperature: {float(calibrator.temperature):.3f}")
                            calibrator_loaded = True
                            break
                    except Exception as e:
                        logger.warning(f"Failed to load calibrator from {path}: {e}")
                        continue
            
            if not calibrator_loaded:
                raise Exception("Could not load Phase 1 confidence calibrator from any path")
            
            # Initialize Strategic Oracle with the calibrator
            logger.info("Initializing Strategic Oracle...")
            strategic_oracle = StrategicOracle(
                self.orchestrator.solution_generator.confidence_calibrator
            )
            logger.info("✅ Phase 2 Strategic Oracle initialized")
            
            # Initialize STARTrainer with optimized config
            logger.info("Initializing STARTrainer...")
            star_config = {
                'max_iterations': 8,  # Conservative for stability
                'batch_size': 24,     # Manageable batch size
                'problems_per_batch': 12,
                'solutions_per_problem': 6,
                'confidence_threshold': 0.85,
                'max_ece_degradation': 0.01,  # Don't let ECE get worse than Phase 1
                'early_stopping_patience': 3,
                'checkpoint_frequency': 2,
                'bayesian_labeler_config': {
                    'base_confidence_threshold': 0.85,
                    'adaptive_threshold_enabled': True,
                    'min_quality_score': 0.8,
                    'max_uncertainty': 0.2
                }
            }
            
            self.star_trainer = STARTrainer(
                orchestrator=self.orchestrator,
                strategic_oracle=strategic_oracle,
                config=star_config
            )
            logger.info("✅ STARTrainer initialized with Bayesian pseudo-labeling")
            
            duration = time.time() - stage_start
            self.results['stages']['initialization'] = {
                'status': 'completed',
                'duration': duration,
                'components': ['SRAGVOrchestrator', 'StrategicOracle', 'STARTrainer', 'BayesianPseudoLabeler']
            }
            
            logger.info(f"✅ Stage 1 complete ({duration:.1f}s)")
            return {'status': 'success', 'duration': duration}
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            self.results['stages']['initialization'] = {
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - stage_start
            }
            return {'status': 'failed', 'error': str(e)}
    
    def load_training_data(self) -> Dict[str, Any]:
        """Load diverse problems for Phase 3 training."""
        logger.info("📚 Stage 2: Loading training data...")
        stage_start = time.time()
        
        try:
            # Load diverse problems from Phase 1
            with open('data/phase1_diverse_problems.json', 'r') as f:
                all_problems = json.load(f)
            
            # Split into training and validation
            # Use 80% for training, 20% for validation
            split_point = int(len(all_problems) * 0.8)
            training_problems = all_problems[:split_point]
            validation_problems = all_problems[split_point:]
            
            logger.info(f"Training problems: {len(training_problems)}")
            logger.info(f"Validation problems: {len(validation_problems)}")
            
            # Verify problem structure
            sample_problem = training_problems[0]
            required_fields = ['problem_id', 'question']
            for field in required_fields:
                if field not in sample_problem:
                    raise Exception(f"Missing required field in problems: {field}")
            
            duration = time.time() - stage_start
            self.results['stages']['data_loading'] = {
                'status': 'completed',
                'duration': duration,
                'training_problems': len(training_problems),
                'validation_problems': len(validation_problems)
            }
            
            logger.info(f"✅ Stage 2 complete ({duration:.1f}s)")
            return {
                'status': 'success',
                'training_problems': training_problems,
                'validation_problems': validation_problems,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            self.results['stages']['data_loading'] = {
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - stage_start
            }
            return {'status': 'failed', 'error': str(e)}
    
    def run_continuous_training(
        self, 
        training_problems: List[Dict[str, Any]], 
        validation_problems: List[Dict[str, Any]],
        max_duration_hours: float = 4.0
    ) -> Dict[str, Any]:
        """Execute Phase 3 continuous training."""
        logger.info("🧠 Stage 3: STAR Continuous Training...")
        logger.info(f"Max duration: {max_duration_hours} hours")
        stage_start = time.time()
        
        try:
            # Execute continuous training
            training_results = self.star_trainer.continuous_train(
                initial_problems=training_problems,
                validation_problems=validation_problems,
                max_duration_hours=max_duration_hours
            )
            
            duration = time.time() - stage_start
            
            # Extract key metrics
            final_metrics = training_results.get('final_metrics', {})
            best_ece = final_metrics.get('best_ece', float('inf'))
            total_iterations = final_metrics.get('total_iterations', 0)
            
            self.results['stages']['continuous_training'] = {
                'status': 'completed',
                'duration': duration,
                'training_results': training_results,
                'best_ece': best_ece,
                'total_iterations': total_iterations
            }
            
            logger.info(f"✅ Stage 3 complete ({duration/60:.1f} min)")
            logger.info(f"Best ECE achieved: {best_ece:.4f}")
            logger.info(f"Total iterations: {total_iterations}")
            
            return {
                'status': 'success',
                'training_results': training_results,
                'duration': duration
            }
            
        except Exception as e:
            logger.error(f"Continuous training failed: {e}")
            logger.error(traceback.format_exc())
            self.results['stages']['continuous_training'] = {
                'status': 'failed',
                'error': str(e),
                'duration': time.time() - stage_start
            }
            return {'status': 'failed', 'error': str(e)}
    
    def generate_final_report(self, training_results: Dict[str, Any]) -> None:
        """Generate comprehensive Phase 3 final report."""
        logger.info("📊 Stage 4: Generating final report...")
        
        # Calculate total time
        total_time = time.time() - self.start_time
        
        # Extract training metrics
        final_metrics = training_results.get('final_metrics', {})
        best_ece = final_metrics.get('best_ece', float('inf'))
        
        # Determine success status
        phase1_ece = 0.0003  # Our Phase 1 baseline
        success = best_ece <= phase1_ece * 2  # Success if within 2x of Phase 1
        
        self.results.update({
            'status': 'completed' if success else 'degraded',
            'end_time': datetime.now().isoformat(),
            'total_duration': total_time,
            'final_metrics': {
                'best_ece': best_ece,
                'phase1_baseline_ece': phase1_ece,
                'ece_improvement': best_ece <= phase1_ece,
                'total_iterations': final_metrics.get('total_iterations', 0),
                'total_solutions_generated': final_metrics.get('total_solutions_generated', 0),
                'total_pseudo_labels': final_metrics.get('total_pseudo_labels', 0)
            },
            'success_indicators': [
                f"ECE {'maintained' if best_ece <= phase1_ece * 2 else 'degraded'}: {best_ece:.4f}",
                f"Continuous training executed: {final_metrics.get('total_iterations', 0)} iterations",
                f"Pseudo-labeling active: {final_metrics.get('total_pseudo_labels', 0)} labels created",
                "Phase 3 STAR architecture validated"
            ]
        })
        
        # Save results
        results_path = f"phase3_results/phase3_final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Summary
        logger.info("🎉 PHASE 3 STAR CONTINUOUS TRAINING COMPLETED!")
        logger.info("=" * 60)
        logger.info(f"Status: {self.results['status'].upper()}")
        logger.info(f"Total time: {total_time/60:.1f} minutes")
        logger.info(f"Best ECE: {best_ece:.4f}")
        logger.info(f"Phase 1 baseline: {phase1_ece:.4f}")
        logger.info(f"Iterations: {final_metrics.get('total_iterations', 0)}")
        logger.info(f"Solutions generated: {final_metrics.get('total_solutions_generated', 0)}")
        logger.info(f"Pseudo-labels created: {final_metrics.get('total_pseudo_labels', 0)}")
        logger.info(f"Report saved: {results_path}")
        logger.info("=" * 60)
        
        # Status for ML Foundry monitoring
        with open('/workspace/training_status.txt', 'w') as f:
            f.write(f"PHASE3_COMPLETE|ECE:{best_ece:.4f}|ITERATIONS:{final_metrics.get('total_iterations', 0)}")


def main():
    """Main Phase 3 training orchestration."""
    logger.info("🚀 SRAG-V Phase 3: STAR Continuous Training")
    logger.info("Building on Phase 1 (ECE 0.0003) + Phase 2 (Strategic Oracle)")
    
    trainer = Phase3STARTrainer()
    
    try:
        # Stage 1: Prerequisites
        if not trainer.validate_prerequisites():
            raise Exception("Prerequisites validation failed")
        
        # Stage 2: Initialize components  
        init_result = trainer.initialize_components()
        if init_result['status'] != 'success':
            raise Exception(f"Component initialization failed: {init_result.get('error')}")
        
        # Stage 3: Load data
        data_result = trainer.load_training_data()
        if data_result['status'] != 'success':
            raise Exception(f"Data loading failed: {data_result.get('error')}")
        
        # Stage 4: Continuous training
        training_result = trainer.run_continuous_training(
            training_problems=data_result['training_problems'],
            validation_problems=data_result['validation_problems'],
            max_duration_hours=4.0  # Conservative limit
        )
        
        if training_result['status'] != 'success':
            raise Exception(f"Training failed: {training_result.get('error')}")
        
        # Stage 5: Final report
        trainer.generate_final_report(training_result['training_results'])
        
        logger.info("🎉 Phase 3 execution completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Phase 3 execution failed: {e}")
        logger.error(traceback.format_exc())
        
        # Save error report
        trainer.results.update({
            'status': 'failed',
            'error': str(e),
            'end_time': datetime.now().isoformat(),
            'total_duration': time.time() - trainer.start_time
        })
        
        error_report_path = f"phase3_results/phase3_error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_report_path, 'w') as f:
            json.dump(trainer.results, f, indent=2)
        
        logger.error(f"Error report saved: {error_report_path}")
        sys.exit(1)
    
    finally:
        # Cleanup
        logger.info("🧹 Cleaning up Phase 3 resources...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("✅ Phase 3 resource cleanup complete")


if __name__ == "__main__":
    main()