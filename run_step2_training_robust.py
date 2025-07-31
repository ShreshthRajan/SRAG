#!/usr/bin/env python3
"""
SRAG-V Step 2: Robust Core Training Loop Implementation
Fixed with SOTA July 2025 techniques for stable code generation.
"""

import os
import sys
import logging
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragv.orchestrator import SRAGVOrchestrator
from sragv.training.self_play_trainer import SelfPlayTrainer
from sragv.training.grpo_trainer import GRPOTrainer, GRPOConfig
from sragv.training.map_elites import MAPElites, MAPElitesConfig
from sragv.training.reward_system import RoleConditionedRewardSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/step2_training_robust.log", mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class RobustStep2Trainer:
    """Robust Step 2 trainer with improved error handling and monitoring."""
    
    def __init__(self):
        self.orchestrator = None
        self.trainer = None
        self.start_time = None
        self.iteration_start_time = None
        self.performance_metrics = []
        
    def validate_prerequisites(self) -> bool:
        """Comprehensive prerequisite validation."""
        logger.info("🔍 Validating Step 2 prerequisites...")
        
        checks = {
            "Configuration file": Path("config/config.yaml").exists(),
            "Expanded dataset": Path("data/expanded_apps.json").exists(),
            "Source: src/sragv/__init__.py": Path("src/sragv/__init__.py").exists(),
            "Source: src/sragv/orchestrator.py": Path("src/sragv/orchestrator.py").exists(),
            "Source: src/sragv/models/__init__.py": Path("src/sragv/models/__init__.py").exists(),
            "Source: src/sragv/training/grpo_trainer.py": Path("src/sragv/training/grpo_trainer.py").exists(),
            "Source: src/sragv/training/map_elites.py": Path("src/sragv/training/map_elites.py").exists(),
            "Source: src/sragv/training/reward_system.py": Path("src/sragv/training/reward_system.py").exists(),
            "Source: src/sragv/training/self_play_trainer.py": Path("src/sragv/training/self_play_trainer.py").exists(),
        }
        
        # Additional dataset size check
        if checks["Expanded dataset"]:
            try:
                with open("data/expanded_apps.json", 'r') as f:
                    data = json.load(f)
                    dataset_size = len(data)
                    checks["Dataset size (≥2000)"] = dataset_size >= 2000
                    logger.info(f"Dataset contains {dataset_size} problems")
            except Exception as e:
                logger.error(f"Failed to load dataset: {e}")
                checks["Dataset size (≥2000)"] = False
        
        # Cache check for models
        qwen_models = [
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-Coder-7B-Instruct", 
            "Qwen/Qwen2.5-0.5B-Instruct"
        ]
        
        cached_models = 0
        for model_name in qwen_models:
            cache_path = Path.home() / ".cache/huggingface/hub" / f"models--{model_name.replace('/', '--')}"
            if cache_path.exists():
                cached_models += 1
        
        checks["Qwen models cached"] = cached_models >= 2  # At least 2 models cached
        
        # Report results
        passed = 0
        total = len(checks)
        
        logger.info("Prerequisites validation:")
        for check, status in checks.items():
            status_symbol = "✅ PASS" if status else "❌ FAIL"
            logger.info(f"  {check:<40} {status_symbol}")
            if status:
                passed += 1
        
        success_rate = (passed / total) * 100
        logger.info(f"Prerequisites: {passed}/{total} checks passed ({success_rate:.1f}%)")
        
        if success_rate == 100.0:
            logger.info("✅ Prerequisites validation successful")
            return True
        else:
            logger.error(f"❌ Prerequisites validation failed ({total - passed} failures)")
            return False
    
    def initialize_training_system(self) -> bool:
        """Initialize the SRAG-V training system with robust error handling."""
        try:
            logger.info("🚀 Initializing SRAG-V training system...")
            
            # Initialize orchestrator
            logger.info("  Initializing SRAG-V orchestrator...")
            self.orchestrator = SRAGVOrchestrator(config_path="config/config.yaml")
            logger.info("  ✅ SRAG-V orchestrator initialized")
            
            # Initialize self-play trainer
            logger.info("  Initializing self-play trainer...")
            self.trainer = SelfPlayTrainer(
                orchestrator=self.orchestrator,
                num_iterations=12,
                problems_per_iteration=32,
                solutions_per_problem=14,
                enable_emergent_tracking=True
            )
            logger.info("  ✅ Self-play trainer initialized")
            
            logger.info("✅ Training system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize training system: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_training_config(self) -> Dict:
        """Create optimized training configuration."""
        config = {
            "total_iterations": 12,
            "problems_per_iteration": 32,
            "solutions_per_problem": 14,
            "emergent_behavior_tracking": True,
            "enable_checkpointing": True,
            "checkpoint_interval": 2,  # Save every 2 iterations
            "enable_detailed_logging": True,
            "enable_performance_monitoring": True
        }
        
        logger.info("📋 Training configuration:")
        for key, value in config.items():
            logger.info(f"  - {key.replace('_', ' ').title()}: {value}")
        
        return config
    
    def monitor_iteration_progress(self, iteration: int, total_iterations: int, 
                                 solutions_generated: int, solutions_validated: int,
                                 avg_score: float, success_rate: float) -> Dict:
        """Monitor and log iteration progress with detailed metrics."""
        
        elapsed_time = time.time() - self.iteration_start_time if self.iteration_start_time else 0
        total_elapsed = time.time() - self.start_time if self.start_time else 0
        
        progress_info = {
            "iteration": iteration,
            "total_iterations": total_iterations,
            "progress_percent": (iteration / total_iterations) * 100,
            "solutions_generated": solutions_generated,
            "solutions_validated": solutions_validated,
            "avg_score": avg_score,
            "success_rate": success_rate,
            "iteration_duration_minutes": elapsed_time / 60,
            "total_duration_hours": total_elapsed / 3600,
            "estimated_remaining_hours": ((total_elapsed / iteration) * (total_iterations - iteration)) / 3600 if iteration > 0 else 0
        }
        
        # Log progress
        logger.info(f"📊 ITERATION {iteration}/{total_iterations} PROGRESS:")
        logger.info(f"  🎯 Progress: {progress_info['progress_percent']:.1f}%")
        logger.info(f"  🔧 Solutions: {solutions_generated} generated, {solutions_validated} validated")
        logger.info(f"  📈 Success Rate: {success_rate:.1f}%")
        logger.info(f"  ⭐ Average Score: {avg_score:.3f}")
        logger.info(f"  ⏱️  Duration: {elapsed_time/60:.1f} minutes")
        logger.info(f"  🕐 ETA: {progress_info['estimated_remaining_hours']:.1f} hours remaining")
        
        # Store metrics
        self.performance_metrics.append(progress_info)
        
        return progress_info
    
    def save_checkpoint(self, iteration: int, additional_data: Dict = None) -> bool:
        """Save training checkpoint with full state."""
        try:
            checkpoint_dir = Path("checkpoints/step2_training_robust")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            checkpoint_data = {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "total_training_time_hours": (time.time() - self.start_time) / 3600 if self.start_time else 0,
                "performance_metrics": self.performance_metrics,
                "additional_data": additional_data or {}
            }
            
            checkpoint_file = checkpoint_dir / f"checkpoint_iteration_{iteration}.json"
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.info(f"💾 Checkpoint saved: {checkpoint_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            return False
    
    def run_robust_training(self) -> bool:
        """Run the robust Step 2 training with comprehensive monitoring."""
        logger.info("🚀 SRAG-V STEP 2: ROBUST CORE TRAINING LOOP IMPLEMENTATION")
        logger.info("=" * 80)
        logger.info("Implementing breakthrough-level self-play training with:")
        logger.info("  ✓ GRPO (Group Relative Policy Optimization) - July 2025 SOTA")
        logger.info("  ✓ MAP-Elites diversity mechanism")
        logger.info("  ✓ Role-conditioned reward system")
        logger.info("  ✓ Emergent behavior tracking")
        logger.info("  ✓ Improved solution generator with Qwen2.5-Coder chat templates")
        logger.info("  ✓ Bulletproof error handling and recovery")
        logger.info("=" * 80)
        
        self.start_time = time.time()
        
        # Step 1: Validate prerequisites
        if not self.validate_prerequisites():
            logger.error("❌ Prerequisites validation failed. Aborting training.")
            return False
        
        # Step 2: Create training configuration
        config = self.create_training_config()
        
        # Step 3: Initialize training system
        if not self.initialize_training_system():
            logger.error("❌ Training system initialization failed. Aborting training.")
            return False
        
        # Step 4: Launch monitored training
        logger.info("🎯 Starting monitored training execution...")
        
        try:
            success = self.trainer.train(
                progress_callback=self.monitor_iteration_progress,
                checkpoint_callback=self.save_checkpoint,
                error_callback=self._handle_training_error
            )
            
            if success:
                total_time = (time.time() - self.start_time) / 3600
                logger.info("🎉 STEP 2 TRAINING COMPLETED SUCCESSFULLY!")
                logger.info(f"📊 Total training time: {total_time:.2f} hours")
                logger.info(f"📈 Final performance metrics: {len(self.performance_metrics)} iterations completed")
                return True
            else:
                logger.error("❌ Training failed to complete successfully")
                return False
                
        except KeyboardInterrupt:
            logger.info("⌨️ Training interrupted by user")
            self.save_checkpoint(len(self.performance_metrics), {"status": "interrupted"})
            return False
        except Exception as e:
            logger.error(f"💥 Unexpected training error: {e}")
            import traceback
            traceback.print_exc()
            self.save_checkpoint(len(self.performance_metrics), {"status": "error", "error": str(e)})
            return False
    
    def _handle_training_error(self, error: Exception, iteration: int, context: Dict) -> bool:
        """Handle training errors with recovery logic."""
        logger.error(f"🚨 Training error in iteration {iteration}: {error}")
        
        # Save error checkpoint
        self.save_checkpoint(iteration, {
            "status": "error", 
            "error": str(error),
            "context": context
        })
        
        # Implement recovery logic based on error type
        if "out of memory" in str(error).lower() or "cuda" in str(error).lower():
            logger.warning("💾 Memory error detected - implementing memory recovery")
            # Clear GPU cache
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return True  # Retry
        elif "syntax error" in str(error).lower():
            logger.warning("🔧 Syntax error detected - this should be handled by improved solution generator")
            return True  # Continue with improved error handling
        else:
            logger.error("❌ Unrecoverable error - stopping training")
            return False  # Stop training

def main():
    """Main entry point for robust Step 2 training."""
    # Ensure required directories exist
    Path("logs").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Set up environment
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "8"
    
    # Run training
    trainer = RobustStep2Trainer()
    success = trainer.run_robust_training()
    
    if success:
        logger.info("🎉 ROBUST STEP 2 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("Ready to proceed to Step 3: Peak Performance Training")
        sys.exit(0)
    else:
        logger.error("❌ ROBUST STEP 2 TRAINING FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()