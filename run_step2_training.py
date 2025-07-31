#!/usr/bin/env python3
"""
BULLETPROOF SRAG-V Step 2: Core Training Loop Implementation
Complete integration of GRPO, MAP-Elites, and 4-player self-play training.

This script implements the breakthrough-level self-play training system
with emergent verification learning capabilities and bulletproof error handling.
"""

import logging
import sys
import os
import time
import signal
import psutil
import gc
from pathlib import Path
import json
import traceback
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import with bulletproof error handling
try:
    from sragv.orchestrator import SRAGVOrchestrator
    from sragv.training.self_play_trainer import SelfPlayTrainer, SelfPlayConfig
    from sragv.training.grpo_trainer import GRPOConfig
    from sragv.training.map_elites import MAPElitesConfig
    logger_import_success = True
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR: {e}")
    logger_import_success = False

# Create comprehensive directory structure
for dir_name in ["logs", "checkpoints", "monitoring", "artifacts"]:
    Path(dir_name).mkdir(exist_ok=True)

# Setup bulletproof logging with multiple outputs
log_filename = f"logs/step2_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.DEBUG,  # More verbose for debugging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_filename),
        logging.FileHandler('logs/step2_training_latest.log')  # Always latest
    ]
)
logger = logging.getLogger(__name__)

# Global training state for graceful shutdown
training_state = {
    "trainer": None,
    "should_stop": False,
    "current_iteration": 0,
    "last_checkpoint": None
}

def signal_handler(signum, frame):
    """Handle graceful shutdown on signals."""
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    training_state["should_stop"] = True
    
    if training_state["trainer"]:
        try:
            logger.info("💾 Creating emergency checkpoint...")
            training_state["trainer"].save_checkpoint(f"checkpoints/emergency_checkpoint_iter_{training_state['current_iteration']}.pt")
        except Exception as e:
            logger.error(f"❌ Emergency checkpoint failed: {e}")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def log_system_info():
    """Log comprehensive system information."""
    logger.info("🖥️ SYSTEM INFORMATION:")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  PID: {os.getpid()}")
    logger.info(f"  Working Directory: {os.getcwd()}")
    
    try:
        import torch
        logger.info(f"  PyTorch: {torch.__version__}")
        logger.info(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    except Exception as e:
        logger.error(f"❌ Failed to get PyTorch info: {e}")
    
    # System resources
    try:
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        logger.info(f"  CPU Cores: {cpu_count}")
        logger.info(f"  Memory: {memory.total / 1e9:.1f} GB total, {memory.available / 1e9:.1f} GB available")
        logger.info(f"  Disk: {disk.total / 1e9:.1f} GB total, {disk.free / 1e9:.1f} GB free")
    except Exception as e:
        logger.error(f"❌ Failed to get system info: {e}")


def validate_prerequisites():
    """Validate that all Step 1 components are ready."""
    logger.info("🔍 Validating Step 2 prerequisites...")
    
    checks = []
    
    # Check configuration
    config_path = Path("config/config.yaml")
    checks.append(("Configuration file", config_path.exists()))
    
    # Check expanded dataset
    dataset_path = Path("data/expanded_apps.json")
    checks.append(("Expanded dataset", dataset_path.exists()))
    
    if dataset_path.exists():
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        checks.append(("Dataset size (≥2000)", len(data) >= 2000))
    
    # Check model cache
    cache_dir = Path.home() / ".cache/huggingface/hub"
    qwen_models = list(cache_dir.glob("models--Qwen--Qwen2.5*"))
    checks.append(("Qwen models cached", len(qwen_models) >= 3))
    
    # Check src directory structure
    src_checks = [
        "src/sragv/__init__.py",
        "src/sragv/orchestrator.py",
        "src/sragv/models/__init__.py",
        "src/sragv/training/grpo_trainer.py",
        "src/sragv/training/map_elites.py",
        "src/sragv/training/reward_system.py",
        "src/sragv/training/self_play_trainer.py"
    ]
    
    for check_path in src_checks:
        checks.append((f"Source: {check_path}", Path(check_path).exists()))
    
    # Report results
    logger.info("Prerequisites validation:")
    passed = 0
    for check_name, passed_check in checks:
        status = "✅ PASS" if passed_check else "❌ FAIL"
        logger.info(f"  {check_name:<40} {status}")
        if passed_check:
            passed += 1
    
    success_rate = passed / len(checks)
    logger.info(f"Prerequisites: {passed}/{len(checks)} checks passed ({success_rate*100:.1f}%)")
    
    if success_rate < 0.8:
        logger.error("❌ Insufficient prerequisites for Step 2 training")
        return False
    
    logger.info("✅ Prerequisites validation successful")
    return True


def create_optimized_training_config() -> SelfPlayConfig:
    """Create optimized training configuration based on research."""
    logger.info("📋 Creating optimized training configuration...")
    
    config = SelfPlayConfig(
        # Training schedule (12 iterations as per research)
        num_iterations=12,
        bootstrap_iterations=3,
        competitive_iterations=6,
        league_iterations=3,
        
        # Problem generation (research-validated optimization)
        problems_per_iteration=32,  # Maintains statistical significance (July 2025 research)
        dynamic_problem_generation=True,
        problem_difficulty_scaling=True,
        
        # Self-play dynamics (conservative optimization)
        solutions_per_problem=14,  # 12% reduction - maintains MAP-Elites diversity
        test_cases_per_problem=8,   # Full verification coverage (safety requirement)
        archive_sampling_probability=0.3,
        
        # Training optimization (memory and speed)
        gradient_accumulation_steps=4,
        mixed_precision=True,
        checkpoint_every_iterations=2,
        
        # Emergent behavior tracking (breakthrough detection)
        track_emergent_behaviors=True,
        behavior_analysis_frequency=2,  # Frequent analysis
        novel_pattern_threshold=0.8,
        
        # Quality gates (ensure robust training)
        min_solution_accuracy=0.6,  # Reasonable threshold
        min_test_validity=0.5,      # Allow learning from mistakes
        convergence_threshold=0.05  # Detect convergence
    )
    
    logger.info(f"Training configuration:")
    logger.info(f"  - {config.num_iterations} total iterations")
    logger.info(f"  - {config.problems_per_iteration} problems per iteration")
    logger.info(f"  - {config.solutions_per_problem} solutions per problem")
    logger.info(f"  - Emergent behavior tracking: {config.track_emergent_behaviors}")
    
    return config


def initialize_training_system(config: SelfPlayConfig) -> tuple:
    """Initialize the complete training system."""
    logger.info("🚀 Initializing SRAG-V training system...")
    
    # Initialize orchestrator
    logger.info("  Initializing SRAG-V orchestrator...")
    orchestrator = SRAGVOrchestrator("config/config.yaml")
    
    # Initialize self-play trainer
    logger.info("  Initializing self-play trainer...")
    trainer = SelfPlayTrainer(
        orchestrator=orchestrator,
        config=config,
        checkpoint_dir="checkpoints/step2_training"
    )
    
    logger.info("✅ Training system initialized successfully")
    return orchestrator, trainer


def run_training_with_monitoring(trainer: SelfPlayTrainer) -> dict:
    """Run training with comprehensive monitoring and error handling."""
    logger.info("🎯 Starting monitored training execution...")
    
    start_time = time.time()
    
    try:
        # Run the complete training
        results = trainer.run_self_play_training()
        
        training_time = time.time() - start_time
        results["total_training_time"] = training_time
        
        # Log final results
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Total training time: {training_time:.1f}s ({training_time/60:.1f}min)")
        
        # Extract key metrics
        if "breakthrough_indicators" in results:
            breakthrough = results["breakthrough_indicators"]
            breakthrough_score = breakthrough.get("breakthrough_score", 0.0)
            logger.info(f"Breakthrough score: {breakthrough_score:.3f}/1.0")
            
            if breakthrough_score > 0.75:
                logger.info("🏆 BREAKTHROUGH ACHIEVED! Novel verification behaviors detected!")
            elif breakthrough_score > 0.5:
                logger.info("🌟 Significant emergent behaviors observed")
            else:
                logger.info("📈 Training successful, baseline behaviors established")
        
        if "final_archive_stats" in results:
            archive_stats = results["final_archive_stats"]
            logger.info(f"Final archive: {archive_stats.get('total_elites', 0)} elites, "
                       f"{archive_stats.get('occupied_niches', 0)} niches")
        
        return results
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        # Save partial results
        results = {
            "status": "interrupted",
            "training_time": time.time() - start_time,
            "error": "User interrupted"
        }
        return results
        
    except Exception as e:
        logger.error(f"💥 Training failed: {e}")
        logger.error("Full traceback:")
        for line in traceback.format_exc().splitlines():
            logger.error(f"  {line}")
        
        # Save error results
        results = {
            "status": "failed",
            "training_time": time.time() - start_time,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        return results


def analyze_and_report_results(results: dict):
    """Analyze and report comprehensive training results."""
    logger.info("📊 Analyzing training results...")
    
    status = results.get("status", "completed")
    training_time = results.get("total_training_time", 0)
    
    logger.info("=" * 80)
    logger.info("SRAG-V STEP 2 TRAINING RESULTS")
    logger.info("=" * 80)
    
    # Basic metrics
    logger.info(f"Status: {status.upper()}")
    logger.info(f"Training time: {training_time:.1f}s ({training_time/60:.1f}min)")
    
    if status == "completed":
        # Iteration results
        iterations = results.get("iterations", [])
        if iterations:
            logger.info(f"Iterations completed: {len(iterations)}")
            avg_quality = sum(it.get("quality_score", 0) for it in iterations) / len(iterations)
            logger.info(f"Average quality score: {avg_quality:.3f}")
        
        # Convergence analysis
        convergence = results.get("convergence_analysis", {})
        if convergence.get("converged", False):
            logger.info(f"✅ Training converged in {convergence.get('iterations_to_convergence', 'N/A')} iterations")
        else:
            logger.info("⏳ Training completed without full convergence")
        
        # Emergent behaviors
        behaviors = results.get("emergent_behaviors", {})
        novel_patterns = behaviors.get("novel_patterns_discovered", 0)
        logger.info(f"Novel patterns discovered: {novel_patterns}")
        
        breakthrough = results.get("breakthrough_indicators", {})
        breakthrough_score = breakthrough.get("breakthrough_score", 0.0)
        logger.info(f"Breakthrough score: {breakthrough_score:.3f}/1.0")
        
        # Archive statistics
        archive = results.get("final_archive_stats", {})
        logger.info(f"Archive: {archive.get('total_elites', 0)} elites in {archive.get('occupied_niches', 0)} niches")
        
        # Quality assessment
        if breakthrough_score > 0.75:
            logger.info("🏆 EINSTEIN-LEVEL BREAKTHROUGH ACHIEVED!")
            logger.info("   Novel verification behaviors detected")
            logger.info("   Ready for paper submission and industry adoption")
        elif breakthrough_score > 0.5:
            logger.info("🌟 SIGNIFICANT PROGRESS ACHIEVED!")
            logger.info("   Strong emergent behaviors observed")
            logger.info("   Promising foundation for further development")
        elif avg_quality > 0.7:
            logger.info("📈 SOLID BASELINE ESTABLISHED!")
            logger.info("   Training successful, system functional")
            logger.info("   Ready for Step 3 scaling")
        else:
            logger.info("⚠️ TRAINING COMPLETED WITH ISSUES")
            logger.info("   Consider parameter tuning or additional iterations")
    
    elif status == "interrupted":
        logger.info("⏹️ Training was interrupted")
        logger.info("   Partial results may be available in checkpoints")
        
    elif status == "failed":
        error = results.get("error", "Unknown error")
        logger.info(f"💥 Training failed: {error}")
        logger.info("   Check logs for detailed error information")
    
    logger.info("=" * 80)
    
    # Save detailed report
    report_path = Path("logs/step2_final_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"📋 Detailed report saved to: {report_path}")


def cleanup_resources():
    """Clean up resources and memory."""
    logger.info("🧹 Cleaning up resources...")
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("  ✅ CUDA cache cleared")
        except:
            pass
        
        # Clear training state
        training_state["trainer"] = None
        training_state["should_stop"] = False
        
        logger.info("  ✅ Resource cleanup complete")
    except Exception as e:
        logger.error(f"  ❌ Cleanup error: {e}")

def main():
    """BULLETPROOF main execution function for Step 2 training."""
    start_time = time.time()
    exit_code = 1  # Default to failure
    
    try:
        logger.info("🛡️ BULLETPROOF SRAG-V STEP 2: CORE TRAINING LOOP IMPLEMENTATION")
        logger.info("=" * 80)
        logger.info("Implementing breakthrough-level self-play training with:")
        logger.info("  ✓ GRPO (Group Relative Policy Optimization)")
        logger.info("  ✓ MAP-Elites diversity mechanism")
        logger.info("  ✓ Role-conditioned reward system")
        logger.info("  ✓ Emergent behavior tracking")
        logger.info("  ✓ 12-iteration self-play protocol")
        logger.info("  ✓ Bulletproof error handling & recovery")
        logger.info("  ✓ Comprehensive monitoring & logging")
        logger.info("=" * 80)
        
        # Log system information
        log_system_info()
        
        # Check import success
        if not logger_import_success:
            logger.error("❌ Critical imports failed - cannot proceed")
            with open('/workspace/training_status.txt', 'w') as f:
                f.write('FAILED:IMPORT_ERROR')
            return 1
        
        # Step 1: Validate prerequisites with bulletproof error handling
        logger.info("🔍 Step 1: Prerequisites Validation")
        try:
            if not validate_prerequisites():
                logger.error("❌ Prerequisites validation failed")
                with open('/workspace/training_status.txt', 'w') as f:
                    f.write('FAILED:PREREQUISITES')
                return 1
            logger.info("✅ Prerequisites validated successfully")
        except Exception as e:
            logger.error(f"❌ Prerequisites validation error: {e}")
            with open('/workspace/training_status.txt', 'w') as f:
                f.write(f'FAILED:PREREQUISITES_ERROR:{str(e)[:100]}')
            return 1
        
        # Step 2: Create training configuration
        logger.info("⚙️ Step 2: Configuration Creation")
        try:
            config = create_optimized_training_config()
            logger.info("✅ Training configuration created successfully")
        except Exception as e:
            logger.error(f"❌ Configuration creation failed: {e}")
            with open('/workspace/training_status.txt', 'w') as f:
                f.write(f'FAILED:CONFIG_ERROR:{str(e)[:100]}')
            return 1
        
        # Step 3: Initialize training system
        logger.info("🚀 Step 3: Training System Initialization")
        try:
            orchestrator, trainer = initialize_training_system(config)
            training_state["trainer"] = trainer  # Store for signal handler
            logger.info("✅ Training system initialized successfully")
        except Exception as e:
            logger.error(f"❌ Training system initialization failed: {e}")
            logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                logger.error(f"  {line}")
            with open('/workspace/training_status.txt', 'w') as f:
                f.write(f'FAILED:INIT_ERROR:{str(e)[:100]}')
            return 1
        
        # Step 4: Run training with comprehensive monitoring
        logger.info("🎯 Step 4: Bulletproof Training Execution")
        try:
            # Update status to running
            with open('/workspace/training_status.txt', 'w') as f:
                f.write('RUNNING')
            
            results = run_training_with_monitoring(trainer)
            logger.info("✅ Training execution completed")
        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
            results = {"status": "interrupted", "error": "User interrupt"}
        except Exception as e:
            logger.error(f"❌ Training execution failed: {e}")
            logger.error("Full traceback:")
            for line in traceback.format_exc().splitlines():
                logger.error(f"  {line}")
            results = {"status": "failed", "error": str(e)}
        
        # Step 5: Analyze and report results
        logger.info("📊 Step 5: Results Analysis and Reporting")
        try:
            analyze_and_report_results(results)
            logger.info("✅ Results analysis completed")
        except Exception as e:
            logger.error(f"❌ Results analysis failed: {e}")
            # Don't fail the entire training for reporting errors
        
        # Determine final status and exit code
        status = results.get("status", "completed")
        if status == "completed":
            breakthrough_score = results.get("breakthrough_indicators", {}).get("breakthrough_score", 0.0)
            if breakthrough_score > 0.5:
                logger.info("🎉 STEP 2 SUCCESSFULLY COMPLETED WITH BREAKTHROUGH POTENTIAL!")
                with open('/workspace/training_status.txt', 'w') as f:
                    f.write('SUCCESS:BREAKTHROUGH')
                exit_code = 0
            else:
                logger.info("✅ STEP 2 COMPLETED - READY FOR STEP 3")
                with open('/workspace/training_status.txt', 'w') as f:
                    f.write('SUCCESS:COMPLETED')
                exit_code = 0
        elif status == "interrupted":
            logger.info("⏹️ STEP 2 INTERRUPTED - PARTIAL COMPLETION")
            with open('/workspace/training_status.txt', 'w') as f:
                f.write('INTERRUPTED')
            exit_code = 2
        else:
            logger.error("❌ STEP 2 FAILED")
            error_msg = results.get("error", "Unknown error")
            with open('/workspace/training_status.txt', 'w') as f:
                f.write(f'FAILED:{error_msg[:100]}')
            exit_code = 1
        
        # Log final timing
        total_time = time.time() - start_time
        logger.info(f"⏱️ Total execution time: {total_time/3600:.2f} hours")
        
    except Exception as e:
        logger.error(f"💥 CRITICAL ERROR in bulletproof main: {e}")
        logger.error("Full traceback:")
        for line in traceback.format_exc().splitlines():
            logger.error(f"  {line}")
        
        with open('/workspace/training_status.txt', 'w') as f:
            f.write(f'FAILED:CRITICAL_ERROR:{str(e)[:100]}')
        exit_code = 1
    
    finally:
        # Always clean up resources
        cleanup_resources()
        
        # Log final status
        logger.info("🏁 BULLETPROOF TRAINING SESSION COMPLETE")
        logger.info(f"📊 Final exit code: {exit_code}")
        
        if exit_code == 0:
            logger.info("🎉 Training succeeded! Ready for next steps.")
        elif exit_code == 2:
            logger.info("⏸️ Training was interrupted but may be resumable.")
        else:
            logger.info("❌ Training failed. Check logs for details.")
    
    return exit_code


if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir('/workspace/srag-training')
    
    # Run bulletproof training
    exit_code = main()
    
    # Final status update
    logger.info(f"🚪 Exiting with code {exit_code}")
    sys.exit(exit_code)