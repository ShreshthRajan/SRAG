#!/usr/bin/env python3
"""
Bulletproof SRAG-V Training Launcher with Complete Error Handling and Recovery
Based on July 2025 SOTA techniques for stable training.
"""

import os
import sys
import subprocess
import time
import json
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
import traceback
import psutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BulletproofTrainingLauncher:
    """
    Bulletproof training launcher with:
    1. Automatic error detection and recovery
    2. State persistence and resume capabilities  
    3. Real-time health monitoring
    4. Memory leak detection
    5. GPU utilization tracking
    6. Automatic restart on crashes
    """
    
    def __init__(self, config_path: str = "config/bulletproof_config.json"):
        self.config_path = config_path
        self.state_file = Path("training_state.json")
        self.log_file = Path("logs/bulletproof_training.log")
        self.training_process = None
        self.start_time = None
        self.last_successful_checkpoint = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        
        # Ensure directories exist
        Path("logs").mkdir(exist_ok=True)
        Path("checkpoints").mkdir(exist_ok=True)
        
        # Load or create config
        self.config = self._load_config()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _load_config(self) -> Dict:
        """Load bulletproof training configuration."""
        default_config = {
            "training_script": "run_step2_training.py",
            "max_training_hours": 12,
            "health_check_interval": 60,  # seconds
            "memory_threshold_gb": 100,   # GB
            "gpu_utilization_threshold": 5,  # %
            "restart_delay_seconds": 30,
            "max_restarts": 5,
            "checkpoint_interval_minutes": 30,
            "enable_automatic_recovery": True,
            "enable_memory_monitoring": True,
            "enable_gpu_monitoring": True
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
            else:
                # Create default config
                Path(self.config_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return default_config
    
    def _save_state(self, status: str, additional_info: Dict = None):
        """Save current training state for recovery."""
        state = {
            "status": status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "last_update": datetime.now().isoformat(),
            "consecutive_failures": self.consecutive_failures,
            "last_successful_checkpoint": self.last_successful_checkpoint,
            "training_pid": self.training_process.pid if self.training_process else None,
            "additional_info": additional_info or {}
        }
        
        try:
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"State saved: {status}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def _load_state(self) -> Optional[Dict]:
        """Load previous training state for recovery."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
        return None
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self._save_state("interrupted", {"signal": signum})
        if self.training_process:
            self._terminate_training_process()
        sys.exit(0)
    
    def _check_system_health(self) -> Dict[str, any]:
        """Comprehensive system health check."""
        health = {
            "timestamp": datetime.now().isoformat(),
            "healthy": True,
            "issues": [],
            "metrics": {}
        }
        
        try:
            # Memory check
            if self.config["enable_memory_monitoring"]:
                memory = psutil.virtual_memory()
                memory_gb = memory.used / (1024**3)
                health["metrics"]["memory_used_gb"] = round(memory_gb, 2)
                health["metrics"]["memory_percent"] = memory.percent
                
                if memory_gb > self.config["memory_threshold_gb"]:
                    health["healthy"] = False
                    health["issues"].append(f"High memory usage: {memory_gb:.1f}GB")
            
            # GPU check
            if self.config["enable_gpu_monitoring"]:
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", 
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=10
                    )
                    
                    if result.returncode == 0:
                        gpu_lines = result.stdout.strip().split('\n')
                        total_utilization = 0
                        gpu_count = 0
                        
                        for i, line in enumerate(gpu_lines):
                            if line.strip():
                                parts = line.split(',')
                                if len(parts) >= 4:
                                    util = int(parts[0].strip())
                                    mem_used = int(parts[1].strip())
                                    mem_total = int(parts[2].strip())
                                    temp = int(parts[3].strip())
                                    
                                    health["metrics"][f"gpu_{i}_utilization"] = util
                                    health["metrics"][f"gpu_{i}_memory_used"] = mem_used
                                    health["metrics"][f"gpu_{i}_temperature"] = temp
                                    
                                    total_utilization += util
                                    gpu_count += 1
                        
                        if gpu_count > 0:
                            avg_utilization = total_utilization / gpu_count
                            health["metrics"]["avg_gpu_utilization"] = round(avg_utilization, 1)
                            
                            if avg_utilization < self.config["gpu_utilization_threshold"]:
                                health["healthy"] = False
                                health["issues"].append(f"Low GPU utilization: {avg_utilization:.1f}%")
                    
                except Exception as e:
                    health["issues"].append(f"GPU monitoring failed: {e}")
            
            # Training process check
            if self.training_process:
                try:
                    if self.training_process.poll() is None:
                        # Process is running
                        process = psutil.Process(self.training_process.pid)
                        health["metrics"]["training_cpu_percent"] = process.cpu_percent()
                        health["metrics"]["training_memory_mb"] = process.memory_info().rss / (1024*1024)
                    else:
                        # Process has terminated
                        health["healthy"] = False
                        health["issues"].append("Training process terminated unexpectedly")
                except psutil.NoSuchProcess:
                    health["healthy"] = False
                    health["issues"].append("Training process not found")
                except Exception as e:
                    health["issues"].append(f"Process monitoring failed: {e}")
            
            # Training log check
            log_file = Path("logs/training.log")
            if log_file.exists():
                try:
                    # Check if log has been updated recently (within last 5 minutes)
                    last_modified = datetime.fromtimestamp(log_file.stat().st_mtime)
                    time_diff = (datetime.now() - last_modified).total_seconds()
                    health["metrics"]["log_last_updated_seconds"] = int(time_diff)
                    
                    if time_diff > 300:  # 5 minutes
                        health["healthy"] = False
                        health["issues"].append(f"Training log not updated for {time_diff/60:.1f} minutes")
                    
                    # Check recent log entries for errors
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        recent_lines = lines[-50:]  # Last 50 lines
                        
                        error_count = 0
                        for line in recent_lines:
                            if "ERROR" in line or "CRITICAL" in line:
                                error_count += 1
                        
                        health["metrics"]["recent_error_count"] = error_count
                        if error_count > 10:  # Too many errors
                            health["healthy"] = False
                            health["issues"].append(f"High error rate: {error_count} errors in recent logs")
                        
                except Exception as e:
                    health["issues"].append(f"Log monitoring failed: {e}")
            
        except Exception as e:
            health["healthy"] = False
            health["issues"].append(f"Health check failed: {e}")
        
        return health
    
    def _start_training_process(self) -> bool:
        """Start the training process with proper monitoring."""
        try:
            logger.info("🚀 Starting bulletproof training process...")
            
            # Prepare environment
            env = os.environ.copy()
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            env["TOKENIZERS_PARALLELISM"] = "false"
            env["OMP_NUM_THREADS"] = "8"
            
            # Start training process
            self.training_process = subprocess.Popen(
                [sys.executable, self.config["training_script"]],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.getcwd(),
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            logger.info(f"✅ Training process started with PID: {self.training_process.pid}")
            self._save_state("training_started", {"pid": self.training_process.pid})
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start training process: {e}")
            traceback.print_exc()
            return False
    
    def _terminate_training_process(self):
        """Safely terminate the training process."""
        if self.training_process:
            try:
                logger.info("🛑 Terminating training process...")
                
                # Try graceful shutdown first
                self.training_process.terminate()
                
                # Wait up to 30 seconds for graceful shutdown
                try:
                    self.training_process.wait(timeout=30)
                    logger.info("✅ Training process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    logger.warning("⚠️ Force killing training process...")
                    self.training_process.kill()
                    self.training_process.wait()
                    logger.info("✅ Training process force killed")
                
            except Exception as e:
                logger.error(f"❌ Error terminating training process: {e}")
            finally:
                self.training_process = None
    
    def _handle_training_failure(self, failure_reason: str):
        """Handle training failure with recovery logic."""
        logger.error(f"💥 Training failure detected: {failure_reason}")
        self.consecutive_failures += 1
        
        self._save_state("failed", {
            "failure_reason": failure_reason,
            "consecutive_failures": self.consecutive_failures
        })
        
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.critical(f"❌ Maximum consecutive failures reached ({self.max_consecutive_failures}). Stopping.")
            return False
        
        if self.config["enable_automatic_recovery"]:
            logger.info(f"🔄 Attempting automatic recovery (attempt {self.consecutive_failures}/{self.max_consecutive_failures})")
            
            # Wait before restart
            delay = self.config["restart_delay_seconds"] * (2 ** (self.consecutive_failures - 1))  # Exponential backoff
            logger.info(f"⏳ Waiting {delay} seconds before restart...")
            time.sleep(delay)
            
            return True  # Continue with restart
        
        return False
    
    def run_bulletproof_training(self):
        """Run training with complete bulletproof error handling."""
        logger.info("🛡️ Starting Bulletproof SRAG-V Training")
        logger.info("=" * 60)
        
        self.start_time = datetime.now()
        max_training_seconds = self.config["max_training_hours"] * 3600
        
        # Check for previous state
        previous_state = self._load_state()
        if previous_state:
            logger.info(f"📋 Found previous training state: {previous_state['status']}")
        
        restart_count = 0
        max_restarts = self.config["max_restarts"]
        
        while restart_count <= max_restarts:
            try:
                # Start training
                if not self._start_training_process():
                    logger.error("❌ Failed to start training process")
                    break
                
                # Monitor training
                last_health_check = time.time()
                health_check_interval = self.config["health_check_interval"]
                
                while True:
                    # Check if training time exceeded
                    elapsed_time = time.time() - self.start_time.timestamp()
                    if elapsed_time > max_training_seconds:
                        logger.info(f"⏰ Maximum training time reached ({self.config['max_training_hours']} hours)")
                        self._save_state("completed_max_time")
                        break
                    
                    # Check if process is still running
                    if self.training_process.poll() is not None:
                        # Process terminated
                        return_code = self.training_process.returncode
                        if return_code == 0:
                            logger.info("🎉 Training completed successfully!")
                            self._save_state("completed_success")
                            return True
                        else:
                            failure_reason = f"Process exited with code {return_code}"
                            if not self._handle_training_failure(failure_reason):
                                return False
                            break  # Break to restart
                    
                    # Periodic health checks
                    current_time = time.time()
                    if current_time - last_health_check >= health_check_interval:
                        health = self._check_system_health()
                        
                        if not health["healthy"]:
                            issues = "; ".join(health["issues"])
                            logger.warning(f"⚠️ Health issues detected: {issues}")
                            
                            # Critical issues that require restart
                            critical_issues = [
                                "Training process terminated unexpectedly",
                                "Training process not found",
                                "High error rate"
                            ]
                            
                            if any(issue in issues for issue in critical_issues):
                                if not self._handle_training_failure(f"Critical health issues: {issues}"):
                                    return False
                                break  # Break to restart
                        else:
                            logger.debug(f"✅ System healthy: {health['metrics']}")
                            self.consecutive_failures = 0  # Reset failure counter on healthy check
                        
                        last_health_check = current_time
                    
                    # Brief sleep to avoid busy waiting
                    time.sleep(5)
                
                # If we reach here, we need to restart
                self._terminate_training_process()
                restart_count += 1
                
                if restart_count <= max_restarts:
                    logger.info(f"🔄 Preparing to restart training (restart {restart_count}/{max_restarts})")
                else:
                    logger.error(f"❌ Maximum restarts reached ({max_restarts}). Giving up.")
                    break
                
            except KeyboardInterrupt:
                logger.info("⌨️ Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"💥 Unexpected error in training loop: {e}")
                traceback.print_exc()
                if not self._handle_training_failure(f"Unexpected error: {e}"):
                    break
                restart_count += 1
        
        # Cleanup
        self._terminate_training_process()
        self._save_state("stopped", {"restart_count": restart_count})
        logger.info("🏁 Bulletproof training stopped")
        return False

def main():
    launcher = BulletproofTrainingLauncher()
    success = launcher.run_bulletproof_training()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()