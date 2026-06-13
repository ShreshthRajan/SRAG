#!/usr/bin/env python3
"""
Launch raw model (no calibration) HumanEval evaluation on ML Foundry.
This proves the calibration boost by comparing to baseline.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from bulletproof_foundry_launcher import BulletproofFoundryLauncher
import logging

logger = logging.getLogger(__name__)

class RawModelLauncher(BulletproofFoundryLauncher):
    """Launcher for raw model evaluation (no calibration)."""

    def create_bulletproof_startup_script(self) -> str:
        """Create startup script for raw model evaluation."""

        startup_script = '''#!/bin/bash
set -euo pipefail

log_ts() { echo "$(date '+%H:%M:%S') - $1" | tee -a /workspace/deploy.log; }
trap 'log_ts "EXIT CODE $?"' EXIT

log_ts "🚀 RAW MODEL EVALUATION STARTING"
cd /workspace && mkdir -p logs checkpoints phase4_results

# Install dependencies
log_ts "📦 Installing dependencies..."
pip install torch>=2.0.0 transformers>=4.40.0 accelerate>=0.25.0 peft>=0.16.0 bitsandbytes>=0.41.0 datasets>=2.0.0 numpy>=1.24.0 pyyaml>=6.0 tqdm>=4.66.0 scikit-learn>=1.3.0 python-dotenv>=1.0.0 matplotlib>=3.5.0 seaborn>=0.11.0 jinja2>=3.1.0 human-eval --no-cache-dir --timeout=300

# Clone repository
log_ts "📥 Cloning repository..."
git clone https://github.com/ShreshthRajan/SRAG.git srag || exit 1
cd srag

# Environment setup
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_VERBOSITY=error

# GPU health check
log_ts "🔍 GPU Health Check..."
nvidia-smi || exit 1

# Run RAW MODEL evaluation (no calibration)
log_ts "🎯 STARTING RAW MODEL EVALUATION (NO CALIBRATION)..."
exec > >(tee -a logs/raw_model_eval_$(date +%Y%m%d_%H%M%S).log) 2>&1

if python3 run_phase4_step1_humaneval.py --no-calibration --phase1-only; then
    log_ts "🎉 RAW MODEL EVALUATION SUCCESS!"
    echo "SUCCESS" > phase4_raw_model_status.txt
else
    log_ts "❌ RAW MODEL EVALUATION FAILED"
    echo "FAILED:$?" > phase4_raw_model_status.txt
    exit $?
fi

log_ts "🏁 DEPLOYMENT COMPLETE"
'''

        return startup_script.strip()

    def deploy_to_foundry(self):
        """Override bid name for raw model evaluation."""
        import time

        # Update bid name before calling parent
        original_method = super().deploy_to_foundry

        # Temporarily modify the bid name generation
        self._bid_name_prefix = f"sragv-raw-model-{int(time.time())}"

        return original_method()


if __name__ == "__main__":
    launcher = RawModelLauncher()

    logger.info("🎯 LAUNCHING RAW MODEL EVALUATION (NO CALIBRATION)")
    logger.info("This will prove calibration boost by testing baseline performance")
    logger.info("=" * 70)

    if not launcher.validate_prerequisites():
        sys.exit(1)

    bid_id = launcher.deploy_to_foundry()

    if bid_id:
        logger.info("✅ Deployment successful!")
        logger.info("Waiting for instance IP...")

        ip_address = launcher.wait_for_instance_ip(bid_id, timeout_minutes=20)

        if ip_address:
            logger.info("=" * 70)
            logger.info("🎉 RAW MODEL EVALUATION INSTANCE READY!")
            logger.info("=" * 70)
            logger.info(f"SSH Command: ssh -i ~/.ssh/mlfoundry_temp ubuntu@{ip_address}")
            logger.info(f"Check results: cd srag && tail -f logs/phase4_step1_latest.log")
            logger.info("=" * 70)
        else:
            logger.error("❌ Failed to get IP address")
            sys.exit(1)
    else:
        logger.error("❌ Deployment failed")
        sys.exit(1)
