#!/usr/bin/env python3
"""
Launch SRAG-V Step 2 Training on ML Foundry.
Uses 4×L4 GPUs for breakthrough-level self-play training.
"""

import logging
import sys
import os
import time
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragv.foundry import MLFoundryClient

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,  # Enable debug for API debugging
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/foundry_launch.log')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
Path("logs").mkdir(exist_ok=True)


def check_environment():
    """Check required environment variables."""
    logger.info("🔍 Checking ML Foundry environment...")
    
    # Only check for API key - project ID will be auto-resolved
    api_key = os.getenv("ML_FOUNDRY_API_KEY")
    
    if not api_key:
        logger.error(f"❌ Missing ML_FOUNDRY_API_KEY environment variable")
        logger.error("Please set: export ML_FOUNDRY_API_KEY=your_api_key")
        return False
    
    logger.info(f"✅ ML_FOUNDRY_API_KEY: {'*' * 8}")
    
    # Check if project ID is set (optional)
    project_id = os.getenv("ML_FOUNDRY_PROJECT_ID")  
    if project_id:
        logger.info(f"✅ ML_FOUNDRY_PROJECT_ID: {project_id}")
    else:
        logger.info("📋 Project ID will be auto-resolved from project name 'ARG'")
    
    logger.info("✅ Environment check passed")
    return True


def launch_training():
    """Launch Step 2 training on ML Foundry."""
    logger.info("🚀 SRAG-V STEP 2: ML FOUNDRY GPU TRAINING LAUNCH")
    logger.info("=" * 80)
    logger.info("Deploying breakthrough-level self-play training to 4×L4 GPUs")
    logger.info("=" * 80)
    
    try:
        # Environment check
        if not check_environment():
            return False
        
        # Initialize ML Foundry client
        logger.info("🔧 Initializing ML Foundry client...")
        client = MLFoundryClient()
        
        # Launch training job
        logger.info("🎯 Launching Step 2 training job...")
        job_id = client.launch_step2_training()
        
        logger.info("🎉 TRAINING LAUNCHED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"📋 Job ID: {job_id}")
        logger.info(f"⚡ Compute: 4×L4 GPUs + 64GB RAM")
        logger.info(f"⏱️  Estimated completion: 2-3 hours")
        logger.info(f"💰 Estimated cost: $40-60 (6% of budget)")
        logger.info("=" * 80)
        
        # Monitor training progress
        logger.info("👀 Monitoring training progress...")
        start_time = time.time()
        
        while True:
            try:
                job_data = client.monitor_job(job_id)
                status = job_data.get('status', 'unknown')
                
                elapsed = time.time() - start_time
                elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
                
                logger.info(f"⏱️  Status: {status.upper()} (elapsed: {elapsed_str})")
                
                if status == 'completed':
                    logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                    logger.info("🎯 Step 2 breakthrough-level training finished")
                    logger.info("✅ Ready to proceed to Step 3: Peak Performance Training")
                    return True
                    
                elif status == 'failed':
                    error_info = job_data.get('error', 'Unknown error')
                    logger.error(f"💥 Training failed: {error_info}")
                    return False
                    
                elif status == 'running':
                    progress = job_data.get('progress', {})
                    if progress:
                        iteration = progress.get('current_iteration', 0)
                        total_iterations = progress.get('total_iterations', 12)
                        logger.info(f"📈 Progress: Iteration {iteration}/{total_iterations}")
                
                # Wait before next check
                time.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("⏹️  Monitoring interrupted by user")
                logger.info(f"🔄 Training continues on ML Foundry (Job ID: {job_id})")
                logger.info("💡 Resume monitoring with: python -c \"from sragv.foundry import MLFoundryClient; MLFoundryClient().monitor_job('{job_id}')\"")
                return True
                
    except Exception as e:
        logger.error(f"💥 Launch failed: {e}")
        logger.error("🔧 Please check your ML Foundry configuration and try again")
        return False


def main():
    """Main launch function."""
    try:
        success = launch_training()
        if success:
            logger.info("✅ SRAG-V Step 2 training launched successfully!")
            return 0
        else:
            logger.error("❌ Training launch failed")
            return 1
    except Exception as e:
        logger.error(f"💥 Critical error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)