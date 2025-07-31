#!/usr/bin/env python3
"""
Simple ML Foundry launcher that bypasses import issues.
"""

import os
import requests
import json
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ML Foundry configuration
api_key = os.getenv("ML_FOUNDRY_API_KEY")
base_url = "https://api.mlfoundry.com/v2"

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def get_project_id():
    """Get the ARG project ID."""
    response = requests.get(f"{base_url}/projects", headers=headers)
    if response.status_code == 200:
        projects_data = response.json()
        for project in projects_data:
            if project.get('name') == 'ARG':
                return project.get('fid')
    return None

def get_ssh_key_id(project_id):
    """Get existing SSH key ID."""
    response = requests.get(f"{base_url}/ssh-keys?project={project_id}", headers=headers)
    if response.status_code == 200:
        ssh_keys = response.json()
        if ssh_keys:
            return ssh_keys[0].get('fid') or ssh_keys[0].get('id')
    return None

def create_startup_script():
    """Create startup script for training."""
    script_parts = [
        "#!/bin/bash",
        "set -e",
        "",
        "# Install dependencies",
        "pip install torch>=2.0.0 transformers>=4.46.0 accelerate>=0.25.0 peft>=0.16.0",
        "pip install bitsandbytes>=0.41.0 datasets>=2.0.0 numpy>=1.24.0 pyyaml>=6.0",
        "pip install tqdm>=4.66.0 scikit-learn>=1.3.0 python-dotenv>=1.0.0",
        "",
        "# Clone the training code",
        "cd /workspace",
        "git clone https://github.com/ShreshthRajan/SRAG.git srag-training",
        "cd srag-training",
        "",
        "# Create logs and checkpoints directories",
        "mkdir -p logs checkpoints",
        "",
        "# Set environment variables",
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",  
        "export TOKENIZERS_PARALLELISM=false",
        "export OMP_NUM_THREADS=8",
        "",
        "# Run training",
        "echo 'Starting SRAG-V Step 2 training...'",
        "python run_step2_training.py 2>&1 | tee logs/training.log",
        "",
        "echo 'Training completed successfully!'"
    ]
    
    return "\n".join(script_parts)

def launch_training():
    """Launch Step 2 training on ML Foundry."""
    logger.info("🚀 SRAG-V STEP 2: ML FOUNDRY GPU TRAINING LAUNCH")
    logger.info("=" * 80)
    
    # Get project ID
    project_id = get_project_id()
    if not project_id:
        raise Exception("Could not find project 'ARG'")
    logger.info(f"✅ Using project: {project_id}")
    
    # Get SSH key ID
    ssh_key_id = get_ssh_key_id(project_id)
    if not ssh_key_id:
        raise Exception("No SSH key found")
    logger.info(f"✅ Using SSH key: {ssh_key_id}")
    
    # Get 4×A100 instance type
    response = requests.get(f"{base_url}/instance-types", headers=headers)
    if response.status_code != 200:
        raise Exception("Could not get instance types")
    
    instance_types = response.json()
    instance_type = None
    for it in instance_types:
        if it.get('num_gpus') == 4 and it.get('gpu_type') == 'A100':
            instance_type = it['fid']
            logger.info(f"✅ Using instance: {it['name']} ({instance_type})")
            break
    
    if not instance_type:
        raise Exception("Could not find 4×A100 instance type")
    
    # Create startup script
    startup_script = create_startup_script()
    
    # Try different region configurations
    region_configs = [
        {"region": "us-east-1"},
        {"region": "us-west-1"}, 
        {"region": "us-central-1"},
        {"region": "europe-west-1"},
        {},  # No region specified
    ]
    
    for config in region_configs:
        # Create spot bid payload
        bid_payload = {
            "project": project_id,
            "name": f"sragv-step2-training-{int(time.time())}",
            "instance_type": instance_type,
            "limit_price": "$20.00",
            "instance_quantity": 1,
            "launch_specification": {
                "startup_script": startup_script,
                "volumes": [],
                "ssh_keys": [ssh_key_id]
            }
        }
        
        if "region" in config:
            bid_payload["region"] = config["region"]
            logger.info(f"🌍 Trying region: {config['region']}")
        else:
            logger.info("🌍 Trying auto-region selection")
        
        response = requests.post(f"{base_url}/spot/bids", headers=headers, json=bid_payload)
        
        if response.status_code in [200, 201]:
            bid_data = response.json()
            bid_id = bid_data.get('id') or bid_data.get('fid')
            region_info = config.get("region", "auto-selected")
            
            logger.info("🎉 TRAINING LAUNCHED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"📋 Spot Bid ID: {bid_id}")
            logger.info(f"🌍 Region: {region_info}")
            logger.info(f"⚡ Compute: 4×A100 GPUs + 900GB RAM")
            logger.info(f"⏱️  Estimated completion: 2-3 hours")
            logger.info(f"💰 Estimated cost: $60-120 (6-12% of budget)")
            logger.info("=" * 80)
            
            return bid_id
            
        elif "not available in this region" in response.text or "Region not found" in response.text:
            region_name = config.get("region", "auto-select")
            logger.warning(f"❌ Configuration failed for {region_name}")
            continue
        else:
            region_name = config.get("region", "auto-select")
            logger.warning(f"❌ {region_name} failed: {response.status_code} - {response.text}")
            continue
    
    # All configurations failed
    raise Exception(f"Failed to create spot bid in all regions: {response.status_code} - {response.text}")

def main():
    """Main function."""
    try:
        if not api_key:
            logger.error("❌ Missing ML_FOUNDRY_API_KEY")
            return 1
        
        bid_id = launch_training()
        logger.info(f"✅ Training launched with bid ID: {bid_id}")
        logger.info("🔗 Monitor progress in ML Foundry console")
        return 0
        
    except Exception as e:
        logger.error(f"💥 Launch failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())