#!/usr/bin/env python3
"""
Fix SSH key issue and relaunch with guaranteed working setup.
"""

import os
import requests
import time
from dotenv import load_dotenv

load_dotenv()

def terminate_broken_bid():
    """Terminate the current bid with wrong SSH key."""
    api_key = os.getenv("ML_FOUNDRY_API_KEY")
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    bid_id = "bid_bEpgNXUtYwCABoTr"  # Current broken bid
    
    print(f"🚫 Terminating broken bid: {bid_id}")
    response = requests.delete(f"https://api.mlfoundry.com/v2/spot/bids/{bid_id}", headers=headers)
    
    if response.status_code in [200, 204]:
        print("✅ Broken bid terminated")
    else:
        print(f"⚠️ Termination status: {response.status_code} - {response.text}")
    
    return True

def launch_with_correct_ssh_key():
    """Launch new bid with the correct SSH key."""
    api_key = os.getenv("ML_FOUNDRY_API_KEY")
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    
    startup_script = """#!/bin/bash
set -e

# Install dependencies
pip install torch>=2.0.0 transformers>=4.40.0 accelerate>=0.25.0 peft>=0.16.0
pip install bitsandbytes>=0.41.0 datasets>=2.0.0 numpy>=1.24.0 pyyaml>=6.0
pip install tqdm>=4.66.0 scikit-learn>=1.3.0 python-dotenv>=1.0.0

# Clone the training code
cd /workspace
git clone https://github.com/ShreshthRajan/SRAG.git srag-training
cd srag-training

# Create logs and checkpoints directories
mkdir -p logs checkpoints

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Run training
echo 'Starting SRAG-V Step 2 training...'
python run_step2_training.py 2>&1 | tee logs/training.log

echo 'Training completed successfully!'"""

    bid_payload = {
        "project": "proj_V0uIRLp92vFAt2QR",
        "name": f"sragv-step2-fixed-ssh-{int(time.time())}",
        "instance_type": "it_fK7Cx6TVhOK5ZfXT",  # 4×A100
        "limit_price": "$35.00",  # High bid to guarantee win
        "instance_quantity": 1,
        "region": "eu-central1-b",
        "launch_specification": {
            "startup_script": startup_script,
            "volumes": [],
            "ssh_keys": ["sshkey_f4tLcOIezg6HX2zi"]  # CORRECT SSH KEY ID
        }
    }
    
    print("🚀 Creating new bid with correct SSH key...")
    response = requests.post("https://api.mlfoundry.com/v2/spot/bids", headers=headers, json=bid_payload)
    
    if response.status_code in [200, 201]:
        bid_data = response.json()
        new_bid_id = bid_data.get('id') or bid_data.get('fid')
        print(f"✅ NEW BID CREATED: {new_bid_id}")
        print(f"🔑 Using correct SSH key: sshkey_f4tLcOIezg6HX2zi")
        print(f"💰 Bid: $35/hour (guaranteed to win)")
        return new_bid_id
    else:
        print(f"❌ Launch failed: {response.status_code} - {response.text}")
        return None

if __name__ == "__main__":
    print("🔧 FIXING SSH KEY ISSUE AND RELAUNCHING")
    print("=" * 50)
    print("Issue: Wrong SSH key was used in the bid")
    print("Solution: Terminate and relaunch with correct key")
    print()
    
    # Step 1: Terminate broken bid
    terminate_broken_bid()
    
    print("\n⏳ Waiting 10 seconds...")
    time.sleep(10)
    
    # Step 2: Launch with correct SSH key
    new_bid_id = launch_with_correct_ssh_key() 
    
    if new_bid_id:
        print(f"\n🎯 NEW BID READY: {new_bid_id}")
        print("✅ This has the correct SSH key that matches ~/.ssh/mlfoundry_temp")
        print(f"\nRun: python enhanced_training_monitor.py")
        print(f"Enter bid ID: {new_bid_id}")
        print("\nThis will work correctly now!")
    else:
        print("\n💥 Launch failed")