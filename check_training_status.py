#!/usr/bin/env python3
"""
Simple status checker that only queries ML Foundry API - no interference with training.
"""

import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ML Foundry configuration
api_key = os.getenv("ML_FOUNDRY_API_KEY")
base_url = "https://api.mlfoundry.com/v2"

headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}

def check_bid_status(bid_id):
    """Check status of specific bid - READ ONLY."""
    print(f"🔍 Checking status of bid: {bid_id}")
    
    try:
        response = requests.get(f"{base_url}/spot/bids/{bid_id}", headers=headers)
        if response.status_code == 200:
            bid_data = response.json()
            
            status = bid_data.get('status', 'unknown')
            instance_status = bid_data.get('instance_status', 'unknown')
            region = bid_data.get('region', 'unknown')
            
            print(f"✅ Bid Status: {status}")
            print(f"✅ Instance Status: {instance_status}")
            print(f"✅ Region: {region}")
            
            # Show VM IP if available
            if 'instance' in bid_data and bid_data['instance']:
                instance = bid_data['instance']
                vm_ip = instance.get('public_ip') or instance.get('ip')
                if vm_ip:
                    print(f"✅ VM IP: {vm_ip}")
            
            return bid_data
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("📊 SRAG-V TRAINING STATUS CHECK")
    print("="*40)
    print("(READ-ONLY - No interference with training)")
    print()
    
    # Your current training bid
    current_bid = "bid_AY5L98ZeUfzGcDc7"
    
    bid_data = check_bid_status(current_bid)
    
    if bid_data:
        instance_status = bid_data.get('instance_status', 'unknown')
        
        if instance_status == 'running':
            print("\n🚀 Great! VM is running - training should be in progress")
            print("💡 The 'unknown' status you saw earlier was just an API display issue")
        elif instance_status == 'provisioning':
            print("\n⏳ VM is still being provisioned...")
        else:
            print(f"\n⚠️  Instance status: {instance_status}")