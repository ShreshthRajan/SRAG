#!/usr/bin/env python3
"""
Check detailed spot bid status and cancel if stuck.
"""

import os
import requests
import json
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
    """Check detailed spot bid status."""
    print(f"🔍 Checking spot bid: {bid_id}")
    
    try:
        response = requests.get(f"{base_url}/spot/bids/{bid_id}", headers=headers)
        if response.status_code == 200:
            bid_data = response.json()
            print("📋 Full bid details:")
            print(json.dumps(bid_data, indent=2))
            
            status = bid_data.get('status', 'unknown')
            instance_status = bid_data.get('instance_status', 'unknown')
            region = bid_data.get('region', 'unknown')
            
            print(f"\n📊 Summary:")
            print(f"  Status: {status}")
            print(f"  Instance: {instance_status}")
            print(f"  Region: {region}")
            
            # Check if stuck
            if status == 'Allocated' and instance_status == 'unknown':
                print(f"\n⚠️  STUCK: Bid allocated but instance not starting (likely due to {region} outage)")
                return True  # Stuck
            elif instance_status == 'running':
                print(f"\n✅ GOOD: Training is running!")
                return False  # Not stuck
            else:
                print(f"\n❓ Status unclear")
                return False
                
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error checking bid: {e}")
        return False

def cancel_bid(bid_id):
    """Cancel a stuck spot bid."""
    print(f"🚫 Cancelling spot bid: {bid_id}")
    
    try:
        response = requests.delete(f"{base_url}/spot/bids/{bid_id}", headers=headers)
        if response.status_code in [200, 204]:
            print("✅ Bid cancelled successfully")
            return True
        else:
            print(f"❌ Cancel failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error cancelling bid: {e}")
        return False

if __name__ == "__main__":
    # Check the current bid
    bid_id = "bid_5BouuMzEOJPjMRKX"  # From your logs
    
    print("🔧 ML FOUNDRY BID STATUS CHECK")
    print("="*50)
    
    is_stuck = check_bid_status(bid_id)
    
    if is_stuck:
        print("\n💥 RECOMMENDATION: Cancel this bid and relaunch in working region")
        print("Run: python launch_foundry_training.py (it will use us-central2-a instead)")
        
        cancel_choice = input("\nCancel stuck bid now? (y/N): ")
        if cancel_choice.lower() == 'y':
            if cancel_bid(bid_id):
                print("\n🚀 Now run: python launch_foundry_training.py")
            else:
                print("\n❌ Manual cancellation needed via ML Foundry console")
    else:
        print("\n✅ Bid looks okay - continue monitoring")