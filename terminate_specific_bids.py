#!/usr/bin/env python3
"""
Terminate specific old faulty bids as requested by ML Foundry staff.
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

def terminate_bid(bid_id):
    """Terminate a specific bid."""
    print(f"🚫 Terminating bid: {bid_id}")
    
    try:
        response = requests.delete(f"{base_url}/spot/bids/{bid_id}", headers=headers)
        if response.status_code in [200, 204]:
            print(f"✅ Successfully terminated {bid_id}")
            return True
        else:
            print(f"❌ Failed to terminate {bid_id}: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error terminating {bid_id}: {e}")
        return False

if __name__ == "__main__":
    print("🔧 TERMINATE SPECIFIC OLD FAULTY BIDS")
    print("="*50)
    
    # The 2 old faulty bids ML Foundry staff identified
    old_bids = [
        "bid_MDglwmlidHsq4byi",  # First stuck bid
        "bid_DJsmXjTwpAQ6L4dq"   # Another old bid  
    ]
    
    current_bid = "bid_AY5L98ZeUfzGcDc7"  # Current running training
    
    print(f"🎯 Terminating 2 old faulty bids:")
    print(f"  - bid_MDglwmlidHsq4byi (first stuck bid)")
    print(f"  - bid_DJsmXjTwpAQ6L4dq (second old bid)")
    print(f"✅ Keeping current training bid: {current_bid}")
    
    # Confirm before terminating
    confirm = input(f"\nTerminate the 2 old bids? (y/N): ")
    if confirm.lower() == 'y':
        success_count = 0
        for bid_id in old_bids:
            if terminate_bid(bid_id):
                success_count += 1
        
        print(f"\n✅ Successfully terminated {success_count}/{len(old_bids)} old bids")
        print("🚀 Old faulty bids cleaned up as requested by ML Foundry staff")
        print(f"🎯 Your training continues on: {current_bid}")
    else:
        print("\n⏸️ Termination cancelled")