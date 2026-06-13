#!/usr/bin/env python3
"""
Cancel stuck bid and relaunch fresh training in working region.
"""

import os
import requests
import time
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

def cancel_stuck_bid(bid_id):
    """Cancel the stuck bid."""
    print(f"🚫 Cancelling stuck bid: {bid_id}")
    
    try:
        response = requests.delete(f"{base_url}/spot/bids/{bid_id}", headers=headers)
        if response.status_code in [200, 204]:
            print("✅ Stuck bid cancelled successfully")
            return True
        else:
            print(f"❌ Cancel failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error cancelling bid: {e}")
        return False

if __name__ == "__main__":
    print("🔧 CANCEL STUCK BID AND PREPARE FOR RELAUNCH")
    print("="*60)
    
    # Cancel the stuck bid
    stuck_bid_id = "bid_nLYGh8eWZlc7l3pC"
    
    if cancel_stuck_bid(stuck_bid_id):
        print("\n🚀 Stuck bid cancelled successfully!")
        print("✅ Ready to relaunch with fresh VM")
        print("\nNext step: Run 'python launch_foundry_training.py' for fresh launch")
    else:
        print("\n❌ Please manually cancel bid in ML Foundry console")
        print("Then run: python launch_foundry_training.py")