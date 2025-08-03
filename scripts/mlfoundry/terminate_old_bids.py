#!/usr/bin/env python3
"""
Terminate old faulty bids as requested by ML Foundry staff.
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

def list_all_bids():
    """List all spot bids to identify old ones."""
    print("🔍 Listing all spot bids...")
    
    try:
        # Get project ID first
        response = requests.get(f"{base_url}/projects", headers=headers)
        if response.status_code == 200:
            projects_data = response.json()
            project_id = None
            for project in projects_data:
                if project.get('name') == 'ARG':
                    project_id = project.get('fid')
                    break
            
            if not project_id:
                print("❌ Could not find project 'ARG'")
                return []
            
            # List all bids for the project
            response = requests.get(f"{base_url}/spot/bids?project={project_id}", headers=headers)
            if response.status_code == 200:
                bids = response.json()
                print(f"📋 Found {len(bids)} total bids:")
                
                current_bid = "bid_AY5L98ZeUfzGcDc7"  # Current active bid
                old_bids = []
                
                for bid in bids:
                    bid_id = bid.get('fid') or bid.get('id')
                    status = bid.get('status', 'unknown')
                    region = bid.get('region', 'unknown')
                    
                    print(f"  - {bid_id}: {status} in {region}")
                    
                    # Mark as old if not the current bid
                    if bid_id != current_bid:
                        old_bids.append(bid_id)
                
                return old_bids
            else:
                print(f"❌ Error listing bids: {response.status_code} - {response.text}")
                return []
        else:
            print(f"❌ Error getting project: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Error: {e}")
        return []

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
    print("🔧 TERMINATE OLD FAULTY BIDS")
    print("="*50)
    
    # List all bids
    old_bids = list_all_bids()
    
    if old_bids:
        print(f"\n🎯 Found {len(old_bids)} old bids to terminate:")
        for bid_id in old_bids:
            print(f"  - {bid_id}")
        
        # Confirm before terminating
        confirm = input(f"\nTerminate all {len(old_bids)} old bids? (y/N): ")
        if confirm.lower() == 'y':
            success_count = 0
            for bid_id in old_bids:
                if terminate_bid(bid_id):
                    success_count += 1
            
            print(f"\n✅ Successfully terminated {success_count}/{len(old_bids)} old bids")
            print("🚀 Old faulty bids cleaned up as requested by ML Foundry staff")
        else:
            print("\n⏸️ Termination cancelled")
    else:
        print("\n✅ No old bids found to terminate")
        print("Current bid (bid_AY5L98ZeUfzGcDc7) is the only active bid")