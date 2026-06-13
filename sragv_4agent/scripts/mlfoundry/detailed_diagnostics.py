#!/usr/bin/env python3
"""
Detailed diagnostics for ML Foundry training - get ALL available information.
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def get_detailed_diagnostics(bid_id):
    api_key = os.getenv("ML_FOUNDRY_API_KEY")
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    base_url = "https://api.mlfoundry.com/v2"
    
    print(f"🔍 DETAILED DIAGNOSTICS FOR: {bid_id}")
    print("=" * 60)
    
    # 1. Get full bid details
    print("1. FULL BID DETAILS:")
    try:
        response = requests.get(f"{base_url}/spot/bids/{bid_id}", headers=headers)
        if response.status_code == 200:
            bid_data = response.json()
            print(json.dumps(bid_data, indent=2))
            
            # Extract instance IDs if they exist
            instances = bid_data.get('instances', [])
            print(f"\n🖥️  Found {len(instances)} instances: {instances}")
            
            return bid_data, instances
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return None, []
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None, []

def check_instance_details(instance_ids):
    if not instance_ids:
        print("\n2. NO INSTANCES TO CHECK")
        return
        
    api_key = os.getenv("ML_FOUNDRY_API_KEY")
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    base_url = "https://api.mlfoundry.com/v2"
    
    print("\n2. INSTANCE DETAILS:")
    for i, instance_id in enumerate(instance_ids):
        print(f"\n   Instance {i+1}/{len(instance_ids)}: {instance_id}")
        try:
            response = requests.get(f"{base_url}/instances/{instance_id}", headers=headers)
            print(f"   Status: {response.status_code}")
            if response.status_code == 200:
                instance_data = response.json()
                print(f"   Data: {json.dumps(instance_data, indent=4)}")
            else:
                print(f"   Error: {response.text}")
        except Exception as e:
            print(f"   Exception: {e}")

def check_spot_market():
    api_key = os.getenv("ML_FOUNDRY_API_KEY")
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    base_url = "https://api.mlfoundry.com/v2"
    
    print("\n3. SPOT MARKET STATUS:")
    try:
        # Check spot availability
        response = requests.get(f"{base_url}/spot/availability", headers=headers)
        print(f"   Availability API: {response.status_code}")
        if response.status_code == 200:
            print(f"   Data: {json.dumps(response.json(), indent=4)}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Exception: {e}")

def check_project_activity():
    api_key = os.getenv("ML_FOUNDRY_API_KEY")
    headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
    base_url = "https://api.mlfoundry.com/v2"
    project_id = "proj_V0uIRLp92vFAt2QR"
    
    print("\n4. PROJECT ACTIVITY:")
    try:
        # List all bids for project
        response = requests.get(f"{base_url}/spot/bids?project={project_id}", headers=headers)
        print(f"   All bids: {response.status_code}")
        if response.status_code == 200:
            bids = response.json()
            print(f"   Found {len(bids)} total bids:")
            for bid in bids:
                bid_id = bid.get('fid', bid.get('id'))
                status = bid.get('status')
                instance_status = bid.get('instance_status')
                name = bid.get('name')
                created = bid.get('created_at')
                print(f"     - {bid_id}: {status}/{instance_status} | {name} | {created}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   Exception: {e}")

if __name__ == "__main__":
    bid_id = "bid_bEpgNXUtYwCABoTr"
    
    # Run all diagnostics
    bid_data, instances = get_detailed_diagnostics(bid_id)
    check_instance_details(instances)
    check_spot_market()
    check_project_activity()
    
    print("\n" + "=" * 60)
    print("🎯 DIAGNOSIS COMPLETE")
    
    if bid_data:
        status = bid_data.get('status')
        instance_status = bid_data.get('instance_status')
        
        if status == 'Allocated' and instance_status == 'unknown':
            print("💡 LIKELY CAUSE: VM is still being provisioned by ML Foundry")
            print("⏳ RECOMMENDATION: Wait 5-10 more minutes, this is normal")
        elif instances:
            print("💡 INSTANCES EXIST: Check instance details above for IP/status")
        else:
            print("⚠️  NO INSTANCES: Provisioning may have failed")