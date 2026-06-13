#!/usr/bin/env python3
"""
Check ML Foundry spot availability to find valid regions.
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

def check_spot_availability():
    """Check spot availability to find valid regions."""
    print("🔍 Checking ML Foundry spot availability...")
    
    try:
        response = requests.get(f"{base_url}/spot/availability", headers=headers)
        print(f"Spot availability endpoint status: {response.status_code}")
        
        if response.status_code == 200:
            availability_data = response.json()
            print("Spot availability data:")
            print(json.dumps(availability_data, indent=2))
            
            # Extract regions from availability data
            regions = set()
            if isinstance(availability_data, list):
                for item in availability_data:
                    if 'region' in item:
                        regions.add(item['region'])
            elif isinstance(availability_data, dict):
                # Handle different response formats
                for key, value in availability_data.items():
                    if 'region' in str(key).lower():
                        regions.add(value)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict) and 'region' in item:
                                regions.add(item['region'])
            
            if regions:
                print(f"\n✅ Found valid regions: {sorted(regions)}")
                return sorted(regions)
            else:
                print("\n❌ No regions found in availability data")
                return []
        else:
            print(f"Spot availability error: {response.text}")
            return []
    except Exception as e:
        print(f"Spot availability check failed: {e}")
        return []

if __name__ == "__main__":
    print("🔧 ML FOUNDRY SPOT AVAILABILITY CHECK")
    print("="*50)
    
    if not api_key:
        print("❌ Missing ML_FOUNDRY_API_KEY")
        exit(1)
    
    valid_regions = check_spot_availability()
    
    if valid_regions:
        print(f"\n🎯 VALID REGIONS FOR SPOT BIDS:")
        for region in valid_regions:
            print(f"  - {region}")
    else:
        print("\n❌ Could not determine valid regions")
        print("📋 Try contacting ML Foundry support or check their console")