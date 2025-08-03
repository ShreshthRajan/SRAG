#!/usr/bin/env python3
"""
Exhaustively check all possible ML Foundry API endpoints for training completion info.
"""

import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('ML_FOUNDRY_API_KEY')
headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
project_id = 'proj_V0uIRLp92vFAt2QR'
bid_id = 'bid_AY5L98ZeUfzGcDc7'

# Try every possible endpoint that might have completion info
endpoints_to_try = [
    f'/spot/bids/{bid_id}/logs',
    f'/spot/bids/{bid_id}/artifacts', 
    f'/spot/bids/{bid_id}/history',
    f'/spot/bids/{bid_id}/status',
    f'/spot/bids/{bid_id}/events',
    f'/projects/{project_id}/runs',
    f'/projects/{project_id}/experiments', 
    f'/projects/{project_id}/logs',
    f'/projects/{project_id}/artifacts',
    f'/projects/{project_id}/files',
    f'/runs?project={project_id}',
    f'/experiments?project={project_id}',
    f'/logs?project={project_id}',
    f'/files?project={project_id}',
    f'/storage?project={project_id}',
    f'/models?project={project_id}',
    f'/datasets?project={project_id}',
    '/user/activity',
    '/user/history',
    '/billing/usage'
]

base_url = 'https://api.mlfoundry.com/v2'

print("🔍 CHECKING ALL POSSIBLE ML FOUNDRY ENDPOINTS")
print("=" * 60)

for endpoint in endpoints_to_try:
    url = base_url + endpoint
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:  # Not empty
                print(f'✅ SUCCESS: {endpoint}')
                print(f'   Response: {json.dumps(data, indent=2)[:500]}...')
                print()
        elif response.status_code != 404:
            print(f'⚠️  WARNING {endpoint}: {response.status_code} - {response.text[:100]}')
    except Exception as e:
        print(f'❌ ERROR {endpoint}: {str(e)[:50]}')

print("\n" + "=" * 60)
print("Search complete. Any SUCCESS results above contain potential training info.")