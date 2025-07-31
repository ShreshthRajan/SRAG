#!/usr/bin/env python3
import os, requests, time
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
api_key = os.getenv('ML_FOUNDRY_API_KEY')
headers = {'Authorization': f'Bearer {api_key}'}
bid_id = 'bid_RbufiFage5c9VKc7'

print(f"🔍 Checking for IP every 10 seconds - Bid: {bid_id}")
print("Press Ctrl+C to stop")
print("-" * 50)

while True:
    try:
        timestamp = datetime.now().strftime('%H:%M:%S')
        response = requests.get(f'https://api.mlfoundry.com/v2/spot/bids/{bid_id}', headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get('status')
            instances = data.get('instances', [])
            
            if status == 'Terminated':
                print(f"[{timestamp}] 🔴 TERMINATED!")
                break
                
            if instances:
                instance_id = instances[0]
                inst_resp = requests.get(f'https://api.mlfoundry.com/v2/instances/{instance_id}', headers=headers)
                
                if inst_resp.status_code == 200:
                    inst_data = inst_resp.json()
                    ip = inst_data.get('public_ip_address')
                    
                    if ip:
                        print(f"[{timestamp}] 🌐 IP READY: {ip}")
                        print(f"🔑 SSH COMMAND: ssh -i ~/.ssh/mlfoundry_temp ubuntu@{ip}")
                        print("🚀 You can now SSH in and check training!")
                        break
                    else:
                        print(f"[{timestamp}] ⏳ Status: {status} | Instance: {instance_id} | IP: Still provisioning...")
                else:
                    print(f"[{timestamp}] ⏳ Status: {status} | Instance: {instance_id} | Getting details...")
            else:
                print(f"[{timestamp}] ⏳ Status: {status} | No instances yet...")
        else:
            print(f"[{timestamp}] ❌ API Error: {response.status_code}")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Stopped checking")
        break
    except Exception as e:
        print(f"[{timestamp}] ⚠️ Error: {e}")
    
    time.sleep(10)