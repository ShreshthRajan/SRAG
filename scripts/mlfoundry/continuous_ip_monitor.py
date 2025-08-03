#!/usr/bin/env python3
"""
Continuous IP Monitor for ML Foundry Training
Provides real-time visibility into instance provisioning and IP assignment.
"""

import os
import sys
import requests
import time
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class ContinuousIPMonitor:
    def __init__(self, bid_id="bid_sQCZbrBRSSXHjlJz"):
        self.bid_id = bid_id
        self.api_key = os.getenv("ML_FOUNDRY_API_KEY")
        self.base_url = "https://api.mlfoundry.com/v2"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        self.ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")
        self.last_status = None
        self.start_time = time.time()
        
        print(f"🔍 CONTINUOUS IP MONITOR STARTED")
        print(f"📊 Monitoring Bid: {self.bid_id}")
        print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)
    
    def get_timestamp(self):
        return datetime.now().strftime('%H:%M:%S')
    
    def get_elapsed_time(self):
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def check_bid_status(self):
        """Check bid status and return detailed info."""
        try:
            response = requests.get(f"{self.base_url}/spot/bids/{self.bid_id}", headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                return {
                    'success': True,
                    'status': data.get('status', 'unknown'),
                    'instances': data.get('instances', []),
                    'region': data.get('region', 'unknown'),
                    'created_at': data.get('created_at', ''),
                    'deactivated_at': data.get('deactivated_at', None)
                }
            else:
                return {
                    'success': False,
                    'error': f"API Error: {response.status_code} - {response.text[:100]}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': f"Network Error: {str(e)}"
            }
    
    def check_instance_ip(self, instance_id):
        """Try multiple methods to get instance IP."""
        endpoints_to_try = [
            f"{self.base_url}/instances/{instance_id}",
            f"{self.base_url}/spot/bids/{self.bid_id}/instances",
            f"{self.base_url}/instances/{instance_id}/details"
        ]
        
        for endpoint in endpoints_to_try:
            try:
                response = requests.get(endpoint, headers=self.headers)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Handle different response formats
                    if isinstance(data, list) and data:
                        data = data[0]
                    
                    # Try different IP field names
                    ip_fields = ['public_ip_address', 'ip_address', 'external_ip', 'public_ip']
                    for field in ip_fields:
                        ip = data.get(field)
                        if ip:
                            return {
                                'success': True,
                                'ip': ip,
                                'method': endpoint,
                                'instance_status': data.get('status', 'unknown')
                            }
                    
                    return {
                        'success': True,
                        'ip': None,
                        'method': endpoint,
                        'instance_status': data.get('status', 'provisioning'),
                        'raw_data': data
                    }
                        
            except Exception as e:
                continue
        
        return {
            'success': False,
            'error': "All instance endpoints failed"
        }
    
    def test_ssh_connection(self, ip_address):
        """Test if SSH connection is ready."""
        import subprocess
        try:
            ssh_cmd = [
                "ssh", "-i", self.ssh_key_path, 
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-o", "BatchMode=yes",  # Non-interactive
                f"ubuntu@{ip_address}",
                "echo 'SSH_READY'"
            ]
            
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0 and "SSH_READY" in result.stdout:
                return True
            return False
        except:
            return False
    
    def print_status_update(self, bid_info, instance_info=None, ip_ready=False):
        """Print formatted status update."""
        timestamp = self.get_timestamp()
        elapsed = self.get_elapsed_time()
        
        # Status line
        status = bid_info.get('status', 'unknown')
        status_emoji = {
            'Allocated': '🟢',
            'Pending': '🟡', 
            'Provisioning': '🟡',
            'Terminated': '🔴',
            'Failed': '🔴'
        }.get(status, '⚪')
        
        print(f"[{timestamp}] {status_emoji} {status} | Elapsed: {elapsed}")
        
        # Instance details
        instances = bid_info.get('instances', [])
        if instances:
            instance_id = instances[0]
            print(f"         📱 Instance: {instance_id}")
            
            if instance_info:
                if instance_info.get('success'):
                    ip = instance_info.get('ip')
                    if ip:
                        ssh_status = "✅ Ready" if ip_ready else "🔄 Testing..."
                        print(f"         🌐 IP: {ip} | SSH: {ssh_status}")
                        if ip_ready:
                            print(f"         🔑 SSH Command: ssh -i {self.ssh_key_path} ubuntu@{ip}")
                    else:
                        inst_status = instance_info.get('instance_status', 'provisioning')
                        print(f"         ⏳ Instance Status: {inst_status} (no IP yet)")
                else:
                    print(f"         ❌ Instance Check: {instance_info.get('error', 'unknown error')}")
        
        # Region info
        region = bid_info.get('region', 'unknown')
        print(f"         🌍 Region: {region}")
        
        print()  # Empty line for readability
    
    def monitor_continuously(self, max_wait_minutes=10):
        """Main monitoring loop."""
        max_wait_seconds = max_wait_minutes * 60
        check_interval = 15  # Check every 15 seconds
        
        print(f"🕐 Will monitor for {max_wait_minutes} minutes")
        print(f"🔄 Checking every {check_interval} seconds")
        print()
        
        while time.time() - self.start_time < max_wait_seconds:
            # Check bid status
            bid_info = self.check_bid_status()
            
            if not bid_info['success']:
                print(f"[{self.get_timestamp()}] ❌ Bid check failed: {bid_info['error']}")
                time.sleep(check_interval)
                continue
            
            # Check if terminated
            if bid_info['status'] == 'Terminated':
                print(f"[{self.get_timestamp()}] 🔴 TRAINING TERMINATED!")
                if bid_info.get('deactivated_at'):
                    print(f"         Terminated at: {bid_info['deactivated_at']}")
                return False
            
            # Check instances
            instances = bid_info.get('instances', [])
            instance_info = None
            ip_ready = False
            
            if instances:
                instance_id = instances[0]
                instance_info = self.check_instance_ip(instance_id)
                
                if instance_info.get('success') and instance_info.get('ip'):
                    ip_address = instance_info['ip']
                    ip_ready = self.test_ssh_connection(ip_address)
                    
                    if ip_ready:
                        self.print_status_update(bid_info, instance_info, ip_ready)
                        print("🎉 SUCCESS! IP RESOLVED AND SSH READY!")
                        print("=" * 60)
                        print(f"🆔 Bid ID: {self.bid_id}")
                        print(f"🌐 IP Address: {ip_address}")
                        print(f"🔑 SSH Command: ssh -i {self.ssh_key_path} ubuntu@{ip_address}")
                        print(f"📊 Training Status Check: ssh -i {self.ssh_key_path} ubuntu@{ip_address} 'cat /workspace/training_status.txt 2>/dev/null || echo STARTING'")
                        print(f"📋 View Logs: ssh -i {self.ssh_key_path} ubuntu@{ip_address} 'tail -f /workspace/logs/training_*.log'")
                        print("=" * 60)
                        return True
            
            # Print status update
            self.print_status_update(bid_info, instance_info, ip_ready)
            
            # Wait before next check
            time.sleep(check_interval)
        
        print(f"⏰ TIMEOUT: IP not ready after {max_wait_minutes} minutes")
        print("💬 Time to message the ML Foundry team!")
        return False

def main():
    monitor = ContinuousIPMonitor()
    
    try:
        success = monitor.monitor_continuously(max_wait_minutes=10)
        
        if success:
            print("🚀 You can now monitor your training with SSH!")
            print("🔄 Or run: python bulletproof_foundry_launcher.py (it will detect existing deployment)")
        else:
            print("📧 Consider messaging ML Foundry team about the IP assignment delay")
            
    except KeyboardInterrupt:
        print(f"\n🛑 Monitoring stopped by user")
        print(f"⏱️ Total monitoring time: {monitor.get_elapsed_time()}")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    main()