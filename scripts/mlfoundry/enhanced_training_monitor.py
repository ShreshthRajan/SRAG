#!/usr/bin/env python3
"""
Enhanced SRAG-V Training Monitor with Complete Visibility
100% reliable monitoring with SSH access and real-time progress tracking.
"""

import os
import subprocess
import time
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class EnhancedTrainingMonitor:
    def __init__(self):
        self.api_key = os.getenv("ML_FOUNDRY_API_KEY")
        self.headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        self.base_url = "https://api.mlfoundry.com/v2"
        self.ssh_key = os.path.expanduser("~/.ssh/mlfoundry_temp")
        
    def test_ssh_connection(self, vm_ip):
        """Test SSH connection with the passwordless key."""
        print(f"🔑 Testing SSH connection to {vm_ip}...")
        
        try:
            result = subprocess.run([
                "ssh", "-i", self.ssh_key, "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
                "echo 'SSH_SUCCESS'"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and "SSH_SUCCESS" in result.stdout:
                print("✅ SSH connection established successfully!")
                return True
            else:
                print(f"❌ SSH failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ SSH error: {e}")
            return False
    
    def get_vm_ip_from_bid(self, bid_id):
        """Extract VM IP from instances API (correct method)."""
        try:
            # Get all instances for the project
            project_id = "proj_V0uIRLp92vFAt2QR"
            response = requests.get(f"{self.base_url}/instances?project={project_id}", headers=self.headers)
            
            if response.status_code == 200:
                result = response.json()
                instances = result.get('data', [])
                
                # Find instance that matches this bid
                for instance in instances:
                    if instance.get('bid') == bid_id:
                        # Check if running
                        if instance.get('status') == 'STATUS_RUNNING':
                            ip = instance.get('ssh_destination')
                            if ip:
                                print(f"✅ Found running instance: {ip}")
                                return ip
                        else:
                            status = instance.get('status', 'unknown')
                            print(f"⏳ Instance status: {status}")
                            return None
                            
            print("🔍 No matching instance found yet")
            return None
        except Exception as e:
            print(f"Error getting VM IP: {e}")
            return None
    
    def check_training_progress(self, vm_ip):
        """Check comprehensive training progress via SSH."""
        commands = [
            ("System uptime", "uptime"),
            ("Check training process", "ps aux | grep 'python.*run_step2_training' | grep -v grep || echo 'No training process found'"),
            ("GPU utilization", "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"),
            ("Training directory", "ls -la /workspace/srag-training/ 2>/dev/null || echo 'Training directory not found'"),
            ("Recent training logs", "tail -10 /workspace/srag-training/logs/training.log 2>/dev/null || tail -10 /workspace/srag-training/logs/step2_training.log 2>/dev/null || echo 'No training logs found yet'"),
            ("Checkpoint files", "ls -la /workspace/srag-training/checkpoints/ 2>/dev/null || echo 'No checkpoints directory'"),
            ("Disk usage", "df -h /workspace"),
            ("Memory usage", "free -h"),
            ("Network connectivity", "ping -c 1 github.com >/dev/null 2>&1 && echo 'Internet: OK' || echo 'Internet: FAILED'")
        ]
        
        print(f"\n{'='*60}")
        print(f"📊 TRAINING PROGRESS CHECK - {datetime.now().strftime('%H:%M:%S')}")
        print(f"🖥️  VM: {vm_ip}")
        print(f"{'='*60}")
        
        ssh_success = True
        
        for description, command in commands:
            print(f"\n📋 {description}:")
            try:
                result = subprocess.run([
                    "ssh", "-i", self.ssh_key, "-o", "ConnectTimeout=10",
                    "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
                    command
                ], capture_output=True, text=True, timeout=15)
                
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if output:
                        print(f"✅ {output}")
                    else:
                        print("📝 No output")
                else:
                    print(f"⚠️ Command failed: {result.stderr.strip()}")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                ssh_success = False
                
        return ssh_success
    
    def monitor_bid_status(self, bid_id):
        """Monitor bid status using correct instances API."""
        try:
            # Get bid info for basic details
            response = requests.get(f"{self.base_url}/spot/bids/{bid_id}", headers=self.headers)
            if response.status_code == 200:
                bid_data = response.json()
                bid_status = bid_data.get('status', 'unknown')
                region = bid_data.get('region', 'unknown')
            else:
                bid_status = "api_error"
                region = "unknown"
                bid_data = {}
            
            # Get instance status from instances API
            project_id = "proj_V0uIRLp92vFAt2QR"
            inst_response = requests.get(f"{self.base_url}/instances?project={project_id}", headers=self.headers)
            
            instance_status = "not_found"
            if inst_response.status_code == 200:
                result = inst_response.json()
                instances = result.get('data', [])
                
                for instance in instances:
                    if instance.get('bid') == bid_id:
                        instance_status = instance.get('status', 'unknown')
                        break
            
            print(f"\n🎯 BID STATUS UPDATE:")
            print(f"   Bid: {bid_status}")
            print(f"   Instance: {instance_status}")
            print(f"   Region: {region}")
            
            return bid_status, instance_status, bid_data
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
            return "error", "unknown", {}
    
    def wait_for_vm_ready(self, bid_id, max_wait_minutes=20):
        """Wait for VM to be ready with SSH access."""
        print(f"\n⏳ Waiting for VM to be ready (max {max_wait_minutes} minutes)...")
        
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        
        while time.time() - start_time < max_wait_seconds:
            # Check bid status
            bid_status, instance_status, bid_data = self.monitor_bid_status(bid_id)
            
            if bid_status == "Allocated" and instance_status not in ["unknown", "provisioning"]:
                # Try to get VM IP
                vm_ip = self.get_vm_ip_from_bid(bid_id)
                if vm_ip:
                    print(f"🔍 Found VM IP: {vm_ip}")
                    
                    # Test SSH connection
                    if self.test_ssh_connection(vm_ip):
                        print(f"🚀 VM ready! SSH access confirmed.")
                        return vm_ip
                    else:
                        print("⏳ VM found but SSH not ready yet...")
                else:
                    print("⏳ VM provisioning, no IP yet...")
            elif bid_status != "Allocated":
                print(f"❌ Bid status changed to: {bid_status}")
                return None
            
            print(f"⏳ Still waiting... ({int(time.time() - start_time)}s elapsed)")
            time.sleep(30)  # Check every 30 seconds
        
        print(f"❌ Timeout: VM not ready after {max_wait_minutes} minutes")
        return None
    
    def continuous_monitoring(self, bid_id, vm_ip):
        """Continuously monitor training with comprehensive checks."""
        print(f"\n🔄 Starting continuous monitoring...")
        print(f"📊 Bid: {bid_id}")
        print(f"🖥️  VM: {vm_ip}")
        print("Press Ctrl+C to stop monitoring")
        
        start_time = time.time()
        check_count = 0
        
        try:
            while True:
                check_count += 1
                elapsed_hours = (time.time() - start_time) / 3600
                
                print(f"\n{'='*80}")
                print(f"🕐 MONITORING CHECK #{check_count} - Elapsed: {elapsed_hours:.1f}h")
                print(f"{'='*80}")
                
                # 1. Check bid status
                bid_status, instance_status, bid_data = self.monitor_bid_status(bid_id)
                
                # 2. Check training progress via SSH
                if vm_ip:
                    ssh_success = self.check_training_progress(vm_ip)
                    if not ssh_success:
                        print("⚠️ SSH connection lost - VM may have been terminated")
                
                # 3. Check for completion indicators
                if instance_status == "completed":
                    print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
                    break
                elif instance_status == "failed":
                    print("💥 TRAINING FAILED!")
                    break
                elif bid_status != "Allocated":
                    print(f"⚠️ Bid status changed: {bid_status}")
                    if bid_status in ["Preempted", "Terminated"]:
                        print("💥 TRAINING INTERRUPTED!")
                        break
                
                # 4. Wait for next check
                print(f"\n⏳ Next check in 60 seconds...")
                time.sleep(60)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ Monitoring stopped by user")
            print(f"🎯 Training continues running on VM: {vm_ip}")
            print(f"📊 Total monitoring time: {elapsed_hours:.1f} hours")

def main():
    print("🔧 ENHANCED SRAG-V TRAINING MONITOR")
    print("=" * 50)
    
    monitor = EnhancedTrainingMonitor()
    
    # Get bid ID from user
    bid_id = input("Enter bid ID to monitor: ").strip()
    
    if not bid_id:
        print("❌ No bid ID provided")
        return
    
    print(f"🎯 Monitoring bid: {bid_id}")
    
    # Check if we already have a VM IP
    vm_ip = monitor.get_vm_ip_from_bid(bid_id)
    
    if vm_ip:
        print(f"🔍 Found existing VM: {vm_ip}")
        if monitor.test_ssh_connection(vm_ip):
            print("✅ SSH access confirmed")
            monitor.continuous_monitoring(bid_id, vm_ip)
        else:
            print("❌ SSH access failed")
    else:
        # Wait for VM to be ready
        vm_ip = monitor.wait_for_vm_ready(bid_id)
        if vm_ip:
            monitor.continuous_monitoring(bid_id, vm_ip)
        else:
            print("❌ Could not establish monitoring")

if __name__ == "__main__":
    main()