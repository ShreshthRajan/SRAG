#!/usr/bin/env python3
"""
Monitor SRAG-V training directly via SSH to the running VM.
"""

import subprocess
import time
import os
from pathlib import Path

def check_ssh_connection():
    """Test SSH connection to the training VM."""
    vm_ip = "52.39.194.7"
    ssh_key = os.path.expanduser("~/.ssh/id_ed25519")
    
    print(f"🔍 Testing SSH connection to training VM: {vm_ip}")
    
    try:
        # Test basic connection
        result = subprocess.run([
            "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10", 
            "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}", 
            "echo 'SSH connection successful'"
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("✅ SSH connection successful!")
            return True
        else:
            print(f"❌ SSH connection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ SSH error: {e}")
        return False

def monitor_training_progress():
    """Monitor training progress via SSH."""
    vm_ip = "52.39.194.7"
    ssh_key = os.path.expanduser("~/.ssh/id_ed25519")
    
    commands = [
        ("Check if training process is running", "ps aux | grep python | grep -v grep"),
        ("Check GPU usage", "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv"),
        ("Check training logs (last 20 lines)", "tail -20 /workspace/srag-training/logs/training.log 2>/dev/null || echo 'No training log found yet'"),
        ("Check if training script exists", "ls -la /workspace/srag-training/ 2>/dev/null || echo 'Training directory not found'"),
        ("Check current directory contents", "pwd && ls -la"),
        ("Check system uptime", "uptime"),
    ]
    
    print(f"🔍 Monitoring training progress on VM: {vm_ip}")
    print("=" * 60)
    
    for description, command in commands:
        print(f"\n📋 {description}:")
        try:
            result = subprocess.run([
                "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10", 
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
                print(f"❌ Command failed: {result.stderr.strip()}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        time.sleep(1)  # Brief pause between commands

def continuous_monitoring():
    """Continuously monitor training every 30 seconds."""
    print("🔄 Starting continuous training monitoring...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            print(f"\n{'='*60}")
            print(f"🕐 Training Status Check - {time.strftime('%H:%M:%S')}")
            print(f"{'='*60}")
            
            monitor_training_progress()
            
            print(f"\n⏳ Waiting 30 seconds for next check...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped by user")
        print("🎯 Training continues running on the VM")

if __name__ == "__main__":
    print("🔧 SRAG-V TRAINING DIRECT MONITORING")
    print("="*50)
    
    # First test connection
    if check_ssh_connection():
        print(f"\n🚀 Connection established! Monitoring training...")
        
        # Ask user what they want to do
        print(f"\nOptions:")
        print(f"1. One-time status check")
        print(f"2. Continuous monitoring (every 30 seconds)")
        
        choice = input(f"\nChoose option (1 or 2): ").strip()
        
        if choice == "2":
            continuous_monitoring()
        else:
            monitor_training_progress()
            
    else:
        print(f"\n❌ Cannot connect to training VM")
        print(f"🔧 This could mean:")
        print(f"   - VM is still starting up")
        print(f"   - SSH key needs to be added to VM")
        print(f"   - Network connectivity issues")
        print(f"\n💡 ML Foundry staff confirmed VM is running at 52.39.194.7")
        print(f"   Try again in a few minutes if VM is still provisioning")