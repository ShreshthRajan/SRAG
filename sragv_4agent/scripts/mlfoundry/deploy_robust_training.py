#!/usr/bin/env python3
"""
Deploy Robust SRAG-V Training to ML Foundry VM
Uploads fixed solution generator and starts bulletproof training.
"""

import os
import subprocess
import time
from pathlib import Path

def deploy_to_vm():
    """Deploy the robust training system to the ML Foundry VM."""
    vm_ip = "34.219.245.10"
    ssh_key = os.path.expanduser("~/.ssh/mlfoundry_temp")
    
    print("🚀 DEPLOYING ROBUST SRAG-V TRAINING")
    print("=" * 50)
    
    # Files to upload
    files_to_upload = [
        ("src/sragv/models/solution_generator.py", "/workspace/srag-training/src/sragv/models/solution_generator.py"),
        ("src/sragv/models/base_player.py", "/workspace/srag-training/src/sragv/models/base_player.py"),
        ("run_step2_training_robust.py", "/workspace/srag-training/run_step2_training_robust.py"),
        ("bulletproof_training_launcher.py", "/workspace/srag-training/bulletproof_training_launcher.py"),
    ]
    
    print("📂 Uploading fixed files...")
    
    for local_file, remote_file in files_to_upload:
        if not Path(local_file).exists():
            print(f"❌ Local file not found: {local_file}")
            continue
            
        print(f"📤 Uploading {local_file} -> {remote_file}")
        
        # Create directory if needed
        remote_dir = str(Path(remote_file).parent)
        subprocess.run([
            "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
            f"mkdir -p {remote_dir}"
        ], check=True)
        
        # Upload file
        result = subprocess.run([
            "scp", "-i", ssh_key, "-o", "StrictHostKeyChecking=no",
            local_file, f"ubuntu@{vm_ip}:{remote_file}"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Uploaded: {local_file}")
        else:
            print(f"❌ Failed to upload {local_file}: {result.stderr}")
            return False
    
    print("\n🔧 Setting up bulletproof training environment...")
    
    # Create bulletproof config
    config_setup = '''
cat > /workspace/srag-training/config/bulletproof_config.json << EOF
{
  "training_script": "run_step2_training_robust.py",
  "max_training_hours": 8,
  "health_check_interval": 45,
  "memory_threshold_gb": 200,
  "gpu_utilization_threshold": 3,
  "restart_delay_seconds": 20,
  "max_restarts": 3,
  "checkpoint_interval_minutes": 20,
  "enable_automatic_recovery": true,
  "enable_memory_monitoring": true,
  "enable_gpu_monitoring": true
}
EOF
'''
    
    result = subprocess.run([
        "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
        config_setup
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Bulletproof config created")
    else:
        print(f"❌ Failed to create config: {result.stderr}")
        return False
    
    # Make scripts executable
    subprocess.run([
        "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
        "chmod +x /workspace/srag-training/*.py"
    ])
    
    print("\n🛡️ Starting bulletproof training...")
    
    # Start bulletproof training in background
    start_command = '''
cd /workspace/srag-training && 
nohup python bulletproof_training_launcher.py > logs/bulletproof_launcher.log 2>&1 &
echo "Bulletproof training started with PID: $!"
'''
    
    result = subprocess.run([
        "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
        start_command
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Bulletproof training launched!")
        print(f"📊 Output: {result.stdout.strip()}")
        
        # Verify it's running
        time.sleep(5)
        check_result = subprocess.run([
            "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
            "ps aux | grep bulletproof_training_launcher | grep -v grep"
        ], capture_output=True, text=True)
        
        if check_result.returncode == 0 and check_result.stdout.strip():
            print("✅ Confirmed: Bulletproof training is running")
            print("\n🎯 DEPLOYMENT SUCCESSFUL!")
            print("📊 Monitor with: python enhanced_training_monitor.py")
            print("🆔 Bid ID: bid_MWrvNizxEIaTCgY8")
            return True
        else:
            print("⚠️ Warning: Could not confirm training is running")
            return False
    else:
        print(f"❌ Failed to start training: {result.stderr}")
        return False

def show_monitoring_instructions():
    """Show instructions for monitoring the robust training."""
    print("\n" + "=" * 60)
    print("🔍 MONITORING THE ROBUST TRAINING")
    print("=" * 60)
    print()
    print("1. Real-time monitoring:")
    print("   python enhanced_training_monitor.py")
    print("   Enter bid ID: bid_MWrvNizxEIaTCgY8")
    print()
    print("2. Direct SSH monitoring:")
    print("   ssh -i ~/.ssh/mlfoundry_temp ubuntu@34.219.245.10")
    print("   tail -f /workspace/srag-training/logs/bulletproof_launcher.log")
    print()
    print("3. Check training logs:")
    print("   tail -f /workspace/srag-training/logs/step2_training_robust.log")
    print()
    print("4. Expected improvements:")
    print("   ✅ Syntax error rate: <10% (down from 90%+)")
    print("   ✅ Automatic error recovery")
    print("   ✅ Training state persistence")
    print("   ✅ Real-time health monitoring")
    print("   ✅ Memory leak detection")
    print("   ✅ GPU utilization tracking")
    print()
    print("5. Success indicators:")
    print("   ✅ 'Generated 14/14 solutions (success rate: 85%+)'")
    print("   ✅ 'System healthy' in monitoring checks")
    print("   ✅ 'ITERATION X/12' progress messages")
    print("   ✅ Stable GPU utilization 15-30%")

if __name__ == "__main__":
    success = deploy_to_vm()
    
    if success:
        show_monitoring_instructions()
        print("\n🎉 ROBUST TRAINING DEPLOYMENT COMPLETE!")
        print("🛡️ Your training is now bulletproof with SOTA techniques.")
    else:
        print("\n❌ DEPLOYMENT FAILED")
        print("Check the error messages above and retry.")