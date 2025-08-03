#!/usr/bin/env python3
"""
Diagnose what happened to the SRAG-V training
"""
import subprocess
import os

def check_training_completion():
    ssh_key = os.path.expanduser("~/.ssh/mlfoundry_temp")
    vm_ip = "34.219.245.10"
    
    commands = [
        ("Check final training logs", "tail -20 /workspace/srag-training/logs/training.log"),
        ("Check for completion message", "grep -i 'training.*complet' /workspace/srag-training/logs/training.log"),
        ("Check for error/crash", "grep -i 'error\\|exception\\|crash\\|failed' /workspace/srag-training/logs/training.log | tail -5"),
        ("Check checkpoint contents", "ls -la /workspace/srag-training/checkpoints/step2_training/"),
        ("Check process exit code", "echo $? || echo 'No exit code available'"),
        ("Check system logs for OOM", "dmesg | grep -i 'killed process' | tail -3"),
        ("Check iteration progress", "grep -i 'iteration.*12' /workspace/srag-training/logs/training.log"),
    ]
    
    print("🔍 DIAGNOSING TRAINING COMPLETION STATUS")
    print("=" * 60)
    
    for description, command in commands:
        print(f"\n📋 {description}:")
        try:
            result = subprocess.run([
                "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
                command
            ], capture_output=True, text=True, timeout=15)
            
            if result.stdout.strip():
                print(f"✅ {result.stdout.strip()}")
            elif result.stderr.strip():
                print(f"⚠️ {result.stderr.strip()}")
            else:
                print("📝 No output")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_training_completion()