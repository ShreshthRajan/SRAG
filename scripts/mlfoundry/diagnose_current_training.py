#!/usr/bin/env python3
"""
Diagnose the CURRENT training run that started at 2pm (bid_MWrvNizxEIaTCgY8)
"""
import subprocess
import os

def check_current_training():
    ssh_key = os.path.expanduser("~/.ssh/mlfoundry_temp")
    vm_ip = "34.219.245.10"
    
    commands = [
        ("Check FULL training log from 2pm start", "cat /workspace/srag-training/logs/training.log | grep '2025-07-31 1[4-9]\\|2025-07-31 2[0-9]' | tail -50"),
        ("Check if training completed all 12 iterations", "grep -c 'ITERATION.*12' /workspace/srag-training/logs/training.log"),
        ("Check final completion status", "tail -100 /workspace/srag-training/logs/training.log | grep -i 'complet\\|finish\\|done\\|success'"),
        ("Check what caused training to stop", "tail -50 /workspace/srag-training/logs/training.log | grep -v 'syntax error'"),
        ("Check actual checkpoint files saved", "find /workspace/srag-training/checkpoints/ -name '*.pt' -o -name '*.bin' -o -name '*checkpoint*' | head -10"),
        ("Check system uptime vs training start", "uptime"),
        ("Check if process died with exit code", "ps aux | grep -i python | grep -v grep || echo 'No Python processes running'"),
    ]
    
    print("🔍 DIAGNOSING CURRENT TRAINING RUN (2PM-8:42PM)")
    print("=" * 60)
    
    for description, command in commands:
        print(f"\n📋 {description}:")
        try:
            result = subprocess.run([
                "ssh", "-i", ssh_key, "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=no", f"ubuntu@{vm_ip}",
                command
            ], capture_output=True, text=True, timeout=20)
            
            if result.stdout.strip():
                print(f"✅ {result.stdout.strip()}")
            elif result.stderr.strip():
                print(f"⚠️ {result.stderr.strip()}")
            else:
                print("📝 No output")
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    check_current_training()