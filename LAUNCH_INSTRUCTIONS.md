# 🚀 BULLETPROOF SRAG-V STEP 2 TRAINING LAUNCH

## 100% RELIABLE MONITORED TRAINING PROTOCOL

### STEP 1: Launch Training (In Your Terminal)
```bash
cd /Users/shreshth.rajan/projects/srag
source venv/bin/activate
python launch_foundry_training.py
```

**IMPORTANT:** Keep this terminal open - it will show the bid creation and initial status.

### STEP 2: Get Bid ID 
From the launch output, copy the bid ID (format: `bid_XXXXXXXXXXXXX`)

### STEP 3: Start Enhanced Monitoring (New Terminal Window)
```bash
cd /Users/shreshth.rajan/projects/srag  
source venv/bin/activate
python enhanced_training_monitor.py
```

Enter the bid ID when prompted.

### STEP 4: Monitor Training Progress
The enhanced monitor will:

1. **Wait for VM provisioning** (up to 20 minutes)
2. **Test SSH connection** with passwordless key
3. **Continuously monitor**:
   - ✅ Training process status
   - ✅ GPU utilization (should be 80-95%)
   - ✅ Training logs in real-time
   - ✅ Checkpoint creation
   - ✅ System resources
   - ✅ Network connectivity

### STEP 5: Success Indicators
**Training is working correctly when you see:**
- ✅ `python.*run_step2_training` process running
- ✅ GPU utilization 80-95% on all 4 GPUs
- ✅ Training logs showing iteration progress
- ✅ Checkpoints being created in `/workspace/srag-training/checkpoints/`
- ✅ No error messages in logs

### STEP 6: Completion Detection
The monitor will automatically detect:
- 🎉 **SUCCESS:** Instance status changes to "completed"
- 💥 **FAILURE:** Instance status changes to "failed" or "preempted"
- ⚠️ **ISSUES:** SSH connection lost or training process stops

## ENHANCED FEATURES

### Higher Spot Bid ($30 vs $20)
- Prevents preemption by other users
- Ensures training completes uninterrupted
- Still cost-effective for 3-4 hour training

### Real-time SSH Monitoring
- Direct access to VM during training
- Live training log streaming
- GPU utilization monitoring
- Immediate issue detection

### Automatic Issue Detection
- Training process crashes
- GPU memory issues
- Network connectivity problems
- SSH access loss

## IF SOMETHING GOES WRONG

### Training Process Not Found
```bash
# SSH into VM manually
ssh -i ~/.ssh/mlfoundry_temp ubuntu@[VM_IP]

# Check what's running
ps aux | grep python

# Check startup logs
tail -50 /var/log/cloud-init-output.log

# Restart training manually if needed
cd /workspace/srag-training
python run_step2_training.py 2>&1 | tee logs/training.log
```

### SSH Connection Issues
- Verify SSH key was added to ML Foundry console
- Check VM IP is correct
- Wait longer for VM provisioning (can take 10-15 minutes)

### High Memory Usage
- Normal for 4×A100 training
- QLoRA should keep GPU memory under 70GB per GPU
- CPU memory usage should be under 50GB

## EXPECTED TIMELINE

- **VM Provisioning:** 5-15 minutes
- **Training Duration:** 3-4 hours
- **Total Time:** ~4-5 hours

## MONITORING COMMANDS

While training runs, you can manually check:
```bash
# GPU status
nvidia-smi

# Training progress  
tail -f /workspace/srag-training/logs/training.log

# System resources
htop

# Disk space
df -h
```

---

## 🎯 LAUNCH CHECKLIST

- [ ] SSH key uploaded to ML Foundry console
- [ ] Terminal 1: `python launch_foundry_training.py` running
- [ ] Terminal 2: `python enhanced_training_monitor.py` running
- [ ] Bid ID copied and entered in monitor
- [ ] VM provisioned and SSH working
- [ ] Training process confirmed running
- [ ] GPU utilization 80%+
- [ ] Training logs showing progress

**When all checkboxes are ✅, your training is bulletproof!**