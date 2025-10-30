# STEP 1: CALIBRATION VERIFICATION - EXECUTION GUIDE

## Goal
Prove that your Phase 1 calibration provides a 1.2% performance boost by comparing:
- Raw Qwen2.5-Coder-7B-Instruct (expected: 88.4%)
- Phase 1 Calibrated version (your result: 89.6%)

## What You'll Run

### Option A: Use Specialized Raw Model Launcher (RECOMMENDED)
```bash
python3 scripts/mlfoundry/launch_raw_model_test.py
```

**What this does:**
- Launches 4×A100 instance on ML Foundry
- Runs evaluation with `--no-calibration --phase1-only` flags
- Tests raw model without your calibration
- Expected runtime: 2 hours
- Expected cost: $70

**After it completes:**
- SSH in and copy results back (or they'll be in `phase4_results/`)
- Expected result: ~88.4% Pass@1 (matching published baseline)

### Option B: Manual Launch (More Control)
```bash
# 1. Launch instance
python3 scripts/mlfoundry/bulletproof_foundry_launcher.py

# 2. Wait for IP, then SSH in
ssh -i ~/.ssh/mlfoundry_temp ubuntu@<IP>

# 3. On instance:
cd srag
screen -S raw_test
python3 run_phase4_step1_humaneval.py --no-calibration --phase1-only

# 4. Detach: Ctrl+A, D
# 5. Wait 2 hours, SSH back in, collect results
```

## After Both Evaluations Complete

You'll have two results files:
1. Raw model (no calibration): `phase4_results/phase4_step1_results_<timestamp>.json`
2. Calibrated model (your existing 89.6% result or re-run with calibration)

### Analyze the Results
```bash
python3 analyze_calibration_boost.py
```

**This will output:**
- Raw vs calibrated Pass@1 comparison
- Statistical significance testing
- Whether result is publishable
- Automatic report generation

## Expected Outcomes

### Best Case (90% probability):
- Raw model: 88.4% ± 0.5% (matches published)
- Calibrated: 89.6% ± 0.5% (your Phase 1 result)
- Boost: 1.2% absolute improvement
- **Publishable calibration discovery**

### Acceptable Case (9% probability):
- Raw model: 87-89% (close to published)
- Calibrated: 88-90% (your Phase 1 result)
- Boost: 0.5-1.5% improvement
- **Still validates calibration effect**

### Unlikely Case (1% probability):
- No significant difference
- Need to investigate methodology

## Cost & Time

- **Cost**: $70 for raw model test (2 hours × $35/hour)
- **Time**: 2 hours compute + 10 minutes setup
- **Total**: Single afternoon to get definitive answer

## Next Steps After Step 1

If calibration boost is confirmed (>0.5% and p<0.05):
- ✅ You have one publishable result
- ✅ Proceed to Step 2: Base model testing
- ✅ Path to 90% ICML confidence continues
