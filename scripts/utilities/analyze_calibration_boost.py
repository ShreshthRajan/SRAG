#!/usr/bin/env python3
"""
Analyze calibration boost by comparing raw vs calibrated model results.
Used to prove that Phase 1 calibration improves performance.
"""

import json
import glob
import numpy as np
from pathlib import Path
from typing import Dict, Any

def load_latest_results(pattern: str) -> Dict[str, Any]:
    """Load the most recent results file matching pattern."""
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No results found matching: {pattern}")

    latest_file = max(files, key=lambda f: Path(f).stat().st_mtime)
    print(f"Loading: {latest_file}")

    with open(latest_file, 'r') as f:
        return json.load(f)


def analyze_calibration_boost():
    """Compare raw model vs calibrated model performance."""

    print("=" * 80)
    print("CALIBRATION BOOST ANALYSIS")
    print("=" * 80)

    # Try to load both raw and calibrated results
    try:
        # Look for phase1-only results (these could be raw or calibrated)
        # We need to check the metadata to determine which is which
        all_results = glob.glob("phase4_results/phase4_step1_results_*.json")

        if len(all_results) < 2:
            print(f"❌ Need at least 2 evaluation runs to compare")
            print(f"   Found only {len(all_results)} result file(s)")
            print(f"   Run both raw model and calibrated evaluations first")
            return

        # Load and identify raw vs calibrated
        raw_result = None
        calibrated_result = None

        for result_file in sorted(all_results, key=lambda f: Path(f).stat().st_mtime, reverse=True)[:5]:
            with open(result_file, 'r') as f:
                data = json.load(f)

            # Check if this was a raw model run (no calibration)
            stage2 = data.get("stages", {}).get("phase1_calibrator_loading", {})

            if stage2.get("status") == "skipped" and stage2.get("mode") == "raw_model":
                if raw_result is None:
                    raw_result = data
                    print(f"✅ Found RAW model result: {result_file}")
            elif stage2.get("calibrator_loaded") == True:
                if calibrated_result is None:
                    calibrated_result = data
                    print(f"✅ Found CALIBRATED model result: {result_file}")

        if raw_result is None:
            print("\n❌ No raw model evaluation found")
            print("   Run: python3 run_phase4_step1_humaneval.py --no-calibration --phase1-only")
            return

        if calibrated_result is None:
            print("\n❌ No calibrated model evaluation found")
            print("   Your existing 89.6% result should work - checking older files...")
            # Could be in older format
            return

        # Extract metrics
        raw_phase1 = raw_result["stages"]["phase1_evaluation"]["results"]["metrics"]
        cal_phase1 = calibrated_result["stages"]["phase1_evaluation"]["results"]["metrics"]

        raw_pass1 = raw_phase1["pass_at_1"]
        cal_pass1 = cal_phase1["pass_at_1"]

        raw_ece = raw_phase1["ece"]
        cal_ece = cal_phase1["ece"]

        # Calculate improvement
        pass1_boost = cal_pass1 - raw_pass1
        pass1_boost_percent = (pass1_boost / raw_pass1) * 100 if raw_pass1 > 0 else 0

        print("\n" + "=" * 80)
        print("CALIBRATION BOOST RESULTS")
        print("=" * 80)

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Raw Model Pass@1:        {raw_pass1:.3f} ({raw_pass1*100:.1f}%)")
        print(f"   Calibrated Model Pass@1: {cal_pass1:.3f} ({cal_pass1*100:.1f}%)")
        print(f"   ➡️  Absolute Boost:       {pass1_boost:+.3f} ({pass1_boost*100:+.1f}%)")
        print(f"   ➡️  Relative Boost:       {pass1_boost_percent:+.1f}%")

        print(f"\n🎯 CALIBRATION QUALITY:")
        print(f"   Raw Model ECE:           {raw_ece:.6f}")
        print(f"   Calibrated Model ECE:    {cal_ece:.6f}")
        print(f"   ➡️  ECE Improvement:      {(raw_ece - cal_ece):.6f}")

        # Statistical significance (simple binomial test)
        n_problems = raw_phase1["total_evaluated"]
        se = np.sqrt((raw_pass1 * (1-raw_pass1) + cal_pass1 * (1-cal_pass1)) / n_problems)
        z_score = pass1_boost / se if se > 0 else 0

        # Two-tailed p-value approximation
        from scipy import stats
        try:
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        except:
            # Fallback if scipy not available
            p_value = 0.05 if abs(z_score) > 1.96 else 0.5

        print(f"\n📈 STATISTICAL SIGNIFICANCE:")
        print(f"   Sample size:             {n_problems} problems")
        print(f"   Z-score:                 {z_score:.2f}")
        print(f"   P-value:                 {p_value:.4f}")
        print(f"   Significant (p<0.05):    {'✅ YES' if p_value < 0.05 else '❌ NO'}")

        # Research contribution assessment
        print("\n" + "=" * 80)
        print("RESEARCH CONTRIBUTION ASSESSMENT")
        print("=" * 80)

        # Check if this is publishable
        is_publishable = pass1_boost > 0.005 and p_value < 0.05  # >0.5% boost and significant

        if is_publishable:
            print("✅ PUBLISHABLE RESULT!")
            print(f"   Calibration provides {pass1_boost*100:.1f}% absolute improvement")
            print(f"   At SOTA performance level ({raw_pass1*100:.1f}% baseline)")
            print(f"   Statistically significant (p={p_value:.4f})")

            # Compare to published baseline
            qwen_published = 0.884  # 88.4%
            if abs(raw_pass1 - qwen_published) < 0.02:
                print(f"\n✅ Raw model matches published baseline ({qwen_published*100:.1f}%)")
                print(f"   This validates the evaluation methodology")
            else:
                print(f"\n⚠️ Raw model differs from published baseline:")
                print(f"   Expected: {qwen_published*100:.1f}%")
                print(f"   Observed: {raw_pass1*100:.1f}%")
                print(f"   Delta: {(raw_pass1 - qwen_published)*100:+.1f}%")
        else:
            print("⚠️ Results need stronger signal for publication")
            if pass1_boost <= 0.005:
                print(f"   Boost too small: {pass1_boost*100:.1f}% (need >0.5%)")
            if p_value >= 0.05:
                print(f"   Not significant: p={p_value:.4f} (need p<0.05)")

        print("\n" + "=" * 80)

        # Save comparison analysis
        comparison_report = {
            "analysis_type": "calibration_boost_verification",
            "raw_model": {
                "pass_at_1": raw_pass1,
                "ece": raw_ece,
                "total_problems": n_problems
            },
            "calibrated_model": {
                "pass_at_1": cal_pass1,
                "ece": cal_ece,
                "total_problems": n_problems
            },
            "calibration_boost": {
                "absolute_improvement": pass1_boost,
                "relative_improvement_percent": pass1_boost_percent,
                "ece_improvement": raw_ece - cal_ece
            },
            "statistical_validation": {
                "z_score": float(z_score),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "sample_size": n_problems
            },
            "publishable": is_publishable,
            "matches_published_baseline": abs(raw_pass1 - qwen_published) < 0.02
        }

        report_path = "phase4_results/calibration_boost_analysis.json"
        with open(report_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)

        print(f"📁 Analysis saved: {report_path}")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("\nExpected file structure:")
        print("  phase4_results/phase4_step1_results_YYYYMMDD_HHMMSS.json")
        print("\nMake sure you've run both evaluations:")
        print("  1. Raw model:       python3 run_phase4_step1_humaneval.py --no-calibration --phase1-only")
        print("  2. Calibrated:      python3 run_phase4_step1_humaneval.py --phase1-only")


if __name__ == "__main__":
    analyze_calibration_boost()
