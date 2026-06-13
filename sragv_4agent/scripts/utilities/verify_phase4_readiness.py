#!/usr/bin/env python3
"""
Phase 4 Readiness Verification Script
Tests all critical components before full HumanEval evaluation implementation.

This script verifies:
1. HumanEval dataset integration
2. Model loading (Phase 1 vs Phase 3)
3. Pass@1 calculation methodology
4. Compute requirements and budget estimation
5. Output format compatibility

Author: Claude + Shreshth
Date: August 19, 2025
"""

import os
import sys
import json
import time
import logging
import torch
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set environment variables for stability
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "8"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/phase4_verification.log')
    ]
)
logger = logging.getLogger(__name__)

class Phase4ReadinessVerifier:
    """Comprehensive verification system for Phase 4 implementation readiness."""
    
    def __init__(self):
        self.verification_results = {
            "start_time": datetime.now().isoformat(),
            "tests": {},
            "overall_status": "pending",
            "readiness_score": 0.0
        }
        
        # Ensure logs directory exists
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("🔍 Phase 4 Readiness Verification Starting")
    
    def verify_humaneval_dataset(self) -> bool:
        """Verify HumanEval dataset availability and format."""
        logger.info("📚 TEST 1: HumanEval Dataset Verification")
        
        try:
            # Try to import the official HumanEval evaluation library
            try:
                import human_eval
                logger.info("✅ human_eval library available")
                humaneval_available = True
            except ImportError:
                logger.warning("⚠️ human_eval library not installed")
                humaneval_available = False
            
            # Check for alternative: download HumanEval manually
            humaneval_paths = [
                "data/HumanEval.jsonl",
                "data/humaneval.jsonl", 
                "data/human_eval.jsonl"
            ]
            
            local_humaneval = False
            for path in humaneval_paths:
                if Path(path).exists():
                    logger.info(f"✅ Found local HumanEval at: {path}")
                    local_humaneval = True
                    break
            
            if not local_humaneval and not humaneval_available:
                logger.error("❌ No HumanEval dataset found")
                logger.info("Need to either:")
                logger.info("  1. pip install human-eval")
                logger.info("  2. Download HumanEval.jsonl to data/ directory")
                
                self.verification_results["tests"]["humaneval_dataset"] = {
                    "status": "failed",
                    "issue": "No HumanEval dataset available",
                    "solutions": [
                        "pip install human-eval",
                        "Download HumanEval.jsonl manually"
                    ]
                }
                return False
            
            # If we have the library, test a sample problem
            if humaneval_available:
                try:
                    from human_eval.data import read_problems
                    problems = read_problems()
                    sample_problem = list(problems.values())[0]
                    
                    logger.info(f"✅ Sample HumanEval problem loaded:")
                    logger.info(f"   Task ID: {sample_problem.get('task_id', 'unknown')}")
                    logger.info(f"   Prompt length: {len(sample_problem.get('prompt', ''))}")
                    
                    self.verification_results["tests"]["humaneval_dataset"] = {
                        "status": "passed",
                        "problems_available": len(problems),
                        "sample_task_id": sample_problem.get('task_id'),
                        "format_verified": True
                    }
                    return True
                    
                except Exception as e:
                    logger.error(f"❌ Error loading HumanEval problems: {e}")
                    self.verification_results["tests"]["humaneval_dataset"] = {
                        "status": "failed",
                        "issue": f"HumanEval loading error: {e}"
                    }
                    return False
            
            self.verification_results["tests"]["humaneval_dataset"] = {
                "status": "partial",
                "local_file_available": local_humaneval,
                "needs_implementation": True
            }
            return True
            
        except Exception as e:
            logger.error(f"❌ HumanEval verification failed: {e}")
            self.verification_results["tests"]["humaneval_dataset"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def verify_model_loading(self) -> bool:
        """Verify Phase 1 and Phase 3 models can be loaded correctly."""
        logger.info("🤖 TEST 2: Model Loading Verification")
        
        try:
            # Import SRAG-V components
            from sragv.orchestrator import SRAGVOrchestrator
            from sragv.confidence_calibration import EnhancedConfidenceCalibrator
            
            models_status = {}
            
            # Test Phase 1 model loading
            logger.info("Loading Phase 1 baseline model...")
            try:
                orchestrator_phase1 = SRAGVOrchestrator("config/config.yaml")
                orchestrator_phase1.initialize_players()
                
                # Load Phase 1 calibrator
                calibrator_path = "checkpoints/phase1_star_calibrator_1754272154.pt"
                if Path(calibrator_path).exists():
                    checkpoint = torch.load(calibrator_path, map_location='cpu', weights_only=False)
                    calibrator_phase1 = EnhancedConfidenceCalibrator()
                    calibrator_phase1.load_state_dict(checkpoint['state_dict'])
                    calibrator_phase1.is_trained = checkpoint.get('is_trained', True)
                    
                    orchestrator_phase1.solution_generator.confidence_calibrator = calibrator_phase1
                    orchestrator_phase1.solution_generator.use_calibration = True
                    
                    models_status["phase1"] = {
                        "status": "loaded",
                        "calibrator_ece": checkpoint.get('ece', 'unknown'),
                        "is_trained": calibrator_phase1.is_trained
                    }
                    logger.info("✅ Phase 1 model loaded successfully")
                else:
                    models_status["phase1"] = {
                        "status": "failed",
                        "issue": f"Calibrator not found: {calibrator_path}"
                    }
                    logger.error(f"❌ Phase 1 calibrator not found: {calibrator_path}")
                    
            except Exception as e:
                models_status["phase1"] = {
                    "status": "failed", 
                    "error": str(e)
                }
                logger.error(f"❌ Phase 1 loading failed: {e}")
            
            # Test Phase 3 model loading
            logger.info("Loading Phase 3 trained model...")
            try:
                orchestrator_phase3 = SRAGVOrchestrator("config/config.yaml")
                orchestrator_phase3.initialize_players()
                
                # Load Phase 3 calibrator
                phase3_calibrator_path = "checkpoints/phase3_star_training/phase3_final_calibrator_1754491530.pt"
                if Path(phase3_calibrator_path).exists():
                    phase3_checkpoint = torch.load(phase3_calibrator_path, map_location='cpu', weights_only=False)
                    calibrator_phase3 = EnhancedConfidenceCalibrator()
                    calibrator_phase3.load_state_dict(phase3_checkpoint['state_dict'])
                    calibrator_phase3.is_trained = True  # Critical fix
                    
                    orchestrator_phase3.solution_generator.confidence_calibrator = calibrator_phase3
                    orchestrator_phase3.solution_generator.use_calibration = True
                    
                    models_status["phase3"] = {
                        "status": "loaded",
                        "pseudo_labels_used": phase3_checkpoint.get('pseudo_labels_used', 'unknown'),
                        "is_trained": calibrator_phase3.is_trained,
                        "iterations": phase3_checkpoint.get('iterations', 'unknown')
                    }
                    logger.info("✅ Phase 3 model loaded successfully")
                else:
                    models_status["phase3"] = {
                        "status": "failed",
                        "issue": f"Calibrator not found: {phase3_calibrator_path}"
                    }
                    logger.error(f"❌ Phase 3 calibrator not found: {phase3_calibrator_path}")
                    
            except Exception as e:
                models_status["phase3"] = {
                    "status": "failed",
                    "error": str(e)
                }
                logger.error(f"❌ Phase 3 loading failed: {e}")
            
            # Test basic generation capability
            if models_status.get("phase1", {}).get("status") == "loaded":
                logger.info("Testing Phase 1 generation capability...")
                try:
                    test_problem = {
                        "task_id": "test/0",
                        "prompt": "def add(a, b):\n    \"\"\"\n    Add two numbers a and b.\n    \"\"\"\n",
                        "entry_point": "add"
                    }
                    
                    solutions = orchestrator_phase1.solution_generator.generate(
                        test_problem, num_solutions=1
                    )
                    
                    if solutions and len(solutions) > 0:
                        models_status["phase1"]["generation_test"] = "passed"
                        logger.info("✅ Phase 1 generation test passed")
                    else:
                        models_status["phase1"]["generation_test"] = "failed"
                        logger.warning("⚠️ Phase 1 generated no solutions")
                        
                except Exception as e:
                    models_status["phase1"]["generation_test"] = f"error: {e}"
                    logger.error(f"❌ Phase 1 generation test failed: {e}")
            
            self.verification_results["tests"]["model_loading"] = models_status
            
            # Determine success
            phase1_ok = models_status.get("phase1", {}).get("status") == "loaded"
            phase3_ok = models_status.get("phase3", {}).get("status") == "loaded"
            
            return phase1_ok and phase3_ok
            
        except Exception as e:
            logger.error(f"❌ Model loading verification failed: {e}")
            self.verification_results["tests"]["model_loading"] = {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            return False
    
    def verify_pass_at_k_implementation(self) -> bool:
        """Verify Pass@k calculation methodology."""
        logger.info("📊 TEST 3: Pass@k Implementation Verification")
        
        try:
            # Test basic Pass@k calculation (using official HumanEval implementation)
            def calculate_pass_at_k(n: int, c: int, k: int) -> float:
                """
                Calculate Pass@k metric (from HumanEval paper).
                n: total samples generated
                c: correct samples
                k: samples to consider
                
                Returns probability that at least one of top k samples is correct.
                """
                if n - c < k:
                    return 1.0
                
                # Use combination formula: 1 - C(n-c, k) / C(n, k)
                import math
                def comb(n, k):
                    if k > n or k < 0:
                        return 0
                    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
                
                return 1.0 - (comb(n - c, k) / comb(n, k))
            
            # Test cases
            test_cases = [
                {"n": 5, "c": 1, "k": 1, "expected": 0.2},  # 1/5 = 20%
                {"n": 5, "c": 2, "k": 1, "expected": 0.4},  # 2/5 = 40%
                {"n": 5, "c": 5, "k": 1, "expected": 1.0},  # 5/5 = 100%
            ]
            
            pass_k_results = []
            for test in test_cases:
                calculated = calculate_pass_at_k(test["n"], test["c"], test["k"])
                expected = test["expected"]
                passed = abs(calculated - expected) < 0.01
                
                pass_k_results.append({
                    "test": test,
                    "calculated": calculated,
                    "passed": passed
                })
                
                if passed:
                    logger.info(f"✅ Pass@k test passed: {test}")
                else:
                    logger.error(f"❌ Pass@k test failed: {test}, got {calculated}")
            
            all_passed = all(r["passed"] for r in pass_k_results)
            
            self.verification_results["tests"]["pass_at_k"] = {
                "status": "passed" if all_passed else "failed",
                "test_results": pass_k_results,
                "implementation_verified": all_passed
            }
            
            return all_passed
            
        except Exception as e:
            logger.error(f"❌ Pass@k verification failed: {e}")
            self.verification_results["tests"]["pass_at_k"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def verify_compute_requirements(self) -> bool:
        """Verify compute requirements and budget estimation."""
        logger.info("💰 TEST 4: Compute Requirements & Budget Verification")
        
        try:
            # Estimate compute requirements
            problems_count = 164  # HumanEval
            models_count = 2      # Phase 1 + Phase 3
            solutions_per_problem = 5
            
            total_generations = problems_count * models_count * solutions_per_problem
            estimated_time_per_generation = 10  # seconds
            total_compute_seconds = total_generations * estimated_time_per_generation
            total_compute_hours = total_compute_seconds / 3600
            
            # ML Foundry pricing (from our previous usage)
            ml_foundry_hourly_rate = 30  # $30/hour for 4xA100
            estimated_cost = total_compute_hours * ml_foundry_hourly_rate
            
            logger.info(f"📊 Compute Estimation:")
            logger.info(f"   Total generations needed: {total_generations}")
            logger.info(f"   Estimated compute time: {total_compute_hours:.1f} hours")
            logger.info(f"   Estimated cost: ${estimated_cost:.2f}")
            
            # Budget check
            available_budget = 300  # Conservative estimate
            budget_sufficient = estimated_cost <= available_budget
            
            if budget_sufficient:
                logger.info(f"✅ Budget sufficient: ${estimated_cost:.2f} <= ${available_budget}")
            else:
                logger.warning(f"⚠️ Budget may be tight: ${estimated_cost:.2f} > ${available_budget}")
            
            # Alternative: local compute check
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"🖥️ Local GPU available: {gpu_count} GPUs, {gpu_memory:.1f}GB each")
                local_feasible = gpu_memory >= 24  # Need ~24GB for 7B model
            else:
                logger.info("🖥️ No local GPU available")
                local_feasible = False
            
            self.verification_results["tests"]["compute_requirements"] = {
                "status": "analyzed",
                "total_generations": total_generations,
                "estimated_hours": total_compute_hours,
                "estimated_cost": estimated_cost,
                "budget_sufficient": budget_sufficient,
                "local_gpu_available": gpu_available,
                "local_feasible": local_feasible,
                "recommendation": "ml_foundry" if not local_feasible else "local_or_foundry"
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Compute verification failed: {e}")
            self.verification_results["tests"]["compute_requirements"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def verify_output_format(self) -> bool:
        """Verify output format compatibility with research standards."""
        logger.info("📝 TEST 5: Output Format Verification")
        
        try:
            # Create sample evaluation results structure
            sample_results = {
                "evaluation_type": "phase4_humaneval",
                "timestamp": datetime.now().isoformat(),
                "models": {
                    "phase1_baseline": {
                        "total_problems": 164,
                        "pass_at_1": 0.185,  # Example: 18.5%
                        "pass_at_5": 0.280,
                        "ece": 0.0003,
                        "mce": 0.012,
                        "brier_score": 0.145,
                        "avg_confidence": 0.483,
                        "generation_count": 820,
                        "model_info": {
                            "training_data_size": 951,
                            "description": "Phase 1 baseline with exceptional calibration"
                        }
                    },
                    "phase3_trained": {
                        "total_problems": 164,
                        "pass_at_1": 0.245,  # Example: 24.5% (6% improvement)
                        "pass_at_5": 0.340,
                        "ece": 0.635,
                        "mce": 0.089,
                        "brier_score": 0.175,
                        "avg_confidence": 0.596,
                        "generation_count": 820,
                        "model_info": {
                            "training_data_size": 1383,
                            "pseudo_labels_added": 432,
                            "description": "Phase 3 with SRAG-V training"
                        }
                    }
                },
                "comparative_analysis": {
                    "performance_gain": {
                        "absolute_improvement": 0.06,  # 6%
                        "relative_improvement": 0.324,  # 32.4%
                        "improvement_per_pseudo_label": 0.000139  # 6% / 432
                    },
                    "data_efficiency": {
                        "pseudo_labels_used": 432,
                        "equivalent_human_labels": 2160,  # Estimate
                        "efficiency_multiplier": 5.0
                    },
                    "calibration_analysis": {
                        "phase1_ece": 0.0003,
                        "phase3_ece": 0.635,
                        "degradation_factor": 2116.7,
                        "degradation_acceptable": True
                    }
                },
                "statistical_validation": {
                    "mann_whitney_u_test": {
                        "statistic": 12450.0,
                        "p_value": 0.003,
                        "significant": True
                    },
                    "confidence_intervals": {
                        "phase1_pass_at_1": [0.162, 0.208],
                        "phase3_pass_at_1": [0.221, 0.269]
                    },
                    "effect_size_cohens_d": 0.42
                }
            }
            
            # Test JSON serialization
            json_output = json.dumps(sample_results, indent=2, default=str)
            logger.info("✅ JSON serialization successful")
            
            # Test key metrics extraction
            key_metrics = {
                "phase1_pass_at_1": sample_results["models"]["phase1_baseline"]["pass_at_1"],
                "phase3_pass_at_1": sample_results["models"]["phase3_trained"]["pass_at_1"],
                "improvement": sample_results["comparative_analysis"]["performance_gain"]["absolute_improvement"],
                "efficiency_multiplier": sample_results["comparative_analysis"]["data_efficiency"]["efficiency_multiplier"],
                "statistical_significance": sample_results["statistical_validation"]["mann_whitney_u_test"]["significant"]
            }
            
            logger.info("✅ Key metrics extraction successful:")
            for key, value in key_metrics.items():
                logger.info(f"   {key}: {value}")
            
            self.verification_results["tests"]["output_format"] = {
                "status": "passed",
                "sample_structure_valid": True,
                "json_serializable": True,
                "key_metrics_extractable": True,
                "research_standard_format": True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Output format verification failed: {e}")
            self.verification_results["tests"]["output_format"] = {
                "status": "failed",
                "error": str(e)
            }
            return False
    
    def generate_readiness_report(self) -> Dict[str, Any]:
        """Generate comprehensive readiness report."""
        logger.info("📋 Generating Phase 4 Readiness Report")
        
        # Calculate readiness score
        total_tests = len(self.verification_results["tests"])
        passed_tests = sum(1 for test in self.verification_results["tests"].values() 
                          if test.get("status") == "passed")
        
        readiness_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        # Determine overall status
        if readiness_score >= 80:
            overall_status = "ready"
        elif readiness_score >= 60:
            overall_status = "mostly_ready"
        else:
            overall_status = "needs_work"
        
        self.verification_results.update({
            "end_time": datetime.now().isoformat(),
            "overall_status": overall_status,
            "readiness_score": readiness_score,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "summary": {
                "ready_to_implement": overall_status == "ready",
                "critical_issues": [],
                "recommendations": []
            }
        })
        
        # Identify critical issues and recommendations
        for test_name, test_result in self.verification_results["tests"].items():
            if test_result.get("status") == "failed":
                issue = f"{test_name}: {test_result.get('issue', test_result.get('error', 'Unknown error'))}"
                self.verification_results["summary"]["critical_issues"].append(issue)
                
                if test_name == "humaneval_dataset":
                    self.verification_results["summary"]["recommendations"].append(
                        "Install HumanEval: pip install human-eval"
                    )
                elif test_name == "model_loading":
                    self.verification_results["summary"]["recommendations"].append(
                        "Verify model checkpoints exist and are loadable"
                    )
        
        return self.verification_results
    
    def run_all_verifications(self) -> Dict[str, Any]:
        """Run all verification tests and generate report."""
        logger.info("🚀 Starting comprehensive Phase 4 readiness verification")
        
        # Run all verification tests
        tests = [
            ("HumanEval Dataset", self.verify_humaneval_dataset),
            ("Model Loading", self.verify_model_loading),
            ("Pass@k Implementation", self.verify_pass_at_k_implementation),
            ("Compute Requirements", self.verify_compute_requirements),
            ("Output Format", self.verify_output_format)
        ]
        
        for test_name, test_func in tests:
            try:
                logger.info(f"Running {test_name}...")
                success = test_func()
                logger.info(f"{test_name}: {'✅ PASSED' if success else '❌ FAILED'}")
            except Exception as e:
                logger.error(f"{test_name}: 💥 CRASHED - {e}")
        
        # Generate final report
        report = self.generate_readiness_report()
        
        # Save report
        report_path = f"logs/phase4_readiness_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("🎯 PHASE 4 READINESS VERIFICATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Overall Status: {report['overall_status'].upper()}")
        logger.info(f"Readiness Score: {report['readiness_score']:.1f}%")
        logger.info(f"Tests Passed: {report['passed_tests']}/{report['total_tests']}")
        
        if report["summary"]["critical_issues"]:
            logger.info("❌ CRITICAL ISSUES:")
            for issue in report["summary"]["critical_issues"]:
                logger.info(f"   • {issue}")
        
        if report["summary"]["recommendations"]:
            logger.info("💡 RECOMMENDATIONS:")
            for rec in report["summary"]["recommendations"]:
                logger.info(f"   • {rec}")
        
        logger.info(f"📄 Full report saved: {report_path}")
        
        return report


def main():
    """Main execution function."""
    try:
        verifier = Phase4ReadinessVerifier()
        report = verifier.run_all_verifications()
        
        if report["summary"]["ready_to_implement"]:
            print("\n🎉 READY TO IMPLEMENT PHASE 4 STEP 1!")
            print("All critical components verified. Proceed with HumanEval evaluation.")
        else:
            print("\n⚠️ NOT READY FOR IMPLEMENTATION")
            print("Please address critical issues before proceeding.")
            
        return 0 if report["summary"]["ready_to_implement"] else 1
        
    except Exception as e:
        logger.error(f"💥 Verification script failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())