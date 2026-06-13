#!/usr/bin/env python3
"""
Comprehensive Integration Testing for All 4 SRAG-V Players
Tests all players with SOTA July 2025 upgrades for robustness and functionality.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragv.models.problem_generator import ProblemGenerator
from sragv.models.solution_generator import SolutionGenerator  
from sragv.models.verification_generator import VerificationGenerator
from sragv.models.meta_verifier import MetaVerifier

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SRAGVPlayersIntegrationTest:
    """Comprehensive integration test for all 4 SRAG-V players."""
    
    def __init__(self):
        self.test_results = {
            "problem_generator": {"status": "pending", "details": {}},
            "solution_generator": {"status": "pending", "details": {}},
            "verification_generator": {"status": "pending", "details": {}},
            "meta_verifier": {"status": "pending", "details": {}},
            "integration": {"status": "pending", "details": {}}
        }
        
        # Test problem for integration testing
        self.test_problem = {
            "problem_id": "test_001",
            "title": "Sum of Two Numbers",
            "question": "Write a function that takes two integers and returns their sum.",
            "function_signature": "def add_two_numbers(a: int, b: int) -> int:",
            "examples": "add_two_numbers(2, 3) -> 5\nadd_two_numbers(-1, 1) -> 0",
            "difficulty": "easy",
            "type": "arithmetic"
        }
        
        # Sample solutions for testing
        self.test_solutions = [
            {"code": "def add_two_numbers(a, b):\n    return a + b", "solution_id": "sol_001"},
            {"code": "def add_two_numbers(a, b):\n    return a + b + 1", "solution_id": "sol_002"},  # Incorrect
            {"code": "def add_two_numbers(a, b):\n    result = a + b\n    return result", "solution_id": "sol_003"}
        ]
    
    def create_player_configs(self) -> Dict:
        """Create standardized configs for all players."""
        base_config = {
            "max_length": 2048,
            "temperature": 0.3,
            "top_p": 0.9,
            "lora_config": None  # Disable LoRA for testing
        }
        
        return {
            "problem_generator": {
                **base_config,
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
                "temperature": 0.4
            },
            "solution_generator": {
                **base_config, 
                "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "temperature": 0.2
            },
            "verification_generator": {
                **base_config,
                "model_name": "Qwen/Qwen2.5-1.5B-Instruct", 
                "temperature": 0.3
            },
            "meta_verifier": {
                **base_config,
                "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
                "temperature": 0.2
            }
        }
    
    def test_problem_generator(self) -> bool:
        """Test Problem Generator with SOTA techniques."""
        logger.info("🧪 Testing Problem Generator (Player 1)")
        
        try:
            config = self.create_player_configs()["problem_generator"]
            generator = ProblemGenerator(config)
            
            # Test 1: Basic problem generation
            logger.info("  - Testing problem generation...")
            problems = generator.generate(
                target_difficulty="medium",
                target_type="arithmetic", 
                num_problems=2,
                max_retry_attempts=2
            )
            
            if not problems:
                logger.error("  ❌ No problems generated")
                return False
            
            # Test 2: Validate generated problems
            valid_problems = 0
            for problem in problems:
                if all(key in problem for key in ["title", "question"]):
                    valid_problems += 1
            
            success_rate = (valid_problems / len(problems)) * 100
            
            self.test_results["problem_generator"] = {
                "status": "passed" if success_rate >= 75 else "failed",
                "details": {
                    "problems_generated": len(problems),
                    "valid_problems": valid_problems,
                    "success_rate": success_rate,
                    "sample_problem": problems[0] if problems else None
                }
            }
            
            logger.info(f"  ✅ Problem Generator: {len(problems)} problems, {success_rate:.1f}% valid")
            return success_rate >= 75
            
        except Exception as e:
            logger.error(f"  ❌ Problem Generator failed: {e}")
            self.test_results["problem_generator"]["status"] = "error"
            self.test_results["problem_generator"]["details"]["error"] = str(e)
            return False
    
    def test_solution_generator(self) -> bool:
        """Test Solution Generator with SOTA techniques."""
        logger.info("🧪 Testing Solution Generator (Player 2)")
        
        try:
            config = self.create_player_configs()["solution_generator"]
            generator = SolutionGenerator(config)
            
            # Test 1: Basic solution generation
            logger.info("  - Testing solution generation...")
            solutions = generator.generate(
                problem=self.test_problem,
                num_solutions=3,
                max_retry_attempts=2
            )
            
            if not solutions:
                logger.error("  ❌ No solutions generated")
                return False
            
            # Test 2: Validate syntax of generated solutions
            valid_solutions = 0
            for solution in solutions:
                try:
                    compile(solution.get("code", ""), "<string>", "exec")
                    valid_solutions += 1
                except SyntaxError:
                    logger.debug(f"  Syntax error in solution: {solution.get('code', '')[:100]}")
            
            success_rate = (valid_solutions / len(solutions)) * 100
            
            self.test_results["solution_generator"] = {
                "status": "passed" if success_rate >= 75 else "failed",
                "details": {
                    "solutions_generated": len(solutions),
                    "valid_solutions": valid_solutions,
                    "success_rate": success_rate,
                    "sample_solution": solutions[0].get("code", "") if solutions else None
                }
            }
            
            logger.info(f"  ✅ Solution Generator: {len(solutions)} solutions, {success_rate:.1f}% valid syntax")
            return success_rate >= 75
            
        except Exception as e:
            logger.error(f"  ❌ Solution Generator failed: {e}")
            self.test_results["solution_generator"]["status"] = "error"
            self.test_results["solution_generator"]["details"]["error"] = str(e)
            return False
    
    def test_verification_generator(self) -> bool:
        """Test Verification Generator with SOTA techniques."""
        logger.info("🧪 Testing Verification Generator (Player 3)")
        
        try:
            config = self.create_player_configs()["verification_generator"]
            generator = VerificationGenerator(config)
            
            # Test 1: Basic test case generation
            logger.info("  - Testing test case generation...")
            test_cases = generator.generate(
                problem=self.test_problem,
                solutions=self.test_solutions,
                num_test_cases=6,
                max_retry_attempts=2
            )
            
            if not test_cases:
                logger.error("  ❌ No test cases generated")
                return False
            
            # Test 2: Validate test case structure
            valid_test_cases = 0
            categories_found = set()
            
            for test_case in test_cases:
                if all(key in test_case for key in ["name", "category"]):
                    valid_test_cases += 1
                    categories_found.add(test_case.get("category", "unknown"))
            
            success_rate = (valid_test_cases / len(test_cases)) * 100
            coverage_score = len(categories_found) / 8  # 8 possible categories
            
            self.test_results["verification_generator"] = {
                "status": "passed" if success_rate >= 75 and coverage_score >= 0.3 else "failed",
                "details": {
                    "test_cases_generated": len(test_cases),
                    "valid_test_cases": valid_test_cases,
                    "success_rate": success_rate,
                    "coverage_score": coverage_score,
                    "categories_found": list(categories_found),
                    "sample_test_case": test_cases[0] if test_cases else None
                }
            }
            
            logger.info(f"  ✅ Verification Generator: {len(test_cases)} test cases, {success_rate:.1f}% valid, {coverage_score:.2f} coverage")
            return success_rate >= 75 and coverage_score >= 0.3
            
        except Exception as e:
            logger.error(f"  ❌ Verification Generator failed: {e}")
            self.test_results["verification_generator"]["status"] = "error"
            self.test_results["verification_generator"]["details"]["error"] = str(e)
            return False
    
    def test_meta_verifier(self) -> bool:
        """Test Meta-Verifier with SOTA techniques."""
        logger.info("🧪 Testing Meta-Verifier (Player 4)")
        
        try:
            config = self.create_player_configs()["meta_verifier"]
            verifier = MetaVerifier(config)
            
            # Create sample test cases to validate
            sample_test_cases = [
                {
                    "name": "Basic Addition Test",
                    "category": "basic",
                    "input": [2, 3],
                    "expected": 5,
                    "reasoning": "Test basic functionality"
                },
                {
                    "name": "Invalid Test",
                    "category": "unknown",
                    "input": None,
                    "expected": None,
                    "reasoning": ""
                },
                {
                    "name": "Edge Case - Zero",
                    "category": "edge",
                    "input": [0, 0],
                    "expected": 0,
                    "reasoning": "Test with zero values"
                }
            ]
            
            # Test 1: Validate test cases
            logger.info("  - Testing test case validation...")
            validated_test_cases = verifier.validate_test_cases(
                problem=self.test_problem,
                solutions=self.test_solutions,
                test_cases=sample_test_cases,
                max_retry_attempts=2
            )
            
            if not validated_test_cases:
                logger.error("  ❌ No validated test cases returned")
                return False
            
            # Test 2: Check validation quality
            validations_with_confidence = 0
            validations_with_reasons = 0
            avg_confidence = 0
            
            for test_case in validated_test_cases:
                confidence = test_case.get("confidence", 0)
                if confidence > 0:
                    validations_with_confidence += 1
                    avg_confidence += confidence
                
                if test_case.get("validation_explanation"):
                    validations_with_reasons += 1
            
            avg_confidence = avg_confidence / len(validated_test_cases) if validated_test_cases else 0
            confidence_rate = (validations_with_confidence / len(validated_test_cases)) * 100
            explanation_rate = (validations_with_reasons / len(validated_test_cases)) * 100
            
            self.test_results["meta_verifier"] = {
                "status": "passed" if confidence_rate >= 75 and avg_confidence >= 0.5 else "failed",
                "details": {
                    "test_cases_validated": len(validated_test_cases),
                    "confidence_rate": confidence_rate,
                    "explanation_rate": explanation_rate,
                    "avg_confidence": avg_confidence,
                    "sample_validation": validated_test_cases[0] if validated_test_cases else None
                }
            }
            
            logger.info(f"  ✅ Meta-Verifier: {len(validated_test_cases)} validations, {confidence_rate:.1f}% with confidence, avg confidence: {avg_confidence:.2f}")
            return confidence_rate >= 75 and avg_confidence >= 0.5
            
        except Exception as e:
            logger.error(f"  ❌ Meta-Verifier failed: {e}")
            self.test_results["meta_verifier"]["status"] = "error"
            self.test_results["meta_verifier"]["details"]["error"] = str(e)
            return False
    
    def test_full_pipeline_integration(self) -> bool:
        """Test complete 4-player pipeline integration."""
        logger.info("🧪 Testing Full Pipeline Integration")
        
        try:
            configs = self.create_player_configs()
            
            # Initialize all players
            problem_gen = ProblemGenerator(configs["problem_generator"])
            solution_gen = SolutionGenerator(configs["solution_generator"])
            verification_gen = VerificationGenerator(configs["verification_generator"])
            meta_verifier = MetaVerifier(configs["meta_verifier"])
            
            logger.info("  - Step 1: Generate problem...")
            problems = problem_gen.generate(
                target_difficulty="easy",
                target_type="arithmetic",
                num_problems=1,
                max_retry_attempts=1
            )
            
            if not problems:
                logger.error("  ❌ Pipeline failed at problem generation")
                return False
            
            problem = problems[0]
            logger.info(f"  ✅ Generated problem: {problem.get('title', 'Unknown')}")
            
            logger.info("  - Step 2: Generate solutions...")
            solutions = solution_gen.generate(
                problem=problem,
                num_solutions=2,
                max_retry_attempts=1
            )
            
            if not solutions:
                logger.error("  ❌ Pipeline failed at solution generation")
                return False
            
            logger.info(f"  ✅ Generated {len(solutions)} solutions")
            
            logger.info("  - Step 3: Generate test cases...")
            test_cases = verification_gen.generate(
                problem=problem,
                solutions=solutions,
                num_test_cases=4,
                max_retry_attempts=1
            )
            
            if not test_cases:
                logger.error("  ❌ Pipeline failed at test case generation")
                return False
            
            logger.info(f"  ✅ Generated {len(test_cases)} test cases")
            
            logger.info("  - Step 4: Validate test cases...")
            validated_test_cases = meta_verifier.validate_test_cases(
                problem=problem,
                solutions=solutions,
                test_cases=test_cases,
                max_retry_attempts=1
            )
            
            if not validated_test_cases:
                logger.error("  ❌ Pipeline failed at test case validation")
                return False
            
            logger.info(f"  ✅ Validated {len(validated_test_cases)} test cases")
            
            # Calculate pipeline success metrics
            valid_test_cases = sum(1 for tc in validated_test_cases if tc.get("is_valid", True))
            pipeline_success_rate = (valid_test_cases / len(validated_test_cases)) * 100
            
            self.test_results["integration"] = {
                "status": "passed" if pipeline_success_rate >= 50 else "failed",
                "details": {
                    "problems_generated": len(problems),
                    "solutions_generated": len(solutions),
                    "test_cases_generated": len(test_cases),
                    "test_cases_validated": len(validated_test_cases),
                    "valid_test_cases": valid_test_cases,
                    "pipeline_success_rate": pipeline_success_rate
                }
            }
            
            logger.info(f"  ✅ Full Pipeline: {pipeline_success_rate:.1f}% success rate")
            return pipeline_success_rate >= 50
            
        except Exception as e:
            logger.error(f"  ❌ Pipeline integration failed: {e}")
            self.test_results["integration"]["status"] = "error"
            self.test_results["integration"]["details"]["error"] = str(e)
            return False
    
    def run_all_tests(self) -> Dict:
        """Run all integration tests and return results."""
        logger.info("🚀 Starting SRAG-V Players Integration Testing")
        logger.info("=" * 60)
        
        test_functions = [
            ("Problem Generator", self.test_problem_generator),
            ("Solution Generator", self.test_solution_generator), 
            ("Verification Generator", self.test_verification_generator),
            ("Meta-Verifier", self.test_meta_verifier),
            ("Full Pipeline", self.test_full_pipeline_integration)
        ]
        
        passed_tests = 0
        total_tests = len(test_functions)
        
        for test_name, test_func in test_functions:
            try:
                success = test_func()
                if success:
                    passed_tests += 1
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name}: ERROR - {e}")
        
        overall_success_rate = (passed_tests / total_tests) * 100
        
        logger.info("=" * 60)
        logger.info(f"🏁 Integration Testing Complete: {passed_tests}/{total_tests} tests passed ({overall_success_rate:.1f}%)")
        
        # Add overall summary
        self.test_results["overall"] = {
            "tests_passed": passed_tests,
            "total_tests": total_tests,
            "success_rate": overall_success_rate,
            "status": "passed" if overall_success_rate >= 80 else "failed"
        }
        
        # Save results
        results_file = Path("test_results_integration.json")
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"📊 Detailed results saved to: {results_file}")
        
        return self.test_results

def main():
    """Main entry point for integration testing."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
    
    tester = SRAGVPlayersIntegrationTest()
    results = tester.run_all_tests()
    
    # Rate each player out of 10 as requested
    logger.info("\n🎯 PLAYER ROBUSTNESS RATINGS (out of 10):")
    
    ratings = {}
    for player, result in results.items():
        if player == "overall":
            continue
            
        if result["status"] == "passed":
            if player == "integration":
                success_rate = result["details"].get("pipeline_success_rate", 0)
            else:
                success_rate = result["details"].get("success_rate", 0)
                
            if success_rate >= 90:
                rating = 9
            elif success_rate >= 80:
                rating = 8
            elif success_rate >= 70:
                rating = 7
            else:
                rating = 6
        elif result["status"] == "failed":
            rating = 4
        else:  # error
            rating = 2
        
        ratings[player] = rating
        player_name = player.replace("_", " ").title()
        logger.info(f"  {player_name}: {rating}/10")
    
    avg_rating = sum(ratings.values()) / len(ratings)
    logger.info(f"\n📈 Average Player Rating: {avg_rating:.1f}/10")
    
    if avg_rating >= 8.0:
        logger.info("🎉 EXCELLENT: All players are robust and ready for training!")
    elif avg_rating >= 7.0:
        logger.info("✅ GOOD: Players are ready for training with minor improvements possible")
    elif avg_rating >= 6.0:
        logger.info("⚠️ ACCEPTABLE: Players functional but need improvements before production")
    else:
        logger.info("❌ POOR: Players need significant fixes before training")
    
    return results

if __name__ == "__main__":
    main()