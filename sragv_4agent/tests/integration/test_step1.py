"""
Comprehensive test script for SRAG-V Step 1 implementation.
Tests all components of the 4-player architecture and bootstrapping.
"""

import logging
import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from sragv.orchestrator import SRAGVOrchestrator
from sragv.models import (
    ProblemGenerator, 
    SolutionGenerator, 
    VerificationGenerator, 
    MetaVerifier,
    PlayerConfig
)
from sragv.utils.simple_data_loader import SimpleAPPSDataLoader
from sragv.utils.apps_loader import APPSDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/test_step1.log')
    ]
)
logger = logging.getLogger(__name__)


def test_data_loaders():
    """Test both data loaders."""
    logger.info("=== Testing Data Loaders ===")
    
    try:
        # Test simple data loader
        logger.info("Testing SimpleAPPSDataLoader...")
        simple_loader = SimpleAPPSDataLoader(max_problems=5)
        problems, categorized = simple_loader.prepare_bootstrap_data()
        
        assert len(problems) > 0, "No problems loaded"
        assert all(isinstance(p, dict) for p in problems), "Problems not in dict format"
        assert 'easy' in categorized, "No easy problems found"
        
        logger.info(f"✓ Simple loader: {len(problems)} problems, {sum(len(probs) for probs in categorized.values())} categorized")
        
        # Test real APPS loader (if available)
        try:
            logger.info("Testing APPSDataLoader...")
            apps_loader = APPSDataLoader(max_problems=10)
            real_problems, real_categorized = apps_loader.prepare_bootstrap_data()
            
            if real_problems:
                logger.info(f"✓ APPS loader: {len(real_problems)} problems loaded")
            else:
                logger.info("ℹ APPS loader: No problems loaded (expected if dataset not available)")
                
        except Exception as e:
            logger.warning(f"APPS loader failed (expected): {e}")
        
        return problems, categorized
        
    except Exception as e:
        logger.error(f"Data loader test failed: {e}")
        traceback.print_exc()
        return None, None


def test_problem_generator():
    """Test Problem Generator (Player 1)."""
    logger.info("=== Testing Problem Generator ===")
    
    try:
        # Create configuration
        config = {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "max_length": 2048,
            "temperature": 0.8,
            "top_p": 0.95,
            "lora_config": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            }
        }
        
        # Initialize generator
        logger.info("Initializing ProblemGenerator...")
        generator = ProblemGenerator(config)
        
        # Test problem generation
        logger.info("Generating test problems...")
        problems = generator.generate(
            target_difficulty="easy",
            target_type="arithmetic", 
            num_problems=2
        )
        
        assert len(problems) > 0, "No problems generated"
        assert all(isinstance(p, dict) for p in problems), "Problems not in dict format"
        assert all('problem_id' in p for p in problems), "Missing problem_id"
        
        logger.info(f"✓ Problem Generator: Generated {len(problems)} problems")
        for i, problem in enumerate(problems):
            logger.info(f"  Problem {i+1}: {problem['title'][:50]}...")
        
        return problems
        
    except Exception as e:
        logger.error(f"Problem generator test failed: {e}")
        traceback.print_exc()
        return []


def test_solution_generator(problems):
    """Test Solution Generator (Player 2)."""
    logger.info("=== Testing Solution Generator ===")
    
    if not problems:
        logger.warning("No problems provided for solution generation test")
        return []
    
    try:
        # Create configuration
        config = {
            "model_name": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "max_length": 4096,
            "temperature": 0.8,
            "top_p": 0.95,
            "quantization": "4bit",
            "lora_config": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
            }
        }
        
        # Initialize generator
        logger.info("Initializing SolutionGenerator...")
        generator = SolutionGenerator(config)
        
        # Test solution generation
        test_problem = problems[0]
        logger.info(f"Generating solutions for problem: {test_problem['title'][:50]}...")
        
        # Generate fewer solutions for testing
        solutions = generator.generate(
            problem=test_problem,
            num_solutions=4,  # Reduced for testing
            max_attempts_per_solution=1
        )
        
        assert len(solutions) > 0, "No solutions generated"
        assert all(isinstance(s, dict) for s in solutions), "Solutions not in dict format"
        assert all('code' in s for s in solutions), "Missing code in solutions"
        
        logger.info(f"✓ Solution Generator: Generated {len(solutions)} solutions")
        for i, solution in enumerate(solutions):
            logger.info(f"  Solution {i+1}: {len(solution['code'])} chars, score: {solution.get('score', 0):.3f}")
        
        return solutions
        
    except Exception as e:
        logger.error(f"Solution generator test failed: {e}")
        traceback.print_exc()
        return []


def test_verification_generator(problems, solutions):
    """Test Verification Generator (Player 3)."""
    logger.info("=== Testing Verification Generator ===")
    
    if not problems or not solutions:
        logger.warning("No problems or solutions provided for verification test")
        return []
    
    try:
        # Create configuration
        config = {
            "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
            "max_length": 2048,
            "temperature": 0.7,
            "top_p": 0.9,
            "lora_config": {
                "rank": 32,
                "alpha": 64,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            }
        }
        
        # Initialize generator
        logger.info("Initializing VerificationGenerator...")
        generator = VerificationGenerator(config)
        
        # Test verification generation
        test_problem = problems[0]
        logger.info(f"Generating test cases for problem: {test_problem['title'][:50]}...")
        
        test_cases = generator.generate(
            problem=test_problem,
            solutions=solutions,
            num_test_cases=4  # Reduced for testing
        )
        
        assert len(test_cases) > 0, "No test cases generated"
        assert all(isinstance(t, dict) for t in test_cases), "Test cases not in dict format"
        assert all('test_id' in t for t in test_cases), "Missing test_id"
        
        logger.info(f"✓ Verification Generator: Generated {len(test_cases)} test cases")
        for i, test_case in enumerate(test_cases):
            logger.info(f"  Test {i+1}: {test_case['name'][:30]}... (category: {test_case.get('category', 'unknown')})")
        
        return test_cases
        
    except Exception as e:
        logger.error(f"Verification generator test failed: {e}")
        traceback.print_exc()
        return []


def test_meta_verifier(problems, solutions, test_cases):
    """Test Meta-Verifier (Player 4)."""
    logger.info("=== Testing Meta-Verifier ===")
    
    if not problems or not solutions or not test_cases:
        logger.warning("Missing data for meta-verifier test")
        return []
    
    try:
        # Create configuration
        config = {
            "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_length": 1024,
            "temperature": 0.1,
            "top_p": 0.95,
            "lora_config": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            }
        }
        
        # Initialize verifier
        logger.info("Initializing MetaVerifier...")
        verifier = MetaVerifier(config)
        
        # Test validation
        test_problem = problems[0]
        logger.info(f"Validating test cases for problem: {test_problem['title'][:50]}...")
        
        validated_tests = verifier.validate_test_cases(
            problem=test_problem,
            solutions=solutions,
            test_cases=test_cases
        )
        
        assert len(validated_tests) > 0, "No validated tests returned"
        assert all(isinstance(t, dict) for t in validated_tests), "Validated tests not in dict format"
        assert all('is_valid' in t for t in validated_tests), "Missing is_valid field"
        
        # Get statistics
        valid_count = sum(1 for t in validated_tests if t['is_valid'])
        avg_confidence = sum(t.get('confidence', 0.5) for t in validated_tests) / len(validated_tests)
        
        logger.info(f"✓ Meta-Verifier: Validated {len(validated_tests)} tests")
        logger.info(f"  Valid: {valid_count}/{len(validated_tests)} ({valid_count/len(validated_tests)*100:.1f}%)")
        logger.info(f"  Avg confidence: {avg_confidence:.3f}")
        
        return validated_tests
        
    except Exception as e:
        logger.error(f"Meta-verifier test failed: {e}")
        traceback.print_exc()
        return []


def test_orchestrator():
    """Test the complete orchestrator."""
    logger.info("=== Testing Orchestrator ===")
    
    try:
        # Initialize orchestrator
        logger.info("Initializing SRAGVOrchestrator...")
        orchestrator = SRAGVOrchestrator()
        
        # Test configuration loading
        assert orchestrator.config is not None, "Configuration not loaded"
        logger.info("✓ Configuration loaded successfully")
        
        # Test data loader initialization
        orchestrator.initialize_data_loaders()
        assert orchestrator.simple_data_loader is not None, "Simple data loader not initialized"
        logger.info("✓ Data loaders initialized")
        
        # Test bootstrap data loading
        bootstrap_problems = orchestrator.load_bootstrap_data()
        assert bootstrap_problems is not None, "Bootstrap data not loaded"
        problems, categorized = bootstrap_problems
        assert len(problems) > 0, "No bootstrap problems"
        logger.info(f"✓ Bootstrap data loaded: {len(problems)} problems")
        
        # Test single iteration (lightweight)
        logger.info("Running test iteration...")
        metrics = orchestrator.run_single_iteration(
            iteration=1,
            bootstrap_problems=problems[:2]  # Use only 2 problems for testing
        )
        
        assert metrics is not None, "No metrics returned"
        assert metrics.iteration == 1, "Wrong iteration number"
        logger.info(f"✓ Test iteration completed: {metrics.problems_generated} problems, {metrics.solutions_generated} solutions")
        
        return True
        
    except Exception as e:
        logger.error(f"Orchestrator test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all Step 1 tests."""
    logger.info("Starting SRAG-V Step 1 Comprehensive Testing")
    logger.info("=" * 60)
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Test results
    results = {
        "data_loaders": False,
        "problem_generator": False,
        "solution_generator": False,
        "verification_generator": False,
        "meta_verifier": False,
        "orchestrator": False
    }
    
    try:
        # Test 1: Data Loaders
        problems, categorized = test_data_loaders()
        results["data_loaders"] = problems is not None
        
        # Test 2: Problem Generator
        if results["data_loaders"]:
            generated_problems = test_problem_generator()
            results["problem_generator"] = len(generated_problems) > 0
            
            # Use generated problems if available, otherwise use loaded problems
            test_problems = generated_problems if generated_problems else problems
        else:
            test_problems = []
        
        # Test 3: Solution Generator
        if test_problems:
            solutions = test_solution_generator(test_problems)
            results["solution_generator"] = len(solutions) > 0
        else:
            solutions = []
        
        # Test 4: Verification Generator
        if test_problems and solutions:
            test_cases = test_verification_generator(test_problems, solutions)
            results["verification_generator"] = len(test_cases) > 0
        else:
            test_cases = []
        
        # Test 5: Meta-Verifier
        if test_problems and solutions and test_cases:
            validated_tests = test_meta_verifier(test_problems, solutions, test_cases)
            results["meta_verifier"] = len(validated_tests) > 0
        
        # Test 6: Complete Orchestration
        results["orchestrator"] = test_orchestrator()
        
    except Exception as e:
        logger.error(f"Critical error in testing: {e}")
        traceback.print_exc()
    
    # Report results
    logger.info("=" * 60)
    logger.info("STEP 1 TEST RESULTS:")
    logger.info("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name:25} {status}")
    
    logger.info("-" * 60)
    logger.info(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED - Step 1 implementation is ready!")
        return 0
    elif passed_tests >= total_tests * 0.8:
        logger.info("⚠️  MOSTLY WORKING - Step 1 implementation is mostly ready with minor issues")
        return 1
    else:
        logger.error("❌ SIGNIFICANT ISSUES - Step 1 implementation needs work")
        return 2


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)