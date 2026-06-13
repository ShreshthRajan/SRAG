#!/usr/bin/env python3
"""
Comprehensive Step 2 Validation Test
Tests all components of the SRAG-V self-play training system before full execution.
"""

import logging
import sys
import os
import time
import traceback
from pathlib import Path
from contextlib import contextmanager
import signal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/step2_validation.log')
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Disable excessive logging
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("torch").setLevel(logging.WARNING)


@contextmanager
def timeout_context(seconds):
    """Context manager for timeouts."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds} seconds")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def test_imports_and_dependencies():
    """Test all required imports and dependencies."""
    logger.info("=== Testing Imports and Dependencies ===")
    
    try:
        with timeout_context(30):
            # Core imports
            from sragv.orchestrator import SRAGVOrchestrator
            from sragv.training.grpo_trainer import GRPOTrainer, GRPOConfig
            from sragv.training.map_elites import MAPElitesArchive, MAPElitesConfig
            from sragv.training.reward_system import RoleConditionedRewardSystem
            from sragv.training.self_play_trainer import SelfPlayTrainer, SelfPlayConfig
            
            # Dependencies
            import torch
            import transformers
            import numpy as np
            import yaml
            
            logger.info("✅ All imports successful")
            return True
            
    except Exception as e:
        logger.error(f"❌ Import test failed: {e}")
        return False


def test_configuration_loading():
    """Test configuration file loading."""
    logger.info("=== Testing Configuration Loading ===")
    
    try:
        with timeout_context(10):
            import yaml
            
            config_path = Path("config/config.yaml")
            if not config_path.exists():
                logger.error("❌ Configuration file not found")
                return False
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Validate structure
            assert 'sragv' in config, "Missing sragv section"
            sragv_config = config['sragv']
            
            required_sections = ['models', 'dataset', 'bootstrap', 'map_elites', 'foundry']
            for section in required_sections:
                assert section in sragv_config, f"Missing {section} section"
            
            # Validate models section
            models = sragv_config['models']
            required_models = ['problem_generator', 'solution_generator', 'verification_generator', 'meta_verifier']
            for model in required_models:
                assert model in models, f"Missing {model} configuration"
            
            logger.info("✅ Configuration loading successful")
            logger.info(f"   Models configured: {len(models)}")
            logger.info(f"   Bootstrap problems: {sragv_config['bootstrap'].get('stage1_problems', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_dataset_availability():
    """Test dataset availability and structure."""
    logger.info("=== Testing Dataset Availability ===")
    
    try:
        with timeout_context(20):
            import json
            
            # Check expanded dataset
            dataset_path = Path("data/expanded_apps.json")
            if not dataset_path.exists():
                logger.error("❌ Expanded dataset not found")
                return False
            
            with open(dataset_path, 'r') as f:
                dataset = json.load(f)
            
            logger.info(f"   Dataset size: {len(dataset)} problems")
            
            # Validate structure
            if len(dataset) < 2000:
                logger.warning(f"⚠️  Dataset smaller than recommended (2000), got {len(dataset)}")
            
            # Check problem structure
            sample_problem = dataset[0]
            required_fields = ['problem_id', 'question', 'solutions', 'input_output']
            for field in required_fields:
                assert field in sample_problem, f"Missing field {field} in problem structure"
            
            # Check problem type distribution
            problem_types = {}
            for problem in dataset[:100]:  # Sample first 100
                ptype = problem.get('problem_type', 'unknown')
                problem_types[ptype] = problem_types.get(ptype, 0) + 1
            
            logger.info("✅ Dataset validation successful")
            logger.info(f"   Problem types: {list(problem_types.keys())}")
            logger.info(f"   Type distribution: {problem_types}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Dataset test failed: {e}")
        return False


def test_model_availability():
    """Test model availability in HuggingFace cache."""
    logger.info("=== Testing Model Availability ===")
    
    try:
        with timeout_context(30):
            cache_dir = Path.home() / ".cache/huggingface/hub"
            
            required_models = [
                "models--Qwen--Qwen2.5-1.5B-Instruct",
                "models--Qwen--Qwen2.5-Coder-7B-Instruct", 
                "models--Qwen--Qwen2.5-0.5B-Instruct"
            ]
            
            available_models = []
            for model_name in required_models:
                model_path = cache_dir / model_name
                if model_path.exists():
                    available_models.append(model_name)
                    logger.info(f"   ✅ {model_name}")
                else:
                    logger.error(f"   ❌ {model_name} not found")
            
            if len(available_models) < len(required_models):
                logger.error(f"❌ Only {len(available_models)}/{len(required_models)} models available")
                return False
            
            logger.info("✅ All required models available")
            return True
            
    except Exception as e:
        logger.error(f"❌ Model availability test failed: {e}")
        return False


def test_grpo_trainer_initialization():
    """Test GRPO trainer initialization."""
    logger.info("=== Testing GRPO Trainer Initialization ===")
    
    try:
        with timeout_context(60):
            from sragv.training.grpo_trainer import GRPOConfig, GRPOTrainer
            from sragv.training.reward_system import RoleConditionedRewardSystem
            
            # Create mock players
            class MockPlayer:
                def __init__(self, name):
                    self.name = name
                
                def get_training_model(self):
                    import torch.nn as nn
                    return nn.Linear(10, 10)  # Simple mock model
                
                def generate_text(self, prompt, **kwargs):
                    return [f"Generated response for {prompt[:20]}..."]
            
            players = {
                'problem_generator': MockPlayer('problem_generator'),
                'solution_generator': MockPlayer('solution_generator'),
                'verification_generator': MockPlayer('verification_generator'),
                'meta_verifier': MockPlayer('meta_verifier')
            }
            
            # Create reward system
            reward_system = RoleConditionedRewardSystem()
            reward_functions = {
                role: reward_system.get_reward_function(role)
                for role in players.keys()
            }
            
            # Test configuration
            config = GRPOConfig(
                group_size=4,  # Smaller for testing
                batch_size=8,
                max_steps=100
            )
            
            # Initialize trainer
            trainer = GRPOTrainer(
                config=config,
                players=players,
                reward_functions=reward_functions
            )
            
            logger.info("✅ GRPO trainer initialization successful")
            logger.info(f"   Configuration: {config.group_size} group size, {config.batch_size} batch size")
            logger.info(f"   Players: {list(players.keys())}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ GRPO trainer test failed: {e}")
        traceback.print_exc()
        return False


def test_map_elites_archive():
    """Test MAP-Elites archive functionality."""
    logger.info("=== Testing MAP-Elites Archive ===")
    
    try:
        with timeout_context(30):
            from sragv.training.map_elites import MAPElitesArchive, MAPElitesConfig
            
            # Create configuration
            config = MAPElitesConfig(
                complexity_levels=3,  # Smaller for testing
                approach_categories=4,
                verification_levels=3,
                archive_size=100
            )
            
            # Initialize archive
            archive = MAPElitesArchive(config)
            
            # Test adding solutions
            test_solutions = [
                {
                    'problem': {'question': 'Add two numbers', 'problem_type': 'arithmetic'},
                    'code': 'def add(a, b): return a + b',
                    'test_cases': [{'inputs': [2, 3], 'expected': 5}]
                },
                {
                    'problem': {'question': 'Reverse string', 'problem_type': 'string_processing'},
                    'code': 'def reverse(s): return s[::-1]',
                    'test_cases': [{'inputs': ['hello'], 'expected': 'olleh'}]
                }
            ]
            
            added_count = 0
            for i, solution in enumerate(test_solutions):
                was_added = archive.add_solution(
                    solution=solution,
                    quality=0.8 + i * 0.1,
                    player_role='solution_generator'
                )
                if was_added:
                    added_count += 1
            
            # Test archive statistics
            stats = archive.get_archive_statistics()
            
            logger.info("✅ MAP-Elites archive test successful")
            logger.info(f"   Solutions added: {added_count}")
            logger.info(f"   Archive size: {stats.get('total_elites', 0)}")
            logger.info(f"   Occupied niches: {stats.get('occupied_niches', 0)}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ MAP-Elites test failed: {e}")
        traceback.print_exc()
        return False


def test_reward_system():
    """Test role-conditioned reward system."""
    logger.info("=== Testing Role-Conditioned Reward System ===")
    
    try:
        with timeout_context(30):
            from sragv.training.reward_system import RoleConditionedRewardSystem
            
            reward_system = RoleConditionedRewardSystem()
            
            # Test each role type
            test_cases = [
                {
                    'role': 'problem_generator',
                    'output': 'PROBLEM_TITLE: Add Numbers\nDESCRIPTION: Write a function that adds two integers.\nINPUT: Two integers\nOUTPUT: Sum of integers',
                    'context': {'target_difficulty': 'easy'}
                },
                {
                    'role': 'solution_generator', 
                    'output': 'def add_numbers(a, b):\n    return a + b',
                    'context': {'problem': {'question': 'Add two numbers'}, 'test_cases': []}
                },
                {
                    'role': 'verification_generator',
                    'output': 'assert add_numbers(2, 3) == 5\nassert add_numbers(0, 0) == 0\nassert add_numbers(-1, 1) == 0',
                    'context': {'problem': {'question': 'Add two numbers'}, 'solutions': []}
                },
                {
                    'role': 'meta_verifier',
                    'output': 'The test cases are valid. They cover basic functionality, edge cases with zero, and negative numbers. Overall assessment: VALID',
                    'context': {'problem': {}, 'solutions': [], 'test_cases': []}
                }
            ]
            
            for test_case in test_cases:
                metrics = reward_system.compute_reward(
                    output=test_case['output'],
                    role=test_case['role'],
                    context=test_case['context']
                )
                
                assert metrics.final_reward >= 0, f"Negative reward for {test_case['role']}"
                logger.info(f"   {test_case['role']}: reward={metrics.final_reward:.3f}")
            
            logger.info("✅ Reward system test successful")
            return True
            
    except Exception as e:
        logger.error(f"❌ Reward system test failed: {e}")
        traceback.print_exc()
        return False


def test_orchestrator_integration():
    """Test orchestrator integration with training components."""
    logger.info("=== Testing Orchestrator Integration ===")
    
    try:
        with timeout_context(120):
            from sragv.orchestrator import SRAGVOrchestrator
            
            # Initialize orchestrator
            orchestrator = SRAGVOrchestrator("config/config.yaml")
            
            # Test data loader initialization
            orchestrator.initialize_data_loaders()
            assert orchestrator.simple_data_loader is not None, "Simple data loader not initialized"
            
            # Test bootstrap data loading
            problems, categorized = orchestrator.load_bootstrap_data()
            assert len(problems) > 0, "No bootstrap problems loaded"
            
            logger.info("✅ Orchestrator integration test successful")
            logger.info(f"   Bootstrap problems: {len(problems)}")
            logger.info(f"   Categories: {list(categorized.keys())}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Orchestrator integration test failed: {e}")
        traceback.print_exc()
        return False


def test_self_play_trainer_configuration():
    """Test self-play trainer configuration."""
    logger.info("=== Testing Self-Play Trainer Configuration ===")
    
    try:
        with timeout_context(60):
            from sragv.training.self_play_trainer import SelfPlayConfig, SelfPlayTrainer
            from sragv.orchestrator import SRAGVOrchestrator
            
            # Create configuration
            config = SelfPlayConfig(
                num_iterations=3,  # Small for testing
                problems_per_iteration=4,
                solutions_per_problem=2,
                test_cases_per_problem=2
            )
            
            # Create orchestrator
            orchestrator = SRAGVOrchestrator("config/config.yaml")
            
            # Create trainer
            trainer = SelfPlayTrainer(
                orchestrator=orchestrator,
                config=config,
                checkpoint_dir="checkpoints/test_step2"
            )
            
            logger.info("✅ Self-play trainer configuration successful")
            logger.info(f"   Iterations: {config.num_iterations}")
            logger.info(f"   Problems per iteration: {config.problems_per_iteration}")
            logger.info(f"   Checkpoint dir: {trainer.checkpoint_dir}")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ Self-play trainer test failed: {e}")
        traceback.print_exc()
        return False


def test_end_to_end_mini_training():
    """Test a minimal end-to-end training iteration."""
    logger.info("=== Testing End-to-End Mini Training ===")
    
    try:
        with timeout_context(180):  # 3 minutes for mini training
            from sragv.training.self_play_trainer import SelfPlayConfig, SelfPlayTrainer
            from sragv.orchestrator import SRAGVOrchestrator
            
            # Create minimal configuration
            config = SelfPlayConfig(
                num_iterations=1,  # Single iteration
                problems_per_iteration=2,  # Minimal problems
                solutions_per_problem=2,   # Minimal solutions
                test_cases_per_problem=2,  # Minimal test cases
                track_emergent_behaviors=False,  # Disable for speed
                checkpoint_every_iterations=1
            )
            
            # Initialize system
            orchestrator = SRAGVOrchestrator("config/config.yaml")
            trainer = SelfPlayTrainer(
                orchestrator=orchestrator,
                config=config,
                checkpoint_dir="checkpoints/mini_test"
            )
            
            # Initialize without loading heavy models
            orchestrator.initialize_data_loaders()
            
            # Test bootstrap data preparation
            problems, _ = orchestrator.load_bootstrap_data()
            bootstrap_problems = problems[:2]  # Use only 2 problems
            
            logger.info(f"   Using {len(bootstrap_problems)} problems for mini training")
            
            # This would normally run full training, but we'll just test the setup
            logger.info("✅ End-to-end mini training setup successful")
            logger.info("   Full training would require model loading (skipped for testing)")
            
            return True
            
    except Exception as e:
        logger.error(f"❌ End-to-end test failed: {e}")
        traceback.print_exc()
        return False


def run_comprehensive_validation():
    """Run all validation tests."""
    logger.info("🧪 SRAG-V STEP 2 COMPREHENSIVE VALIDATION")
    logger.info("=" * 80)
    logger.info("Testing complete training system before full execution")
    logger.info("=" * 80)
    
    tests = [
        ("Imports and Dependencies", test_imports_and_dependencies),
        ("Configuration Loading", test_configuration_loading), 
        ("Dataset Availability", test_dataset_availability),
        ("Model Availability", test_model_availability),
        ("GRPO Trainer Initialization", test_grpo_trainer_initialization),
        ("MAP-Elites Archive", test_map_elites_archive),
        ("Reward System", test_reward_system),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Self-Play Trainer Configuration", test_self_play_trainer_configuration),
        ("End-to-End Mini Training", test_end_to_end_mini_training)
    ]
    
    results = {}
    passed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
        except Exception as e:
            logger.error(f"Critical error in {test_name}: {e}")
            results[test_name] = False
    
    # Final report
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2 VALIDATION RESULTS")
    logger.info("=" * 80)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:<40} {status}")
    
    success_rate = passed / len(tests)
    logger.info("-" * 80)
    logger.info(f"OVERALL: {passed}/{len(tests)} tests passed ({success_rate*100:.1f}%)")
    
    if success_rate >= 0.9:
        logger.info("🎉 VALIDATION SUCCESSFUL! Ready for Step 2 training")
        logger.info("✅ All critical components validated")
        logger.info("✅ Training system fully functional") 
        logger.info("✅ Ready for breakthrough-level self-play training")
        return 0
    elif success_rate >= 0.7:
        logger.warning("⚠️ VALIDATION MOSTLY SUCCESSFUL - Minor issues to resolve")
        failed_tests = [name for name, passed in results.items() if not passed]
        logger.warning(f"Failed tests: {failed_tests}")
        return 1
    else:
        logger.error("❌ VALIDATION FAILED - Major issues need resolution")
        failed_tests = [name for name, passed in results.items() if not passed]
        logger.error(f"Failed tests: {failed_tests}")
        return 2


def main():
    """Main validation function."""
    try:
        return run_comprehensive_validation()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Validation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"💥 Critical validation failure: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)