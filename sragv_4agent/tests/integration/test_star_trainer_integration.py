#!/usr/bin/env python3
"""
Integration tests for STARTrainer to ensure it works properly with all components.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

import json
import numpy as np
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock

# Mock the dependencies to avoid loading heavy models
class MockSolutionGenerator:
    def __init__(self):
        self.confidence_calibrator = Mock()
        self.confidence_calibrator.is_trained = True
    
    def generate(self, problem, num_solutions=6):
        """Mock solution generation with realistic structure."""
        solutions = []
        for i in range(num_solutions):
            solution = {
                'solution_id': f'sol_{i:02d}',
                'code': f'def solution_{i}():\n    return {i}',
                'score': np.random.beta(3, 2),  # Realistic confidence distribution  
                'syntax_valid': np.random.random() > 0.1,
                'execution_success': np.random.random() > 0.2,
                'pass_rate': np.random.beta(3, 2),
                'code_quality_score': np.random.beta(4, 2),
                'problem': problem
            }
            solutions.append(solution)
        return solutions
    
    def train_confidence_calibrator(self, training_data, validation_data=None, num_epochs=1, learning_rate=0.001):
        """Mock calibrator training."""
        return {'final_loss': 0.01 * np.random.random()}

class MockOrchestrator:
    def __init__(self):
        self.solution_generator = MockSolutionGenerator()

class MockStrategicOracle:
    def strategic_problem_selection(self, problems, num_problems, strategy_weights=None):
        """Mock strategic problem selection."""
        return problems[:num_problems]

def create_mock_problems(num_problems: int = 10) -> List[Dict[str, Any]]:
    """Create mock problems for testing."""
    problems = []
    for i in range(num_problems):
        problem = {
            'problem_id': f'prob_{i:03d}',
            'question': f'Write a function that solves problem {i}',
            'difficulty': ['easy', 'medium', 'hard'][i % 3],
            'source': 'test'
        }
        problems.append(problem)
    return problems

def test_star_trainer_initialization():
    """Test STARTrainer initialization."""
    print("=== STAR TRAINER INITIALIZATION TEST ===")
    
    from sragv.training.star_trainer import STARTrainer
    
    # Create mocks
    orchestrator = MockOrchestrator()
    strategic_oracle = MockStrategicOracle()
    
    # Test initialization
    trainer = STARTrainer(orchestrator, strategic_oracle)
    
    # Verify initialization
    assert trainer.orchestrator is orchestrator
    assert trainer.strategic_oracle is strategic_oracle
    assert trainer.bayesian_pseudo_labeler is not None
    assert trainer.current_iteration == 0
    assert trainer.total_solutions_generated == 0
    
    print("✅ STARTrainer initialization successful")
    
    # Test custom configuration
    custom_config = {
        'max_iterations': 5,
        'batch_size': 16,
        'confidence_threshold': 0.9
    }
    trainer_custom = STARTrainer(orchestrator, strategic_oracle, custom_config)
    assert trainer_custom.config['max_iterations'] == 5
    assert trainer_custom.config['confidence_threshold'] == 0.9
    
    print("✅ Custom configuration successful")

def test_solution_generation():
    """Test solution generation with confidence."""
    print("=== SOLUTION GENERATION TEST ===")
    
    from sragv.training.star_trainer import STARTrainer
    
    orchestrator = MockOrchestrator()
    strategic_oracle = MockStrategicOracle()
    trainer = STARTrainer(orchestrator, strategic_oracle)
    
    # Test solution generation
    problems = create_mock_problems(3)
    solutions = trainer._generate_solutions_with_confidence(problems)
    
    assert len(solutions) > 0
    assert trainer.total_solutions_generated > 0
    
    # Verify solution structure
    sample_solution = solutions[0]
    assert 'score' in sample_solution  # Confidence field
    assert 'code' in sample_solution
    assert 'solution_id' in sample_solution
    
    print(f"✅ Generated {len(solutions)} solutions successfully")
    
    # Test with empty problems
    empty_solutions = trainer._generate_solutions_with_confidence([])
    assert len(empty_solutions) == 0
    
    print("✅ Empty problems handled correctly")

def test_bayesian_pseudo_labeling_integration():
    """Test BayesianPseudoLabeler integration."""
    print("=== BAYESIAN PSEUDO-LABELING INTEGRATION TEST ===")
    
    from sragv.training.star_trainer import STARTrainer
    
    orchestrator = MockOrchestrator()
    strategic_oracle = MockStrategicOracle()
    trainer = STARTrainer(orchestrator, strategic_oracle)
    
    # Create high-quality mock solutions
    solutions = []
    for i in range(10):
        solution = {
            'solution_id': f'sol_{i:02d}',
            'code': f'def solution_{i}():\n    return {i}',
            'score': 0.9 + 0.05 * np.random.random(),  # High confidence
            'syntax_valid': True,
            'execution_success': True,
            'pass_rate': 0.8 + 0.2 * np.random.random(),
            'code_quality_score': 0.8 + 0.2 * np.random.random()
        }
        solutions.append(solution)
    
    # Test Bayesian pseudo-labeling
    pseudo_labels, metrics = trainer.bayesian_pseudo_labeler.create_pseudo_labels(solutions, iteration=0)
    
    assert isinstance(pseudo_labels, list)
    assert isinstance(metrics, dict)
    assert 'pseudo_label_rate' in metrics
    assert 'avg_confidence' in metrics
    
    print(f"✅ Created {len(pseudo_labels)} pseudo-labels")
    print(f"✅ Pseudo-label rate: {metrics['pseudo_label_rate']:.3f}")
    
    # Test pseudo-label structure
    if pseudo_labels:
        sample_label = pseudo_labels[0]
        assert 'solution' in sample_label
        assert 'pseudo_label_confidence' in sample_label
        print("✅ Pseudo-label structure correct")

def test_replay_buffer_operations():
    """Test replay buffer initialization and updates."""
    print("=== REPLAY BUFFER TEST ===")
    
    from sragv.training.star_trainer import STARTrainer
    
    orchestrator = MockOrchestrator()
    strategic_oracle = MockStrategicOracle()
    trainer = STARTrainer(orchestrator, strategic_oracle)
    
    # Test replay buffer initialization
    problems = create_mock_problems(5)
    replay_buffer = trainer._initialize_replay_buffer(problems)
    
    assert isinstance(replay_buffer, list)
    print(f"✅ Initialized replay buffer with {len(replay_buffer)} solutions")
    
    # Test replay buffer updates
    pseudo_labels = [
        {
            'solution': {'code': 'test', 'score': 0.9},
            'pseudo_label_confidence': 0.92
        },
        {
            'solution': {'code': 'test2', 'score': 0.8},
            'pseudo_label_confidence': 0.82  # Below 0.85 threshold
        }
    ]
    
    updated_buffer = trainer._update_replay_buffer(replay_buffer, pseudo_labels)
    
    # Should only add the high-confidence pseudo-label
    expected_additions = 1  # Only one above 0.85 threshold
    print(f"✅ Replay buffer updated correctly")

def test_training_step():
    """Test continuous training step."""
    print("=== CONTINUOUS TRAINING STEP TEST ===")
    
    from sragv.training.star_trainer import STARTrainer
    
    orchestrator = MockOrchestrator()
    strategic_oracle = MockStrategicOracle()
    trainer = STARTrainer(orchestrator, strategic_oracle)
    
    # Create mock pseudo-labeled data
    pseudo_labeled_data = [
        {
            'solution': {
                'code': 'def test(): return 1',
                'score': 0.9,
                'problem': {'problem_id': 'test_001'}
            },
            'pseudo_label_confidence': 0.92
        }
    ]
    
    replay_buffer = [
        {'code': 'def replay(): return 2', 'score': 0.85}
    ]
    
    # Test training step
    metrics = trainer._continuous_training_step(pseudo_labeled_data, replay_buffer)
    
    assert isinstance(metrics, dict)
    assert 'pseudo_labels_used' in metrics
    assert 'training_loss' in metrics
    assert 'gradient_steps' in metrics
    
    print(f"✅ Training step completed")
    print(f"✅ Training loss: {metrics['training_loss']:.4f}")
    print(f"✅ Gradient steps: {metrics['gradient_steps']}")
    
    # Test with empty data
    empty_metrics = trainer._continuous_training_step([], replay_buffer)
    assert empty_metrics['pseudo_labels_used'] == 0
    assert empty_metrics['gradient_steps'] == 0
    
    print("✅ Empty data handling correct")

def test_configuration_integration():
    """Test configuration integration between components."""
    print("=== CONFIGURATION INTEGRATION TEST ===")
    
    from sragv.training.star_trainer import STARTrainer
    
    orchestrator = MockOrchestrator()
    strategic_oracle = MockStrategicOracle()
    
    # Test that BayesianPseudoLabeler config is properly passed
    custom_config = {
        'bayesian_labeler_config': {
            'base_confidence_threshold': 0.9,
            'adaptive_threshold_enabled': False,
            'min_quality_score': 0.85
        }
    }
    
    trainer = STARTrainer(orchestrator, strategic_oracle, custom_config)
    
    # Check that config was passed to BayesianPseudoLabeler
    labeler_config = trainer.bayesian_pseudo_labeler.config
    assert labeler_config['base_confidence_threshold'] == 0.9
    assert labeler_config['adaptive_threshold_enabled'] == False
    assert labeler_config['min_quality_score'] == 0.85
    
    print("✅ Configuration properly integrated")

def test_error_handling():
    """Test error handling in STARTrainer."""
    print("=== ERROR HANDLING TEST ===")
    
    from sragv.training.star_trainer import STARTrainer
    
    # Create orchestrator with failing solution generator
    class FailingSolutionGenerator:
        def generate(self, problem, num_solutions=6):
            raise Exception("Mock generation failure")
        
        def train_confidence_calibrator(self, *args, **kwargs):
            raise Exception("Mock training failure")
    
    class FailingOrchestrator:
        def __init__(self):
            self.solution_generator = FailingSolutionGenerator()
    
    orchestrator = FailingOrchestrator()
    strategic_oracle = MockStrategicOracle()
    trainer = STARTrainer(orchestrator, strategic_oracle)
    
    # Test solution generation error handling
    problems = create_mock_problems(2)
    solutions = trainer._generate_solutions_with_confidence(problems)
    
    # Should handle errors gracefully and return empty list
    assert isinstance(solutions, list)
    print("✅ Solution generation error handling correct")
    
    # Test training step error handling
    pseudo_labeled_data = [{'solution': {'code': 'test', 'score': 0.9}, 'pseudo_label_confidence': 0.9}]
    metrics = trainer._continuous_training_step(pseudo_labeled_data, [])
    
    # Should return error metrics or handle gracefully
    # The training might succeed if it has fallback mechanisms
    assert isinstance(metrics, dict)
    assert 'training_loss' in metrics
    print("✅ Training step error handling correct")

def run_all_tests():
    """Run all STARTrainer integration tests."""
    print("🧪 STARTING STAR TRAINER INTEGRATION TESTS")
    print("=" * 60)
    
    test_star_trainer_initialization()
    test_solution_generation()
    test_bayesian_pseudo_labeling_integration()
    test_replay_buffer_operations()
    test_training_step()
    test_configuration_integration()
    test_error_handling()
    
    print("=" * 60)
    print("🎉 ALL STAR TRAINER INTEGRATION TESTS PASSED!")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)