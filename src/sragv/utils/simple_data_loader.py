"""
Simplified data loader for SRAG-V project that works without HuggingFace datasets.
Creates synthetic APPS-like problems for initial bootstrapping and testing.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SimpleAPPSDataLoader:
    """Simplified APPS data loader for initial development."""
    
    def __init__(self, cache_dir: str = "./data", max_problems: int = 100):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_problems = max_problems
        
    def create_synthetic_problems(self) -> List[Dict]:
        """Create synthetic coding problems for initial testing."""
        logger.info("Creating synthetic coding problems for testing...")
        
        synthetic_problems = []
        
        # Problem 1: Simple arithmetic
        problem1 = {
            'problem_id': 'test_001',
            'question': '''Write a function that takes two integers a and b and returns their sum.

Function signature: def add_numbers(a: int, b: int) -> int:

Example:
- add_numbers(2, 3) should return 5
- add_numbers(-1, 1) should return 0
''',
            'solutions': ['''def add_numbers(a: int, b: int) -> int:
    return a + b'''],
            'starter_code': 'def add_numbers(a: int, b: int) -> int:\n    # Your code here\n    pass',
            'input_output': {
                'inputs': [
                    ['2', '3'],
                    ['-1', '1'], 
                    ['0', '0'],
                    ['100', '-50'],
                    ['-10', '-20']
                ],
                'outputs': [
                    '5',
                    '0',
                    '0', 
                    '50',
                    '-30'
                ]
            },
            'difficulty': 'introductory',
            'url': 'synthetic://test_001',
            'source': 'synthetic',
            'test_case_count': 5
        }
        synthetic_problems.append(problem1)
        
        # Problem 2: String manipulation
        problem2 = {
            'problem_id': 'test_002',
            'question': '''Write a function that takes a string and returns its reverse.

Function signature: def reverse_string(s: str) -> str:

Example:
- reverse_string("hello") should return "olleh"
- reverse_string("") should return ""
''',
            'solutions': ['''def reverse_string(s: str) -> str:
    return s[::-1]'''],
            'starter_code': 'def reverse_string(s: str) -> str:\n    # Your code here\n    pass',
            'input_output': {
                'inputs': [
                    ['hello'],
                    [''],
                    ['a'],
                    ['abc def'],
                    ['12345']
                ],
                'outputs': [
                    'olleh',
                    '',
                    'a',
                    'fed cba',
                    '54321'
                ]
            },
            'difficulty': 'introductory',
            'url': 'synthetic://test_002',
            'source': 'synthetic',
            'test_case_count': 5
        }
        synthetic_problems.append(problem2)
        
        # Problem 3: List operations
        problem3 = {
            'problem_id': 'test_003',
            'question': '''Write a function that finds the maximum element in a list of integers.

Function signature: def find_max(numbers: List[int]) -> int:

Example:
- find_max([1, 3, 2]) should return 3
- find_max([-1, -5, -2]) should return -1
''',
            'solutions': ['''def find_max(numbers: List[int]) -> int:
    return max(numbers)'''],
            'starter_code': 'def find_max(numbers: List[int]) -> int:\n    # Your code here\n    pass',
            'input_output': {
                'inputs': [
                    ['[1, 3, 2]'],
                    ['[-1, -5, -2]'],
                    ['[42]'],
                    ['[0, 0, 0]'],
                    ['[100, 99, 101, 50]']
                ],
                'outputs': [
                    '3',
                    '-1',
                    '42',
                    '0',
                    '101'
                ]
            },
            'difficulty': 'interview',
            'url': 'synthetic://test_003',
            'source': 'synthetic',
            'test_case_count': 5
        }
        synthetic_problems.append(problem3)
        
        # Add more problems up to max_problems
        base_problems = [problem1, problem2, problem3]
        while len(synthetic_problems) < min(self.max_problems, 10):  # Cap at 10 for now
            # Create variations
            idx = len(synthetic_problems)
            base_idx = idx % len(base_problems)
            base_problem = base_problems[base_idx].copy()
            base_problem['problem_id'] = f'test_{idx+1:03d}'
            base_problem['url'] = f'synthetic://test_{idx+1:03d}'
            synthetic_problems.append(base_problem)
        
        logger.info(f"Created {len(synthetic_problems)} synthetic problems")
        return synthetic_problems
    
    def try_download_real_apps(self) -> Optional[List[Dict]]:
        """Try to download a small subset of real APPS data from GitHub."""
        logger.info("Attempting to download real APPS data...")
        
        try:
            # Try to get a few problems from the APPS GitHub repo
            apps_url = "https://api.github.com/repos/hendrycks/apps/contents/test"
            response = requests.get(apps_url, timeout=30)
            
            if response.status_code == 200:
                logger.info("Successfully contacted APPS repository")
                # For now, return None to use synthetic data
                # In production, we would parse the GitHub response
                return None
            else:
                logger.warning(f"Failed to access APPS repo: {response.status_code}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to download APPS data: {e}")
            return None
    
    def categorize_problems(self, problems: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize problems by difficulty."""
        categorized = {
            'easy': [],
            'medium': [],
            'hard': []
        }
        
        for problem in problems:
            difficulty = problem.get('difficulty', 'unknown').lower()
            if difficulty in ['introductory', 'interview']:
                categorized['easy'].append(problem)
            elif difficulty in ['competition']:
                categorized['hard'].append(problem)
            else:
                categorized['medium'].append(problem)
        
        logger.info(f"Categorized problems: "
                   f"Easy: {len(categorized['easy'])}, "
                   f"Medium: {len(categorized['medium'])}, "
                   f"Hard: {len(categorized['hard'])}")
        
        return categorized
    
    def save_processed_data(self, problems: List[Dict], filename: str = "processed_apps.json"):
        """Save processed problems to disk."""
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(problems, f, indent=2)
        
        logger.info(f"Saved {len(problems)} processed problems to {filepath}")
    
    def load_processed_data(self, filename: str = "expanded_apps.json") -> Optional[List[Dict]]:
        """Load previously processed problems from disk."""
        filepath = self.cache_dir / filename
        
        # Try expanded dataset first (2000 problems), then fall back to processed
        if not filepath.exists() and filename == "expanded_apps.json":
            logger.info("Expanded dataset not found, trying processed_apps.json")
            return self.load_processed_data("processed_apps.json")
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                problems = json.load(f)
            logger.info(f"Loaded {len(problems)} processed problems from {filepath}")
            return problems
        
        return None
    
    def prepare_bootstrap_data(self) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """Main method to prepare data for bootstrapping."""
        # Try to load cached data first - prioritize expanded dataset
        cached_problems = self.load_processed_data("expanded_apps.json")
        if cached_problems:
            categorized = self.categorize_problems(cached_problems)
            return cached_problems, categorized
        
        # Try to download real APPS data
        real_problems = self.try_download_real_apps()
        
        if real_problems is None:
            # Fall back to synthetic data
            logger.info("Using synthetic problems for bootstrapping")
            problems = self.create_synthetic_problems()
        else:
            problems = real_problems
        
        # Save processed data
        self.save_processed_data(problems)
        
        # Categorize problems
        categorized = self.categorize_problems(problems)
        
        return problems, categorized


def validate_problem_format(problem: Dict) -> bool:
    """Validate that a problem has the required format."""
    required_fields = ['problem_id', 'question', 'solutions', 'input_output']
    
    for field in required_fields:
        if field not in problem:
            return False
    
    # Check that we have valid test cases
    input_output = problem['input_output']
    if not isinstance(input_output, dict):
        return False
    
    inputs = input_output.get('inputs', [])
    outputs = input_output.get('outputs', [])
    
    return len(inputs) > 0 and len(outputs) > 0 and len(inputs) == len(outputs)


if __name__ == "__main__":
    # Test the simple data loader
    logging.basicConfig(level=logging.INFO)
    
    loader = SimpleAPPSDataLoader(max_problems=5)
    problems, categorized = loader.prepare_bootstrap_data()
    
    print(f"Total problems: {len(problems)}")
    for category, probs in categorized.items():
        print(f"{category.title()}: {len(probs)} problems")
    
    # Validate first problem
    if problems:
        is_valid = validate_problem_format(problems[0])
        print(f"First problem valid: {is_valid}")
        print(f"First problem: {problems[0]['question'][:100]}...")