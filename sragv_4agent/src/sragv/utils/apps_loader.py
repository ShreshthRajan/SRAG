"""
Real APPS dataset loader for SRAG-V project.
Downloads and preprocesses the full APPS dataset from HuggingFace.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class APPSDataLoader:
    """Handles loading and preprocessing of APPS dataset."""
    
    def __init__(self, cache_dir: str = "./data", max_problems: int = 500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_problems = max_problems
        
    def load_apps_dataset(self) -> Dataset:
        """Load APPS dataset from HuggingFace."""
        logger.info("Loading APPS dataset from HuggingFace...")
        
        try:
            dataset = load_dataset(
                "codeparrot/apps",
                split="train",
                cache_dir=str(self.cache_dir / "huggingface")
            )
            logger.info(f"Loaded {len(dataset)} problems from APPS dataset")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load APPS dataset: {e}")
            raise
    
    def filter_rich_problems(self, dataset: Dataset, min_test_cases: int = 3) -> List[Dict]:
        """Filter problems with rich test suites for bootstrapping."""
        logger.info(f"Filtering problems with at least {min_test_cases} test cases...")
        
        rich_problems = []
        
        for item in tqdm(dataset, desc="Filtering problems"):
            try:
                # Parse test cases
                input_output = json.loads(item.get('input_output', '{}'))
                inputs = input_output.get('inputs', [])
                outputs = input_output.get('outputs', [])
                
                # Check if we have enough test cases
                if len(inputs) >= min_test_cases and len(outputs) >= min_test_cases:
                    # Parse solutions
                    solutions = json.loads(item.get('solutions', '[]'))
                    if not solutions:  # Skip problems without solutions
                        continue
                        
                    problem_data = {
                        'problem_id': item.get('problem_id', ''),
                        'question': item.get('question', ''),
                        'solutions': solutions,
                        'starter_code': item.get('starter_code', ''),
                        'input_output': input_output,
                        'difficulty': item.get('difficulty', 'unknown'),
                        'url': item.get('url', ''),
                        'source': item.get('source', ''),
                        'test_case_count': len(inputs)
                    }
                    rich_problems.append(problem_data)
                    
                    if len(rich_problems) >= self.max_problems:
                        break
                        
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Skipping malformed problem: {e}")
                continue
        
        logger.info(f"Found {len(rich_problems)} problems with rich test suites")
        return rich_problems
    
    def categorize_problems(self, problems: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize problems by difficulty for structured bootstrapping."""
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
    
    def save_processed_data(self, problems: List[Dict], filename: str = "apps_processed.json"):
        """Save processed problems to disk."""
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(problems, f, indent=2)
        
        logger.info(f"Saved {len(problems)} processed problems to {filepath}")
    
    def load_processed_data(self, filename: str = "apps_processed.json") -> Optional[List[Dict]]:
        """Load previously processed problems from disk."""
        filepath = self.cache_dir / filename
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                problems = json.load(f)
            logger.info(f"Loaded {len(problems)} processed problems from {filepath}")
            return problems
        
        return None
    
    def prepare_bootstrap_data(self) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """Main method to prepare data for bootstrapping."""
        # Try to load cached data first
        cached_problems = self.load_processed_data()
        if cached_problems:
            categorized = self.categorize_problems(cached_problems)
            return cached_problems, categorized
        
        # Load and process dataset
        dataset = self.load_apps_dataset()
        rich_problems = self.filter_rich_problems(dataset)
        
        # Save processed data
        self.save_processed_data(rich_problems)
        
        # Categorize problems
        categorized = self.categorize_problems(rich_problems)
        
        return rich_problems, categorized


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
    # Test the APPS data loader
    logging.basicConfig(level=logging.INFO)
    
    loader = APPSDataLoader(max_problems=100)  # Small test
    problems, categorized = loader.prepare_bootstrap_data()
    
    print(f"Total problems: {len(problems)}")
    for category, probs in categorized.items():
        print(f"{category.title()}: {len(probs)} problems")
    
    # Validate first problem
    if problems:
        is_valid = validate_problem_format(problems[0])
        print(f"First problem valid: {is_valid}")
        print(f"First problem keys: {list(problems[0].keys())}")
        print(f"First problem question: {problems[0]['question'][:200]}...")