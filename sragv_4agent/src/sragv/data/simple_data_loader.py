"""
Simple data loader for SRAG-V when APPS dataset is not available.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class SimpleDataLoader:
    """Simple data loader for bootstrapping when APPS is unavailable."""
    
    def __init__(self, data_path: str = "data/expanded_apps.json", max_problems: int = 500):
        self.data_path = data_path
        self.max_problems = max_problems
        logger.info(f"SimpleDataLoader initialized: {data_path}, max_problems={max_problems}")
    
    def prepare_bootstrap_data(self) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """Load and prepare bootstrap data."""
        try:
            with open(self.data_path, 'r') as f:
                problems = json.load(f)
            
            # Limit to max_problems
            if len(problems) > self.max_problems:
                problems = problems[:self.max_problems]
            
            # Categorize problems
            categorized = self._categorize_problems(problems)
            
            logger.info(f"Loaded {len(problems)} problems")
            return problems, categorized
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return [], {}
    
    def _categorize_problems(self, problems: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize problems by type."""
        categorized = {}
        
        for problem in problems:
            problem_type = problem.get('problem_type', 'unknown')
            if problem_type not in categorized:
                categorized[problem_type] = []
            categorized[problem_type].append(problem)
        
        return categorized