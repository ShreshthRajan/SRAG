"""
Problem Generator - Player 1 of SRAG-V architecture.
Generates coding problems with natural language descriptions.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
import logging
import random
import numpy as np

from .base_player import BasePlayer

logger = logging.getLogger(__name__)


class ProblemGenerator(BasePlayer):
    """
    Player 1: Problem Generator (1.5B parameters)
    Model: Qwen2.5-1.5B-Instruct
    
    Input:
    - Current solver success rate histogram
    - Problem type distribution from archive
    - Difficulty progression signal
    
    Output: Code problems with natural language descriptions
    
    Architecture Details:
    - Base model + 3 specialized LoRA adapters:
      - Difficulty prediction (estimates P(solve))
      - Novelty scoring (maximizes coverage in MAP-Elites grid)
      - Semantic diversity (ensures varied problem types)
    """
    
    def __init__(self, config: Dict):
        super().__init__(
            model_name=config["model_name"],
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            lora_config=config.get("lora_config")
        )
        
        # Problem generation parameters
        self.problem_types = [
            "arithmetic", "string_manipulation", "array_processing",
            "graph_algorithms", "dynamic_programming", "sorting",
            "searching", "tree_traversal", "math_computation",
            "pattern_matching"
        ]
        
        self.difficulty_levels = ["easy", "medium", "hard"]
        
    def process_input(
        self,
        success_rate_histogram: Optional[Dict] = None,
        problem_type_distribution: Optional[Dict] = None,
        difficulty_signal: Optional[str] = None,
        target_difficulty: str = "medium",
        target_type: str = "arithmetic"
    ) -> str:
        """Process input and create a prompt for problem generation."""
        
        # Default success rate histogram if not provided
        if success_rate_histogram is None:
            success_rate_histogram = {
                "easy": 0.8,
                "medium": 0.5,
                "hard": 0.2
            }
        
        # Default problem type distribution if not provided
        if problem_type_distribution is None:
            problem_type_distribution = {ptype: 1.0/len(self.problem_types) for ptype in self.problem_types}
        
        prompt = f"""You are an expert coding problem generator. Generate a programming problem based on the following requirements:

TARGET DIFFICULTY: {target_difficulty}
TARGET TYPE: {target_type}

CURRENT SUCCESS RATES:
{json.dumps(success_rate_histogram, indent=2)}

PROBLEM TYPE DISTRIBUTION:
{json.dumps(problem_type_distribution, indent=2)}

DIFFICULTY GUIDANCE:
- Easy: Basic operations, straightforward logic, minimal edge cases
- Medium: Multiple steps, some algorithmic thinking, moderate edge cases
- Hard: Complex algorithms, optimization required, many edge cases

Generate a coding problem with the following structure:
1. Clear problem statement
2. Function signature
3. 2-3 concrete examples
4. Input/output specifications
5. Constraints

Format your response as:
PROBLEM_TITLE: [Brief title]
DIFFICULTY: {target_difficulty}
TYPE: {target_type}

[Problem description]

Function signature: [function signature]

Examples:
[2-3 examples with input/output]

Constraints:
[Any constraints]

Generate a well-structured, interesting problem now:"""

        return prompt
    
    def parse_output(self, output: str) -> Dict:
        """Parse model output into structured problem format."""
        try:
            # Extract components using regex patterns
            title_match = re.search(r'PROBLEM_TITLE:\s*(.+)', output, re.IGNORECASE)
            difficulty_match = re.search(r'DIFFICULTY:\s*(\w+)', output, re.IGNORECASE)
            type_match = re.search(r'TYPE:\s*(\w+)', output, re.IGNORECASE)
            
            # Extract function signature
            func_sig_match = re.search(r'Function signature:\s*(.+)', output, re.IGNORECASE)
            
            # Extract examples section
            examples_match = re.search(r'Examples?:(.*?)(?:Constraints?:|$)', output, re.DOTALL | re.IGNORECASE)
            
            # Extract constraints
            constraints_match = re.search(r'Constraints?:\s*(.+?)$', output, re.DOTALL | re.IGNORECASE)
            
            # Extract main problem description (between TYPE and Function signature)
            desc_pattern = r'TYPE:\s*\w+\s*\n\n(.*?)(?:Function signature:|$)'
            desc_match = re.search(desc_pattern, output, re.DOTALL | re.IGNORECASE)
            
            problem = {
                "problem_id": f"gen_{random.randint(1000, 9999)}",
                "title": title_match.group(1).strip() if title_match else "Generated Problem",
                "difficulty": difficulty_match.group(1).strip().lower() if difficulty_match else "medium",
                "problem_type": type_match.group(1).strip().lower() if type_match else "arithmetic",
                "description": desc_match.group(1).strip() if desc_match else output[:500],
                "function_signature": func_sig_match.group(1).strip() if func_sig_match else "",
                "examples": examples_match.group(1).strip() if examples_match else "",
                "constraints": constraints_match.group(1).strip() if constraints_match else "",
                "raw_output": output
            }
            
            return problem
            
        except Exception as e:
            logger.warning(f"Failed to parse problem output: {e}")
            return {
                "problem_id": f"gen_{random.randint(1000, 9999)}",
                "title": "Parsing Error",
                "difficulty": "medium",
                "problem_type": "arithmetic",
                "description": output[:500],
                "function_signature": "",
                "examples": "",
                "constraints": "",
                "raw_output": output
            }
    
    def generate(
        self,
        success_rate_histogram: Optional[Dict] = None,
        problem_type_distribution: Optional[Dict] = None,
        difficulty_signal: Optional[str] = None,
        target_difficulty: str = "medium",
        target_type: str = "arithmetic",
        num_problems: int = 1
    ) -> List[Dict]:
        """Generate coding problems."""
        logger.info(f"Generating {num_problems} problems with difficulty={target_difficulty}, type={target_type}")
        
        problems = []
        
        for i in range(num_problems):
            # Create prompt
            prompt = self.process_input(
                success_rate_histogram=success_rate_histogram,
                problem_type_distribution=problem_type_distribution,
                difficulty_signal=difficulty_signal,
                target_difficulty=target_difficulty,
                target_type=target_type
            )
            
            # Generate response
            outputs = self.generate_text(
                prompt=prompt,
                max_new_tokens=1024,
                temperature=self.temperature,
                top_p=self.top_p,
                num_return_sequences=1
            )
            
            # Parse output
            if outputs:
                problem = self.parse_output(outputs[0])
                problems.append(problem)
                logger.debug(f"Generated problem: {problem['title']}")
            else:
                logger.warning(f"Failed to generate problem {i+1}")
        
        logger.info(f"Successfully generated {len(problems)} problems")
        return problems
    
    def estimate_difficulty(self, problem_text: str) -> float:
        """Estimate the probability that a solver will succeed on this problem."""
        # Simple heuristic-based difficulty estimation
        # In practice, this would use a specialized LoRA adapter
        
        difficulty_indicators = {
            "easy": ["add", "sum", "basic", "simple", "count", "length"],
            "medium": ["sort", "search", "filter", "transform", "algorithm"],
            "hard": ["optimize", "complex", "dynamic", "graph", "recursive", "efficient"]
        }
        
        text_lower = problem_text.lower()
        
        easy_score = sum(1 for word in difficulty_indicators["easy"] if word in text_lower)
        medium_score = sum(1 for word in difficulty_indicators["medium"] if word in text_lower)
        hard_score = sum(1 for word in difficulty_indicators["hard"] if word in text_lower)
        
        total_score = easy_score + medium_score + hard_score
        if total_score == 0:
            return 0.5  # Default to medium difficulty
        
        # Convert to probability of solving (higher for easier problems)
        if easy_score > medium_score and easy_score > hard_score:
            return 0.8
        elif hard_score > medium_score and hard_score > easy_score:
            return 0.2
        else:
            return 0.5
    
    def compute_novelty_score(self, problem: Dict, archive: List[Dict]) -> float:
        """Compute novelty score for MAP-Elites grid coverage."""
        if not archive:
            return 1.0
        
        # Simple novelty based on problem type and description similarity
        problem_type = problem.get("problem_type", "")
        problem_desc = problem.get("description", "")
        
        type_overlap = sum(1 for p in archive if p.get("problem_type") == problem_type)
        type_novelty = max(0, 1 - (type_overlap / len(archive)))
        
        # Simple text similarity (could be improved with embeddings)
        desc_similarities = []
        for p in archive:
            archive_desc = p.get("description", "")
            # Jaccard similarity of words
            words1 = set(problem_desc.lower().split())
            words2 = set(archive_desc.lower().split())
            if words1 or words2:
                similarity = len(words1 & words2) / len(words1 | words2)
                desc_similarities.append(similarity)
        
        desc_novelty = 1 - max(desc_similarities) if desc_similarities else 1.0
        
        return (type_novelty + desc_novelty) / 2
    
    def generate_diverse_batch(
        self,
        success_rate_histogram: Dict,
        problem_type_distribution: Dict,
        archive: List[Dict],
        batch_size: int = 32
    ) -> List[Dict]:
        """Generate a diverse batch of problems using MAP-Elites diversity mechanism."""
        logger.info(f"Generating diverse batch of {batch_size} problems")
        
        problems = []
        
        # Select target difficulties and types based on distribution
        targets = []
        for _ in range(batch_size):
            # Sample difficulty based on inverse success rate (harder problems get more samples)
            difficulties = list(success_rate_histogram.keys())
            weights = [1 - success_rate_histogram[d] for d in difficulties]
            weights = [w / sum(weights) for w in weights]  # Normalize
            
            target_difficulty = np.random.choice(difficulties, p=weights)
            
            # Sample problem type based on distribution
            types = list(problem_type_distribution.keys())
            type_weights = list(problem_type_distribution.values())
            type_weights = [w / sum(type_weights) for w in type_weights]  # Normalize
            
            target_type = np.random.choice(types, p=type_weights)
            
            targets.append((target_difficulty, target_type))
        
        # Generate problems
        for target_difficulty, target_type in targets:
            batch_problems = self.generate(
                success_rate_histogram=success_rate_histogram,
                problem_type_distribution=problem_type_distribution,
                target_difficulty=target_difficulty,
                target_type=target_type,
                num_problems=1
            )
            
            if batch_problems:
                problem = batch_problems[0]
                # Add diversity metrics
                problem["estimated_difficulty"] = self.estimate_difficulty(problem["description"])
                problem["novelty_score"] = self.compute_novelty_score(problem, archive)
                problems.append(problem)
        
        logger.info(f"Generated {len(problems)} diverse problems")
        return problems