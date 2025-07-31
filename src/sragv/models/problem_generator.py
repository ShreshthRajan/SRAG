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
    ) -> List[Dict[str, str]]:
        """Process input and create Qwen2.5-1.5B-Instruct chat messages for problem generation."""
        
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
        
        # Create system message optimized for Qwen2.5-1.5B-Instruct
        system_message = """You are Qwen, created by Alibaba Cloud. You are an expert competitive programming problem setter specializing in creating high-quality coding challenges.

Your task is to generate well-structured, engaging coding problems that test algorithmic thinking and programming skills. Follow these critical requirements:
1. ALWAYS create problems with clear, unambiguous descriptions
2. Use proper mathematical notation and formatting
3. Include comprehensive examples with explanations
4. Set realistic constraints based on the target difficulty
5. Ensure problems are solvable and have deterministic solutions"""

        # Create user message with few-shot examples and chain-of-thought reasoning
        user_message = f"""Generate a {target_difficulty} difficulty coding problem of type "{target_type}".

Current success rates show: {json.dumps(success_rate_histogram, indent=1)}
Focus on creating problems that fill gaps in our current distribution.

Here are examples of well-structured problems:

**Example 1 - Easy Array Problem:**
PROBLEM_TITLE: Sum of Even Numbers
DIFFICULTY: easy
TYPE: array_processing

Given an array of integers, return the sum of all even numbers.

Function signature: def sum_even_numbers(nums: List[int]) -> int:

Examples:
- Input: [1, 2, 3, 4, 5, 6] → Output: 12 (2 + 4 + 6)
- Input: [1, 3, 5] → Output: 0 (no even numbers)
- Input: [] → Output: 0 (empty array)

Constraints:
- 0 ≤ len(nums) ≤ 1000
- -1000 ≤ nums[i] ≤ 1000

**Example 2 - Medium String Problem:**
PROBLEM_TITLE: Longest Palindromic Substring
DIFFICULTY: medium  
TYPE: string_manipulation

Find the longest palindromic substring in a given string. If multiple palindromes have the same maximum length, return the first one found.

Function signature: def longest_palindrome(s: str) -> str:

Examples:
- Input: "babad" → Output: "bab" (or "aba", both valid)
- Input: "cbbd" → Output: "bb"
- Input: "a" → Output: "a"

Constraints:
- 1 ≤ len(s) ≤ 1000
- s consists of lowercase English letters only

Now generate a {target_difficulty} difficulty problem of type "{target_type}" following this exact format:

PROBLEM_TITLE:"""

        # Return as chat messages for proper Qwen2.5-1.5B-Instruct formatting
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def parse_output(self, output: str) -> Optional[Dict]:
        """Parse model output into structured problem format with improved error handling."""
        try:
            # Method 1: Extract from properly formatted output
            title_match = re.search(r'PROBLEM_TITLE:\s*(.+)', output, re.IGNORECASE)
            difficulty_match = re.search(r'DIFFICULTY:\s*(\w+)', output, re.IGNORECASE)
            type_match = re.search(r'TYPE:\s*(\w+)', output, re.IGNORECASE)
            
            # Extract function signature with multiple patterns
            func_sig_patterns = [
                r'Function signature:\s*(.+?)(?:\n|$)',
                r'def\s+\w+\([^)]*\)\s*->\s*\w+:',
                r'def\s+\w+\([^)]*\):'
            ]
            
            func_sig = ""
            for pattern in func_sig_patterns:
                match = re.search(pattern, output, re.IGNORECASE | re.MULTILINE)
                if match:
                    func_sig = match.group(1) if match.groups() else match.group(0)
                    break
            
            # Extract examples section with better parsing
            examples_patterns = [
                r'Examples?:\s*(.*?)(?:Constraints?:|$)',
                r'Input:.*?Output:.*?(?=Constraints?:|$)',
            ]
            
            examples = ""
            for pattern in examples_patterns:
                match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
                if match:
                    examples = match.group(1).strip() if match.groups() else match.group(0)
                    break
            
            # Extract constraints
            constraints_match = re.search(r'Constraints?:\s*(.*?)$', output, re.DOTALL | re.IGNORECASE)
            
            # Extract main problem description (more flexible)
            desc_patterns = [
                r'TYPE:\s*\w+\s*\n\n(.*?)(?:Function signature:|def\s+\w+|$)',
                r'DIFFICULTY:\s*\w+\s*\n\n(.*?)(?:Function signature:|def\s+\w+|$)',
                r'PROBLEM_TITLE:.*?\n\n(.*?)(?:Function signature:|def\s+\w+|Examples?:|$)'
            ]
            
            description = ""
            for pattern in desc_patterns:
                match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
                if match:
                    description = match.group(1).strip()
                    if len(description) > 20:  # Reasonable description length
                        break
            
            # Validate that we have essential components
            if not title_match or not description:
                logger.warning("Missing essential problem components in output")
                return None
            
            problem = {
                "problem_id": f"gen_{random.randint(1000, 9999)}",
                "title": title_match.group(1).strip(),
                "difficulty": difficulty_match.group(1).strip().lower() if difficulty_match else "medium",
                "problem_type": type_match.group(1).strip().lower().replace("_", "_") if type_match else "arithmetic",
                "question": description,  # Using 'question' to match dataset format
                "function_signature": func_sig,
                "examples": examples,
                "constraints": constraints_match.group(1).strip() if constraints_match else "",
                "raw_output": output
            }
            
            # Quality validation
            if self._validate_problem_quality(problem):
                return problem
            else:
                logger.warning("Generated problem failed quality validation")
                return None
            
        except Exception as e:
            logger.error(f"Critical error in parse_output: {e}")
            return None
    
    def _validate_problem_quality(self, problem: Dict) -> bool:
        """Validate generated problem quality."""
        try:
            # Must have title and description
            if not problem.get("title") or not problem.get("question"):
                return False
            
            # Title should be reasonable length
            title = problem.get("title", "")
            if len(title) < 3 or len(title) > 100:
                return False
            
            # Description should be substantial
            description = problem.get("question", "")
            if len(description) < 20 or len(description) > 2000:
                return False
            
            # Should have examples if possible
            examples = problem.get("examples", "")
            if not examples or len(examples) < 10:
                logger.debug("Problem missing substantial examples")
            
            # Check for problematic content
            problematic_patterns = [
                "I cannot",
                "I'm sorry",
                "error occurred",
                "cannot generate",
                "inappropriate"
            ]
            
            full_text = f"{title} {description} {examples}".lower()
            for pattern in problematic_patterns:
                if pattern.lower() in full_text:
                    logger.debug(f"Problem contains problematic pattern: {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Problem validation error: {e}")
            return False
    
    def generate(
        self,
        success_rate_histogram: Optional[Dict] = None,
        problem_type_distribution: Optional[Dict] = None,
        difficulty_signal: Optional[str] = None,
        target_difficulty: str = "medium",
        target_type: str = "arithmetic",
        num_problems: int = 1,
        max_retry_attempts: int = 3
    ) -> List[Dict]:
        """Generate coding problems using SOTA Qwen2.5-1.5B-Instruct techniques."""
        logger.info(f"Generating {num_problems} problems with difficulty={target_difficulty}, type={target_type}")
        
        problems = []
        valid_problems_count = 0
        
        for i in range(num_problems):
            problem_generated = False
            
            # Use progressive temperature strategy
            base_temp = 0.3 if target_difficulty == "easy" else 0.5 if target_difficulty == "medium" else 0.7
            
            for attempt in range(max_retry_attempts):
                try:
                    # Create chat messages using proper Qwen2.5-1.5B-Instruct format
                    messages = self.process_input(
                        success_rate_histogram=success_rate_histogram,
                        problem_type_distribution=problem_type_distribution,
                        difficulty_signal=difficulty_signal,
                        target_difficulty=target_difficulty,
                        target_type=target_type
                    )
                    
                    # Apply temperature variation for retries
                    current_temp = base_temp + (attempt * 0.1)
                    current_temp = min(current_temp, 0.9)
                    
                    # Generate using chat template
                    outputs = self.generate_text(
                        messages=messages,
                        max_new_tokens=800,  # Sufficient for problem generation
                        temperature=current_temp,
                        top_p=0.9,
                        do_sample=current_temp > 0.0,
                        num_return_sequences=1
                    )
                    
                    if outputs and len(outputs) > 0:
                        # Parse problem with improved error handling
                        problem = self.parse_output(outputs[0])
                        
                        if problem:
                            # Add generation metadata
                            problem.update({
                                "generation_temperature": current_temp,
                                "retry_attempt": attempt + 1,
                                "target_difficulty": target_difficulty,
                                "target_type": target_type
                            })
                            
                            problems.append(problem)
                            valid_problems_count += 1
                            problem_generated = True
                            
                            logger.debug(f"✅ Generated problem {i+1}/{num_problems}: {problem['title']} (temp: {current_temp:.2f})")
                            break
                        else:
                            logger.debug(f"⚠️ Problem {i+1} attempt {attempt+1} failed validation")
                    else:
                        logger.debug(f"⚠️ No output generated for problem {i+1} attempt {attempt+1}")
                    
                except Exception as e:
                    logger.warning(f"❌ Error generating problem {i+1}, attempt {attempt+1}: {e}")
                    continue
            
            if not problem_generated:
                logger.warning(f"❌ Failed to generate valid problem {i+1}/{num_problems} after {max_retry_attempts} attempts")
        
        success_rate = (valid_problems_count / num_problems) * 100
        logger.info(f"✅ Generated {valid_problems_count}/{num_problems} problems (success rate: {success_rate:.1f}%)")
        
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
        batch_size: int = 32,
        difficulty_scaling: float = 1.0  # Scaling factor for problem difficulty
    ) -> List[Dict]:
        """Generate a diverse batch of problems using MAP-Elites diversity mechanism."""
        logger.info(f"Generating diverse batch of {batch_size} problems")
        
        problems = []
        
        # Select target difficulties and types based on distribution
        targets = []
        for _ in range(batch_size):
            # Sample difficulty based on inverse success rate (harder problems get more samples)
            # Apply difficulty scaling to bias toward harder problems when scaling > 1.0
            difficulties = list(success_rate_histogram.keys())
            weights = [1 - success_rate_histogram[d] for d in difficulties]
            
            # Apply difficulty scaling: higher scaling increases weight of harder difficulties
            if difficulty_scaling > 1.0:
                # Map difficulty names to numeric values for scaling
                difficulty_levels = {"easy": 1, "medium": 2, "hard": 3, "interview": 2, "competition": 3}
                scaled_weights = []
                for i, d in enumerate(difficulties):
                    level = difficulty_levels.get(d, 2)  # Default to medium
                    scale_factor = difficulty_scaling ** (level - 1)  # Exponential scaling
                    scaled_weights.append(weights[i] * scale_factor)
                weights = scaled_weights
            
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