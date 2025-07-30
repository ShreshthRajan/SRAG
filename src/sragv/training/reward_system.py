"""
Role-Conditioned Reward System for SRAG-V 4-Player Self-Play.
Implements sophisticated reward functions for each player role based on
verification learning principles and emergent behavior objectives.
"""

import ast
import re
import subprocess
import tempfile
import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class RewardMetrics:
    """Container for computed reward metrics."""
    base_reward: float
    bonus_reward: float
    penalty: float
    final_reward: float
    components: Dict[str, float]
    metadata: Dict[str, Any]


class CodeExecutionEnvironment:
    """Safe code execution environment for testing solutions."""
    
    def __init__(self, timeout_seconds: int = 5):
        self.timeout_seconds = timeout_seconds
    
    def execute_code(self, code: str, test_inputs: List[Any]) -> Tuple[bool, List[Any], str]:
        """
        Execute code with test inputs safely.
        
        Returns:
            (success, outputs, error_message)
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                # Write the code
                f.write(code)
                f.write('\n\n')
                
                # Add test execution
                f.write('import json\n')
                f.write('import sys\n')
                f.write('results = []\n')
                f.write('errors = []\n')
                
                # Extract function name from code
                function_names = re.findall(r'def\s+(\w+)\s*\(', code)
                if not function_names:
                    return False, [], "No function definition found"
                
                main_function = function_names[0]
                
                for i, test_input in enumerate(test_inputs):
                    f.write(f'try:\n')
                    if isinstance(test_input, (list, tuple)) and len(test_input) > 1:
                        args_str = ', '.join(repr(arg) for arg in test_input)
                        f.write(f'    result = {main_function}({args_str})\n')
                    else:
                        if isinstance(test_input, (list, tuple)) and len(test_input) == 1:
                            test_input = test_input[0]
                        f.write(f'    result = {main_function}({repr(test_input)})\n')
                    f.write(f'    results.append(result)\n')
                    f.write(f'except Exception as e:\n')
                    f.write(f'    results.append(None)\n')
                    f.write(f'    errors.append(str(e))\n')
                
                f.write('print(json.dumps({"results": results, "errors": errors}))\n')
                temp_file = f.name
            
            # Execute with timeout
            try:
                result = subprocess.run(
                    ['python', temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds
                )
                
                if result.returncode == 0:
                    output_data = json.loads(result.stdout.strip())
                    return True, output_data['results'], '\n'.join(output_data['errors'])
                else:
                    return False, [], result.stderr
                    
            except subprocess.TimeoutExpired:
                return False, [], "Code execution timeout"
            except json.JSONDecodeError:
                return False, [], f"Invalid output: {result.stdout}"
            
        except Exception as e:
            return False, [], str(e)
        finally:
            # Clean up temporary file
            if 'temp_file' in locals():
                try:
                    os.unlink(temp_file)
                except:
                    pass


class ProblemGeneratorReward:
    """Reward function for Problem Generator (Player 1)."""
    
    def __init__(self):
        self.execution_env = CodeExecutionEnvironment()
    
    def compute_reward(self, output: str, context: Dict[str, Any]) -> RewardMetrics:
        """
        Compute reward for problem generation.
        
        Evaluates:
        - Problem clarity and completeness
        - Appropriate difficulty level
        - Clear input/output specification
        - Test case adequacy
        """
        components = {}
        
        # Parse the generated problem
        problem_data = self._parse_problem_output(output)
        
        # Base reward: Problem structure completeness
        structure_reward = self._evaluate_problem_structure(problem_data)
        components['structure'] = structure_reward
        
        # Clarity and readability
        clarity_reward = self._evaluate_problem_clarity(problem_data)
        components['clarity'] = clarity_reward
        
        # Difficulty appropriateness
        difficulty_reward = self._evaluate_difficulty_appropriateness(problem_data, context)
        components['difficulty'] = difficulty_reward
        
        # Test case quality
        test_quality_reward = self._evaluate_test_case_quality(problem_data)
        components['test_quality'] = test_quality_reward
        
        # Novelty bonus
        novelty_bonus = self._compute_novelty_bonus(problem_data, context)
        components['novelty'] = novelty_bonus
        
        # Compute base reward
        base_reward = np.mean([structure_reward, clarity_reward, difficulty_reward, test_quality_reward])
        
        # Apply bonuses and penalties
        bonus_reward = novelty_bonus
        penalty = self._compute_penalty(problem_data)
        components['penalty'] = -penalty
        
        final_reward = base_reward + bonus_reward - penalty
        
        return RewardMetrics(
            base_reward=base_reward,
            bonus_reward=bonus_reward,
            penalty=penalty,
            final_reward=final_reward,
            components=components,
            metadata={'problem_data': problem_data}
        )
    
    def _parse_problem_output(self, output: str) -> Dict[str, Any]:
        """Parse problem generator output into structured format."""
        problem_data = {
            'title': '',
            'description': '',
            'input_format': '',
            'output_format': '',
            'examples': [],
            'constraints': '',
            'test_cases': []
        }
        
        lines = output.split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            lower_line = line.lower()
            if 'title' in lower_line and ':' in line:
                problem_data['title'] = line.split(':', 1)[1].strip()
            elif 'description' in lower_line or 'problem' in lower_line:
                current_section = 'description'
                continue
            elif 'input' in lower_line:
                current_section = 'input_format'
                continue
            elif 'output' in lower_line:
                current_section = 'output_format'
                continue
            elif 'example' in lower_line:
                current_section = 'examples'
                continue
            elif 'constraint' in lower_line:
                current_section = 'constraints'
                continue
            
            # Add content to current section
            if current_section:
                if current_section == 'examples':
                    problem_data['examples'].append(line)
                else:
                    if problem_data[current_section]:
                        problem_data[current_section] += ' ' + line
                    else:
                        problem_data[current_section] = line
        
        return problem_data
    
    def _evaluate_problem_structure(self, problem_data: Dict[str, Any]) -> float:
        """Evaluate completeness of problem structure."""
        required_components = ['title', 'description', 'input_format', 'output_format']
        optional_components = ['examples', 'constraints']
        
        # Required components (80% of score)
        required_score = 0.0
        for component in required_components:
            if problem_data[component] and len(problem_data[component]) > 10:
                required_score += 0.2  # 20% per component
        
        # Optional components (20% of score)
        optional_score = 0.0
        for component in optional_components:
            if problem_data[component]:
                optional_score += 0.1  # 10% per component
        
        return required_score + optional_score
    
    def _evaluate_problem_clarity(self, problem_data: Dict[str, Any]) -> float:
        """Evaluate clarity and readability of problem description."""
        description = problem_data['description']
        if not description:
            return 0.0
        
        clarity_score = 0.0
        
        # Length appropriateness (not too short, not too long)
        word_count = len(description.split())
        if 20 <= word_count <= 200:
            clarity_score += 0.3
        elif word_count > 10:
            clarity_score += 0.1
        
        # Clear structure indicators
        structure_keywords = ['input', 'output', 'return', 'given', 'find', 'calculate']
        keyword_count = sum(1 for keyword in structure_keywords if keyword in description.lower())
        clarity_score += min(keyword_count * 0.1, 0.3)
        
        # Examples provided
        if problem_data['examples']:
            clarity_score += 0.2
        
        # Constraints specified
        if problem_data['constraints']:
            clarity_score += 0.2
        
        return min(clarity_score, 1.0)
    
    def _evaluate_difficulty_appropriateness(self, problem_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Evaluate if problem difficulty matches target."""
        target_difficulty = context.get('target_difficulty', 'medium')
        description = problem_data['description'].lower()
        
        # Difficulty indicators
        easy_indicators = ['add', 'sum', 'count', 'find', 'simple', 'basic']
        medium_indicators = ['sort', 'search', 'loop', 'condition', 'array', 'string']
        hard_indicators = ['algorithm', 'optimize', 'complex', 'efficient', 'data structure', 'recursive']
        
        easy_count = sum(1 for indicator in easy_indicators if indicator in description)
        medium_count = sum(1 for indicator in medium_indicators if indicator in description)
        hard_count = sum(1 for indicator in hard_indicators if indicator in description)
        
        # Score based on target difficulty
        if target_difficulty == 'easy':
            if easy_count > 0 and hard_count == 0:
                return 1.0
            elif medium_count > 0 and hard_count == 0:
                return 0.7
            else:
                return 0.3
        elif target_difficulty == 'medium':
            if medium_count > 0:
                return 1.0
            elif easy_count > 0 or hard_count > 0:
                return 0.6
            else:
                return 0.5
        elif target_difficulty == 'hard':
            if hard_count > 0:
                return 1.0
            elif medium_count > 0:
                return 0.6
            else:
                return 0.2
        
        return 0.5  # Default moderate score
    
    def _evaluate_test_case_quality(self, problem_data: Dict[str, Any]) -> float:
        """Evaluate quality of provided test cases."""
        examples = problem_data['examples']
        if not examples:
            return 0.3  # Moderate score for no examples
        
        quality_score = 0.0
        
        # Number of examples
        num_examples = len(examples)
        quality_score += min(num_examples * 0.2, 0.6)
        
        # Example completeness (input and expected output)
        complete_examples = 0
        for example in examples:
            if '->' in example or 'output' in example.lower() or 'return' in example.lower():
                complete_examples += 1
        
        if complete_examples > 0:
            quality_score += 0.4 * (complete_examples / num_examples)
        
        return min(quality_score, 1.0)
    
    def _compute_novelty_bonus(self, problem_data: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Compute bonus for novel problem characteristics."""
        # Check against recent problems in context
        recent_problems = context.get('recent_problems', [])
        if not recent_problems:
            return 0.2  # Bonus for first problem
        
        current_title = problem_data['title'].lower()
        current_description = problem_data['description'].lower()
        
        # Compute similarity to recent problems
        max_similarity = 0.0
        for recent_problem in recent_problems[-10:]:  # Check last 10 problems
            recent_title = recent_problem.get('title', '').lower()
            recent_desc = recent_problem.get('description', '').lower()
            
            # Simple similarity based on common words
            current_words = set(current_description.split())
            recent_words = set(recent_desc.split())
            
            if current_words and recent_words:
                intersection = len(current_words & recent_words)
                union = len(current_words | recent_words)
                similarity = intersection / union if union > 0 else 0
                max_similarity = max(max_similarity, similarity)
        
        # Higher novelty bonus for lower similarity
        novelty_bonus = (1.0 - max_similarity) * 0.3
        return novelty_bonus
    
    def _compute_penalty(self, problem_data: Dict[str, Any]) -> float:
        """Compute penalties for problem quality issues."""
        penalty = 0.0
        
        # Penalty for missing critical components
        if not problem_data['title']:
            penalty += 0.2
        if not problem_data['description'] or len(problem_data['description']) < 20:
            penalty += 0.3
        if not problem_data['input_format']:
            penalty += 0.1
        if not problem_data['output_format']:
            penalty += 0.1
        
        # Penalty for unclear or inconsistent format
        description = problem_data['description'].lower()
        if 'todo' in description or 'implement' in description:
            penalty += 0.1  # Penalty for incomplete descriptions
        
        return penalty


class SolutionGeneratorReward:
    """Reward function for Solution Generator (Player 2)."""
    
    def __init__(self):
        self.execution_env = CodeExecutionEnvironment()
    
    def compute_reward(self, output: str, context: Dict[str, Any]) -> RewardMetrics:
        """
        Compute reward for solution generation.
        
        Evaluates:
        - Code correctness and functionality
        - Code quality and style
        - Efficiency considerations
        - Edge case handling
        """
        components = {}
        
        problem = context.get('problem', {})
        test_cases = context.get('test_cases', [])
        
        # Correctness: Execute code against test cases
        correctness_reward = self._evaluate_correctness(output, problem, test_cases)
        components['correctness'] = correctness_reward
        
        # Code quality and style
        quality_reward = self._evaluate_code_quality(output)
        components['quality'] = quality_reward
        
        # Completeness and structure
        completeness_reward = self._evaluate_completeness(output, problem)
        components['completeness'] = completeness_reward
        
        # Efficiency considerations
        efficiency_reward = self._evaluate_efficiency(output)
        components['efficiency'] = efficiency_reward
        
        # Edge case handling
        edge_case_reward = self._evaluate_edge_case_handling(output, test_cases)
        components['edge_cases'] = edge_case_reward
        
        # Compute base reward (weighted by importance)
        weights = {
            'correctness': 0.4,
            'quality': 0.2,
            'completeness': 0.2,
            'efficiency': 0.1,
            'edge_cases': 0.1
        }
        
        base_reward = sum(components[key] * weights[key] for key in weights)
        
        # Bonus for exceptional solutions
        bonus_reward = self._compute_solution_bonus(output, components)
        components['bonus'] = bonus_reward
        
        # Penalty for problematic code
        penalty = self._compute_solution_penalty(output)
        components['penalty'] = -penalty
        
        final_reward = base_reward + bonus_reward - penalty
        
        return RewardMetrics(
            base_reward=base_reward,
            bonus_reward=bonus_reward,
            penalty=penalty,
            final_reward=final_reward,
            components=components,
            metadata={'execution_results': components.get('execution_results', {})}
        )
    
    def _evaluate_correctness(self, code: str, problem: Dict[str, Any], test_cases: List[Any]) -> float:
        """Evaluate correctness by running test cases."""
        if not test_cases:
            # No test cases provided, use basic syntax check
            try:
                ast.parse(code)
                return 0.5  # Basic syntax correctness
            except SyntaxError:
                return 0.0
        
        # Execute code with test cases
        success, results, error_msg = self.execution_env.execute_code(code, test_cases)
        
        if not success:
            logger.debug(f"Code execution failed: {error_msg}")
            return 0.0
        
        # Compare results with expected outputs
        expected_outputs = problem.get('input_output', {}).get('outputs', [])
        if not expected_outputs:
            # No expected outputs, just check if code runs
            return 0.7 if results else 0.3
        
        # Calculate correctness percentage
        correct_count = 0
        total_count = min(len(results), len(expected_outputs))
        
        for i in range(total_count):
            if results[i] is not None:
                # Handle different types of expected outputs
                expected = expected_outputs[i]
                actual = results[i]
                
                # Try to convert expected output to appropriate type
                try:
                    if isinstance(expected, str):
                        if expected.isdigit():
                            expected = int(expected)
                        elif expected.replace('.', '').isdigit():
                            expected = float(expected)
                        elif expected.lower() in ['true', 'false']:
                            expected = expected.lower() == 'true'
                    
                    if actual == expected:
                        correct_count += 1
                except:
                    # String comparison fallback
                    if str(actual) == str(expected):
                        correct_count += 1
        
        correctness_ratio = correct_count / total_count if total_count > 0 else 0
        return correctness_ratio
    
    def _evaluate_code_quality(self, code: str) -> float:
        """Evaluate code quality and style."""
        quality_score = 0.0
        
        # Basic structure checks
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        # Function definition present
        if any(line.startswith('def ') for line in lines):
            quality_score += 0.3
        
        # Proper indentation (basic check)
        indented_lines = [line for line in code.split('\n') if line.startswith('    ')]
        if indented_lines:
            quality_score += 0.2
        
        # Variable naming (basic check for meaningful names)
        if any(len(word) > 2 for line in lines for word in re.findall(r'\b[a-z_][a-z0-9_]*\b', line)):
            quality_score += 0.2
        
        # Comments or docstrings
        if '#' in code or '"""' in code or "'''" in code:
            quality_score += 0.1
        
        # Return statement present
        if 'return' in code:
            quality_score += 0.2
        
        return min(quality_score, 1.0)
    
    def _evaluate_completeness(self, code: str, problem: Dict[str, Any]) -> float:
        """Evaluate if solution addresses the complete problem."""
        completeness_score = 0.0
        
        # Function definition
        function_matches = re.findall(r'def\s+(\w+)\s*\(', code)
        if function_matches:
            completeness_score += 0.4
        
        # Problem-specific requirements
        problem_desc = problem.get('question', '').lower()
        
        # Check for key operations mentioned in problem
        key_operations = []
        if 'add' in problem_desc or 'sum' in problem_desc:
            key_operations.append('+')
        if 'subtract' in problem_desc:
            key_operations.append('-')
        if 'multiply' in problem_desc:
            key_operations.append('*')
        if 'sort' in problem_desc:
            key_operations.extend(['sort', 'sorted'])
        if 'find' in problem_desc or 'search' in problem_desc:
            key_operations.extend(['find', 'in', 'index'])
        
        if key_operations:
            operations_found = sum(1 for op in key_operations if op in code)
            completeness_score += 0.3 * (operations_found / len(key_operations))
        else:
            completeness_score += 0.3  # Default if no specific operations identified
        
        # Input handling
        if '(' in code and ')' in code:  # Basic parameter handling
            completeness_score += 0.15
        
        # Output generation
        if 'return' in code:
            completeness_score += 0.15
        
        return min(completeness_score, 1.0)
    
    def _evaluate_efficiency(self, code: str) -> float:
        """Evaluate efficiency considerations in the code."""
        efficiency_score = 0.5  # Default moderate score
        
        # Bonus for efficient patterns
        if 'for' in code and 'range' in code:
            efficiency_score += 0.2  # Structured iteration
        
        # Penalty for obvious inefficiencies
        if code.count('for') > 3:  # Multiple nested loops indicator
            efficiency_score -= 0.2
        
        # Bonus for built-in functions usage
        efficient_functions = ['sum', 'max', 'min', 'sorted', 'len', 'any', 'all']
        efficient_count = sum(1 for func in efficient_functions if func in code)
        efficiency_score += min(efficient_count * 0.1, 0.3)
        
        return max(0.0, min(efficiency_score, 1.0))
    
    def _evaluate_edge_case_handling(self, code: str, test_cases: List[Any]) -> float:
        """Evaluate how well the code handles edge cases."""
        edge_case_score = 0.5  # Default moderate score
        
        # Check for explicit edge case handling
        edge_case_patterns = ['if', 'elif', 'else', 'try', 'except']
        pattern_count = sum(1 for pattern in edge_case_patterns if pattern in code)
        edge_case_score += min(pattern_count * 0.1, 0.3)
        
        # Check for empty/null handling
        null_checks = ['is None', '== None', 'not ', 'len(', 'empty']
        null_check_count = sum(1 for check in null_checks if check in code)
        edge_case_score += min(null_check_count * 0.05, 0.2)
        
        return min(edge_case_score, 1.0)
    
    def _compute_solution_bonus(self, code: str, components: Dict[str, float]) -> float:
        """Compute bonus for exceptional solutions."""
        bonus = 0.0
        
        # Bonus for high correctness
        if components.get('correctness', 0) >= 0.9:
            bonus += 0.1
        
        # Bonus for clean, readable code
        if components.get('quality', 0) >= 0.8:
            bonus += 0.05
        
        # Bonus for concise solutions
        line_count = len([line for line in code.split('\n') if line.strip()])
        if 3 <= line_count <= 10:  # Sweet spot for conciseness
            bonus += 0.05
        
        return bonus
    
    def _compute_solution_penalty(self, code: str) -> float:
        """Compute penalties for problematic code."""
        penalty = 0.0
        
        # Syntax error penalty
        try:
            ast.parse(code)
        except SyntaxError:
            penalty += 0.5
        
        # Incomplete code penalty
        if 'TODO' in code or 'FIXME' in code or '...' in code:
            penalty += 0.2
        
        # No function definition penalty
        if 'def ' not in code:
            penalty += 0.3
        
        # No return statement penalty
        if 'return' not in code and 'def ' in code:
            penalty += 0.2
        
        return penalty


class VerificationGeneratorReward:
    """Reward function for Verification Generator (Player 3)."""
    
    def __init__(self):
        self.execution_env = CodeExecutionEnvironment()
    
    def compute_reward(self, output: str, context: Dict[str, Any]) -> RewardMetrics:
        """
        Compute reward for test case generation.
        
        Evaluates:
        - Test case coverage and diversity
        - Edge case identification
        - Test case validity and executability
        - Discrimination power (ability to catch bugs)
        """
        components = {}
        
        problem = context.get('problem', {})
        solutions = context.get('solutions', [])
        
        # Parse generated test cases
        test_cases = self._parse_test_cases(output)
        
        # Coverage and diversity
        coverage_reward = self._evaluate_test_coverage(test_cases, problem)
        components['coverage'] = coverage_reward
        
        # Edge case identification
        edge_case_reward = self._evaluate_edge_case_coverage(test_cases, problem)
        components['edge_cases'] = edge_case_reward
        
        # Test case validity
        validity_reward = self._evaluate_test_validity(test_cases, problem)
        components['validity'] = validity_reward
        
        # Discrimination power
        discrimination_reward = self._evaluate_discrimination_power(test_cases, solutions, problem)
        components['discrimination'] = discrimination_reward
        
        # Compute base reward
        weights = {
            'coverage': 0.3,
            'edge_cases': 0.3,
            'validity': 0.2,
            'discrimination': 0.2
        }
        
        base_reward = sum(components[key] * weights[key] for key in weights)
        
        # Bonus for exceptional test suites
        bonus_reward = self._compute_verification_bonus(test_cases, components)
        components['bonus'] = bonus_reward
        
        # Penalty for invalid or poor test cases
        penalty = self._compute_verification_penalty(test_cases, output)
        components['penalty'] = -penalty
        
        final_reward = base_reward + bonus_reward - penalty
        
        return RewardMetrics(
            base_reward=base_reward,
            bonus_reward=bonus_reward,
            penalty=penalty,
            final_reward=final_reward,
            components=components,
            metadata={'test_cases': test_cases}
        )
    
    def _parse_test_cases(self, output: str) -> List[Dict[str, Any]]:
        """Parse test cases from verification generator output."""
        test_cases = []
        
        lines = output.split('\n')
        current_test = {}
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for test case patterns
            if 'assert' in line.lower():
                # Extract assertion
                test_cases.append({
                    'type': 'assertion',
                    'content': line,
                    'inputs': self._extract_inputs_from_assertion(line),
                    'expected': self._extract_expected_from_assertion(line)
                })
            elif 'test' in line.lower() and '(' in line:
                # Function call pattern
                test_cases.append({
                    'type': 'function_call',
                    'content': line,
                    'inputs': self._extract_inputs_from_call(line),
                    'expected': None
                })
            elif '->' in line or '=' in line:
                # Input/output pattern
                parts = line.split('->' if '->' in line else '=')
                if len(parts) == 2:
                    test_cases.append({
                        'type': 'input_output',
                        'content': line,
                        'inputs': self._parse_inputs(parts[0].strip()),
                        'expected': self._parse_expected(parts[1].strip())
                    })
        
        return test_cases
    
    def _extract_inputs_from_assertion(self, assertion: str) -> List[Any]:
        """Extract inputs from assertion statement."""
        # Simple pattern matching for function calls in assertions
        match = re.search(r'\w+\((.*?)\)', assertion)
        if match:
            args_str = match.group(1)
            try:
                # Try to evaluate as Python literals
                return [ast.literal_eval(arg.strip()) for arg in args_str.split(',') if arg.strip()]
            except:
                return [args_str]
        return []
    
    def _extract_expected_from_assertion(self, assertion: str) -> Any:
        """Extract expected value from assertion."""
        # Look for == pattern
        if '==' in assertion:
            parts = assertion.split('==')
            if len(parts) == 2:
                try:
                    return ast.literal_eval(parts[1].strip())
                except:
                    return parts[1].strip()
        return None
    
    def _extract_inputs_from_call(self, call: str) -> List[Any]:
        """Extract inputs from function call."""
        match = re.search(r'\w+\((.*?)\)', call)
        if match:
            args_str = match.group(1)
            try:
                return [ast.literal_eval(arg.strip()) for arg in args_str.split(',') if arg.strip()]
            except:
                return [args_str]
        return []
    
    def _parse_inputs(self, input_str: str) -> List[Any]:
        """Parse input string into list of values."""
        try:
            # Try to evaluate as Python literal
            return [ast.literal_eval(input_str)]
        except:
            # Fall back to string parsing
            return [input_str.strip()]
    
    def _parse_expected(self, expected_str: str) -> Any:
        """Parse expected output string."""
        try:
            return ast.literal_eval(expected_str)
        except:
            return expected_str.strip()
    
    def _evaluate_test_coverage(self, test_cases: List[Dict[str, Any]], problem: Dict[str, Any]) -> float:
        """Evaluate test coverage and diversity."""
        if not test_cases:
            return 0.0
        
        coverage_score = 0.0
        
        # Number of test cases
        num_tests = len(test_cases)
        coverage_score += min(num_tests * 0.1, 0.4)  # Up to 0.4 for many tests
        
        # Input diversity
        unique_inputs = set()
        for test_case in test_cases:
            inputs = test_case.get('inputs', [])
            if inputs:
                # Create hashable representation
                input_repr = str(tuple(inputs) if isinstance(inputs, list) else inputs)
                unique_inputs.add(input_repr)
        
        diversity_ratio = len(unique_inputs) / max(num_tests, 1)
        coverage_score += diversity_ratio * 0.3
        
        # Type coverage (different input types)
        input_types = set()
        for test_case in test_cases:
            inputs = test_case.get('inputs', [])
            for inp in inputs:
                input_types.add(type(inp).__name__)
        
        type_diversity = min(len(input_types) * 0.1, 0.3)
        coverage_score += type_diversity
        
        return min(coverage_score, 1.0)
    
    def _evaluate_edge_case_coverage(self, test_cases: List[Dict[str, Any]], problem: Dict[str, Any]) -> float:
        """Evaluate edge case coverage."""
        if not test_cases:
            return 0.0
        
        edge_case_score = 0.0
        total_tests = len(test_cases)
        edge_case_count = 0
        
        for test_case in test_cases:
            inputs = test_case.get('inputs', [])
            
            for inp in inputs:
                # Check for common edge cases
                if self._is_edge_case_input(inp):
                    edge_case_count += 1
                    break  # Count each test case only once
        
        # Edge case ratio
        if total_tests > 0:
            edge_case_ratio = edge_case_count / total_tests
            edge_case_score = min(edge_case_ratio * 2.0, 1.0)  # Up to 50% edge cases is ideal
        
        return edge_case_score
    
    def _is_edge_case_input(self, inp: Any) -> bool:
        """Check if input represents an edge case."""
        if inp is None:
            return True
        
        if isinstance(inp, (int, float)):
            return inp == 0 or inp < 0 or abs(inp) > 1000
        
        if isinstance(inp, str):
            return len(inp) == 0 or len(inp) > 100
        
        if isinstance(inp, (list, tuple)):
            return len(inp) == 0 or len(inp) > 100
        
        return False
    
    def _evaluate_test_validity(self, test_cases: List[Dict[str, Any]], problem: Dict[str, Any]) -> float:
        """Evaluate validity of test cases."""
        if not test_cases:
            return 0.0
        
        valid_count = 0
        
        for test_case in test_cases:
            if self._is_valid_test_case(test_case, problem):
                valid_count += 1
        
        return valid_count / len(test_cases) if test_cases else 0
    
    def _is_valid_test_case(self, test_case: Dict[str, Any], problem: Dict[str, Any]) -> bool:
        """Check if a test case is valid."""
        # Basic validity checks
        content = test_case.get('content', '')
        
        # Must have some content
        if not content or len(content) < 3:
            return False
        
        # Should have inputs
        inputs = test_case.get('inputs', [])
        if not inputs:
            return False
        
        # Should be syntactically valid Python (basic check)
        try:
            if 'assert' in content:
                # For assertions, try to parse as Python statement
                ast.parse(content)
            return True
        except:
            return False
    
    def _evaluate_discrimination_power(self, test_cases: List[Dict[str, Any]], 
                                     solutions: List[str], problem: Dict[str, Any]) -> float:
        """Evaluate ability of test cases to discriminate between solutions."""
        if not test_cases or not solutions:
            return 0.5  # Default moderate score
        
        # This is a simplified version - in practice, you'd run test cases against solutions
        discrimination_score = 0.5
        
        # Bonus for having both positive and negative test cases
        has_positive = any('true' in str(tc).lower() or 'pass' in str(tc).lower() 
                          for tc in test_cases)
        has_negative = any('false' in str(tc).lower() or 'fail' in str(tc).lower() 
                          for tc in test_cases)
        
        if has_positive and has_negative:
            discrimination_score += 0.3
        elif has_positive or has_negative:
            discrimination_score += 0.1
        
        # Bonus for variety in test case types
        test_types = set(tc.get('type', '') for tc in test_cases)
        type_bonus = min(len(test_types) * 0.1, 0.2)
        discrimination_score += type_bonus
        
        return min(discrimination_score, 1.0)
    
    def _compute_verification_bonus(self, test_cases: List[Dict[str, Any]], 
                                   components: Dict[str, float]) -> float:
        """Compute bonus for exceptional test suites."""
        bonus = 0.0
        
        # Bonus for comprehensive coverage
        if components.get('coverage', 0) >= 0.8:
            bonus += 0.1
        
        # Bonus for good edge case coverage
        if components.get('edge_cases', 0) >= 0.6:
            bonus += 0.1
        
        # Bonus for large number of valid test cases
        if len(test_cases) >= 8:
            bonus += 0.05
        
        return bonus
    
    def _compute_verification_penalty(self, test_cases: List[Dict[str, Any]], output: str) -> float:
        """Compute penalties for poor test generation."""
        penalty = 0.0
        
        # Penalty for no test cases
        if not test_cases:
            penalty += 0.5
        
        # Penalty for invalid syntax
        invalid_count = sum(1 for tc in test_cases if not self._is_valid_test_case(tc, {}))
        if test_cases:
            penalty += (invalid_count / len(test_cases)) * 0.3
        
        # Penalty for very short output
        if len(output.strip()) < 20:
            penalty += 0.2
        
        return penalty


class MetaVerifierReward:
    """Reward function for Meta-Verifier (Player 4)."""
    
    def compute_reward(self, output: str, context: Dict[str, Any]) -> RewardMetrics:
        """
        Compute reward for meta-verification.
        
        Evaluates:
        - Verification accuracy
        - Analysis depth and quality
        - Consistency of judgments
        - Identification of test case issues
        """
        components = {}
        
        problem = context.get('problem', {})
        solutions = context.get('solutions', [])
        test_cases = context.get('test_cases', [])
        
        # Parse verification output
        verification_result = self._parse_verification_output(output)
        
        # Accuracy of verification judgments
        accuracy_reward = self._evaluate_verification_accuracy(verification_result, context)
        components['accuracy'] = accuracy_reward
        
        # Analysis depth and quality
        analysis_reward = self._evaluate_analysis_quality(verification_result, output)
        components['analysis'] = analysis_reward
        
        # Consistency of judgments
        consistency_reward = self._evaluate_consistency(verification_result)
        components['consistency'] = consistency_reward
        
        # Issue identification capability  
        issue_identification_reward = self._evaluate_issue_identification(verification_result, context)
        components['issue_identification'] = issue_identification_reward
        
        # Compute base reward
        weights = {
            'accuracy': 0.4,
            'analysis': 0.2,
            'consistency': 0.2,
            'issue_identification': 0.2
        }
        
        base_reward = sum(components[key] * weights[key] for key in weights)
        
        # Bonus for exceptional verification
        bonus_reward = self._compute_metaverifier_bonus(verification_result, components)
        components['bonus'] = bonus_reward
        
        # Penalty for poor verification
        penalty = self._compute_metaverifier_penalty(verification_result, output)
        components['penalty'] = -penalty
        
        final_reward = base_reward + bonus_reward - penalty
        
        return RewardMetrics(
            base_reward=base_reward,
            bonus_reward=bonus_reward,
            penalty=penalty,
            final_reward=final_reward,
            components=components,
            metadata={'verification_result': verification_result}
        )
    
    def _parse_verification_output(self, output: str) -> Dict[str, Any]:
        """Parse meta-verifier output into structured format."""
        result = {
            'overall_validity': None,
            'individual_assessments': [],
            'issues_identified': [],
            'confidence': 0.5,
            'reasoning': ''
        }
        
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip().lower()
            if not line:
                continue
            
            # Overall validity assessment
            if 'valid' in line and 'invalid' not in line:
                result['overall_validity'] = True
            elif 'invalid' in line:
                result['overall_validity'] = False
            
            # Confidence indicators
            if 'confident' in line or 'certain' in line:
                result['confidence'] = 0.9
            elif 'uncertain' in line or 'unsure' in line:
                result['confidence'] = 0.3
            
            # Issue identification
            issue_keywords = ['error', 'problem', 'issue', 'incorrect', 'wrong', 'bug']
            if any(keyword in line for keyword in issue_keywords):
                result['issues_identified'].append(line)
        
        result['reasoning'] = output  # Keep full output as reasoning
        
        return result
    
    def _evaluate_verification_accuracy(self, verification_result: Dict[str, Any], 
                                      context: Dict[str, Any]) -> float:
        """Evaluate accuracy of verification judgments."""
        # This is simplified - in practice, you'd have ground truth verification
        
        # If we have known correct/incorrect test cases, compare against them
        ground_truth = context.get('ground_truth_validity')
        if ground_truth is not None:
            predicted = verification_result.get('overall_validity')
            if predicted == ground_truth:
                return 1.0
            else:
                return 0.0
        
        # Otherwise, use heuristics based on consistency and reasoning quality
        accuracy_score = 0.5  # Default moderate accuracy
        
        # Bonus for high confidence with detailed reasoning
        confidence = verification_result.get('confidence', 0.5)
        reasoning_length = len(verification_result.get('reasoning', ''))
        
        if confidence > 0.7 and reasoning_length > 50:
            accuracy_score += 0.3
        elif confidence > 0.5 and reasoning_length > 20:
            accuracy_score += 0.1
        
        return min(accuracy_score, 1.0)
    
    def _evaluate_analysis_quality(self, verification_result: Dict[str, Any], output: str) -> float:
        """Evaluate quality and depth of analysis."""
        analysis_score = 0.0
        
        # Length and detail of reasoning
        reasoning = verification_result.get('reasoning', '')
        word_count = len(reasoning.split())
        
        if word_count >= 50:
            analysis_score += 0.3
        elif word_count >= 20:
            analysis_score += 0.2
        elif word_count >= 10:
            analysis_score += 0.1
        
        # Specific analysis elements
        analysis_keywords = [
            'because', 'therefore', 'however', 'although', 'specifically',
            'check', 'verify', 'test', 'input', 'output', 'expected', 'actual'
        ]
        
        keyword_count = sum(1 for keyword in analysis_keywords if keyword in reasoning.lower())
        analysis_score += min(keyword_count * 0.05, 0.3)
        
        # Structure and organization
        if any(marker in output for marker in ['1.', '2.', '-', '*']):
            analysis_score += 0.2  # Bonus for structured analysis
        
        # Issue identification specificity
        issues = verification_result.get('issues_identified', [])
        if issues:
            analysis_score += min(len(issues) * 0.1, 0.2)
        
        return min(analysis_score, 1.0)
    
    def _evaluate_consistency(self, verification_result: Dict[str, Any]) -> float:
        """Evaluate consistency of verification judgments."""
        consistency_score = 0.5  # Default moderate consistency
        
        # Check consistency between overall validity and identified issues
        overall_valid = verification_result.get('overall_validity')
        issues = verification_result.get('issues_identified', [])
        
        if overall_valid is True and len(issues) == 0:
            consistency_score = 1.0  # Consistent: valid with no issues
        elif overall_valid is False and len(issues) > 0:
            consistency_score = 1.0  # Consistent: invalid with identified issues
        elif overall_valid is True and len(issues) > 0:
            consistency_score = 0.3  # Inconsistent: valid but has issues
        elif overall_valid is False and len(issues) == 0:
            consistency_score = 0.3  # Inconsistent: invalid but no specific issues
        
        # Confidence consistency
        confidence = verification_result.get('confidence', 0.5)
        reasoning_length = len(verification_result.get('reasoning', ''))
        
        # High confidence should come with detailed reasoning
        if confidence > 0.8 and reasoning_length < 30:
            consistency_score *= 0.7  # Penalty for overconfidence without reasoning
        elif confidence < 0.3 and reasoning_length > 100:
            consistency_score *= 0.8  # Slight penalty for underconfidence with detailed reasoning
        
        return consistency_score
    
    def _evaluate_issue_identification(self, verification_result: Dict[str, Any], 
                                     context: Dict[str, Any]) -> float:
        """Evaluate ability to identify specific issues."""
        identification_score = 0.5  # Default moderate score
        
        issues = verification_result.get('issues_identified', [])
        
        # Bonus for identifying issues when test cases are actually problematic
        test_cases = context.get('test_cases', [])
        if test_cases:
            # Heuristic: if there are very few or very many test cases, that's potentially an issue
            num_tests = len(test_cases)
            if (num_tests < 2 or num_tests > 20) and issues:
                identification_score += 0.3
            elif 2 <= num_tests <= 8 and not issues:
                identification_score += 0.2  # Good number of tests, correctly no issues identified
        
        # Bonus for specific and detailed issue descriptions
        specific_issues = [issue for issue in issues if len(issue.split()) > 3]
        identification_score += min(len(specific_issues) * 0.1, 0.3)
        
        return min(identification_score, 1.0)
    
    def _compute_metaverifier_bonus(self, verification_result: Dict[str, Any], 
                                   components: Dict[str, float]) -> float:
        """Compute bonus for exceptional meta-verification."""
        bonus = 0.0
        
        # Bonus for high accuracy
        if components.get('accuracy', 0) >= 0.9:
            bonus += 0.1
        
        # Bonus for detailed analysis
        if components.get('analysis', 0) >= 0.8:
            bonus += 0.05
        
        # Bonus for high consistency
        if components.get('consistency', 0) >= 0.9:
            bonus += 0.05
        
        # Bonus for high confidence with good reasoning
        confidence = verification_result.get('confidence', 0.5)
        reasoning_length = len(verification_result.get('reasoning', ''))
        if confidence > 0.8 and reasoning_length > 100:
            bonus += 0.1
        
        return bonus
    
    def _compute_metaverifier_penalty(self, verification_result: Dict[str, Any], output: str) -> float:
        """Compute penalties for poor meta-verification."""
        penalty = 0.0
        
        # Penalty for very short output
        if len(output.strip()) < 10:
            penalty += 0.4
        
        # Penalty for no clear validity judgment
        if verification_result.get('overall_validity') is None:
            penalty += 0.3
        
        # Penalty for inconsistent judgments
        if components.get('consistency', 1.0) < 0.5:
            penalty += 0.2
        
        # Penalty for no reasoning
        reasoning = verification_result.get('reasoning', '')
        if len(reasoning.split()) < 5:
            penalty += 0.2
        
        return penalty


class RoleConditionedRewardSystem:
    """Main reward system that coordinates all role-specific reward functions."""
    
    def __init__(self):
        self.reward_functions = {
            'problem_generator': ProblemGeneratorReward(),
            'solution_generator': SolutionGeneratorReward(),
            'verification_generator': VerificationGeneratorReward(),
            'meta_verifier': MetaVerifierReward()
        }
        
        logger.info("Role-conditioned reward system initialized with 4 player types")
    
    def compute_reward(self, output: str, role: str, context: Dict[str, Any]) -> RewardMetrics:
        """
        Compute role-conditioned reward for a player's output.
        
        Args:
            output: Generated output from the player
            role: Player role identifier
            context: Additional context for reward computation
            
        Returns:
            RewardMetrics object with detailed reward breakdown
        """
        if role not in self.reward_functions:
            logger.warning(f"Unknown role {role}, using default reward")
            return RewardMetrics(
                base_reward=0.5,
                bonus_reward=0.0,
                penalty=0.0,
                final_reward=0.5,
                components={'default': 0.5},
                metadata={}
            )
        
        reward_function = self.reward_functions[role]
        metrics = reward_function.compute_reward(output, context)
        
        logger.debug(f"Computed reward for {role}: {metrics.final_reward:.3f} "
                    f"(base: {metrics.base_reward:.3f}, bonus: {metrics.bonus_reward:.3f}, "
                    f"penalty: {metrics.penalty:.3f})")
        
        return metrics
    
    def get_reward_function(self, role: str) -> callable:
        """Get reward function for a specific role."""
        if role in self.reward_functions:
            return lambda output, context: self.compute_reward(output, role, context)
        else:
            return lambda output, context: RewardMetrics(0.5, 0.0, 0.0, 0.5, {'default': 0.5}, {})