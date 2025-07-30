"""
Verification Generator - Player 3 of SRAG-V architecture.
Generates test cases designed to differentiate correct/incorrect solutions.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
import logging
import ast
import random

from .base_player import BasePlayer

logger = logging.getLogger(__name__)


class VerificationGenerator(BasePlayer):
    """
    Player 3: Verification Generator (1.5B parameters)
    Model: Qwen2.5-1.5B-Instruct
    
    Input:
    - Problem specification
    - All 16 solution attempts
    - Execution traces from each solution
    
    Output: Test cases designed to differentiate correct/incorrect solutions
    
    Architecture Details:
    - Trained to maximize bug detection rate
    - Generates 8 test cases per problem
    - Uses coverage-guided generation (inspired by fuzzing literature)
    """
    
    def __init__(self, config: Dict):
        super().__init__(
            model_name=config["model_name"],
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            lora_config=config.get("lora_config")
        )
        
        # Test case categories for comprehensive coverage
        self.test_categories = [
            "basic_functionality",
            "edge_cases",
            "boundary_values",
            "error_conditions",
            "performance_tests",
            "corner_cases",
            "type_variations",
            "large_inputs"
        ]
    
    def process_input(
        self,
        problem: Dict,
        solutions: List[Dict],
        execution_traces: Optional[List] = None
    ) -> str:
        """Process input and create a prompt for test case generation."""
        
        problem_desc = problem.get("description", "")
        function_sig = problem.get("function_signature", "")
        examples = problem.get("examples", "")
        
        prompt = f"""You are an expert test case generator. Your job is to create comprehensive test cases that will help distinguish between correct and incorrect implementations of a coding problem.

PROBLEM DESCRIPTION:
{problem_desc}

{f'FUNCTION SIGNATURE: {function_sig}' if function_sig else ''}

{f'EXAMPLES: {examples}' if examples else ''}

SOLUTION IMPLEMENTATIONS TO TEST:
"""
        
        # Include a sample of solutions for context (up to 5)
        for i, solution in enumerate(solutions[:5], 1):
            code = solution.get("code", "")[:300] + "..." if len(solution.get("code", "")) > 300 else solution.get("code", "")
            prompt += f"\nSolution {i}:\n```python\n{code}\n```\n"
        
        if len(solutions) > 5:
            prompt += f"\n... and {len(solutions) - 5} more solutions to test.\n"
        
        prompt += f"""

GENERATE 8 COMPREHENSIVE TEST CASES:

Your test cases should:
1. Cover basic functionality (normal cases)
2. Test edge cases and boundary conditions
3. Include error conditions and invalid inputs
4. Test performance with larger inputs
5. Verify corner cases that might break incorrect solutions
6. Test different input types and formats

For each test case, provide:
- Test name/description
- Input values
- Expected output
- Test category (basic/edge/boundary/error/performance/corner/type/large)

Format each test case as:
TEST_CASE_N:
Name: [descriptive name]
Category: [category]
Input: [input values]
Expected: [expected output]
Reasoning: [why this test is important]

Generate 8 diverse test cases now:"""

        return prompt
    
    def parse_output(self, output: str) -> List[Dict]:
        """Parse model output to extract test cases."""
        test_cases = []
        
        try:
            # Split output into individual test cases
            test_sections = re.split(r'TEST_CASE_\d+:', output, flags=re.IGNORECASE)
            
            for i, section in enumerate(test_sections[1:], 1):  # Skip first empty section
                try:
                    # Extract components
                    name_match = re.search(r'Name:\s*(.+)', section, re.IGNORECASE)
                    category_match = re.search(r'Category:\s*(\w+)', section, re.IGNORECASE)
                    input_match = re.search(r'Input:\s*(.+)', section, re.IGNORECASE)
                    expected_match = re.search(r'Expected:\s*(.+)', section, re.IGNORECASE)
                    reasoning_match = re.search(r'Reasoning:\s*(.+)', section, re.IGNORECASE | re.DOTALL)
                    
                    test_case = {
                        "test_id": f"test_{i:02d}",
                        "name": name_match.group(1).strip() if name_match else f"Test Case {i}",
                        "category": category_match.group(1).strip().lower() if category_match else "basic",
                        "input": self.parse_test_input(input_match.group(1).strip() if input_match else ""),
                        "expected": self.parse_expected_output(expected_match.group(1).strip() if expected_match else ""),
                        "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
                        "raw_section": section.strip()
                    }
                    
                    test_cases.append(test_case)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse test case {i}: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"Failed to parse test cases: {e}")
            # Fallback: create simple test cases
            test_cases = self.create_fallback_tests()
        
        logger.info(f"Parsed {len(test_cases)} test cases")
        return test_cases
    
    def parse_test_input(self, input_str: str) -> Any:
        """Parse test input string into appropriate Python objects."""
        try:
            # Remove common prefixes
            input_str = input_str.replace("Input:", "").strip()
            
            # Try to evaluate as Python literal
            try:
                return ast.literal_eval(input_str)
            except (ValueError, SyntaxError):
                pass
            
            # Handle common patterns
            if input_str.startswith('[') and input_str.endswith(']'):
                # List format
                try:
                    return json.loads(input_str)
                except json.JSONDecodeError:
                    return input_str
            
            elif input_str.startswith('"') and input_str.endswith('"'):
                # String format
                return input_str[1:-1]
            
            elif input_str.startswith("'") and input_str.endswith("'"):
                # String format
                return input_str[1:-1]
            
            elif input_str.isdigit() or (input_str.startswith('-') and input_str[1:].isdigit()):
                # Integer
                return int(input_str)
            
            elif '.' in input_str:
                # Float
                try:
                    return float(input_str)
                except ValueError:
                    pass
            
            # Return as string if can't parse
            return input_str
            
        except Exception as e:
            logger.warning(f"Failed to parse input '{input_str}': {e}")
            return input_str
    
    def parse_expected_output(self, expected_str: str) -> Any:
        """Parse expected output string into appropriate Python objects."""
        try:
            # Remove common prefixes
            expected_str = expected_str.replace("Expected:", "").replace("Output:", "").strip()
            
            # Try to evaluate as Python literal
            try:
                return ast.literal_eval(expected_str)
            except (ValueError, SyntaxError):
                pass
            
            # Handle common patterns similar to input parsing
            if expected_str.startswith('[') and expected_str.endswith(']'):
                try:
                    return json.loads(expected_str)
                except json.JSONDecodeError:
                    return expected_str
            
            elif expected_str.startswith('"') and expected_str.endswith('"'):
                return expected_str[1:-1]
            
            elif expected_str.startswith("'") and expected_str.endswith("'"):
                return expected_str[1:-1]
            
            elif expected_str.isdigit() or (expected_str.startswith('-') and expected_str[1:].isdigit()):
                return int(expected_str)
            
            elif '.' in expected_str:
                try:
                    return float(expected_str)
                except ValueError:
                    pass
            
            # Special cases
            if expected_str.lower() in ['true', 'false']:
                return expected_str.lower() == 'true'
            
            if expected_str.lower() in ['none', 'null']:
                return None
            
            return expected_str
            
        except Exception as e:
            logger.warning(f"Failed to parse expected output '{expected_str}': {e}")
            return expected_str
    
    def create_fallback_tests(self) -> List[Dict]:
        """Create basic fallback test cases if parsing fails."""
        return [
            {
                "test_id": "test_01",
                "name": "Basic Functionality Test",
                "category": "basic",
                "input": [1, 2],
                "expected": 3,
                "reasoning": "Test basic functionality with simple inputs"
            },
            {
                "test_id": "test_02",
                "name": "Edge Case - Empty Input",
                "category": "edge",
                "input": [],
                "expected": None,
                "reasoning": "Test behavior with empty input"
            }
        ]
    
    def generate(
        self,
        problem: Dict,
        solutions: List[Dict],
        execution_traces: Optional[List] = None,
        num_test_cases: int = 8
    ) -> List[Dict]:
        """Generate test cases for the given problem and solutions."""
        logger.info(f"Generating {num_test_cases} test cases for problem {problem.get('problem_id', 'unknown')}")
        
        # Create prompt
        prompt = self.process_input(
            problem=problem,
            solutions=solutions,
            execution_traces=execution_traces
        )
        
        # Generate response
        outputs = self.generate_text(
            prompt=prompt,
            max_new_tokens=2048,
            temperature=self.temperature,
            top_p=self.top_p,
            num_return_sequences=1
        )
        
        if not outputs:
            logger.warning("Failed to generate test cases, using fallback")
            return self.create_fallback_tests()
        
        # Parse test cases
        test_cases = self.parse_output(outputs[0])
        
        # Ensure we have the right number of test cases
        if len(test_cases) < num_test_cases:
            logger.warning(f"Generated only {len(test_cases)} test cases, expected {num_test_cases}")
            # Add more basic test cases if needed
            while len(test_cases) < num_test_cases:
                test_cases.append({
                    "test_id": f"test_{len(test_cases)+1:02d}",
                    "name": f"Additional Test {len(test_cases)+1}",
                    "category": "basic",
                    "input": None,
                    "expected": None,
                    "reasoning": "Additional test case to meet requirements"
                })
        
        elif len(test_cases) > num_test_cases:
            # Take the first num_test_cases
            test_cases = test_cases[:num_test_cases]
        
        # Add metadata
        for i, test_case in enumerate(test_cases):
            test_case["generation_order"] = i + 1
            test_case["problem_id"] = problem.get("problem_id", "unknown")
        
        logger.info(f"Successfully generated {len(test_cases)} test cases")
        return test_cases
    
    def analyze_solution_differences(self, solutions: List[Dict]) -> Dict:
        """Analyze differences between solutions to guide test generation."""
        analysis = {
            "syntax_errors": 0,
            "different_approaches": [],
            "common_patterns": [],
            "complexity_levels": []
        }
        
        try:
            for solution in solutions:
                code = solution.get("code", "")
                
                # Check syntax
                try:
                    ast.parse(code)
                except SyntaxError:
                    analysis["syntax_errors"] += 1
                
                # Simple approach detection (could be improved)
                if "for " in code:
                    analysis["different_approaches"].append("iterative")
                if "recursion" in code.lower() or "def " in code and code.count("def ") > 1:
                    analysis["different_approaches"].append("recursive")
                if "sorted(" in code or ".sort(" in code:
                    analysis["different_approaches"].append("sorting")
                
                # Complexity estimation
                lines = len([line for line in code.split('\n') if line.strip()])
                if lines < 10:
                    analysis["complexity_levels"].append("simple")
                elif lines < 25:
                    analysis["complexity_levels"].append("medium")
                else:
                    analysis["complexity_levels"].append("complex")
        
        except Exception as e:
            logger.warning(f"Error analyzing solutions: {e}")
        
        return analysis
    
    def compute_coverage_score(self, test_cases: List[Dict]) -> float:
        """Compute coverage score for generated test cases."""
        try:
            categories_covered = set(tc.get("category", "basic") for tc in test_cases)
            total_categories = len(self.test_categories)
            coverage = len(categories_covered) / total_categories
            
            # Bonus for having multiple tests per important category
            important_categories = ["basic", "edge", "boundary"]
            bonus = 0
            for category in important_categories:
                category_count = sum(1 for tc in test_cases if tc.get("category") == category)
                if category_count >= 2:
                    bonus += 0.1
            
            return min(coverage + bonus, 1.0)
            
        except Exception as e:
            logger.warning(f"Error computing coverage score: {e}")
            return 0.5  # Default score
    
    def enhance_test_cases_with_mutation(self, base_test_cases: List[Dict]) -> List[Dict]:
        """Enhance test cases through mutation for better bug detection."""
        enhanced_tests = base_test_cases.copy()
        
        try:
            for test_case in base_test_cases[:3]:  # Mutate first 3 test cases
                original_input = test_case.get("input")
                
                if isinstance(original_input, int):
                    # Create boundary mutations for integers
                    mutations = [
                        original_input - 1,
                        original_input + 1,
                        0,
                        -1,
                        abs(original_input)
                    ]
                    
                    for i, mutated_input in enumerate(mutations[:2]):  # Limit mutations
                        enhanced_tests.append({
                            "test_id": f"mut_{test_case['test_id']}_{i+1}",
                            "name": f"Mutation of {test_case['name']}",
                            "category": "mutation",
                            "input": mutated_input,
                            "expected": None,  # Would need to compute
                            "reasoning": f"Boundary mutation of {original_input}",
                            "parent_test": test_case["test_id"]
                        })
                
                elif isinstance(original_input, list) and original_input:
                    # Create list mutations
                    mutations = [
                        [],  # Empty list
                        [original_input[0]],  # Single element
                        original_input + original_input,  # Doubled
                    ]
                    
                    for i, mutated_input in enumerate(mutations[:2]):
                        enhanced_tests.append({
                            "test_id": f"mut_{test_case['test_id']}_{i+1}",
                            "name": f"List Mutation of {test_case['name']}",
                            "category": "mutation",
                            "input": mutated_input,
                            "expected": None,
                            "reasoning": f"List structure mutation",
                            "parent_test": test_case["test_id"]
                        })
        
        except Exception as e:
            logger.warning(f"Error enhancing test cases: {e}")
        
        return enhanced_tests