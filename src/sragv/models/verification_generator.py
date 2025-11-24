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
import time

from .base_player import BasePlayer

logger = logging.getLogger(__name__)


class TestQualityPredictor:
    """
    STAR Phase 2 Enhancement: Test Quality Prediction
    
    Predicts the quality and reliability of generated test cases
    to support strategic problem selection in the oracle.
    """
    
    def __init__(self):
        self.quality_factors = {
            'coverage_completeness': 0.25,
            'edge_case_detection': 0.25, 
            'discriminative_power': 0.20,
            'execution_reliability': 0.15,
            'input_diversity': 0.15
        }
    
    def predict_test_quality(
        self, 
        test_cases: List[Dict], 
        problem: Dict[str, Any],
        solutions: List[Dict] = None
    ) -> Dict[str, float]:
        """
        Predict overall quality score for generated test cases.
        Higher scores indicate better test suites for strategic learning.
        """
        try:
            if not test_cases:
                return {'overall_quality': 0.0, 'confidence': 0.0}
            
            # Compute quality dimensions
            coverage_score = self._assess_coverage_completeness(test_cases, problem)
            edge_case_score = self._assess_edge_case_detection(test_cases, problem)
            discriminative_score = self._assess_discriminative_power(test_cases, solutions)
            reliability_score = self._assess_execution_reliability(test_cases)
            diversity_score = self._assess_input_diversity(test_cases)
            
            # Weighted combination
            overall_quality = (
                self.quality_factors['coverage_completeness'] * coverage_score +
                self.quality_factors['edge_case_detection'] * edge_case_score +
                self.quality_factors['discriminative_power'] * discriminative_score +
                self.quality_factors['execution_reliability'] * reliability_score +
                self.quality_factors['input_diversity'] * diversity_score
            )
            
            # Confidence based on number of test cases and consistency
            confidence = min(1.0, len(test_cases) / 8.0 * 0.8 + 0.2)
            
            return {
                'overall_quality': overall_quality,
                'confidence': confidence,
                'coverage_completeness': coverage_score,
                'edge_case_detection': edge_case_score,
                'discriminative_power': discriminative_score,
                'execution_reliability': reliability_score,
                'input_diversity': diversity_score
            }
            
        except Exception as e:
            logger.warning(f"Error predicting test quality: {e}")
            return {'overall_quality': 0.5, 'confidence': 0.0}
    
    def _assess_coverage_completeness(self, test_cases: List[Dict], problem: Dict) -> float:
        """Assess how well test cases cover the problem requirements."""
        
        problem_text = problem.get('question', '').lower()
        
        # Check for basic coverage categories
        coverage_categories = {
            'normal_cases': False,
            'empty_input': False,
            'single_element': False,
            'multiple_elements': False,
            'boundary_values': False
        }
        
        for test_case in test_cases:
            test_input = test_case.get('input', '')
            test_str = str(test_input).lower()
            
            # Detect coverage patterns
            if 'empty' in test_str or '[]' in test_str or len(str(test_input)) == 0:
                coverage_categories['empty_input'] = True
            elif isinstance(test_input, (list, str)) and len(test_input) == 1:
                coverage_categories['single_element'] = True
            elif isinstance(test_input, (list, str)) and len(test_input) > 1:
                coverage_categories['multiple_elements'] = True
            
            # Check for boundary indicators
            boundary_indicators = ['max', 'min', '999', '1000', 'large', 'limit']
            if any(indicator in test_str for indicator in boundary_indicators):
                coverage_categories['boundary_values'] = True
                
            # Normal case if none of the above
            if not any([coverage_categories['empty_input'], 
                       coverage_categories['single_element'],
                       coverage_categories['boundary_values']]):
                coverage_categories['normal_cases'] = True
        
        coverage_score = sum(coverage_categories.values()) / len(coverage_categories)
        return coverage_score
    
    def _assess_edge_case_detection(self, test_cases: List[Dict], problem: Dict) -> float:
        """Assess how well test cases target edge cases."""
        
        edge_case_indicators = [
            'empty', 'null', 'zero', 'negative', 'maximum', 'minimum',
            'boundary', 'limit', 'edge', 'corner', 'special'
        ]
        
        edge_case_count = 0
        for test_case in test_cases:
            test_str = str(test_case).lower()
            if any(indicator in test_str for indicator in edge_case_indicators):
                edge_case_count += 1
        
        # Score based on proportion of edge cases
        edge_case_ratio = edge_case_count / len(test_cases) if test_cases else 0
        
        # Optimal ratio is around 30-50% edge cases
        if 0.3 <= edge_case_ratio <= 0.5:
            return 1.0
        elif 0.2 <= edge_case_ratio <= 0.6:
            return 0.8
        elif edge_case_ratio > 0:
            return 0.6
        else:
            return 0.2
    
    def _assess_discriminative_power(self, test_cases: List[Dict], solutions: List[Dict]) -> float:
        """
        Enhanced discriminative power assessment using solution analysis.
        STAR Phase 2: More sophisticated ML-based approach.
        """
        
        if not solutions or len(solutions) < 2:
            return 0.5  # Can't assess without multiple solutions
        
        # 1. Input diversity analysis (enhanced)
        input_types = set()
        input_sizes = set()
        input_complexities = []
        
        for test_case in test_cases:
            test_input = test_case.get('input')
            input_types.add(type(test_input).__name__)
            
            if hasattr(test_input, '__len__'):
                input_sizes.add(len(test_input))
                # Complexity based on nested structure
                complexity = self._compute_input_complexity(test_input)
                input_complexities.append(complexity)
            else:
                input_sizes.add(1)
                input_complexities.append(0.1)  # Simple input
        
        # 2. Solution diversity analysis
        solution_lengths = [len(s.get('code', '')) for s in solutions]
        solution_approaches = set()
        
        for solution in solutions:
            code = solution.get('code', '')
            # Identify algorithmic approach
            if any(keyword in code.lower() for keyword in ['sort', 'sorted']):
                solution_approaches.add('sorting')
            if any(keyword in code.lower() for keyword in ['dict', 'hash', 'map']):
                solution_approaches.add('hashing')
            if any(keyword in code.lower() for keyword in ['for', 'while']):
                solution_approaches.add('iterative')
            if any(keyword in code.lower() for keyword in ['recursive', 'return']):
                solution_approaches.add('recursive')
        
        # 3. Enhanced discriminative scoring
        type_diversity = min(1.0, len(input_types) / 3.0)
        size_diversity = min(1.0, len(input_sizes) / 5.0)
        complexity_diversity = (max(input_complexities) - min(input_complexities)) if input_complexities else 0
        approach_diversity = min(1.0, len(solution_approaches) / 4.0)
        
        # Weighted combination
        discriminative_score = (
            type_diversity * 0.25 +
            size_diversity * 0.25 +
            complexity_diversity * 0.25 +
            approach_diversity * 0.25
        )
        
        return discriminative_score
    
    def _compute_input_complexity(self, test_input) -> float:
        """Compute complexity score for test input."""
        try:
            if isinstance(test_input, (list, tuple)):
                if not test_input:
                    return 0.1  # Empty is simple but important
                
                # Nested structure complexity
                max_depth = self._get_nested_depth(test_input)
                length_complexity = min(1.0, len(test_input) / 10.0)  # Normalize by typical size
                
                return (max_depth * 0.6 + length_complexity * 0.4) / 2.0
                
            elif isinstance(test_input, dict):
                return min(1.0, len(test_input) / 5.0) * 0.8  # Dictionaries are complex
            elif isinstance(test_input, str):
                return min(1.0, len(test_input) / 20.0) * 0.6  # String complexity
            else:
                return 0.3  # Simple primitive
                
        except Exception:
            return 0.5  # Default complexity
    
    def _get_nested_depth(self, obj, current_depth=0) -> int:
        """Get maximum nesting depth of a data structure."""
        if isinstance(obj, (list, tuple)):
            if not obj:
                return current_depth
            return max(self._get_nested_depth(item, current_depth + 1) for item in obj)
        elif isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_nested_depth(value, current_depth + 1) for value in obj.values())
        else:
            return current_depth
    
    def _assess_execution_reliability(self, test_cases: List[Dict]) -> float:
        """Assess the reliability and executability of test cases."""
        
        reliable_count = 0
        for test_case in test_cases:
            # Check if test case has proper structure
            has_input = 'input' in test_case
            has_output = 'output' in test_case or 'expected' in test_case
            
            if has_input and has_output:
                # Additional check: input should be serializable
                try:
                    json.dumps(test_case.get('input'))
                    reliable_count += 1
                except:
                    pass
        
        reliability_score = reliable_count / len(test_cases) if test_cases else 0
        return reliability_score
    
    def _assess_input_diversity(self, test_cases: List[Dict]) -> float:
        """Assess diversity of test inputs."""
        
        if not test_cases:
            return 0.0
        
        # Collect input characteristics
        input_characteristics = []
        for test_case in test_cases:
            test_input = test_case.get('input')
            characteristics = {
                'type': type(test_input).__name__,
                'length': len(test_input) if hasattr(test_input, '__len__') else 1,
                'str_repr': str(test_input)[:50]  # First 50 chars
            }
            input_characteristics.append(characteristics)
        
        # Count unique characteristics
        unique_types = len(set(char['type'] for char in input_characteristics))
        unique_lengths = len(set(char['length'] for char in input_characteristics))
        unique_reprs = len(set(char['str_repr'] for char in input_characteristics))
        
        # Normalize diversity scores
        type_diversity = min(1.0, unique_types / 3.0)
        length_diversity = min(1.0, unique_lengths / 5.0)
        repr_diversity = unique_reprs / len(test_cases)
        
        return (type_diversity + length_diversity + repr_diversity) / 3.0


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
        
        # STAR Phase 2 Enhancement: Test Quality Prediction
        self.test_quality_predictor = TestQualityPredictor()
        self.enable_quality_prediction = True
    
    def process_input(
        self,
        problem: Dict,
        solutions: List[Dict],
        execution_traces: Optional[List] = None
    ) -> List[Dict[str, str]]:
        """Process input and create Qwen2.5-1.5B-Instruct chat messages for test case generation."""
        
        problem_desc = problem.get("question", problem.get("description", ""))
        function_sig = problem.get("function_signature", "")
        examples = problem.get("examples", "")
        problem_title = problem.get("title", "Unknown Problem")
        
        # Create system message optimized for Qwen2.5-1.5B-Instruct
        system_message = """You are Qwen, created by Alibaba Cloud. You are an expert software test engineer specializing in comprehensive test case generation and bug detection.

Your task is to generate high-quality test cases that effectively distinguish between correct and incorrect code implementations. Follow these critical requirements:
1. ALWAYS create comprehensive test coverage including edge cases and boundary conditions
2. Use systematic testing approaches combining normal, edge, boundary, and error cases
3. Generate diverse test inputs that maximize bug detection potential
4. Ensure test cases are executable and have clear expected outputs
5. Focus on corner cases that commonly break incorrect implementations"""

        # Analyze solutions to guide test generation
        solution_analysis = self._analyze_solution_patterns(solutions)
        
        # Create user message with few-shot examples and structured guidance
        # Build optional sections
        function_sig_section = f'Function Signature: {function_sig}' if function_sig else ''
        examples_section = f'Examples: {examples}' if examples else ''

        user_message = f"""Generate comprehensive test cases for the following coding problem:

**Problem: {problem_title}**
Description: {problem_desc}

{function_sig_section}
{examples_section}

**Solution Implementations Analysis:**
- Total solutions: {len(solutions)}
- Detected approaches: {', '.join(solution_analysis.get('approaches', ['various']))}
- Complexity levels: {', '.join(set(solution_analysis.get('complexity_levels', ['mixed'])))}

Here are examples of high-quality test case generation:

**Example 1 - Array Sum Problem:**
Function: def array_sum(arr): return sum(arr)

Test Cases:
1. Basic functionality: array_sum([1, 2, 3]) → 6
2. Empty array: array_sum([]) → 0  
3. Single element: array_sum([5]) → 5
4. Negative numbers: array_sum([-1, -2, 3]) → 0
5. Large numbers: array_sum([10**6, 10**6]) → 2000000
6. Zero values: array_sum([0, 0, 0]) → 0
7. Mixed types edge: array_sum([1, 2.5, 3]) → type handling
8. Boundary case: array_sum([sys.maxsize]) → overflow check

**Example 2 - String Palindrome Problem:**
Function: def is_palindrome(s): return s == s[::-1]

Test Cases:
1. Basic palindrome: is_palindrome("racecar") → True
2. Not palindrome: is_palindrome("hello") → False
3. Empty string: is_palindrome("") → True
4. Single character: is_palindrome("a") → True
5. Case sensitivity: is_palindrome("Racecar") → False
6. Spaces/punctuation: is_palindrome("race car") → False
7. Numbers: is_palindrome("12321") → True
8. Very long string: is_palindrome("a" * 1000 + "a" * 1000) → True

Now generate 8 comprehensive test cases for the given problem following this exact JSON format:

```json
{
  "test_cases": [
    {
      "name": "Basic Functionality Test",
      "category": "basic",
      "input": [example_input],
      "expected": expected_output,
      "reasoning": "Tests core functionality with normal inputs"
    }
  ]
}
```

Categories to use: "basic", "edge", "boundary", "error", "performance", "corner", "type", "large"

Focus on generating test cases that will expose bugs in incorrect implementations while validating correct ones."""

        # Return as chat messages for proper Qwen2.5-1.5B-Instruct formatting
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def _analyze_solution_patterns(self, solutions: List[Dict]) -> Dict:
        """Analyze solution patterns to guide test generation (SOTA technique)."""
        analysis = {
            "approaches": [],
            "complexity_levels": [],
            "common_patterns": [],
            "potential_bugs": []
        }
        
        try:
            for solution in solutions[:10]:  # Analyze up to 10 solutions
                code = solution.get("code", "")
                
                # Detect algorithmic approaches
                if "for " in code or "while " in code:
                    analysis["approaches"].append("iterative")
                if "def " in code and code.count("def ") > 1:
                    analysis["approaches"].append("recursive")
                if "sorted(" in code or ".sort(" in code:
                    analysis["approaches"].append("sorting")
                if "dict" in code.lower() or "hash" in code.lower():
                    analysis["approaches"].append("hashing")
                if "stack" in code.lower() or "queue" in code.lower():
                    analysis["approaches"].append("data_structures")
                
                # Assess complexity
                lines = len([line for line in code.split('\n') if line.strip()])
                if lines < 8:
                    analysis["complexity_levels"].append("simple")
                elif lines < 20:
                    analysis["complexity_levels"].append("medium")
                else:
                    analysis["complexity_levels"].append("complex")
                
                # Identify common patterns that might have bugs
                if "index" in code.lower() and "[" in code:
                    analysis["potential_bugs"].append("index_out_of_bounds")
                if "len(" in code and "range(" in code:
                    analysis["potential_bugs"].append("off_by_one")
                if "except" not in code and ("int(" in code or "float(" in code):
                    analysis["potential_bugs"].append("type_conversion")
                if "return" in code and code.count("return") > 2:
                    analysis["potential_bugs"].append("multiple_returns")
        
        except Exception as e:
            logger.debug(f"Solution analysis error: {e}")
        
        # Remove duplicates and provide defaults
        analysis["approaches"] = list(set(analysis["approaches"])) or ["general"]
        analysis["complexity_levels"] = list(set(analysis["complexity_levels"])) or ["mixed"]
        analysis["potential_bugs"] = list(set(analysis["potential_bugs"]))
        
        return analysis
    
    def generate_with_quality_prediction(
        self,
        problem: Dict,
        solutions: List[Dict],
        num_test_cases: int = 8,
        quality_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        STAR Phase 2: Generate test cases with quality prediction.
        
        Returns both test cases and their predicted quality scores
        to support strategic problem selection.
        """
        logger.info(f"🧪 Generating {num_test_cases} test cases with quality prediction")
        
        # Generate test cases using existing method
        test_cases = self.generate(problem, solutions, num_test_cases)
        
        # Predict test quality if enabled
        quality_metrics = {'overall_quality': 0.5, 'confidence': 0.0}
        if self.enable_quality_prediction and test_cases:
            quality_metrics = self.test_quality_predictor.predict_test_quality(
                test_cases, problem, solutions
            )
            
            logger.info(f"🎯 Test Quality Prediction:")
            logger.info(f"  Overall Quality: {quality_metrics['overall_quality']:.3f}")
            logger.info(f"  Confidence: {quality_metrics['confidence']:.3f}")
            
            # Log detailed quality breakdown
            if 'coverage_completeness' in quality_metrics:
                logger.debug(f"  Coverage: {quality_metrics['coverage_completeness']:.2f}")
                logger.debug(f"  Edge Cases: {quality_metrics['edge_case_detection']:.2f}")
                logger.debug(f"  Discriminative: {quality_metrics['discriminative_power']:.2f}")
                logger.debug(f"  Reliability: {quality_metrics['execution_reliability']:.2f}")
                logger.debug(f"  Diversity: {quality_metrics['input_diversity']:.2f}")
        
        # Determine if test quality meets threshold
        quality_sufficient = quality_metrics['overall_quality'] >= quality_threshold
        
        return {
            'test_cases': test_cases,
            'quality_metrics': quality_metrics,
            'quality_sufficient': quality_sufficient,
            'num_generated': len(test_cases) if test_cases else 0
        }
    
    def parse_output(self, output: str) -> List[Dict]:
        """Parse model output to extract test cases with improved JSON handling."""
        test_cases = []
        
        try:
            # Method 1: Try to extract JSON from the output (SOTA approach)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL | re.IGNORECASE)
            if not json_match:
                # Try without code blocks
                json_match = re.search(r'(\{[^{}]*"test_cases"[^{}]*\[.*?\]\s*\})', output, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(1)
                    # Clean up common JSON formatting issues
                    json_str = self._clean_json_string(json_str)
                    
                    parsed_data = json.loads(json_str)
                    raw_test_cases = parsed_data.get("test_cases", [])
                    
                    for i, raw_test in enumerate(raw_test_cases, 1):
                        if isinstance(raw_test, dict):
                            test_case = {
                                "test_id": f"test_{i:02d}",
                                "name": raw_test.get("name", f"Test Case {i}"),
                                "category": raw_test.get("category", "basic").lower(),
                                "input": raw_test.get("input"),
                                "expected": raw_test.get("expected"),
                                "reasoning": raw_test.get("reasoning", ""),
                                "generation_method": "json_structured"
                            }
                            
                            # Validate essential fields
                            if self._validate_test_case(test_case):
                                test_cases.append(test_case)
                            else:
                                logger.debug(f"Test case {i} failed validation")
                    
                    if test_cases:
                        logger.info(f"Successfully parsed {len(test_cases)} test cases from JSON")
                        return test_cases
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing failed: {e}")
                except Exception as e:
                    logger.debug(f"JSON extraction error: {e}")
            
            # Method 2: Fallback to regex parsing for legacy format
            logger.debug("Falling back to regex parsing")
            test_sections = re.split(r'TEST_CASE_\d+:|(?:^|\n)\d+\.\s*', output, flags=re.IGNORECASE | re.MULTILINE)
            
            for i, section in enumerate(test_sections[1:], 1):  # Skip first empty section
                if not section.strip():
                    continue
                    
                try:
                    # Extract components using multiple patterns
                    name_patterns = [
                        r'[Nn]ame:\s*(.+?)(?:\n|$)',
                        r'[Tt]est\s*[Nn]ame:\s*(.+?)(?:\n|$)',
                        r'^([^:\n]+):\s*'  # First line before colon
                    ]
                    
                    category_patterns = [
                        r'[Cc]ategory:\s*(\w+)',
                        r'[Tt]ype:\s*(\w+)',
                        r'\b(basic|edge|boundary|error|performance|corner|type|large)\b'
                    ]
                    
                    input_patterns = [
                        r'[Ii]nput:\s*(.+?)(?:\n[A-Za-z]|\n\n|$)',
                        r'[Ii]nput\s*values?:\s*(.+?)(?:\n[A-Za-z]|\n\n|$)'
                    ]
                    
                    expected_patterns = [
                        r'[Ee]xpected:\s*(.+?)(?:\n[A-Za-z]|\n\n|$)',
                        r'[Oo]utput:\s*(.+?)(?:\n[A-Za-z]|\n\n|$)',
                        r'→\s*(.+?)(?:\n|$)'  # Arrow format
                    ]
                    
                    name = self._extract_with_patterns(section, name_patterns) or f"Test Case {i}"
                    category = self._extract_with_patterns(section, category_patterns) or "basic"
                    input_str = self._extract_with_patterns(section, input_patterns) or ""
                    expected_str = self._extract_with_patterns(section, expected_patterns) or ""
                    
                    test_case = {
                        "test_id": f"test_{i:02d}",
                        "name": name.strip(),
                        "category": category.strip().lower(),
                        "input": self.parse_test_input(input_str),
                        "expected": self.parse_expected_output(expected_str),
                        "reasoning": "",
                        "generation_method": "regex_parsed",
                        "raw_section": section.strip()
                    }
                    
                    if self._validate_test_case(test_case):
                        test_cases.append(test_case)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse test case {i}: {e}")
                    continue
            
        except Exception as e:
            logger.warning(f"Failed to parse test cases: {e}")
        
        # If no test cases found, create fallback tests
        if not test_cases:
            logger.warning("No valid test cases parsed, using fallback")
            test_cases = self.create_fallback_tests()
        
        logger.info(f"Parsed {len(test_cases)} test cases")
        return test_cases
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean common JSON formatting issues."""
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # Fix unquoted keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        # Fix single quotes
        json_str = json_str.replace("'", '"')
        return json_str
    
    def _extract_with_patterns(self, text: str, patterns: List[str]) -> Optional[str]:
        """Extract text using multiple regex patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None
    
    def _validate_test_case(self, test_case: Dict) -> bool:
        """Validate that a test case has essential components."""
        try:
            # Must have name and category
            if not test_case.get("name") or not test_case.get("category"):
                return False
            
            # Must have some kind of input (can be None but field must exist)
            if "input" not in test_case:
                return False
            
            # Category must be valid
            valid_categories = {"basic", "edge", "boundary", "error", "performance", "corner", "type", "large", "mutation"}
            if test_case.get("category") not in valid_categories:
                test_case["category"] = "basic"  # Fix invalid category
            
            return True
            
        except Exception as e:
            logger.debug(f"Test case validation error: {e}")
            return False
    
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
        num_test_cases: int = 8,
        max_retry_attempts: int = 3
    ) -> List[Dict]:
        """Generate test cases using SOTA Qwen2.5-1.5B-Instruct techniques."""
        logger.info(f"Generating {num_test_cases} test cases for problem {problem.get('problem_id', 'unknown')}")
        
        # Determine optimal temperature based on problem complexity
        problem_complexity = len(solutions) > 10 or len(problem.get("question", "")) > 500
        base_temp = 0.4 if problem_complexity else 0.3  # SOTA temperature for test generation
        
        test_cases = []
        
        for attempt in range(max_retry_attempts):
            try:
                # Create chat messages using proper Qwen2.5-1.5B-Instruct format
                messages = self.process_input(
                    problem=problem,
                    solutions=solutions,
                    execution_traces=execution_traces
                )
                
                # Apply progressive temperature strategy
                current_temp = base_temp + (attempt * 0.1)
                current_temp = min(current_temp, 0.8)  # Cap at 0.8 for consistency
                
                # Generate using chat template
                outputs = self.generate_text(
                    messages=messages,
                    max_new_tokens=2048,
                    temperature=current_temp,
                    top_p=0.9,
                    do_sample=current_temp > 0.0,
                    num_return_sequences=1
                )
                
                if outputs and len(outputs) > 0:
                    # Parse test cases with improved error handling
                    parsed_test_cases = self.parse_output(outputs[0])
                    
                    if parsed_test_cases and len(parsed_test_cases) >= 4:  # At least 4 valid test cases
                        test_cases = parsed_test_cases
                        logger.debug(f"✅ Generated {len(test_cases)} test cases (temp: {current_temp:.2f})")
                        break
                    else:
                        logger.debug(f"⚠️ Attempt {attempt+1} generated only {len(parsed_test_cases)} valid test cases")
                else:
                    logger.debug(f"⚠️ No output generated for attempt {attempt+1}")
                
            except Exception as e:
                logger.warning(f"❌ Error generating test cases, attempt {attempt+1}: {e}")
                continue
        
        # If no test cases generated, use fallback
        if not test_cases:
            logger.warning("Failed to generate test cases after all attempts, using enhanced fallback")
            test_cases = self.create_enhanced_fallback_tests(problem, solutions)
        
        # Ensure we have the right number of test cases
        test_cases = self._normalize_test_case_count(test_cases, num_test_cases)
        
        # Add metadata and generation info
        for i, test_case in enumerate(test_cases):
            test_case.update({
                "generation_order": i + 1,
                "problem_id": problem.get("problem_id", "unknown"),
                "generation_temperature": current_temp if 'current_temp' in locals() else base_temp,
                "solutions_analyzed": len(solutions),
                "generation_timestamp": time.time() if 'time' in globals() else None
            })
        
        # Validate test coverage
        coverage_score = self.compute_coverage_score(test_cases)
        logger.info(f"✅ Generated {len(test_cases)} test cases (coverage: {coverage_score:.2f})")
        
        return test_cases
    
    def _normalize_test_case_count(self, test_cases: List[Dict], target_count: int) -> List[Dict]:
        """Ensure we have the target number of test cases."""
        if len(test_cases) < target_count:
            logger.debug(f"Padding test cases from {len(test_cases)} to {target_count}")
            # Generate additional simple test cases
            while len(test_cases) < target_count:
                test_cases.append({
                    "test_id": f"test_{len(test_cases)+1:02d}",
                    "name": f"Additional Test {len(test_cases)+1}",
                    "category": "basic",
                    "input": None,
                    "expected": None,
                    "reasoning": "Additional test case to meet requirements",
                    "generation_method": "padding"
                })
        
        elif len(test_cases) > target_count:
            # Keep the best test cases (prioritize diverse categories)
            logger.debug(f"Trimming test cases from {len(test_cases)} to {target_count}")
            # Sort by category diversity first, then by generation order
            category_counts = {}
            sorted_tests = []
            
            for test_case in test_cases:
                category = test_case.get("category", "basic")
                category_counts[category] = category_counts.get(category, 0) + 1
                test_case["category_rarity"] = 1.0 / category_counts[category]
            
            # Prioritize diverse categories
            test_cases.sort(key=lambda x: (-x.get("category_rarity", 0), x.get("generation_order", 0)))
            test_cases = test_cases[:target_count]
        
        return test_cases
    
    def create_enhanced_fallback_tests(self, problem: Dict, solutions: List[Dict]) -> List[Dict]:
        """Create enhanced fallback test cases based on problem analysis."""
        logger.info("Creating enhanced fallback test cases")
        
        # Analyze solutions to create better fallback tests
        analysis = self._analyze_solution_patterns(solutions)
        
        fallback_tests = [
            {
                "test_id": "test_01",
                "name": "Basic Functionality Test",
                "category": "basic",
                "input": None,  # Would need problem-specific default
                "expected": None,
                "reasoning": "Test core functionality with normal inputs",
                "generation_method": "fallback_enhanced"
            },
            {
                "test_id": "test_02", 
                "name": "Empty Input Edge Case",
                "category": "edge",
                "input": [],
                "expected": None,
                "reasoning": "Test behavior with empty input"
            },
            {
                "test_id": "test_03",
                "name": "Single Element Test",
                "category": "edge", 
                "input": [1],
                "expected": None,
                "reasoning": "Test minimal valid input"
            },
            {
                "test_id": "test_04",
                "name": "Boundary Value Test",
                "category": "boundary",
                "input": 0,
                "expected": None,
                "reasoning": "Test boundary conditions"
            }
        ]
        
        # Add pattern-specific tests based on analysis
        if "index_out_of_bounds" in analysis.get("potential_bugs", []):
            fallback_tests.append({
                "test_id": "test_05",
                "name": "Index Boundary Test",
                "category": "boundary",
                "input": [1, 2, 3],
                "expected": None,
                "reasoning": "Test for index out of bounds errors"
            })
        
        if "type_conversion" in analysis.get("potential_bugs", []):
            fallback_tests.append({
                "test_id": "test_06", 
                "name": "Type Conversion Test",
                "category": "error",
                "input": "123",
                "expected": None,
                "reasoning": "Test type conversion handling"
            })
        
        return fallback_tests[:8]  # Return up to 8 tests
    
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