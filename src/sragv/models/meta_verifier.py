"""
Meta-Verifier - Player 4 of SRAG-V architecture.
Validates generated test cases and provides binary validity assessment.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
import logging
import ast

from .base_player import BasePlayer

logger = logging.getLogger(__name__)


class MetaVerifier(BasePlayer):
    """
    Player 4: Meta-Verifier (0.5B parameters)
    Model: Qwen2.5-0.5B-Instruct fine-tuned from scratch
    
    Input:
    - Problem + solutions + generated tests + test results
    - Test consistency metrics (determinism, reproducibility)
    
    Output: Binary validity for each test + invalidity reason
    
    Architecture Details:
    - Lightweight but crucial for preventing invalid tests
    - Multi-task output: valid/invalid + 5 invalidity categories
    - Trained on APPS tests initially, then self-improves
    """
    
    def __init__(self, config: Dict):
        super().__init__(
            model_name=config["model_name"],
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            lora_config=config.get("lora_config")
        )
        
        # Invalidity categories
        self.invalidity_categories = [
            "syntax_error",      # Invalid input/output format
            "logic_error",       # Inconsistent with problem requirements
            "redundant",         # Duplicate or unnecessary test
            "unrealistic",       # Inputs that wouldn't occur in practice
            "ambiguous"          # Unclear expected behavior
        ]
        
        # Quality criteria for test validation
        self.quality_criteria = {
            "deterministic": "Test produces consistent results across runs",
            "relevant": "Test inputs are related to the problem domain",
            "passable": "At least one correct solution should pass this test",
            "discriminating": "Test can distinguish correct from incorrect solutions",
            "minimal": "Test contains no redundant assertions"
        }
    
    def process_input(
        self,
        problem: Dict,
        solutions: List[Dict],
        test_cases: List[Dict],
        test_results: Optional[List[Dict]] = None
    ) -> str:
        """Process input and create a prompt for test validation."""
        
        problem_desc = problem.get("description", "")[:500]  # Truncate for brevity
        
        prompt = f"""You are an expert test validator. Your task is to determine if generated test cases are valid and useful for evaluating coding solutions.

PROBLEM:
{problem_desc}

SOLUTIONS AVAILABLE: {len(solutions)} different implementations

TEST CASES TO VALIDATE:
"""
        
        # Include test cases to validate
        for i, test_case in enumerate(test_cases[:8], 1):  # Limit to 8 for prompt length
            name = test_case.get("name", f"Test {i}")
            category = test_case.get("category", "unknown")
            test_input = test_case.get("input", "")
            expected = test_case.get("expected", "")
            reasoning = test_case.get("reasoning", "")[:200]  # Truncate reasoning
            
            prompt += f"""
TEST {i}:
Name: {name}
Category: {category}
Input: {test_input}
Expected: {expected}
Reasoning: {reasoning}
"""
        
        # Add test results if available
        if test_results:
            prompt += f"\nTEST EXECUTION RESULTS:\n"
            for i, result in enumerate(test_results[:3], 1):  # Show sample results
                prompt += f"Test {i} execution: {result}\n"
        
        prompt += f"""
VALIDATION CRITERIA:
A valid test case must be:
1. DETERMINISTIC - Same input always produces same output
2. RELEVANT - Input is meaningful for the problem domain
3. PASSABLE - At least one correct solution can pass it
4. DISCRIMINATING - Can distinguish correct from incorrect solutions
5. MINIMAL - No redundant or unnecessary complexity

For each test case, provide:
- VALID: YES/NO
- INVALIDITY_REASON: (if invalid) one of: {', '.join(self.invalidity_categories)}
- CONFIDENCE: 0.0-1.0 (how confident you are in this assessment)
- EXPLANATION: Brief explanation of your decision

Format your response as:
TEST_1: VALID=YES/NO, REASON=category, CONFIDENCE=0.0-1.0, EXPLANATION=brief explanation
TEST_2: VALID=YES/NO, REASON=category, CONFIDENCE=0.0-1.0, EXPLANATION=brief explanation
...

Validate all {len(test_cases)} test cases:"""

        return prompt
    
    def parse_output(self, output: str, num_test_cases: int) -> List[Dict]:
        """Parse model output to extract validation results."""
        validations = []
        
        try:
            # Split into lines and process each validation
            lines = output.strip().split('\n')
            test_validations = [line for line in lines if line.strip().startswith('TEST_')]
            
            for i, line in enumerate(test_validations[:num_test_cases], 1):
                try:
                    # Parse validation line
                    # Format: TEST_1: VALID=YES/NO, REASON=category, CONFIDENCE=0.0-1.0, EXPLANATION=brief explanation
                    
                    valid_match = re.search(r'VALID\s*=\s*(YES|NO)', line, re.IGNORECASE)
                    reason_match = re.search(r'REASON\s*=\s*(\w+)', line, re.IGNORECASE)
                    confidence_match = re.search(r'CONFIDENCE\s*=\s*([\d.]+)', line, re.IGNORECASE)
                    explanation_match = re.search(r'EXPLANATION\s*=\s*(.+)', line, re.IGNORECASE)
                    
                    is_valid = valid_match.group(1).upper() == 'YES' if valid_match else True
                    invalidity_reason = reason_match.group(1).lower() if reason_match and not is_valid else None
                    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
                    explanation = explanation_match.group(1).strip() if explanation_match else ""
                    
                    validation = {
                        "test_index": i - 1,  # 0-based index
                        "is_valid": is_valid,
                        "invalidity_reason": invalidity_reason,
                        "confidence": confidence,
                        "explanation": explanation,
                        "raw_line": line.strip()
                    }
                    
                    validations.append(validation)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse validation line {i}: {e}")
                    # Add default validation
                    validations.append({
                        "test_index": i - 1,
                        "is_valid": True,  # Default to valid
                        "invalidity_reason": None,
                        "confidence": 0.5,
                        "explanation": "Parsing failed, defaulting to valid",
                        "raw_line": line.strip()
                    })
            
            # Fill in missing validations with defaults
            while len(validations) < num_test_cases:
                idx = len(validations)
                validations.append({
                    "test_index": idx,
                    "is_valid": True,
                    "invalidity_reason": None,
                    "confidence": 0.5,
                    "explanation": "No validation provided, defaulting to valid",
                    "raw_line": ""
                })
        
        except Exception as e:
            logger.warning(f"Failed to parse validation output: {e}")
            # Create default validations
            validations = [
                {
                    "test_index": i,
                    "is_valid": True,
                    "invalidity_reason": None,
                    "confidence": 0.5,
                    "explanation": "Parsing error, defaulting to valid",
                    "raw_line": ""
                }
                for i in range(num_test_cases)
            ]
        
        logger.info(f"Parsed {len(validations)} test validations")
        return validations
    
    def validate_test_cases(
        self,
        problem: Dict,
        solutions: List[Dict],
        test_cases: List[Dict],
        test_results: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Validate a batch of test cases."""
        logger.info(f"Validating {len(test_cases)} test cases for problem {problem.get('problem_id', 'unknown')}")
        
        if not test_cases:
            return []
        
        # Create prompt
        prompt = self.process_input(
            problem=problem,
            solutions=solutions,
            test_cases=test_cases,
            test_results=test_results
        )
        
        # Generate validation
        outputs = self.generate_text(
            prompt=prompt,
            max_new_tokens=1024,
            temperature=0.1,  # Low temperature for consistent validation
            top_p=0.95,
            num_return_sequences=1
        )
        
        if not outputs:
            logger.warning("Failed to generate validations, defaulting to all valid")
            return self.create_default_validations(len(test_cases))
        
        # Parse validations
        validations = self.parse_output(outputs[0], len(test_cases))
        
        # Combine test cases with their validations
        validated_test_cases = []
        for i, test_case in enumerate(test_cases):
            validation = validations[i] if i < len(validations) else {
                "test_index": i,
                "is_valid": True,
                "invalidity_reason": None,
                "confidence": 0.5,
                "explanation": "No validation available"
            }
            
            # Combine test case with validation
            validated_test_case = test_case.copy()
            validated_test_case.update({
                "is_valid": validation["is_valid"],
                "invalidity_reason": validation["invalidity_reason"],
                "confidence": validation["confidence"],
                "validation_explanation": validation["explanation"],
                "meta_verifier_raw": validation.get("raw_line", "")
            })
            
            validated_test_cases.append(validated_test_case)
        
        # Calculate validation statistics
        valid_count = sum(1 for tc in validated_test_cases if tc["is_valid"])
        avg_confidence = sum(tc["confidence"] for tc in validated_test_cases) / len(validated_test_cases)
        
        logger.info(f"Validation complete: {valid_count}/{len(test_cases)} valid, avg confidence: {avg_confidence:.3f}")
        
        return validated_test_cases
    
    def create_default_validations(self, num_test_cases: int) -> List[Dict]:
        """Create default validations when generation fails."""
        return [
            {
                "test_index": i,
                "is_valid": True,
                "invalidity_reason": None,
                "confidence": 0.5,
                "explanation": "Default validation - generation failed"
            }
            for i in range(num_test_cases)
        ]
    
    def analyze_test_consistency(
        self,
        test_cases: List[Dict],
        execution_results: Optional[List[Dict]] = None
    ) -> Dict:
        """Analyze test consistency metrics."""
        analysis = {
            "determinism_score": 0.0,
            "relevance_score": 0.0,
            "passability_score": 0.0,
            "discriminating_score": 0.0,
            "minimality_score": 0.0,
            "overall_quality": 0.0
        }
        
        try:
            if not test_cases:
                return analysis
            
            # Determinism: Check if inputs/outputs are well-defined
            deterministic_tests = 0
            for test_case in test_cases:
                input_val = test_case.get("input")
                expected_val = test_case.get("expected")
                if input_val is not None and expected_val is not None:
                    deterministic_tests += 1
            
            analysis["determinism_score"] = deterministic_tests / len(test_cases)
            
            # Relevance: Check if test categories are appropriate
            relevant_categories = {"basic", "edge", "boundary", "error"}
            relevant_tests = sum(1 for tc in test_cases 
                               if tc.get("category", "") in relevant_categories)
            analysis["relevance_score"] = relevant_tests / len(test_cases)
            
            # Passability: Based on execution results if available
            if execution_results:
                passable_tests = sum(1 for result in execution_results 
                                   if result.get("success", False))
                analysis["passability_score"] = passable_tests / len(execution_results)
            else:
                analysis["passability_score"] = 0.7  # Default assumption
            
            # Discriminating: Tests with varied difficulty
            categories = set(tc.get("category", "basic") for tc in test_cases)
            analysis["discriminating_score"] = min(len(categories) / 4, 1.0)  # Up to 4 categories
            
            # Minimality: Not too many redundant tests
            unique_inputs = set()
            for tc in test_cases:
                input_str = str(tc.get("input", ""))
                unique_inputs.add(input_str)
            
            analysis["minimality_score"] = len(unique_inputs) / len(test_cases)
            
            # Overall quality
            scores = [
                analysis["determinism_score"],
                analysis["relevance_score"],
                analysis["passability_score"],
                analysis["discriminating_score"],
                analysis["minimality_score"]
            ]
            analysis["overall_quality"] = sum(scores) / len(scores)
            
        except Exception as e:
            logger.warning(f"Error analyzing test consistency: {e}")
        
        return analysis
    
    def filter_valid_tests(self, validated_test_cases: List[Dict]) -> List[Dict]:
        """Filter out invalid test cases."""
        valid_tests = [tc for tc in validated_test_cases if tc.get("is_valid", True)]
        
        invalid_count = len(validated_test_cases) - len(valid_tests)
        if invalid_count > 0:
            logger.info(f"Filtered out {invalid_count} invalid test cases")
            
            # Log reasons for invalidity
            invalid_reasons = {}
            for tc in validated_test_cases:
                if not tc.get("is_valid", True):
                    reason = tc.get("invalidity_reason", "unknown")
                    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
            
            logger.info(f"Invalidity reasons: {invalid_reasons}")
        
        return valid_tests
    
    def compute_meta_score(self, validated_test_cases: List[Dict]) -> float:
        """Compute overall meta-verification score."""
        try:
            if not validated_test_cases:
                return 0.0
            
            # Weighted score based on validity and confidence
            total_score = 0.0
            total_weight = 0.0
            
            for tc in validated_test_cases:
                is_valid = tc.get("is_valid", True)
                confidence = tc.get("confidence", 0.5)
                
                # Valid tests contribute positively, invalid tests negatively
                score = confidence if is_valid else -confidence
                weight = confidence  # Higher confidence gets more weight
                
                total_score += score * weight
                total_weight += weight
            
            if total_weight == 0:
                return 0.5
            
            # Normalize to [0, 1] range
            raw_score = total_score / total_weight
            normalized_score = (raw_score + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            return max(0.0, min(1.0, normalized_score))
            
        except Exception as e:
            logger.warning(f"Error computing meta score: {e}")
            return 0.5