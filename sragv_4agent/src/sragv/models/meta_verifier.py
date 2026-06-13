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
            device=config.get("device", "auto"),
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
        
        # STAR Phase 2 Enhancement: Strategic feedback capability
        self.strategic_feedback_enabled = True
        self.verification_confidence_history = []
        self.problem_difficulty_estimates = {}
        
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
    ) -> List[Dict[str, str]]:
        """Process input and create Qwen2.5-0.5B-Instruct chat messages for test validation."""
        
        problem_desc = problem.get("question", problem.get("description", ""))[:400]  # Optimized for 0.5B model
        problem_title = problem.get("title", "Coding Problem")
        
        # Create system message optimized for Qwen2.5-0.5B-Instruct binary classification
        system_message = """You are Qwen, created by Alibaba Cloud. You are an expert test case validator specializing in binary classification of test quality and validity.

Your task is to evaluate test cases and provide binary validity assessments. Focus on accuracy and consistency in your validation decisions."""

        # Analyze solutions and test results for context
        execution_context = self._create_execution_context(solutions, test_results)
        
        # Create user message with few-shot examples and JSON output format
        user_message = f"""Validate the quality and correctness of test cases for this coding problem:

**Problem: {problem_title}**
{problem_desc}

**Context:**
- Available solutions: {len(solutions)}
- Solution patterns: {execution_context}

**Examples of Valid vs Invalid Test Cases:**

**Example 1 - Valid Test:**
```
Name: "Basic Sum Test"
Input: [1, 2, 3]
Expected: 6
Category: basic
→ VALID: This tests core functionality with clear inputs/outputs
```

**Example 2 - Invalid Test:**
```
Name: "Confusing Test"
Input: "unclear input"
Expected: "ambiguous"
Category: unknown
→ INVALID: Ambiguous inputs and expected outputs
```

**Test Cases to Validate:**
"""

        # Add test cases in compact format for 0.5B model efficiency
        for i, test_case in enumerate(test_cases[:8], 1):  # Limit for small model context
            name = test_case.get("name", f"Test {i}")[:50]
            category = test_case.get("category", "unknown")
            test_input = str(test_case.get("input", ""))[:100]  # Truncate for efficiency
            expected = str(test_case.get("expected", ""))[:100]
            
            user_message += f"""
**Test {i}:**
Name: "{name}"
Input: {test_input}
Expected: {expected}
Category: {category}
"""

        user_message += f"""

**Validation Criteria:**
1. DETERMINISTIC: Clear input → consistent output
2. RELEVANT: Matches problem domain
3. EXECUTABLE: Can be run on solutions
4. DISCRIMINATING: Helps identify correct vs incorrect code

**Required JSON Output Format:**
```json
{{
  "validations": [
    {{
      "test_index": 1,
      "is_valid": true,
      "confidence": 0.9,
      "reason": "clear_and_executable",
      "explanation": "Well-defined test case"
    }}
  ]
}}
```

**Reason Categories:** "clear_and_executable", "syntax_error", "logic_error", "redundant", "unrealistic", "ambiguous"

Validate all {len(test_cases)} test cases and respond with the exact JSON format above:"""

        # Return as chat messages for proper Qwen2.5-0.5B-Instruct formatting
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def _create_execution_context(self, solutions: List[Dict], test_results: Optional[List[Dict]]) -> str:
        """Create compact execution context for 0.5B model."""
        context_parts = []
        
        # Analyze solution complexity
        if solutions:
            avg_length = sum(len(s.get("code", "")) for s in solutions[:5]) / min(len(solutions), 5)
            if avg_length < 200:
                context_parts.append("simple")
            elif avg_length < 500:
                context_parts.append("medium")
            else:
                context_parts.append("complex")
        
        # Add test results summary if available
        if test_results:
            success_rate = sum(1 for r in test_results if r.get("success", False)) / len(test_results)
            if success_rate > 0.8:
                context_parts.append("high_success")
            elif success_rate > 0.5:
                context_parts.append("medium_success")
            else:
                context_parts.append("low_success")
        
        return ", ".join(context_parts) if context_parts else "general"
    
    def parse_output(self, output: str, num_test_cases: int) -> List[Dict]:
        """Parse model output with improved JSON handling for 0.5B model."""
        validations = []
        
        try:
            # Method 1: Try to extract JSON from the output (SOTA approach for 0.5B models)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL | re.IGNORECASE)
            if not json_match:
                # Try without code blocks
                json_match = re.search(r'(\{[^{}]*"validations"[^{}]*\[.*?\]\s*\})', output, re.DOTALL)
            
            if json_match:
                try:
                    json_str = json_match.group(1)
                    # Clean up common JSON formatting issues for small models
                    json_str = self._clean_json_string(json_str)
                    
                    parsed_data = json.loads(json_str)
                    raw_validations = parsed_data.get("validations", [])
                    
                    for i, validation in enumerate(raw_validations):
                        if isinstance(validation, dict):
                            # Convert reason to invalidity_reason format
                            reason = validation.get("reason", "clear_and_executable")
                            is_valid = validation.get("is_valid", True)
                            invalidity_reason = None if is_valid else reason
                            
                            validation_result = {
                                "test_index": validation.get("test_index", i) - 1,  # Convert to 0-based
                                "is_valid": is_valid,
                                "invalidity_reason": invalidity_reason,
                                "confidence": float(validation.get("confidence", 0.8)),
                                "explanation": validation.get("explanation", ""),
                                "generation_method": "json_structured",
                                "raw_validation": validation
                            }
                            
                            # Validate the validation result
                            if self._validate_validation_result(validation_result):
                                validations.append(validation_result)
                            else:
                                logger.debug(f"Validation {i+1} failed validation check")
                    
                    if validations:
                        logger.info(f"Successfully parsed {len(validations)} validations from JSON")
                        return self._normalize_validations(validations, num_test_cases)
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parsing failed: {e}")
                except Exception as e:
                    logger.debug(f"JSON extraction error: {e}")
            
            # Method 2: Fallback to regex parsing for legacy format
            logger.debug("Falling back to regex parsing for validation")
            test_validations = []
            
            # Look for various patterns
            patterns = [
                r'TEST[_\s]*(\d+):\s*VALID\s*=\s*(YES|NO)',
                r'Test\s*(\d+).*?valid[:\s]*(\w+)',
                r'(\d+)[\.)].*?(valid|invalid)',
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, output, re.IGNORECASE)
                if matches:
                    test_validations = matches
                    break
            
            for i, match in enumerate(test_validations[:num_test_cases]):
                try:
                    if len(match) >= 2:
                        test_num = int(match[0]) if match[0].isdigit() else i + 1
                        validity = match[1].upper()
                        is_valid = validity in ['YES', 'VALID', 'TRUE']
                        
                        validation = {
                            "test_index": test_num - 1,  # Convert to 0-based
                            "is_valid": is_valid,
                            "invalidity_reason": "ambiguous" if not is_valid else None,
                            "confidence": 0.7,  # Default confidence for regex parsing
                            "explanation": f"Regex parsed: {validity}",
                            "generation_method": "regex_parsed"
                        }
                        
                        validations.append(validation)
                    
                except Exception as e:
                    logger.debug(f"Failed to parse validation match {i}: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to parse validation output: {e}")
        
        # If no validations found, create defaults optimized for 0.5B model behavior
        if not validations:
            logger.warning("No validations parsed, using enhanced fallback")
            validations = self._create_fallback_validations(num_test_cases, output)
        
        return self._normalize_validations(validations, num_test_cases)
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean JSON string for 0.5B model output (more aggressive cleaning)."""
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        # Fix unquoted boolean values
        json_str = re.sub(r':\s*true\b', ': true', json_str)
        json_str = re.sub(r':\s*false\b', ': false', json_str)
        # Fix unquoted keys
        json_str = re.sub(r'(\w+):', r'"\1":', json_str)
        # Fix single quotes
        json_str = json_str.replace("'", '"')
        # Remove newlines inside JSON
        json_str = re.sub(r'\n\s*', ' ', json_str)
        return json_str
    
    def _validate_validation_result(self, validation: Dict) -> bool:
        """Validate that a validation result has essential components."""
        try:
            # Must have test_index
            if "test_index" not in validation:
                return False
            
            # Must have is_valid boolean
            if not isinstance(validation.get("is_valid"), bool):
                return False
            
            # Confidence must be reasonable
            confidence = validation.get("confidence", 0.5)
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                validation["confidence"] = 0.5  # Fix invalid confidence
            
            return True
            
        except Exception as e:
            logger.debug(f"Validation result validation error: {e}")
            return False
    
    def _normalize_validations(self, validations: List[Dict], target_count: int) -> List[Dict]:
        """Ensure we have the target number of validations."""
        # Fill missing validations
        while len(validations) < target_count:
            validations.append({
                "test_index": len(validations),
                "is_valid": True,  # Conservative default for 0.5B model
                "invalidity_reason": None,
                "confidence": 0.6,
                "explanation": "Missing validation, defaulting to valid",
                "generation_method": "padding"
            })
        
        # Trim excess validations
        if len(validations) > target_count:
            validations = validations[:target_count]
        
        # Ensure proper indexing
        for i, validation in enumerate(validations):
            validation["test_index"] = i
        
        return validations
    
    def _create_fallback_validations(self, num_test_cases: int, output: str) -> List[Dict]:
        """Create intelligent fallback validations based on output analysis."""
        fallback_validations = []
        
        # Analyze output for sentiment (simple heuristic for 0.5B model behavior)
        positive_indicators = ["valid", "good", "correct", "appropriate", "clear"]
        negative_indicators = ["invalid", "bad", "incorrect", "unclear", "ambiguous", "error"]
        
        positive_count = sum(1 for word in positive_indicators if word.lower() in output.lower())
        negative_count = sum(1 for word in negative_indicators if word.lower() in output.lower())
        
        # Default validity based on sentiment analysis
        default_validity = positive_count > negative_count
        default_confidence = 0.6 if default_validity else 0.7
        
        for i in range(num_test_cases):
            fallback_validations.append({
                "test_index": i,
                "is_valid": default_validity,
                "invalidity_reason": "ambiguous" if not default_validity else None,
                "confidence": default_confidence,
                "explanation": f"Fallback based on output sentiment analysis",
                "generation_method": "sentiment_fallback"
            })
        
        return fallback_validations
    
    def generate(self, **kwargs) -> Any:
        """Main generation method - delegates to validate_test_cases."""
        return self.validate_test_cases(**kwargs)
    
    def validate_test_cases(
        self,
        problem: Dict,
        solutions: List[Dict],
        test_cases: List[Dict],
        test_results: Optional[List[Dict]] = None,
        max_retry_attempts: int = 2
    ) -> List[Dict]:
        """Validate test cases using SOTA Qwen2.5-0.5B-Instruct techniques."""
        logger.info(f"Validating {len(test_cases)} test cases for problem {problem.get('problem_id', 'unknown')}")
        
        if not test_cases:
            return []
        
        # SOTA temperature strategy for 0.5B models (binary classification optimized)
        base_temp = 0.2  # Very low for consistent binary decisions
        validations = []
        
        for attempt in range(max_retry_attempts):
            try:
                # Create chat messages using proper Qwen2.5-0.5B-Instruct format
                messages = self.process_input(
                    problem=problem,
                    solutions=solutions,
                    test_cases=test_cases,
                    test_results=test_results
                )
                
                # Apply TURN (Temperature Using Relevance Network) inspired strategy
                current_temp = base_temp + (attempt * 0.05)  # Very small increments for 0.5B
                current_temp = min(current_temp, 0.3)  # Cap at 0.3 for consistency
                
                # Generate using chat template with optimized parameters for small model
                outputs = self.generate_text(
                    messages=messages,
                    max_new_tokens=1024,  # Sufficient for JSON validation output
                    temperature=current_temp,
                    top_p=0.95,
                    do_sample=current_temp > 0.0,
                    num_return_sequences=1
                )
                
                if outputs and len(outputs) > 0:
                    # Parse validations with improved error handling
                    parsed_validations = self.parse_output(outputs[0], len(test_cases))
                    
                    if parsed_validations and len(parsed_validations) >= len(test_cases) * 0.8:  # At least 80% parsed
                        validations = parsed_validations
                        logger.debug(f"✅ Validated {len(validations)} test cases (temp: {current_temp:.3f})")
                        break
                    else:
                        logger.debug(f"⚠️ Attempt {attempt+1} parsed only {len(parsed_validations)} validations")
                else:
                    logger.debug(f"⚠️ No output generated for validation attempt {attempt+1}")
                
            except Exception as e:
                logger.warning(f"❌ Error validating test cases, attempt {attempt+1}: {e}")
                continue
        
        # If no validations generated, use intelligent fallback
        if not validations:
            logger.warning("Failed to generate validations after all attempts, using intelligent fallback")
            validations = self.create_intelligent_fallback_validations(test_cases, solutions)
        
        # STAR Phase 2 Enhancement: Extract strategic feedback during validation
        strategic_feedback = {}
        if self.strategic_feedback_enabled:
            strategic_feedback = self._extract_strategic_feedback(
                problem, solutions, test_cases, validations
            )
        
        # Combine test cases with their validations
        validated_test_cases = []
        for i, test_case in enumerate(test_cases):
            validation = validations[i] if i < len(validations) else {
                "test_index": i,
                "is_valid": True,
                "invalidity_reason": None,
                "confidence": 0.6,
                "explanation": "No validation available, defaulting to valid",
                "generation_method": "missing_fallback"
            }
            
            # Combine test case with validation
            validated_test_case = test_case.copy()
            validated_test_case.update({
                "is_valid": validation["is_valid"],
                "invalidity_reason": validation["invalidity_reason"],
                "confidence": validation["confidence"],
                "validation_explanation": validation["explanation"],
                "meta_verifier_raw": validation.get("raw_validation", ""),
                "validation_method": validation.get("generation_method", "standard"),
                "validation_temperature": current_temp if 'current_temp' in locals() else base_temp
            })
            
            validated_test_cases.append(validated_test_case)
        
        # Calculate validation statistics
        valid_count = sum(1 for tc in validated_test_cases if tc["is_valid"])
        avg_confidence = sum(tc["confidence"] for tc in validated_test_cases) / len(validated_test_cases)
        
        # Log validation quality metrics
        method_counts = {}
        for tc in validated_test_cases:
            method = tc.get("validation_method", "standard")
            method_counts[method] = method_counts.get(method, 0) + 1
        
        logger.info(f"✅ Validation complete: {valid_count}/{len(test_cases)} valid, avg confidence: {avg_confidence:.3f}")
        logger.debug(f"Validation methods used: {method_counts}")
        
        # STAR Phase 2: Add strategic feedback to results
        if strategic_feedback:
            return {
                'validated_test_cases': validated_test_cases,
                'strategic_feedback': strategic_feedback
            }
        
        return validated_test_cases
    
    def create_intelligent_fallback_validations(self, test_cases: List[Dict], solutions: List[Dict]) -> List[Dict]:
        """Create intelligent fallback validations based on heuristic analysis."""
        logger.info("Creating intelligent fallback validations using heuristic analysis")
        
        fallback_validations = []
        
        for i, test_case in enumerate(test_cases):
            # Heuristic validation based on test case properties
            name = test_case.get("name", "")
            category = test_case.get("category", "unknown")
            test_input = test_case.get("input")
            expected = test_case.get("expected")
            
            # Initialize validation
            is_valid = True
            confidence = 0.7
            invalidity_reason = None
            explanation = "Heuristic analysis: "
            
            # Rule 1: Check for essential fields
            if test_input is None and expected is None:
                is_valid = False
                invalidity_reason = "ambiguous"
                confidence = 0.9
                explanation += "missing input and expected output"
            
            # Rule 2: Check category validity
            elif category not in ["basic", "edge", "boundary", "error", "performance", "corner", "type", "large", "mutation"]:
                confidence *= 0.8  # Reduce confidence but don't invalidate
                explanation += "unusual category, "
            
            # Rule 3: Check for reasonable input types
            elif isinstance(test_input, str) and len(str(test_input)) > 1000:
                confidence *= 0.7  # Very long inputs might be unrealistic
                explanation += "very long input, "
            
            # Rule 4: Basic functionality tests should be simple
            elif category == "basic" and isinstance(test_input, (list, dict)) and len(test_input) > 100:
                confidence *= 0.8  # Basic tests shouldn't be too complex
                explanation += "complex input for basic test, "
            
            # Rule 5: Check for meaningful names
            if not name or len(name.strip()) < 5:
                confidence *= 0.9
                explanation += "brief test name, "
            
            # Rule 6: Positive bias for well-structured tests
            if test_input is not None and expected is not None and category in ["basic", "edge", "boundary"]:
                confidence = min(confidence * 1.1, 0.95)  # Boost confidence
                explanation += "well-structured test case"
            
            explanation += f" (confidence: {confidence:.2f})"
            
            fallback_validations.append({
                "test_index": i,
                "is_valid": is_valid,
                "invalidity_reason": invalidity_reason,
                "confidence": round(confidence, 3),
                "explanation": explanation,
                "generation_method": "heuristic_fallback"
            })
        
        logger.info(f"Generated {len(fallback_validations)} heuristic validations")
        return fallback_validations
    
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
    
    def _extract_strategic_feedback(
        self, 
        problem: Dict[str, Any], 
        solutions: List[Dict], 
        test_cases: List[Dict], 
        validations: List[Dict]
    ) -> Dict[str, Any]:
        """
        STAR Phase 2: Extract strategic feedback from validation process.
        Provides information for Strategic Oracle to improve problem selection.
        """
        try:
            problem_id = problem.get('problem_id', 'unknown')
            
            # 1. Validation confidence assessment
            validation_scores = [v.get('confidence', 0.5) for v in validations if isinstance(v, dict)]
            avg_validation_confidence = sum(validation_scores) / len(validation_scores) if validation_scores else 0.5
            validation_consistency = 1.0 - (max(validation_scores) - min(validation_scores)) if len(validation_scores) > 1 else 1.0
            
            # 2. Problem difficulty estimation from validation patterns
            invalid_test_ratio = sum(1 for v in validations if not v.get('is_valid', True)) / len(validations) if validations else 0
            difficulty_from_invalidity = min(1.0, invalid_test_ratio * 2.0)  # High invalidity suggests difficult problem
            
            # 3. Test quality assessment
            test_quality_indicators = {
                'has_edge_cases': any('empty' in str(tc).lower() or '[]' in str(tc) for tc in test_cases),
                'has_boundary_values': any(any(indicator in str(tc).lower() for indicator in ['max', 'min', 'limit']) for tc in test_cases),
                'input_diversity': len(set(str(tc.get('input', '')) for tc in test_cases)) / len(test_cases) if test_cases else 0
            }
            test_quality_score = sum(test_quality_indicators.values()) / len(test_quality_indicators)
            
            # 4. Learning signal strength
            clear_validations = sum(1 for v in validations if v.get('confidence', 0) > 0.7 or v.get('confidence', 0) < 0.3)
            learning_signal_strength = clear_validations / len(validations) if validations else 0
            
            # 5. Strategic recommendations
            recommendations = []
            if avg_validation_confidence < 0.5:
                recommendations.append("improve_test_quality")
            if difficulty_from_invalidity > 0.6:
                recommendations.append("focus_on_this_difficulty_level")
            if test_quality_score < 0.4:
                recommendations.append("generate_better_edge_cases")
            if learning_signal_strength > 0.7:
                recommendations.append("good_learning_signal")
            
            # Cache problem difficulty estimate
            combined_difficulty = (difficulty_from_invalidity + (1.0 - avg_validation_confidence)) / 2.0
            self.problem_difficulty_estimates[problem_id] = combined_difficulty
            
            # Record in history
            feedback_record = {
                'problem_id': problem_id,
                'validation_confidence': avg_validation_confidence,
                'problem_difficulty': combined_difficulty,
                'test_quality': test_quality_score,
                'learning_signal': learning_signal_strength
            }
            self.verification_confidence_history.append(feedback_record)
            
            return {
                'validation_confidence': avg_validation_confidence,
                'validation_consistency': validation_consistency,
                'problem_difficulty_estimate': combined_difficulty,
                'test_quality_score': test_quality_score,
                'learning_signal_strength': learning_signal_strength,
                'strategic_recommendations': recommendations,
                'oracle_feedback': {
                    'increase_uncertainty_weight': combined_difficulty > 0.7,
                    'increase_quality_weight': test_quality_score < 0.4,
                    'problem_is_good_candidate': learning_signal_strength > 0.6 and test_quality_score > 0.5
                }
            }
            
        except Exception as e:
            logger.warning(f"Error extracting strategic feedback: {e}")
            return {
                'validation_confidence': 0.5,
                'problem_difficulty_estimate': 0.5,
                'test_quality_score': 0.5,
                'learning_signal_strength': 0.5,
                'strategic_recommendations': [],
                'oracle_feedback': {}
            }
    
    def get_strategic_analytics(self) -> Dict[str, Any]:
        """Get strategic analytics from verification history for oracle feedback."""
        if not self.verification_confidence_history:
            return {'total_verifications': 0}
        
        recent_verifications = self.verification_confidence_history[-20:]  # Last 20
        
        return {
            'total_verifications': len(self.verification_confidence_history),
            'recent_avg_confidence': sum(v['validation_confidence'] for v in recent_verifications) / len(recent_verifications),
            'recent_avg_difficulty': sum(v['problem_difficulty'] for v in recent_verifications) / len(recent_verifications),
            'recent_avg_test_quality': sum(v['test_quality'] for v in recent_verifications) / len(recent_verifications),
            'cached_difficulty_estimates': len(self.problem_difficulty_estimates),
            'strategic_recommendations_count': {
                'improve_test_quality': sum(1 for v in recent_verifications if 'improve_test_quality' in str(v)),
                'focus_on_difficulty': sum(1 for v in recent_verifications if 'focus_on_this_difficulty_level' in str(v)),
                'good_learning_signal': sum(1 for v in recent_verifications if 'good_learning_signal' in str(v))
            }
        }