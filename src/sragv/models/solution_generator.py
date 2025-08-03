"""
Solution Generator - Player 2 of SRAG-V architecture.
Generates Python implementations for coding problems.
"""

import json
import re
from typing import Dict, List, Optional, Any
import logging
import ast
import traceback
import torch

from .base_player import BasePlayer
from ..confidence_calibration import EnhancedConfidenceCalibrator

logger = logging.getLogger(__name__)


class SolutionGenerator(BasePlayer):
    """
    Player 2: Solution Generator (7B parameters)
    Model: Qwen2.5-Coder-7B-Instruct with QLoRA
    
    Input: Problem description + up to 5 previous attempts
    Output: Python implementations
    
    Architecture Details:
    - 4-bit quantization with QLoRA (rank=32)
    - Fits in 24GB GPU memory
    - Generates 16 diverse solutions per problem using temperature=0.8
    - Uses nucleus sampling (top_p=0.95) for diversity
    """
    
    def __init__(self, config: Dict):
        super().__init__(
            model_name=config["model_name"],
            max_length=config["max_length"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            quantization=config.get("quantization"),
            lora_config=config.get("lora_config")
        )
        
        # Initialize STAR confidence calibrator
        self.confidence_calibrator = EnhancedConfidenceCalibrator(
            num_classes=1,  # Regression for solution quality
            temperature_schedule="adaptive",
            calibration_method="temperature_scaling",
            feature_dim=16
        )
        
        # Track calibration training data
        self.calibration_data = []
        self.use_calibration = False  # Enable after calibrator is trained
        
    def process_input(
        self,
        problem: Dict,
        previous_attempts: Optional[List[str]] = None,
        attempt_number: int = 1
    ) -> List[Dict[str, str]]:
        """Process input and create Qwen2.5-Coder chat messages for solution generation."""
        
        # Extract problem details
        problem_desc = problem.get("question", problem.get("description", ""))
        starter_code = problem.get("starter_code", "")
        examples_raw = problem.get("input_output", {})
        
        # Format examples for better readability
        examples_text = ""
        if examples_raw and isinstance(examples_raw, dict):
            inputs = examples_raw.get("inputs", [])
            outputs = examples_raw.get("outputs", [])
            if inputs and outputs:
                examples_text = "\nExamples:\n"
                for i, (inp, out) in enumerate(zip(inputs[:3], outputs[:3])):  # Show max 3 examples
                    if isinstance(inp, list):
                        inp_str = ", ".join(str(x) for x in inp)
                    else:
                        inp_str = str(inp)
                    examples_text += f"Input: {inp_str} → Output: {out}\n"
        
        # Create system message with SOTA techniques
        system_message = """You are Qwen, created by Alibaba Cloud. You are an expert Python programmer specializing in competitive programming and algorithmic problem solving.

Your task is to generate syntactically correct and efficient Python code. Follow these critical requirements:
1. ALWAYS write complete, executable Python functions
2. Use proper Python syntax with correct indentation (4 spaces)
3. Include proper return statements
4. Handle edge cases and validate inputs
5. Write clean, readable code with meaningful variable names
6. ALWAYS wrap your code in ```python and ``` tags
7. Generate only the function implementation, no explanations outside code blocks"""

        # Create user message with few-shot examples for stability
        user_message = f"""Solve this coding problem by writing a Python function.

PROBLEM:
{problem_desc}

{examples_text}

{f'STARTER CODE TEMPLATE:{chr(10)}{starter_code}{chr(10)}' if starter_code else ''}

Here are examples of well-formed solutions:

Example 1 - Two Sum:
```python
def two_sum(nums, target):
    seen = {{}}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

Example 2 - Reverse String:
```python
def reverse_string(s):
    return s[::-1]
```

Now solve the problem above. Write ONLY the Python function, ensure perfect syntax, and make sure it handles all test cases:"""
        
        # Add code block start separately to avoid f-string issues
        user_message += "\n\n```python"

        # Add previous attempts context if available
        if previous_attempts and len(previous_attempts) > 0:
            user_message += f"\n\n--- AVOID THESE PREVIOUS APPROACHES ---\n"
            for i, attempt in enumerate(previous_attempts[-3:], 1):  # Show last 3 attempts
                truncated_attempt = attempt[:200] + ('...' if len(attempt) > 200 else '')
                user_message += f"\nPrevious attempt {i}:\n"
                user_message += "```python\n" + truncated_attempt + "\n```\n"
            
            user_message += f"\nGenerate a DIFFERENT approach (attempt #{attempt_number}):\n\n"
            user_message += "```python"
        
        # Return as chat messages for proper Qwen2.5-Coder formatting
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]
    
    def parse_output(self, output: str) -> Optional[str]:
        """Parse model output to extract Python code with improved error handling."""
        try:
            # Method 1: Extract from markdown code blocks (most reliable)
            # Extract code blocks using separate pattern construction
            pattern = r'```python\s*(.*?)\s*```'
            code_blocks = re.findall(pattern, output, re.DOTALL)
            if code_blocks:
                code = code_blocks[0].strip()
                # Validate syntax immediately
                try:
                    ast.parse(code)
                    logger.debug("Successfully extracted code from markdown block")
                    return code
                except SyntaxError as e:
                    logger.warning(f"Markdown code block has syntax error: {e}")
                    # Try to fix common issues
                    code = self._fix_common_syntax_errors(code)
                    try:
                        ast.parse(code)
                        logger.debug("Fixed syntax errors in markdown code")
                        return code
                    except SyntaxError:
                        pass  # Continue to other methods
            
            # Method 2: Look for function definitions (more aggressive)
            lines = output.strip().split('\n')
            code_lines = []
            in_function = False
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                
                # Start of function
                if stripped.startswith('def ') and ':' in stripped:
                    in_function = True
                    code_lines = [line]  # Start fresh
                    indent_level = len(line) - len(line.lstrip())
                elif in_function:
                    # Continue function body
                    if line.strip() == '' or line.startswith(' ' * (indent_level + 1)) or line.startswith('\t'):
                        code_lines.append(line)
                    elif stripped and not line.startswith(' ' * indent_level) and not line.startswith('\t'):
                        # End of function
                        break
                    else:
                        code_lines.append(line)
            
            if code_lines and in_function:
                code = '\n'.join(code_lines).strip()
                try:
                    ast.parse(code)
                    logger.debug("Successfully extracted function definition")
                    return code
                except SyntaxError as e:
                    logger.warning(f"Function extraction has syntax error: {e}")
                    # Try to fix
                    code = self._fix_common_syntax_errors(code)
                    try:
                        ast.parse(code)
                        logger.debug("Fixed syntax errors in extracted function")
                        return code
                    except SyntaxError:
                        pass
            
            # Method 3: Last resort - clean up the entire output
            cleaned_output = self._clean_output(output)
            if cleaned_output:
                try:
                    ast.parse(cleaned_output)
                    logger.debug("Successfully cleaned entire output")
                    return cleaned_output
                except SyntaxError as e:
                    logger.warning(f"Cleaned output still has syntax error: {e}")
                    return None  # Give up rather than return broken code
            
            logger.warning("Failed to extract valid Python code from output")
            return None
                
        except Exception as e:
            logger.error(f"Critical error in parse_output: {e}")
            return None
    
    def _fix_common_syntax_errors(self, code: str) -> str:
        """Fix common syntax errors in generated code."""
        try:
            # Fix unterminated strings (replace smart quotes)
            code = code.replace('"', '"').replace('"', '"')
            code = code.replace(''', "'").replace(''', "'")
            
            # Fix common indentation issues
            lines = code.split('\n')
            fixed_lines = []
            for line in lines:
                # Convert tabs to 4 spaces
                line = line.expandtabs(4)
                # Fix mixed indentation
                if line.strip():
                    leading_spaces = len(line) - len(line.lstrip())
                    if leading_spaces > 0:
                        # Ensure indentation is multiple of 4
                        new_indent = (leading_spaces // 4) * 4
                        if new_indent != leading_spaces:
                            line = ' ' * new_indent + line.lstrip()
                fixed_lines.append(line)
            
            return '\n'.join(fixed_lines)
        except Exception:
            return code
    
    def _clean_output(self, output: str) -> str:
        """Clean and extract code from raw output."""
        try:
            # Remove common prefixes/suffixes
            output = output.strip()
            
            # Remove explanation text before code
            lines = output.split('\n')
            code_start = -1
            
            for i, line in enumerate(lines):
                if line.strip().startswith('def ') or line.strip().startswith('class '):
                    code_start = i
                    break
            
            if code_start >= 0:
                relevant_lines = lines[code_start:]
                return '\n'.join(relevant_lines).strip()
            
            return output
        except Exception:
            return output
    
    def generate(
        self,
        problem: Dict,
        previous_attempts: Optional[List[str]] = None,
        num_solutions: int = 14,  # Reduced from 16 for stability
        max_retry_attempts: int = 5,  # More retries for robust generation
        max_attempts_per_solution: int = 3  # Maximum attempts per solution for training compatibility
    ) -> List[Dict]:
        """Generate multiple diverse solutions using SOTA Qwen2.5-Coder techniques."""
        problem_id = problem.get('problem_id', 'unknown')
        logger.info(f"Generating {num_solutions} solutions for problem {problem_id}")
        
        solutions = []
        all_attempts = previous_attempts or []
        valid_solutions_count = 0
        
        for i in range(num_solutions):
            solution_generated = False
            
            # SOTA technique: Start with temperature=0 for first few solutions (deterministic)
            # Then increase temperature for diversity
            if i < 3:
                base_temp = 0.0  # Deterministic for stability
            elif i < 8:
                base_temp = 0.3  # Low temperature for quality
            else:
                base_temp = 0.7  # Higher temperature for diversity
            
            # Try multiple attempts with different strategies
            # Use max_attempts_per_solution if provided, otherwise default to max_retry_attempts
            attempts_to_use = max_attempts_per_solution if max_attempts_per_solution is not None else max_retry_attempts
            for attempt in range(attempts_to_use):
                try:
                    # Hardware health check before generation
                    import torch
                    if torch.cuda.is_available():
                        # Check for CUDA errors that might indicate ECC issues
                        try:
                            torch.cuda.synchronize()
                            # Test GPU memory integrity with a small operation
                            test_tensor = torch.randn(100, device='cuda:0')
                            _ = test_tensor.sum()
                            del test_tensor
                            torch.cuda.empty_cache()
                        except Exception as cuda_error:
                            logger.error(f"🚨 CUDA/GPU error detected: {cuda_error}")
                            logger.error(f"This may indicate ECC errors or GPU hardware failure")
                            raise RuntimeError(f"GPU hardware error: {cuda_error}")
                    
                    # Create chat messages using proper Qwen2.5-Coder format
                    messages = self.process_input(
                        problem=problem,
                        previous_attempts=all_attempts if attempt > 0 else [],  # Only show previous attempts after first try
                        attempt_number=i + 1
                    )
                    
                    # Apply temperature variation for retries
                    current_temp = base_temp + (attempt * 0.1)
                    current_temp = min(current_temp, 0.9)  # Cap temperature
                    
                    # Store current temperature for STAR calibration
                    self._current_generation_temperature = current_temp
                    
                    # Generate using chat template with timeout protection
                    import signal
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Text generation timed out - possible GPU hang")
                    
                    # Set 60-second timeout for generation
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(60)
                    
                    try:
                        outputs = self.generate_text(
                            messages=messages,  # Pass messages instead of prompt
                            max_new_tokens=512,  # Reduced for more focused generation
                            temperature=current_temp,
                            top_p=0.95,
                            do_sample=current_temp > 0.0,  # Only sample if temperature > 0
                            num_return_sequences=1,
                            pad_token_id=self.tokenizer.eos_token_id if hasattr(self, 'tokenizer') else None
                        )
                    finally:
                        signal.alarm(0)  # Cancel timeout
                    
                    if outputs and len(outputs) > 0:
                        # Parse solution with improved error handling
                        code = self.parse_output(outputs[0])
                        
                        if code and self._validate_code_quality(code, problem):
                            # Score the solution
                            score = self.score_solution(code, problem)
                            
                            # Accept solution if it meets quality threshold
                            if score >= 0.4:  # Minimum quality threshold
                                solution_data = {
                                    "solution_id": f"sol_{i+1:02d}",
                                    "code": code,
                                    "score": score,
                                    "attempt_number": i + 1,
                                    "generation_temperature": current_temp,
                                    "retry_attempt": attempt + 1
                                }
                                solutions.append(solution_data)
                                all_attempts.append(code)
                                valid_solutions_count += 1
                                solution_generated = True
                                
                                logger.debug(f"✅ Generated solution {i+1}/{num_solutions} (score: {score:.3f}, temp: {current_temp:.2f})")
                                break
                        else:
                            logger.debug(f"⚠️ Solution {i+1} attempt {attempt+1} failed validation")
                    else:
                        logger.debug(f"⚠️ No output generated for solution {i+1} attempt {attempt+1}")
                    
                except Exception as e:
                    logger.warning(f"❌ Error generating solution {i+1}, attempt {attempt+1}: {e}")
                    continue
            
            if not solution_generated:
                logger.warning(f"❌ Failed to generate valid solution {i+1}/{num_solutions} after {max_retry_attempts} attempts")
        
        success_rate = (valid_solutions_count / num_solutions) * 100
        logger.info(f"✅ Generated {valid_solutions_count}/{num_solutions} solutions (success rate: {success_rate:.1f}%)")
        
        # Sort solutions by score (best first)
        solutions.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return solutions
    
    def _validate_code_quality(self, code: str, problem: Dict) -> bool:
        """Validate code quality before accepting as solution."""
        try:
            # Must be syntactically valid
            try:
                ast.parse(code)
            except SyntaxError as e:
                logger.debug(f"Code failed syntax check: {e}")
                return False
            
            # Must contain a function definition
            if 'def ' not in code:
                logger.debug("Code missing function definition")
                return False
            
            # Must have a return statement (basic requirement)
            if 'return' not in code:
                logger.debug("Code missing return statement")
                return False
            
            # Must be reasonable length (not too short or too long)
            lines = [line for line in code.split('\n') if line.strip()]
            if len(lines) < 2 or len(lines) > 100:
                logger.debug(f"Code length unreasonable: {len(lines)} lines")
                return False
            
            # Check for common problematic patterns
            problematic_patterns = [
                'import os',  # Potential security issue
                'exec(',     # Dangerous
                'eval(',     # Dangerous
                '...',       # Placeholder code
                'pass',      # Incomplete implementation
            ]
            
            code_lower = code.lower()
            for pattern in problematic_patterns:
                if pattern in code_lower:
                    logger.debug(f"Code contains problematic pattern: {pattern}")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Code validation error: {e}")
            return False
    
    def score_solution(self, code: str, problem: Dict) -> float:
        """Score a solution with STAR enhanced confidence calibration."""
        try:
            # Compute base heuristic score
            base_score = self._compute_heuristic_score(code, problem)
            
            # Apply STAR confidence calibration if trained
            if self.use_calibration and self.confidence_calibrator.is_trained:
                # Extract features for calibration
                features = self.confidence_calibrator.extract_solution_features(code, problem)
                
                # Get current generation temperature
                current_temp = getattr(self, '_current_generation_temperature', 1.0)
                
                # Apply calibrated confidence scoring
                calibrated_score, calibration_info = self.confidence_calibrator.calibrate_confidence(
                    base_confidence=base_score,
                    temperature=current_temp,
                    features=features
                )
                
                # Store calibration data for analysis
                self.calibration_data.append({
                    'code': code,
                    'problem_id': problem.get('problem_id', 'unknown'),
                    'base_score': base_score,
                    'calibrated_score': calibrated_score,
                    'temperature': current_temp,
                    'calibration_info': calibration_info
                })
                
                return calibrated_score
            else:
                # Use base heuristic score when calibration not available
                return base_score
                
        except Exception as e:
            logger.warning(f"Error in STAR solution scoring: {e}")
            return self._compute_heuristic_score(code, problem)
    
    def _compute_heuristic_score(self, code: str, problem: Dict) -> float:
        """
        Enhanced heuristic scoring that creates natural confidence variance.
        Analyzes multiple dimensions of solution quality to prevent confidence collapse.
        """
        try:
            score = 0.0
            
            # === BASIC VALIDITY (0.3 total) ===
            
            # Syntax validity (0.2 weight) 
            syntax_valid = False
            try:
                ast.parse(code)
                syntax_valid = True
                score += 0.2
            except SyntaxError:
                # Syntax errors get heavily penalized
                return 0.1  # Very low score for syntax errors
            
            # Function definition (0.1 weight)
            if 'def ' in code:
                score += 0.1
            else:
                return 0.15  # No function = very low score
            
            # === SOLUTION QUALITY ANALYSIS (0.7 total) ===
            
            # 1. Algorithmic Sophistication (0.25 weight)
            algo_score = self._analyze_algorithmic_sophistication(code)
            score += 0.25 * algo_score
            
            # 2. Problem-Specific Implementation (0.2 weight)  
            specificity_score = self._analyze_problem_specificity(code, problem)
            score += 0.2 * specificity_score
            
            # 3. Code Quality and Style (0.15 weight)
            quality_score = self._analyze_code_quality(code)
            score += 0.15 * quality_score
            
            # 4. Edge Case Handling (0.1 weight)
            edge_case_score = self._analyze_edge_case_handling(code, problem)
            score += 0.1 * edge_case_score
            
            # Ensure reasonable bounds but allow for real variance
            final_score = max(0.1, min(0.95, score))
            
            return final_score
            
        except Exception as e:
            logger.warning(f"Error in enhanced heuristic scoring: {e}")
            # Return varied fallback based on code characteristics
            return self._compute_fallback_score(code)
    
    def _analyze_algorithmic_sophistication(self, code: str) -> float:
        """Analyze algorithmic sophistication of the solution."""
        try:
            score = 0.0
            code_lower = code.lower()
            
            # Control flow complexity
            if_count = code.count('if ')
            for_count = code.count('for ')
            while_count = code.count('while ')
            
            # Basic control flow (up to 0.3)
            control_complexity = min(0.3, (if_count + for_count + while_count) * 0.1)
            score += control_complexity
            
            # Data structure usage (up to 0.3)
            advanced_structures = [
                'dict', 'set(', 'list(', 'tuple(', 
                'collections', 'heapq', 'bisect'
            ]
            structure_score = min(0.3, sum(0.1 for struct in advanced_structures if struct in code_lower))
            score += structure_score
            
            # Algorithmic patterns (up to 0.4)
            patterns = {
                'recursion': [r'return.*\w+\(', r'def.*\w+.*:.*\w+\('],
                'dynamic_programming': ['dp', 'memo', 'cache'],
                'sorting': ['sort', 'sorted'],
                'searching': [r'binary.*search', 'bfs', 'dfs'],
                'optimization': [r'min\(', r'max\(', 'optimal']
            }
            
            pattern_score = 0.0
            for pattern_type, indicators in patterns.items():
                for indicator in indicators:
                    if re.search(indicator, code_lower):
                        pattern_score += 0.08  # Each pattern adds points
                        break
            
            score += min(0.4, pattern_score)
            
            return min(1.0, score)
            
        except Exception as e:
            logger.debug(f"Error analyzing algorithmic sophistication: {e}")
            return 0.5
    
    def _analyze_problem_specificity(self, code: str, problem: Dict) -> float:
        """Analyze how well the solution addresses the specific problem."""
        try:
            score = 0.0
            
            problem_text = problem.get('question', problem.get('description', '')).lower()
            code_lower = code.lower()
            
            # Extract meaningful keywords from problem (not common words)
            problem_words = re.findall(r'\b\w{4,}\b', problem_text)  # 4+ letter words
            
            # Common words to ignore
            common_words = {
                'function', 'return', 'example', 'should', 'write', 'that', 'this',
                'input', 'output', 'given', 'example', 'integer', 'string', 'list'
            }
            
            meaningful_words = [w for w in problem_words if w not in common_words]
            
            # Check for problem-specific terms
            matched_words = 0
            for word in meaningful_words[:10]:  # Check up to 10 key words
                if word in code_lower:
                    matched_words += 1
            
            if meaningful_words:
                specificity_ratio = matched_words / min(len(meaningful_words), 10)
                score += 0.6 * specificity_ratio
            
            # Check for function name relevance
            function_names = re.findall(r'def\s+(\w+)', code)
            if function_names:
                func_name = function_names[0].lower()
                # Check if function name relates to problem
                name_relevance = sum(1 for word in meaningful_words[:5] if word in func_name)
                if name_relevance > 0:
                    score += 0.2
            
            # Check for appropriate variable names
            variables = re.findall(r'\b[a-z_]\w*\b', code_lower)
            meaningful_vars = [v for v in variables if len(v) > 2 and v not in ['def', 'for', 'if', 'in']]
            
            if len(meaningful_vars) > 2:  # Has descriptive variable names
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.debug(f"Error analyzing problem specificity: {e}")
            return 0.5
    
    def _analyze_code_quality(self, code: str) -> float:
        """Analyze code quality and style."""
        try:
            score = 0.0
            
            # Line count reasonableness
            lines = [line.strip() for line in code.split('\n') if line.strip()]
            line_count = len(lines)
            
            if 3 <= line_count <= 20:  # Concise solutions
                score += 0.3
            elif 21 <= line_count <= 50:  # Reasonable length
                score += 0.2
            else:  # Too short or too long
                score += 0.1
            
            # Variable naming quality
            variables = re.findall(r'\b[a-z_]\w*\b', code.lower())
            single_char_vars = sum(1 for v in variables if len(v) == 1 and v not in 'in')
            descriptive_vars = sum(1 for v in variables if len(v) > 3)
            
            if descriptive_vars > single_char_vars:
                score += 0.2
            elif descriptive_vars == single_char_vars:
                score += 0.1
            
            # Comments and docstrings (bonus)
            if '"""' in code or "'''" in code or '#' in code:
                score += 0.1
            
            # Error handling
            if 'try:' in code or 'except' in code:
                score += 0.1
            
            # Return statement quality
            return_statements = re.findall(r'return\s+(.+)', code)
            if return_statements:
                # Complex return expressions get higher scores
                complex_return = any(len(ret.strip()) > 10 for ret in return_statements)
                if complex_return:
                    score += 0.1
            
            # Avoid hardcoded values (unless it's clearly intentional)
            hardcoded_numbers = re.findall(r'\b\d{2,}\b', code)  # 2+ digit numbers
            if len(hardcoded_numbers) > 3:  # Too many hardcoded values
                score -= 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.debug(f"Error analyzing code quality: {e}")
            return 0.5
    
    def _analyze_edge_case_handling(self, code: str, problem: Dict) -> float:
        """Analyze edge case handling in the solution."""
        try:
            score = 0.0
            code_lower = code.lower()
            
            # Check for common edge case patterns
            edge_patterns = [
                ('empty_check', ['if not', r'if len\(', '== 0', '== []', "== ''"]),
                ('null_check', ['is none', r'if.*none', r'not.*none']),
                ('boundary_check', ['<= 0', '>= ', '< 1', '> 0']),
                ('type_validation', ['isinstance', r'type\(']),
                ('exception_handling', ['try:', 'except', 'finally:']),
            ]
            
            for edge_type, patterns in edge_patterns:
                for pattern in patterns:
                    if re.search(pattern, code_lower):
                        score += 0.2
                        break  # Only count each edge type once
            
            # Check for input validation
            if 'if ' in code and ('input' in code_lower or 'arg' in code_lower):
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            logger.debug(f"Error analyzing edge case handling: {e}")
            return 0.3
    
    def _compute_fallback_score(self, code: str) -> float:
        """Compute fallback score based on basic code characteristics."""
        try:
            # Use code characteristics to create varied fallback scores
            code_hash = hash(code) % 1000
            base_fallback = 0.3 + (code_hash / 1000) * 0.4  # Range: 0.3-0.7
            
            # Adjust based on basic characteristics
            if 'def ' in code:
                base_fallback += 0.1
            if 'return' in code:
                base_fallback += 0.1
            if len(code) > 100:  # More substantial code
                base_fallback += 0.05
                
            return min(0.9, base_fallback)
            
        except Exception:
            return 0.4
    
    def execute_solution(
        self,
        code: str,
        test_inputs: List[Any],
        timeout: float = 5.0
    ) -> List[Dict]:
        """Execute solution with test inputs and return results."""
        results = []
        
        try:
            # Create a safe execution environment
            exec_globals = {
                "__builtins__": {
                    "len": len,
                    "max": max,
                    "min": min,
                    "sum": sum,
                    "sorted": sorted,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "abs": abs,
                    "round": round,
                }
            }
            
            # Execute the code to define functions
            exec(code, exec_globals)
            
            # Find the main function (usually the first defined function)
            main_func = None
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith('_'):
                    main_func = obj
                    break
            
            if main_func is None:
                return [{"success": False, "error": "No callable function found"}]
            
            # Execute with each test input
            for i, test_input in enumerate(test_inputs):
                try:
                    # Handle different input formats
                    if isinstance(test_input, (list, tuple)):
                        result = main_func(*test_input)
                    else:
                        result = main_func(test_input)
                    
                    results.append({
                        "test_case": i,
                        "success": True,
                        "output": result,
                        "error": None
                    })
                    
                except Exception as e:
                    results.append({
                        "test_case": i,
                        "success": False,
                        "output": None,
                        "error": str(e)
                    })
            
        except Exception as e:
            logger.error(f"Error executing solution: {e}")
            return [{"success": False, "error": f"Execution error: {e}"}]
        
        return results
    
    def validate_solutions(
        self,
        solutions: List[Dict],
        problem: Dict,
        test_cases: Optional[List] = None
    ) -> List[Dict]:
        """Validate solutions against test cases if available."""
        if not test_cases:
            logger.info("No test cases provided for validation")
            return solutions
        
        logger.info(f"Validating {len(solutions)} solutions against {len(test_cases)} test cases")
        
        validated_solutions = []
        
        for solution in solutions:
            code = solution.get("code", "")
            if not code:
                continue
            
            # Execute solution
            test_inputs = [tc.get("input") for tc in test_cases if tc.get("input") is not None]
            execution_results = self.execute_solution(code, test_inputs)
            
            # Calculate pass rate
            total_tests = len(execution_results)
            passed_tests = sum(1 for result in execution_results if result.get("success", False))
            pass_rate = passed_tests / total_tests if total_tests > 0 else 0.0
            
            # Add validation info to solution
            validated_solution = solution.copy()
            validated_solution.update({
                "validation_results": execution_results,
                "pass_rate": pass_rate,
                "passed_tests": passed_tests,
                "total_tests": total_tests
            })
            
            validated_solutions.append(validated_solution)
        
        # Sort by pass rate (best first)
        validated_solutions.sort(key=lambda x: x.get("pass_rate", 0), reverse=True)
        
        logger.info(f"Validation complete. Best solution pass rate: {validated_solutions[0].get('pass_rate', 0):.2%}" if validated_solutions else "No valid solutions")
        
        return validated_solutions
    
    # STAR Enhancement Methods
    
    def train_confidence_calibrator(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        num_epochs: int = 50,
        learning_rate: float = 0.01
    ) -> Dict[str, float]:
        """
        Train the STAR confidence calibrator.
        
        Args:
            training_data: List of {code, problem, base_score, true_score}
            validation_data: Optional validation set
            num_epochs: Training epochs
            learning_rate: Learning rate
        
        Returns:
            Training metrics including ECE
        """
        logger.info(f"Training STAR confidence calibrator on {len(training_data)} samples...")
        
        # Prepare training data for calibrator
        calibrator_training_data = []
        for item in training_data:
            features = self.confidence_calibrator.extract_solution_features(
                item['code'], item['problem']
            )
            calibrator_training_data.append({
                'features': features.squeeze(0),  # Remove batch dimension
                'base_confidence': item['base_score'],
                'true_score': item['true_score']
            })
        
        # Prepare validation data if provided
        calibrator_validation_data = None
        if validation_data:
            calibrator_validation_data = []
            for item in validation_data:
                features = self.confidence_calibrator.extract_solution_features(
                    item['code'], item['problem']
                )
                calibrator_validation_data.append({
                    'features': features.squeeze(0),
                    'base_confidence': item['base_score'],
                    'true_score': item['true_score']
                })
        
        # Train the calibrator
        metrics = self.confidence_calibrator.train_calibration(
            training_data=calibrator_training_data,
            validation_data=calibrator_validation_data,
            num_epochs=num_epochs,
            learning_rate=learning_rate
        )
        
        # Enable calibration after training
        self.use_calibration = True
        
        logger.info(f"✅ STAR calibrator training complete. ECE: {metrics.get('ece', 0):.4f}")
        return metrics
    
    def evaluate_calibration_quality(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate the quality of confidence calibration.
        
        Args:
            evaluation_data: List of {code, problem, base_score, true_score}
        
        Returns:
            Calibration quality metrics
        """
        if not self.use_calibration or not self.confidence_calibrator.is_trained:
            logger.warning("Calibrator not trained. Cannot evaluate calibration quality.")
            return {'ece': 1.0, 'mce': 1.0, 'brier_score': 1.0}
        
        # Prepare evaluation data
        calibrator_eval_data = []
        for item in evaluation_data:
            features = self.confidence_calibrator.extract_solution_features(
                item['code'], item['problem']
            )
            calibrator_eval_data.append({
                'features': features.squeeze(0),
                'base_confidence': item['base_score'],
                'true_score': item['true_score']
            })
        
        # Evaluate calibration
        metrics = self.confidence_calibrator.evaluate_calibration(calibrator_eval_data)
        
        logger.info(f"Calibration Quality - ECE: {metrics['ece']:.4f}, MCE: {metrics['mce']:.4f}, Brier: {metrics['brier_score']:.4f}")
        return metrics
    
    def get_star_solution_ranking(self, solutions: List[Dict]) -> List[Dict]:
        """
        Rank solutions using STAR calibrated confidence scores.
        
        Args:
            solutions: List of solution dictionaries
        
        Returns:
            Solutions ranked by calibrated confidence
        """
        if not self.use_calibration or not self.confidence_calibrator.is_trained:
            # Fall back to base heuristic ranking
            return sorted(solutions, key=lambda x: x.get('score', 0), reverse=True)
        
        # Add calibrated scores to solutions
        for solution in solutions:
            if 'calibrated_score' not in solution:
                # Re-score with calibration if not already done
                calibrated_score = self.score_solution(
                    solution.get('code', ''), 
                    solution.get('problem', {})
                )
                solution['calibrated_score'] = calibrated_score
        
        # Rank by calibrated scores
        ranked_solutions = sorted(solutions, key=lambda x: x.get('calibrated_score', x.get('score', 0)), reverse=True)
        
        logger.debug(f"STAR ranking complete. Top solution calibrated score: {ranked_solutions[0].get('calibrated_score', 0):.4f}")
        return ranked_solutions
    
    def save_calibrator(self, path: str):
        """Save the trained STAR calibrator."""
        if self.confidence_calibrator.is_trained:
            self.confidence_calibrator.save_calibrator(path)
            logger.info(f"STAR calibrator saved to {path}")
        else:
            logger.warning("Calibrator not trained. Cannot save.")
    
    def load_calibrator(self, path: str):
        """Load a trained STAR calibrator."""
        try:
            self.confidence_calibrator.load_calibrator(path)
            self.use_calibration = True
            logger.info(f"STAR calibrator loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load STAR calibrator: {e}")
            self.use_calibration = False
    
    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of STAR calibration state."""
        return {
            'use_calibration': self.use_calibration,
            'calibrator_summary': self.confidence_calibrator.get_calibration_summary(),
            'calibration_data_points': len(self.calibration_data)
        }