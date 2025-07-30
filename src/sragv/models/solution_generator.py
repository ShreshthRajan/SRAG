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

from .base_player import BasePlayer

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
        
    def process_input(
        self,
        problem: Dict,
        previous_attempts: Optional[List[str]] = None,
        attempt_number: int = 1
    ) -> str:
        """Process input and create a prompt for solution generation."""
        
        problem_desc = problem.get("description", "")
        function_sig = problem.get("function_signature", "")
        examples = problem.get("examples", "")
        constraints = problem.get("constraints", "")
        
        prompt = f"""You are an expert Python programmer. Solve the following coding problem with a correct and efficient implementation.

PROBLEM:
{problem_desc}

{f'FUNCTION SIGNATURE: {function_sig}' if function_sig else ''}

{f'EXAMPLES: {examples}' if examples else ''}

{f'CONSTRAINTS: {constraints}' if constraints else ''}"""

        # Add previous attempts if available
        if previous_attempts and len(previous_attempts) > 0:
            prompt += f"\n\nPREVIOUS ATTEMPTS (avoid these approaches):\n"
            for i, attempt in enumerate(previous_attempts[-5:], 1):  # Show last 5 attempts
                prompt += f"\nAttempt {i}:\n```python\n{attempt}\n```\n"
            
            prompt += f"\nGenerate a NEW solution (attempt #{attempt_number}) that uses a different approach:"
        else:
            prompt += "\n\nProvide your solution:"
        
        prompt += """

Requirements:
1. Write clean, readable Python code
2. Include proper error handling where appropriate
3. Add brief comments for complex logic
4. Ensure the solution handles all edge cases
5. Optimize for correctness first, then efficiency

```python
"""
        
        return prompt
    
    def parse_output(self, output: str) -> Optional[str]:
        """Parse model output to extract Python code."""
        try:
            # Try to extract code from markdown code blocks
            code_blocks = re.findall(r'```python(.*?)```', output, re.DOTALL)
            if code_blocks:
                code = code_blocks[0].strip()
            else:
                # Try to extract code without markdown
                # Look for function definitions
                lines = output.strip().split('\n')
                code_lines = []
                in_code = False
                
                for line in lines:
                    if line.strip().startswith('def ') or in_code:
                        code_lines.append(line)
                        in_code = True
                    elif in_code and line.strip() == '':
                        code_lines.append(line)
                    elif in_code and not line.strip().startswith(' ') and not line.strip().startswith('\t'):
                        break
                
                if code_lines:
                    code = '\n'.join(code_lines).strip()
                else:
                    # Last resort: take the entire output
                    code = output.strip()
            
            # Validate that it's syntactically correct Python
            try:
                ast.parse(code)
                return code
            except SyntaxError as e:
                logger.warning(f"Generated code has syntax error: {e}")
                return code  # Return anyway, might be fixable
                
        except Exception as e:
            logger.warning(f"Failed to parse solution output: {e}")
            return output.strip()  # Return raw output as fallback
    
    def generate(
        self,
        problem: Dict,
        previous_attempts: Optional[List[str]] = None,
        num_solutions: int = 16,
        max_attempts_per_solution: int = 3
    ) -> List[Dict]:
        """Generate multiple diverse solutions for a problem."""
        logger.info(f"Generating {num_solutions} solutions for problem {problem.get('problem_id', 'unknown')}")
        
        solutions = []
        all_attempts = previous_attempts or []
        
        for i in range(num_solutions):
            best_solution = None
            best_score = -1
            
            # Try multiple attempts for each solution to get the best one
            for attempt in range(max_attempts_per_solution):
                try:
                    # Create prompt
                    prompt = self.process_input(
                        problem=problem,
                        previous_attempts=all_attempts,
                        attempt_number=len(all_attempts) + 1
                    )
                    
                    # Generate response with some temperature for diversity
                    temp = self.temperature + (i * 0.05)  # Increase temp for later solutions
                    temp = min(temp, 1.0)  # Cap at 1.0
                    
                    outputs = self.generate_text(
                        prompt=prompt,
                        max_new_tokens=1024,
                        temperature=temp,
                        top_p=self.top_p,
                        num_return_sequences=1
                    )
                    
                    if outputs:
                        # Parse solution
                        code = self.parse_output(outputs[0])
                        if code:
                            # Score the solution
                            score = self.score_solution(code, problem)
                            
                            if score > best_score:
                                best_solution = code
                                best_score = score
                    
                except Exception as e:
                    logger.warning(f"Error generating solution {i+1}, attempt {attempt+1}: {e}")
                    continue
            
            # Add best solution to results
            if best_solution:
                solution_data = {
                    "solution_id": f"sol_{i+1:02d}",
                    "code": best_solution,
                    "score": best_score,
                    "attempt_number": len(all_attempts) + 1,
                    "generation_temperature": temp
                }
                solutions.append(solution_data)
                all_attempts.append(best_solution)
                
                logger.debug(f"Generated solution {i+1}/{num_solutions} (score: {best_score:.3f})")
            else:
                logger.warning(f"Failed to generate solution {i+1}/{num_solutions}")
        
        logger.info(f"Successfully generated {len(solutions)} solutions")
        return solutions
    
    def score_solution(self, code: str, problem: Dict) -> float:
        """Score a solution based on various criteria."""
        try:
            score = 0.0
            
            # Syntax validity (0.3 weight)
            try:
                ast.parse(code)
                score += 0.3
            except SyntaxError:
                pass  # No points for syntax errors
            
            # Function definition present (0.2 weight)
            if 'def ' in code:
                score += 0.2
            
            # Reasonable length (0.1 weight)
            line_count = len([line for line in code.split('\n') if line.strip()])
            if 5 <= line_count <= 50:  # Reasonable function length
                score += 0.1
            
            # Has return statement (0.1 weight)
            if 'return ' in code:
                score += 0.1
            
            # Contains problem-relevant keywords (0.2 weight)
            problem_text = (problem.get("description", "") + " " + 
                          problem.get("examples", "")).lower()
            
            code_lower = code.lower()
            relevant_keywords = []
            
            # Extract potential keywords from problem
            words = re.findall(r'\b\w+\b', problem_text)
            for word in words:
                if len(word) > 3 and word in code_lower:
                    relevant_keywords.append(word)
            
            if relevant_keywords:
                score += 0.2
            
            # Diversity bonus (0.1 weight) - simple heuristic
            # More unique tokens = more diverse
            unique_tokens = len(set(re.findall(r'\b\w+\b', code_lower)))
            if unique_tokens > 10:
                score += 0.1
            
            return score
            
        except Exception as e:
            logger.warning(f"Error scoring solution: {e}")
            return 0.0
    
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