#!/usr/bin/env python3
"""
Strategic selection of 200 diverse problems for Phase 1 redo.
Uses Strategic Oracle principles to ensure algorithmic diversity.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DiverseProblemSelector:
    """Selects diverse problems for Phase 1 training."""
    
    def __init__(self):
        self.algorithm_keywords = {
            'string': ['string', 'substring', 'character', 'text', 'word'],
            'array': ['array', 'list', 'sort', 'search', 'element'],
            'tree': ['tree', 'binary', 'node', 'leaf', 'root'],
            'graph': ['graph', 'edge', 'vertex', 'path', 'cycle'],
            'dynamic_programming': ['dp', 'dynamic', 'optimal', 'subsequence', 'fibonacci'],
            'recursion': ['recursive', 'recursion', 'factorial', 'permutation'],
            'hash': ['hash', 'dictionary', 'map', 'key', 'lookup'],
            'math': ['mathematical', 'prime', 'gcd', 'lcm', 'modulo'],
            'greedy': ['greedy', 'minimum', 'maximum', 'optimal'],
            'backtracking': ['backtrack', 'combination', 'permutation', 'generate']
        }
        
    def analyze_problem_type(self, problem: Dict[str, Any]) -> Dict[str, float]:
        """Analyze what type of algorithms a problem requires."""
        question = problem.get('question', '').lower()
        solution = problem.get('solutions', [''])[0].lower()
        text = question + ' ' + solution
        
        type_scores = {}
        for algo_type, keywords in self.algorithm_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            type_scores[algo_type] = score
            
        return type_scores
    
    def assess_problem_complexity(self, problem: Dict[str, Any]) -> float:
        """Assess problem complexity based on various factors."""
        question = problem.get('question', '')
        solution = problem.get('solutions', [''])[0]
        
        complexity_score = 0.0
        
        # Length indicators
        if len(question) > 200:
            complexity_score += 0.2
        if len(solution) > 100:
            complexity_score += 0.3
            
        # Algorithmic complexity indicators
        complexity_indicators = [
            'optimal', 'efficient', 'O(n)', 'complexity', 'algorithm',
            'dynamic programming', 'recursive', 'backtrack', 'binary search'
        ]
        
        text = (question + ' ' + solution).lower()
        for indicator in complexity_indicators:
            if indicator in text:
                complexity_score += 0.1
                
        # Code complexity (nested loops, conditions)
        nested_patterns = ['for.*for', 'while.*while', 'if.*if.*else']
        for pattern in nested_patterns:
            if re.search(pattern, solution):
                complexity_score += 0.15
                
        return min(1.0, complexity_score)
    
    def compute_diversity_score(self, problem: Dict[str, Any], selected_problems: List[Dict[str, Any]]) -> float:
        """Compute diversity score compared to already selected problems."""
        if not selected_problems:
            return 1.0
            
        current_types = self.analyze_problem_type(problem)
        
        # Calculate distance from already selected problems
        min_distance = float('inf')
        
        for selected in selected_problems:
            selected_types = self.analyze_problem_type(selected)
            
            # Compute cosine distance between type vectors
            current_vec = np.array([current_types.get(t, 0) for t in self.algorithm_keywords.keys()])
            selected_vec = np.array([selected_types.get(t, 0) for t in self.algorithm_keywords.keys()])
            
            # Avoid division by zero
            current_norm = np.linalg.norm(current_vec)
            selected_norm = np.linalg.norm(selected_vec)
            
            if current_norm == 0 or selected_norm == 0:
                distance = 1.0
            else:
                similarity = np.dot(current_vec, selected_vec) / (current_norm * selected_norm)
                distance = 1.0 - similarity
                
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 1.0
    
    def select_diverse_problems(self, all_problems: List[Dict[str, Any]], target_count: int = 200) -> List[Dict[str, Any]]:
        """Select diverse problems using strategic selection."""
        logger.info(f"Selecting {target_count} diverse problems from {len(all_problems)} available")
        
        # First, categorize by difficulty
        easy_problems = [p for p in all_problems if p.get('difficulty', '').lower() in ['easy', 'introductory']]
        medium_problems = [p for p in all_problems if p.get('difficulty', '').lower() in ['medium', 'interview']]
        hard_problems = [p for p in all_problems if p.get('difficulty', '').lower() in ['hard', 'competition']]
        
        logger.info(f"Problems by difficulty: Easy={len(easy_problems)}, Medium={len(medium_problems)}, Hard={len(hard_problems)}")
        
        # Target distribution for diversity - REALISTIC EXACT COUNTS
        target_distribution = {
            'easy': 54,      # Exactly 54 easy (27%)
            'medium': 91,    # Exactly 91 medium (45.5%) - use all available
            'hard': 55       # Exactly 55 hard (27.5%)
        }
        
        logger.info(f"REQUIRED EXACT distribution: {target_distribution}")
        logger.info(f"Available problems: Easy={len(easy_problems)}, Medium={len(medium_problems)}, Hard={len(hard_problems)}")
        
        # Enforce minimum requirements - fail if we don't have enough
        missing_categories = []
        if len(easy_problems) < target_distribution['easy']:
            missing_categories.append(f"easy (need {target_distribution['easy']}, have {len(easy_problems)})")
        if len(medium_problems) < target_distribution['medium']:
            missing_categories.append(f"medium (need {target_distribution['medium']}, have {len(medium_problems)})")
        if len(hard_problems) < target_distribution['hard']:
            missing_categories.append(f"hard (need {target_distribution['hard']}, have {len(hard_problems)})")
        
        if missing_categories:
            logger.error(f"INSUFFICIENT PROBLEMS in categories: {missing_categories}")
            logger.error("This will compromise data quality. Aborting selection.")
            raise ValueError(f"Insufficient problems in categories: {missing_categories}")
        
        logger.info(f"✅ Sufficient problems available for target distribution")
        logger.info(f"Final target distribution: {target_distribution}")
        
        selected_problems = []
        
        # Select from each difficulty category
        for difficulty, problems_list in [('easy', easy_problems), ('medium', medium_problems), ('hard', hard_problems)]:
            target_from_difficulty = target_distribution[difficulty]
            
            if target_from_difficulty == 0 or not problems_list:
                continue
                
            logger.info(f"Selecting {target_from_difficulty} {difficulty} problems from {len(problems_list)} available")
            
            # Strategic selection within this difficulty
            selected_from_category = self._strategic_select_from_category(
                problems_list, target_from_difficulty, selected_problems
            )
            
            selected_problems.extend(selected_from_category)
            logger.info(f"Selected {len(selected_from_category)} {difficulty} problems")
        
        # If we still need more, fill from most available category
        while len(selected_problems) < target_count and (easy_problems or medium_problems or hard_problems):
            remaining = target_count - len(selected_problems)
            
            # Select from largest remaining pool
            remaining_pools = [(easy_problems, 'easy'), (medium_problems, 'medium'), (hard_problems, 'hard')]
            remaining_pools.sort(key=lambda x: len(x[0]), reverse=True)
            
            largest_pool, pool_name = remaining_pools[0]
            if not largest_pool:
                break
                
            additional = self._strategic_select_from_category(
                largest_pool, min(remaining, 10), selected_problems
            )
            selected_problems.extend(additional)
            
            # Remove selected from pool to avoid duplicates
            for selected in additional:
                if selected in largest_pool:
                    largest_pool.remove(selected)
        
        logger.info(f"Final selection: {len(selected_problems)} problems")
        
        # Analyze final selection
        self._analyze_selection_quality(selected_problems)
        
        return selected_problems[:target_count]  # Ensure exact count
    
    def _strategic_select_from_category(self, problems: List[Dict[str, Any]], count: int, already_selected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select problems from a single difficulty category using strategic scoring."""
        if count >= len(problems):
            return problems.copy()
            
        # Score each problem
        scored_problems = []
        
        for problem in problems:
            # Skip if already selected
            if any(p.get('problem_id') == problem.get('problem_id') for p in already_selected):
                continue
                
            complexity = self.assess_problem_complexity(problem)
            diversity = self.compute_diversity_score(problem, already_selected)
            
            # Combined strategic score (favor diversity and complexity)
            strategic_score = 0.6 * diversity + 0.4 * complexity
            
            scored_problems.append({
                'problem': problem,
                'score': strategic_score,
                'complexity': complexity,
                'diversity': diversity
            })
        
        # Sort by strategic score
        scored_problems.sort(key=lambda x: x['score'], reverse=True)
        
        # Select top problems, ensuring we pick diverse types
        selected = []
        selected_types = set()
        
        for scored in scored_problems:
            if len(selected) >= count:
                break
                
            problem = scored['problem']
            types = self.analyze_problem_type(problem)
            dominant_type = max(types.keys(), key=lambda k: types[k]) if types else 'unknown'
            
            # Prefer problems of types we haven't selected much
            type_count = sum(1 for s in selected if self._get_dominant_type(s) == dominant_type)
            max_per_type = max(1, count // len(self.algorithm_keywords))
            
            if type_count < max_per_type or len(selected) < count // 2:
                selected.append(problem)
                selected_types.add(dominant_type)
        
        # Fill remaining slots if needed
        while len(selected) < count and len(selected) < len(scored_problems):
            for scored in scored_problems:
                if len(selected) >= count:
                    break
                if scored['problem'] not in selected:
                    selected.append(scored['problem'])
        
        return selected
    
    def _get_dominant_type(self, problem: Dict[str, Any]) -> str:
        """Get the dominant algorithm type for a problem."""
        types = self.analyze_problem_type(problem)
        return max(types.keys(), key=lambda k: types[k]) if types else 'unknown'
    
    def _analyze_selection_quality(self, selected_problems: List[Dict[str, Any]]):
        """Analyze and report the quality of problem selection."""
        logger.info("=" * 80)
        logger.info("🔍 COMPREHENSIVE SELECTION QUALITY ANALYSIS")
        logger.info("=" * 80)
        
        # CRITICAL: Difficulty distribution validation
        difficulties = {}
        for prob in selected_problems:
            diff = prob.get('difficulty', 'unknown')
            difficulties[diff] = difficulties.get(diff, 0) + 1
        
        logger.info(f"📊 FINAL DIFFICULTY DISTRIBUTION:")
        logger.info(f"   Easy: {difficulties.get('easy', 0)} problems (target: 54)")
        logger.info(f"   Medium: {difficulties.get('medium', 0)} problems (target: 91)")  
        logger.info(f"   Hard: {difficulties.get('hard', 0)} problems (target: 55)")
        logger.info(f"   Total: {sum(difficulties.values())} problems (target: 200)")
        
        # VALIDATION: Check if we hit targets
        target_validation = {
            'easy': (difficulties.get('easy', 0), 54),
            'medium': (difficulties.get('medium', 0), 91),
            'hard': (difficulties.get('hard', 0), 55)
        }
        
        validation_passed = True
        for diff, (actual, target) in target_validation.items():
            if actual != target:
                logger.error(f"❌ {diff.upper()} MISMATCH: got {actual}, expected {target}")
                validation_passed = False
            else:
                logger.info(f"✅ {diff.upper()} TARGET MET: {actual}/{target}")
        
        if validation_passed:
            logger.info("🎉 ALL DIFFICULTY TARGETS SUCCESSFULLY MET!")
        else:
            logger.error("💥 DIFFICULTY TARGETS NOT MET - DATA QUALITY COMPROMISED")
            raise ValueError("Selection failed to meet difficulty distribution targets")
        
        # Algorithm type distribution  
        type_counts = {}
        for prob in selected_problems:
            types = self.analyze_problem_type(prob)
            dominant_type = max(types.keys(), key=lambda k: types[k]) if types else 'unknown'
            type_counts[dominant_type] = type_counts.get(dominant_type, 0) + 1
        logger.info(f"🧮 Algorithm type distribution: {type_counts}")
        
        # Complexity analysis
        complexities = [self.assess_problem_complexity(p) for p in selected_problems]
        logger.info(f"📈 Complexity stats: mean={np.mean(complexities):.3f}, std={np.std(complexities):.3f}")
        
        # Source distribution
        sources = {}
        for prob in selected_problems:
            source = prob.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        logger.info(f"🔗 Source distribution: {sources}")
        
        logger.info("=" * 80)
        logger.info("✅ SELECTION QUALITY ANALYSIS COMPLETE")
        logger.info("=" * 80)

def main():
    """Main selection process."""
    
    # Load real problems
    problems_path = Path("data/real_coding_problems.json")
    if not problems_path.exists():
        logger.error("Real coding problems not found. Run download_real_datasets.py first!")
        return
    
    with open(problems_path, 'r') as f:
        all_problems = json.load(f)
    
    logger.info(f"Loaded {len(all_problems)} total problems")
    
    # Select diverse subset
    selector = DiverseProblemSelector()
    selected_problems = selector.select_diverse_problems(all_problems, target_count=200)
    
    # Save selected problems
    output_path = Path("data/phase1_diverse_problems.json")
    with open(output_path, 'w') as f:
        json.dump(selected_problems, f, indent=2)
    
    logger.info(f"Saved {len(selected_problems)} diverse problems to {output_path}")
    
    return selected_problems

if __name__ == "__main__":
    selected = main()
    print(f"Selected {len(selected)} diverse problems for Phase 1 redo!")