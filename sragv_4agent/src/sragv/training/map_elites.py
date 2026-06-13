"""
MAP-Elites Diversity Mechanism for SRAG-V.
Implements neural network-compatible MAP-Elites for maintaining behavioral diversity
in 4-player self-play code generation and verification.
Based on July 2025 research: PGA-MAP-Elites and descriptor-conditioned gradients.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import json
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class MAPElitesConfig:
    """Configuration for MAP-Elites diversity mechanism."""
    
    # Archive dimensions (3D grid as per research)
    complexity_levels: int = 5      # Problem complexity: 1-5
    approach_categories: int = 16   # Solution approaches (DP, greedy, graph, etc.)
    verification_levels: int = 10   # Verification difficulty: 1-10
    
    # Total archive size: 5 × 16 × 10 = 800 base niches
    # With redundancy: 8000 total slots (10x redundancy for diversity)
    archive_size: int = 8000
    redundancy_factor: int = 10  # Multiple solutions per niche
    
    # Quality-diversity trade-off
    quality_weight: float = 0.7   # Weight for solution quality
    diversity_weight: float = 0.3 # Weight for behavioral diversity
    
    # Descriptor computation
    descriptor_functions: Dict[str, str] = field(default_factory=lambda: {
        "complexity": "compute_complexity_descriptor",
        "approach": "compute_approach_descriptor", 
        "verification": "compute_verification_descriptor"
    })
    
    # Archive management
    elite_replacement_threshold: float = 0.1  # Replace if 10% better
    diversity_bonus: float = 0.2  # Bonus for novel behaviors
    archive_update_frequency: int = 10  # Update every N generations
    
    # Neural network integration
    descriptor_embedding_dim: int = 64  # Embedding dimension for descriptors
    use_descriptor_conditioning: bool = True  # Condition gradients on descriptors
    archive_distillation_steps: int = 1000  # Steps for policy distillation


class BehavioralDescriptor:
    """Represents behavioral characteristics of a solution."""
    
    def __init__(
        self, 
        complexity: float, 
        approach: str, 
        verification_difficulty: float,
        additional_features: Optional[Dict[str, float]] = None
    ):
        self.complexity = complexity  # 0.0 - 1.0
        self.approach = approach     # String identifier
        self.verification_difficulty = verification_difficulty  # 0.0 - 1.0
        self.additional_features = additional_features or {}
        
        # Compute hash for quick comparison
        self.hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash for behavioral descriptor."""
        desc_str = f"{self.complexity:.3f}_{self.approach}_{self.verification_difficulty:.3f}"
        for key, value in sorted(self.additional_features.items()):
            desc_str += f"_{key}:{value:.3f}"
        return hashlib.md5(desc_str.encode()).hexdigest()[:16]
    
    def to_grid_indices(self, config: MAPElitesConfig) -> Tuple[int, int, int]:
        """Convert descriptor to grid indices."""
        complexity_idx = min(int(self.complexity * config.complexity_levels), 
                            config.complexity_levels - 1)
        
        # Map approach to category index - ensure it fits within config bounds
        approach_categories = [
            "arithmetic", "dynamic_programming", "greedy", "graph_algorithms",
            "string_processing", "array_manipulation", "sorting", "searching",
            "recursion", "iteration", "mathematical", "optimization",
            "data_structures", "algorithms", "simulation", "other"
        ]
        
        if self.approach in approach_categories:
            approach_idx = approach_categories.index(self.approach)
        else:
            approach_idx = len(approach_categories) - 1  # "other"
        
        # Ensure approach index fits within configured bounds
        approach_idx = min(approach_idx, config.approach_categories - 1)
        
        verification_idx = min(int(self.verification_difficulty * config.verification_levels),
                              config.verification_levels - 1)
        
        return complexity_idx, approach_idx, verification_idx
    
    def distance(self, other: 'BehavioralDescriptor') -> float:
        """Compute behavioral distance to another descriptor."""
        # Euclidean distance in descriptor space
        complexity_dist = (self.complexity - other.complexity) ** 2
        verification_dist = (self.verification_difficulty - other.verification_difficulty) ** 2
        
        # Approach distance (0 if same, 1 if different)
        approach_dist = 0.0 if self.approach == other.approach else 1.0
        
        # Additional features distance
        additional_dist = 0.0
        all_keys = set(self.additional_features.keys()) | set(other.additional_features.keys())
        for key in all_keys:
            val1 = self.additional_features.get(key, 0.0)
            val2 = other.additional_features.get(key, 0.0)
            additional_dist += (val1 - val2) ** 2
        
        return np.sqrt(complexity_dist + approach_dist + verification_dist + additional_dist)


@dataclass
class Elite:
    """Represents an elite solution in the archive."""
    solution: Dict[str, Any]  # Complete solution data
    descriptor: BehavioralDescriptor
    quality: float  # Objective quality score
    fitness: float  # Combined quality-diversity fitness
    generation: int  # When this elite was created
    player_role: str  # Which player generated this
    metadata: Dict[str, Any] = field(default_factory=dict)


class MAPElitesArchive:
    """
    Neural network-compatible MAP-Elites archive for behavioral diversity.
    
    Maintains a diverse set of high-quality solutions across behavioral dimensions.
    Supports descriptor-conditioned gradients and archive distillation.
    """
    
    def __init__(self, config: MAPElitesConfig):
        self.config = config
        
        # Main archive: behavioral descriptor hash -> List[Elite]
        self.archive: Dict[str, List[Elite]] = defaultdict(list)
        
        # Grid-based archive for visualization and analysis
        self.grid = np.full(
            (config.complexity_levels, config.approach_categories, config.verification_levels),
            None, dtype=object
        )
        
        # Archive statistics
        self.generation = 0
        self.total_elites = 0
        self.quality_history: List[float] = []
        self.diversity_history: List[float] = []
        
        # Descriptor computation functions
        self.descriptor_functions = self._setup_descriptor_functions()
        
        logger.info(f"MAP-Elites archive initialized: {config.archive_size} slots, "
                   f"grid size {config.complexity_levels}×{config.approach_categories}×{config.verification_levels}")
    
    def _setup_descriptor_functions(self) -> Dict[str, Callable]:
        """Setup behavioral descriptor computation functions."""
        return {
            "complexity": self._compute_complexity_descriptor,
            "approach": self._compute_approach_descriptor,
            "verification": self._compute_verification_descriptor
        }
    
    def _compute_complexity_descriptor(self, solution: Dict[str, Any]) -> float:
        """Compute complexity descriptor (0.0 - 1.0)."""
        # Analyze solution complexity based on multiple factors
        code = solution.get("code", "")
        problem = solution.get("problem", {})
        
        complexity_factors = []
        
        # Code length complexity
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        line_complexity = min(len(lines) / 20.0, 1.0)  # Normalize to max 20 lines
        complexity_factors.append(line_complexity)
        
        # Algorithmic complexity indicators
        complexity_keywords = [
            "for", "while", "if", "elif", "def", "class", "import",
            "recursion", "dynamic", "sort", "search", "graph", "tree"
        ]
        keyword_count = sum(1 for keyword in complexity_keywords if keyword in code.lower())
        keyword_complexity = min(keyword_count / 10.0, 1.0)
        complexity_factors.append(keyword_complexity)
        
        # Problem difficulty indicators
        problem_text = problem.get("question", "").lower()
        difficulty_keywords = [
            "optimize", "efficient", "complex", "advanced", "algorithm",
            "data structure", "dynamic programming", "graph theory"
        ]
        difficulty_count = sum(1 for keyword in difficulty_keywords if keyword in problem_text)
        difficulty_complexity = min(difficulty_count / 5.0, 1.0)
        complexity_factors.append(difficulty_complexity)
        
        # Average complexity
        return np.mean(complexity_factors)
    
    def _compute_approach_descriptor(self, solution: Dict[str, Any]) -> str:
        """Compute approach category descriptor."""
        code = solution.get("code", "").lower()
        problem = solution.get("problem", {}).get("question", "").lower()
        
        # Define approach patterns
        approach_patterns = {
            "dynamic_programming": ["dp", "dynamic", "memo", "cache", "table"],
            "greedy": ["greedy", "local", "immediate", "best choice"],
            "graph_algorithms": ["graph", "node", "edge", "vertex", "bfs", "dfs", "dijkstra"],
            "string_processing": ["string", "char", "substring", "pattern", "text"],
            "array_manipulation": ["array", "list", "index", "element", "sort"],
            "sorting": ["sort", "order", "sorted", "arrange"],
            "searching": ["search", "find", "binary", "linear", "lookup"],
            "recursion": ["recursion", "recursive", "recur", "base case"],
            "iteration": ["loop", "for", "while", "iterate"],
            "mathematical": ["math", "formula", "equation", "calculate"],
            "optimization": ["optimize", "minimize", "maximize", "efficient"],
            "data_structures": ["stack", "queue", "tree", "heap", "hash"],
            "simulation": ["simulate", "model", "step", "process"],
        }
        
        # Find matching approach
        for approach, keywords in approach_patterns.items():
            if any(keyword in code or keyword in problem for keyword in keywords):
                return approach
        
        # Default to arithmetic for simple problems
        if any(op in code for op in ["+", "-", "*", "/", "%"]):
            return "arithmetic"
        
        return "other"
    
    def _compute_verification_descriptor(self, solution: Dict[str, Any]) -> float:
        """Compute verification difficulty descriptor (0.0 - 1.0)."""
        test_cases = solution.get("test_cases", [])
        problem = solution.get("problem", {})
        
        difficulty_factors = []
        
        # Test case complexity
        if test_cases:
            avg_test_length = np.mean([len(str(tc)) for tc in test_cases])
            test_complexity = min(avg_test_length / 100.0, 1.0)
            difficulty_factors.append(test_complexity)
            
            # Edge case coverage
            edge_case_indicators = ["empty", "null", "zero", "negative", "large", "boundary"]
            edge_case_count = sum(
                1 for tc in test_cases 
                for indicator in edge_case_indicators 
                if indicator in str(tc).lower()
            )
            edge_complexity = min(edge_case_count / len(test_cases), 1.0)
            difficulty_factors.append(edge_complexity)
        else:
            difficulty_factors.extend([0.3, 0.3])  # Default moderate difficulty
        
        # Problem verification complexity
        problem_text = problem.get("question", "").lower()
        verification_keywords = [
            "verify", "validate", "check", "test", "correct", "accuracy",
            "edge case", "boundary", "constraint", "requirement"
        ]
        verification_mentions = sum(1 for keyword in verification_keywords if keyword in problem_text)
        verification_complexity = min(verification_mentions / 5.0, 1.0)
        difficulty_factors.append(verification_complexity)
        
        return np.mean(difficulty_factors)
    
    def compute_descriptor(self, solution: Dict[str, Any]) -> BehavioralDescriptor:
        """Compute behavioral descriptor for a solution."""
        complexity = self.descriptor_functions["complexity"](solution)
        approach = self.descriptor_functions["approach"](solution)
        verification = self.descriptor_functions["verification"](solution)
        
        # Additional features based on solution characteristics
        additional_features = {}
        
        # Code style features
        code = solution.get("code", "")
        additional_features["code_density"] = len(code.replace(" ", "")) / max(len(code), 1)
        additional_features["function_count"] = code.count("def ")
        additional_features["comment_ratio"] = code.count("#") / max(len(code.split("\n")), 1)
        
        return BehavioralDescriptor(
            complexity=complexity,
            approach=approach,
            verification_difficulty=verification,
            additional_features=additional_features
        )
    
    def add_solution(
        self, 
        solution: Dict[str, Any], 
        quality: float, 
        player_role: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a solution to the archive if it's good enough.
        
        Returns True if solution was added, False otherwise.
        """
        # Compute behavioral descriptor
        descriptor = self.compute_descriptor(solution)
        
        # Compute diversity bonus
        diversity_bonus = self._compute_diversity_bonus(descriptor)
        
        # Combined fitness: quality + diversity
        fitness = (self.config.quality_weight * quality + 
                  self.config.diversity_weight * diversity_bonus)
        
        # Create elite
        elite = Elite(
            solution=solution,
            descriptor=descriptor,
            quality=quality,
            fitness=fitness,
            generation=self.generation,
            player_role=player_role,
            metadata=metadata or {}
        )
        
        # Check if we should add this elite
        should_add = self._should_add_elite(elite)
        
        if should_add:
            self._add_elite_to_archive(elite)
            self.total_elites += 1
            
            logger.debug(f"Added elite: quality={quality:.3f}, fitness={fitness:.3f}, "
                        f"descriptor={descriptor.hash}, role={player_role}")
            return True
        
        return False
    
    def _compute_diversity_bonus(self, descriptor: BehavioralDescriptor) -> float:
        """Compute diversity bonus for a behavioral descriptor."""
        if not self.archive:
            return 1.0  # Maximum bonus for first solution
        
        # Find minimum distance to existing elites
        min_distance = float('inf')
        
        for elite_list in self.archive.values():
            for elite in elite_list:
                distance = descriptor.distance(elite.descriptor)
                min_distance = min(min_distance, distance)
        
        # Convert distance to bonus (higher distance = higher bonus)
        # Use exponential decay to reward novel behaviors
        diversity_bonus = np.exp(-min_distance)
        
        return diversity_bonus
    
    def _should_add_elite(self, elite: Elite) -> bool:
        """Determine if an elite should be added to the archive."""
        descriptor_hash = elite.descriptor.hash
        
        # If this behavioral niche is empty, always add
        if descriptor_hash not in self.archive:
            return True
        
        existing_elites = self.archive[descriptor_hash]
        
        # If niche has space, add
        if len(existing_elites) < self.config.redundancy_factor:
            return True
        
        # Find weakest elite in this niche
        weakest_elite = min(existing_elites, key=lambda e: e.fitness)
        
        # Replace if new elite is significantly better
        improvement = elite.fitness - weakest_elite.fitness
        threshold = self.config.elite_replacement_threshold
        
        return improvement > threshold
    
    def _add_elite_to_archive(self, elite: Elite):
        """Add elite to the archive, managing capacity."""
        descriptor_hash = elite.descriptor.hash
        
        # Add to main archive
        self.archive[descriptor_hash].append(elite)
        
        # Maintain niche capacity
        if len(self.archive[descriptor_hash]) > self.config.redundancy_factor:
            # Remove weakest elite
            self.archive[descriptor_hash].sort(key=lambda e: e.fitness, reverse=True)
            removed_elite = self.archive[descriptor_hash].pop()
            logger.debug(f"Removed weaker elite: fitness={removed_elite.fitness:.3f}")
        
        # Update grid for visualization
        indices = elite.descriptor.to_grid_indices(self.config)
        if self.grid[indices] is None or elite.fitness > self.grid[indices].fitness:
            self.grid[indices] = elite
        
        # Maintain total archive size
        self._maintain_archive_size()
    
    def _maintain_archive_size(self):
        """Maintain maximum archive size by removing least fit elites."""
        current_size = sum(len(elite_list) for elite_list in self.archive.values())
        
        if current_size > self.config.archive_size:
            # Collect all elites with their fitness
            all_elites = []
            for descriptor_hash, elite_list in self.archive.items():
                for elite in elite_list:
                    all_elites.append((elite.fitness, descriptor_hash, elite))
            
            # Sort by fitness (ascending)
            all_elites.sort(key=lambda x: x[0])
            
            # Remove worst elites
            elites_to_remove = current_size - self.config.archive_size
            for i in range(elites_to_remove):
                fitness, descriptor_hash, elite = all_elites[i]
                self.archive[descriptor_hash].remove(elite)
                
                # Clean up empty niches
                if not self.archive[descriptor_hash]:
                    del self.archive[descriptor_hash]
    
    def sample_diverse_elites(self, n: int) -> List[Elite]:
        """Sample n diverse elites from the archive."""
        if not self.archive:
            return []
        
        # Collect all elites
        all_elites = []
        for elite_list in self.archive.values():
            all_elites.extend(elite_list)
        
        if len(all_elites) <= n:
            return all_elites
        
        # Use farthest-first traversal to select diverse elites
        selected = []
        remaining = all_elites.copy()
        
        # Start with highest quality elite
        first_elite = max(remaining, key=lambda e: e.quality)
        selected.append(first_elite)
        remaining.remove(first_elite)
        
        # Greedily select most diverse elites
        while len(selected) < n and remaining:
            best_elite = None
            best_min_distance = -1
            
            for candidate in remaining:
                # Find minimum distance to already selected elites
                min_distance = min(
                    candidate.descriptor.distance(selected_elite.descriptor)
                    for selected_elite in selected
                )
                
                if min_distance > best_min_distance:
                    best_min_distance = min_distance
                    best_elite = candidate
            
            if best_elite:
                selected.append(best_elite)
                remaining.remove(best_elite)
        
        return selected
    
    def get_archive_statistics(self) -> Dict[str, Any]:
        """Get comprehensive archive statistics."""
        if not self.archive:
            return {"total_elites": 0, "occupied_niches": 0}
        
        all_elites = []
        for elite_list in self.archive.values():
            all_elites.extend(elite_list)
        
        qualities = [elite.quality for elite in all_elites]
        fitnesses = [elite.fitness for elite in all_elites]
        
        # Behavioral diversity metrics
        complexity_values = [elite.descriptor.complexity for elite in all_elites]
        approach_counts = defaultdict(int)
        for elite in all_elites:
            approach_counts[elite.descriptor.approach] += 1
        
        verification_values = [elite.descriptor.verification_difficulty for elite in all_elites]
        
        stats = {
            "total_elites": len(all_elites),
            "occupied_niches": len(self.archive),
            "generation": self.generation,
            
            # Quality statistics
            "mean_quality": np.mean(qualities),
            "max_quality": np.max(qualities),
            "min_quality": np.min(qualities),
            "std_quality": np.std(qualities),
            
            # Fitness statistics
            "mean_fitness": np.mean(fitnesses),
            "max_fitness": np.max(fitnesses),
            "std_fitness": np.std(fitnesses),
            
            # Diversity statistics
            "complexity_mean": np.mean(complexity_values),
            "complexity_std": np.std(complexity_values),
            "verification_mean": np.mean(verification_values),
            "verification_std": np.std(verification_values),
            "approach_diversity": len(approach_counts),
            "approach_distribution": dict(approach_counts),
            
            # Grid occupancy
            "grid_occupancy": np.sum(self.grid != None) / self.grid.size,
        }
        
        return stats
    
    def advance_generation(self):
        """Advance to next generation."""
        self.generation += 1
        
        # Update history
        stats = self.get_archive_statistics()
        if stats["total_elites"] > 0:
            self.quality_history.append(stats["mean_quality"])
            self.diversity_history.append(stats["approach_diversity"])
    
    def save_archive(self, file_path: str):
        """Save archive to disk."""
        archive_data = {
            "config": self.config.__dict__,
            "generation": self.generation,
            "total_elites": self.total_elites,
            "quality_history": self.quality_history,
            "diversity_history": self.diversity_history,
            "archive": {}
        }
        
        # Serialize archive
        for descriptor_hash, elite_list in self.archive.items():
            serialized_elites = []
            for elite in elite_list:
                serialized_elite = {
                    "solution": elite.solution,
                    "descriptor": {
                        "complexity": elite.descriptor.complexity,
                        "approach": elite.descriptor.approach,
                        "verification_difficulty": elite.descriptor.verification_difficulty,
                        "additional_features": elite.descriptor.additional_features,
                        "hash": elite.descriptor.hash
                    },
                    "quality": elite.quality,
                    "fitness": elite.fitness,
                    "generation": elite.generation,
                    "player_role": elite.player_role,
                    "metadata": elite.metadata
                }
                serialized_elites.append(serialized_elite)
            
            archive_data["archive"][descriptor_hash] = serialized_elites
        
        # Save to file
        with open(file_path, "w") as f:
            json.dump(archive_data, f, indent=2)
        
        logger.info(f"MAP-Elites archive saved to {file_path}")
    
    def load_archive(self, file_path: str):
        """Load archive from disk."""
        with open(file_path, "r") as f:
            archive_data = json.load(f)
        
        # Restore state
        self.generation = archive_data["generation"]
        self.total_elites = archive_data["total_elites"]
        self.quality_history = archive_data["quality_history"]
        self.diversity_history = archive_data["diversity_history"]
        
        # Restore archive
        self.archive.clear()
        for descriptor_hash, serialized_elites in archive_data["archive"].items():
            elite_list = []
            for serialized_elite in serialized_elites:
                descriptor_data = serialized_elite["descriptor"]
                descriptor = BehavioralDescriptor(
                    complexity=descriptor_data["complexity"],
                    approach=descriptor_data["approach"],
                    verification_difficulty=descriptor_data["verification_difficulty"],
                    additional_features=descriptor_data["additional_features"]
                )
                
                elite = Elite(
                    solution=serialized_elite["solution"],
                    descriptor=descriptor,
                    quality=serialized_elite["quality"],
                    fitness=serialized_elite["fitness"],
                    generation=serialized_elite["generation"],
                    player_role=serialized_elite["player_role"],
                    metadata=serialized_elite["metadata"]
                )
                
                elite_list.append(elite)
            
            self.archive[descriptor_hash] = elite_list
        
        # Rebuild grid
        self.grid.fill(None)
        for elite_list in self.archive.values():
            for elite in elite_list:
                indices = elite.descriptor.to_grid_indices(self.config)
                if self.grid[indices] is None or elite.fitness > self.grid[indices].fitness:
                    self.grid[indices] = elite
        
        logger.info(f"MAP-Elites archive loaded from {file_path}")