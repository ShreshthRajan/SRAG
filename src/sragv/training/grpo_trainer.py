"""
GRPO (Group Relative Policy Optimization) Trainer for SRAG-V.
Implements July 2025 state-of-the-art GRPO with QLoRA integration.
Based on research: async-grpo, GRPO-Zero, and DeepSeek-R1 algorithms.
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from typing import List, Dict, Tuple, Optional, Any
import logging
import time
import numpy as np
from dataclasses import dataclass, field
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO trainer based on July 2025 research."""
    
    # Core GRPO parameters (research-optimized)
    group_size: int = 8  # Number of responses per prompt
    batch_size: int = 32  # Training batch size
    ppo_epochs: int = 4  # Training epochs per batch
    learning_rate: float = 1e-5  # Conservative for stability
    kl_penalty: float = 0.0  # Often disabled in 2025 (GRPO-Zero approach)
    loss_aggregation: str = "token-mean"  # More stable than "seq-mean"
    
    # Memory optimization (July 2025 best practices)
    use_gradient_checkpointing: bool = True
    use_8bit_optimizer: bool = True
    max_memory_per_gpu: str = "48GB"
    sequence_parallel: bool = True
    
    # LoRA/QLoRA configuration (research-optimized)
    lora_rank: int = 64  # Research shows minimal performance difference above r=8
    lora_alpha: int = 128  # 2×rank for optimal results
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"      # MLP
    ])
    
    # Training schedule
    warmup_steps: int = 100
    max_steps: int = 10000
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Group sampling and reward computation
    reward_model_updates: int = 1000  # Prevent overfitting
    baseline_samples: int = 1000  # For stable group statistics
    
    # Role-specific parameters
    role_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "problem_generator": 1.0,     # Problem generation quality
        "solution_generator": 1.0,    # Code correctness
        "verification_generator": 1.2, # Test case quality (weighted higher)
        "meta_verifier": 1.1          # Verification accuracy
    })


class GRPOTrainer:
    """
    GRPO Trainer implementing July 2025 research findings.
    
    Key features:
    - No critic network (50% memory reduction)
    - Group-relative baselines for advantage computation
    - QLoRA integration for efficient fine-tuning
    - Role-conditioned rewards for 4-player system
    - Memory-optimized for large models
    """
    
    def __init__(
        self,
        config: GRPOConfig,
        players: Dict[str, Any],  # Dictionary of player models
        reward_functions: Dict[str, callable],
        device: str = "auto"
    ):
        self.config = config
        self.players = players
        self.reward_functions = reward_functions
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history: List[Dict] = []
        
        # Initialize optimizers and schedulers for each player
        self.optimizers = {}
        self.schedulers = {}
        self._setup_training_components()
        
        logger.info(f"GRPO Trainer initialized with {len(players)} players")
        logger.info(f"Configuration: {config}")
    
    def _setup_training_components(self):
        """Initialize optimizers and schedulers for each player."""
        for player_name, player in self.players.items():
            # Get the training model (PEFT model if available)
            training_model = player.get_training_model()
            
            # Setup optimizer with 8-bit optimization if configured
            if self.config.use_8bit_optimizer:
                try:
                    import bitsandbytes as bnb
                    # Check if 8-bit optimizers are actually available
                    if hasattr(bnb.functional, 'cadam32bit_grad_fp32'):
                        optimizer_class = bnb.optim.AdamW8bit
                        logger.info(f"Using 8-bit optimizer for {player_name}")
                    else:
                        optimizer_class = AdamW
                        logger.warning(f"8-bit optimizer not available (no GPU support), using standard AdamW for {player_name}")
                except (ImportError, AttributeError):
                    optimizer_class = AdamW
                    logger.warning(f"8-bit optimizer not available, using standard AdamW for {player_name}")
            else:
                optimizer_class = AdamW
            
            # Create optimizer
            optimizer = optimizer_class(
                training_model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=0.01
            )
            
            # Create scheduler
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=self.config.max_steps
            )
            
            self.optimizers[player_name] = optimizer
            self.schedulers[player_name] = scheduler
            
            logger.info(f"Training components setup for {player_name}")
    
    def compute_group_advantages(
        self, 
        rewards: List[float], 
        role_weights: Optional[List[float]] = None
    ) -> List[float]:
        """
        Compute group-relative advantages using July 2025 GRPO formulation.
        
        Advantage: A(s,a) = (R - μ_group) / σ_group
        
        Args:
            rewards: List of rewards for the group
            role_weights: Optional role-specific multipliers
            
        Returns:
            List of normalized advantages
        """
        if len(rewards) == 0:
            return []
        
        # Apply role weights if provided
        if role_weights:
            weighted_rewards = [r * w for r, w in zip(rewards, role_weights)]
        else:
            weighted_rewards = rewards
        
        # Compute group statistics
        rewards_tensor = torch.tensor(weighted_rewards, dtype=torch.float32)
        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std()
        
        # Prevent division by zero
        if std_reward < 1e-8:
            std_reward = 1.0
        
        # Compute normalized advantages
        advantages = (rewards_tensor - mean_reward) / std_reward
        
        return advantages.tolist()
    
    def compute_role_conditioned_reward(
        self, 
        output: str, 
        role: str, 
        context: Dict[str, Any]
    ) -> float:
        """
        Compute role-specific rewards for 4-player system.
        
        Args:
            output: Generated output from player
            role: Player role (problem_generator, solution_generator, etc.)
            context: Additional context for reward computation
            
        Returns:
            Role-conditioned reward value
        """
        # Get base reward from role-specific reward function
        reward_fn = self.reward_functions.get(role)
        if reward_fn is None:
            logger.warning(f"No reward function found for role {role}, using default")
            return 0.0
        
        base_reward_metrics = reward_fn(output, context)

        # Extract final_reward float from RewardMetrics object or use numeric value directly
        import numpy as np
        if isinstance(base_reward_metrics, (float, int, np.floating, np.integer)):
            # Already a numeric value - use it directly
            base_reward = float(base_reward_metrics)
        elif hasattr(base_reward_metrics, 'final_reward'):
            # RewardMetrics object - extract final_reward
            base_reward = base_reward_metrics.final_reward
        else:
            logger.error(f"Invalid reward type: {type(base_reward_metrics)}")
            base_reward = 0.0

        # Apply role multiplier
        role_multiplier = self.config.role_multipliers.get(role, 1.0)

        # Compute role-specific bonus
        role_bonus = self._compute_role_bonus(output, role, context)

        final_reward = base_reward * role_multiplier + role_bonus
        
        logger.debug(f"Role reward for {role}: base={base_reward:.3f}, "
                    f"multiplier={role_multiplier}, bonus={role_bonus:.3f}, "
                    f"final={final_reward:.3f}")
        
        return final_reward
    
    def _compute_role_bonus(self, output: str, role: str, context: Dict[str, Any]) -> float:
        """Compute role-specific bonus rewards."""
        bonus = 0.0
        
        if role == "problem_generator":
            # Bonus for problem diversity and clarity
            if len(output.split()) > 50:  # Detailed problem description
                bonus += 0.1
            if any(keyword in output.lower() for keyword in ["example", "constraint", "input", "output"]):
                bonus += 0.15
        
        elif role == "solution_generator":
            # Bonus for code quality and completeness
            if "def " in output and "return" in output:  # Complete function
                bonus += 0.2
            if len([line for line in output.split('\n') if line.strip()]) > 3:  # Multi-line solution
                bonus += 0.1
        
        elif role == "verification_generator":
            # Bonus for test coverage and edge cases
            test_cases = len([line for line in output.split('\n') if 'assert' in line or 'test' in line.lower()])
            bonus += min(test_cases * 0.05, 0.3)  # Up to 0.3 bonus for many test cases
        
        elif role == "meta_verifier":
            # Bonus for detailed verification analysis
            if any(keyword in output.lower() for keyword in ["valid", "invalid", "correct", "incorrect"]):
                bonus += 0.1
            if len(output.split()) > 20:  # Detailed analysis
                bonus += 0.05
        
        return bonus
    
    def train_step(
        self, 
        prompts: List[str], 
        player_roles: List[str],
        contexts: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Execute a single GRPO training step.
        
        Args:
            prompts: List of input prompts
            player_roles: Corresponding player roles for each prompt
            contexts: Additional context for each prompt
            
        Returns:
            Training metrics dictionary
        """
        batch_size = len(prompts)
        all_outputs = []
        all_rewards = []
        all_log_probs = []
        all_roles = []
        
        # Generate group responses for each prompt
        for prompt, role, context in zip(prompts, player_roles, contexts):
            player = self.players[role]
            
            # Generate multiple responses (group sampling)
            group_outputs = []
            group_log_probs = []
            
            for _ in range(self.config.group_size):
                # Generate response (no grad for generation, but we'll recompute log_probs with grad)
                with torch.no_grad():
                    outputs = player.generate_text(
                        prompt,
                        max_new_tokens=512,
                        temperature=0.8,
                        do_sample=True,
                        num_return_sequences=1
                    )
                    output = outputs[0]

                group_outputs.append(output)

            # Compute log probabilities WITH gradients enabled (outside no_grad context)
            for output in group_outputs:
                log_prob = self._compute_log_probability(player, prompt, output)
                group_log_probs.append(log_prob)
            
            # Compute rewards for the group
            group_rewards = []
            for output in group_outputs:
                reward = self.compute_role_conditioned_reward(output, role, context)
                group_rewards.append(reward)
            
            # Store group data
            all_outputs.extend(group_outputs)
            all_rewards.extend(group_rewards)
            all_log_probs.extend(group_log_probs)
            all_roles.extend([role] * self.config.group_size)
        
        # Compute group-relative advantages
        # Group advantages by role for fair comparison
        role_advantages = {}
        for unique_role in set(all_roles):
            role_indices = [i for i, r in enumerate(all_roles) if r == unique_role]
            role_rewards = [all_rewards[i] for i in role_indices]
            role_weights = [self.config.role_multipliers[unique_role]] * len(role_rewards)
            
            advantages = self.compute_group_advantages(role_rewards, role_weights)
            role_advantages[unique_role] = advantages
        
        # Reconstruct advantages in original order
        advantages = []
        role_counters = {role: 0 for role in set(all_roles)}
        for role in all_roles:
            advantages.append(role_advantages[role][role_counters[role]])
            role_counters[role] += 1
        
        # Compute policy loss using GRPO objective
        total_loss = 0.0
        role_losses = {}
        
        for unique_role in set(all_roles):
            role_indices = [i for i, r in enumerate(all_roles) if r == unique_role]
            role_log_probs = torch.stack([all_log_probs[i] for i in role_indices])
            role_advantages_tensor = torch.tensor(
                [advantages[i] for i in role_indices],
                device=role_log_probs.device,  # Match device of log_probs
                dtype=role_log_probs.dtype  # Match dtype for efficiency
            )

            # GRPO loss: -mean(log_prob * advantage)
            role_loss = -(role_log_probs * role_advantages_tensor).mean()
            role_losses[unique_role] = role_loss
            total_loss += role_loss
        
        # Backward pass and optimization
        total_loss.backward()
        
        # Update each player's parameters
        for role in set(all_roles):
            optimizer = self.optimizers[role]
            scheduler = self.schedulers[role]
            
            # Gradient clipping
            player = self.players[role]
            torch.nn.utils.clip_grad_norm_(player.get_training_model().parameters(), 1.0)
            
            # Optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Compute metrics
        metrics = {
            "total_loss": total_loss.item(),
            "mean_reward": np.mean(all_rewards),
            "std_reward": np.std(all_rewards),
            "mean_advantage": np.mean(advantages),
            "global_step": self.global_step
        }
        
        # Add per-role metrics
        for role in set(all_roles):
            role_indices = [i for i, r in enumerate(all_roles) if r == role]
            role_rewards = [all_rewards[i] for i in role_indices]
            role_advs = [advantages[i] for i in role_indices]
            
            metrics[f"{role}_reward"] = np.mean(role_rewards)
            metrics[f"{role}_advantage"] = np.mean(role_advs)
            metrics[f"{role}_loss"] = role_losses[role].item()
        
        self.global_step += 1
        self.training_history.append(metrics)
        
        return metrics
    
    def _compute_log_probability(self, player, prompt: str, output: str) -> torch.Tensor:
        """Compute log probability of generated output given prompt."""
        # Tokenize input and output
        full_text = prompt + output
        inputs = player.tokenizer(prompt, return_tensors="pt", truncation=True)
        full_inputs = player.tokenizer(full_text, return_tensors="pt", truncation=True)
        
        # Move to device
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            full_inputs = {k: v.cuda() for k, v in full_inputs.items()}
        
        # Get model
        model = player.get_training_model()

        # Compute logits WITH gradients (needed for GRPO backprop)
        outputs = model(**full_inputs)
        logits = outputs.logits

        # Extract output tokens (excluding prompt)
        prompt_length = inputs["input_ids"].shape[1]
        output_logits = logits[0, prompt_length-1:-1]  # Shift for next token prediction
        output_tokens = full_inputs["input_ids"][0, prompt_length:]
        
        # Compute log probabilities
        log_probs = F.log_softmax(output_logits, dim=-1)
        selected_log_probs = log_probs.gather(1, output_tokens.unsqueeze(-1)).squeeze(-1)
        
        # Return mean log probability
        return selected_log_probs.mean()
    
    def save_checkpoint(self, checkpoint_dir: str):
        """Save training checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "config": self.config.__dict__,
            "training_history": self.training_history
        }
        
        with open(checkpoint_path / "training_state.json", "w") as f:
            json.dump(state, f, indent=2)
        
        # Save model adapters
        for player_name, player in self.players.items():
            player_checkpoint_dir = checkpoint_path / f"{player_name}_adapter"
            player.save_adapter(str(player_checkpoint_dir))
        
        # Save optimizer states
        torch.save({
            name: optimizer.state_dict() 
            for name, optimizer in self.optimizers.items()
        }, checkpoint_path / "optimizers.pt")
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_dir)
        
        # Load training state
        with open(checkpoint_path / "training_state.json", "r") as f:
            state = json.load(f)
        
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.training_history = state["training_history"]
        
        # Load model adapters
        for player_name, player in self.players.items():
            player_checkpoint_dir = checkpoint_path / f"{player_name}_adapter"
            if player_checkpoint_dir.exists():
                player.load_adapter(str(player_checkpoint_dir))
        
        # Load optimizer states
        optimizer_states = torch.load(checkpoint_path / "optimizers.pt")
        for name, optimizer in self.optimizers.items():
            if name in optimizer_states:
                optimizer.load_state_dict(optimizer_states[name])
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")