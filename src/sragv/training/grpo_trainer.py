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
    group_size: int = 6  # Number of responses per prompt (reduced from 8 for memory safety, still research-valid 4-8 range)
    batch_size: int = 32  # Training batch size
    ppo_epochs: int = 4  # Training epochs per batch
    learning_rate: float = 1e-5  # Conservative for stability
    kl_penalty: float = 0.0  # Often disabled in 2025 (GRPO-Zero approach)
    loss_aggregation: str = "token-mean"  # More stable than "seq-mean"
    gradient_accumulation_steps: int = 4  # Process prompts in chunks, backward() frequently
    
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

            # Enable gradient checkpointing if configured (critical for memory efficiency)
            if self.config.use_gradient_checkpointing:
                if hasattr(training_model, 'gradient_checkpointing_enable'):
                    training_model.gradient_checkpointing_enable()
                    logger.info(f"✅ Gradient checkpointing enabled for {player_name}")
                elif hasattr(training_model, 'enable_input_require_grads'):
                    # For PEFT models, need to enable input gradients for checkpointing
                    training_model.enable_input_require_grads()
                    if hasattr(training_model.base_model, 'gradient_checkpointing_enable'):
                        training_model.base_model.gradient_checkpointing_enable()
                    logger.info(f"✅ Gradient checkpointing enabled for {player_name} (PEFT)")
                else:
                    logger.warning(f"Gradient checkpointing not available for {player_name}")

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
        Execute a single GRPO training step with gradient accumulation for memory efficiency.

        Processes prompts in chunks to reduce peak memory usage while maintaining
        research-valid group_size=8 for advantage computation.

        Args:
            prompts: List of input prompts
            player_roles: Corresponding player roles for each prompt
            contexts: Additional context for each prompt

        Returns:
            Training metrics dictionary
        """
        batch_size = len(prompts)

        # Calculate chunk size for gradient accumulation
        chunk_size = max(1, batch_size // self.config.gradient_accumulation_steps)
        num_chunks = (batch_size + chunk_size - 1) // chunk_size  # Ceiling division

        # Zero gradients at start
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

        all_role_losses = {role: [] for role in set(player_roles)}
        total_samples_processed = 0

        # Process prompts in chunks (gradient accumulation for memory efficiency)
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, batch_size)

            chunk_prompts = prompts[start_idx:end_idx]
            chunk_roles = player_roles[start_idx:end_idx]
            chunk_contexts = contexts[start_idx:end_idx]

            chunk_outputs = []
            chunk_rewards = []
            chunk_log_probs = []
            chunk_role_list = []

            # Generate group responses for each prompt in this chunk
            for prompt, role, context in zip(chunk_prompts, chunk_roles, chunk_contexts):
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
                chunk_outputs.extend(group_outputs)
                chunk_rewards.extend(group_rewards)
                chunk_log_probs.extend(group_log_probs)
                chunk_role_list.extend([role] * self.config.group_size)

            # Compute group-relative advantages for this chunk
            role_advantages = {}
            for unique_role in set(chunk_role_list):
                role_indices = [i for i, r in enumerate(chunk_role_list) if r == unique_role]
                role_rewards = [chunk_rewards[i] for i in role_indices]
                role_weights = [self.config.role_multipliers[unique_role]] * len(role_rewards)

                advantages = self.compute_group_advantages(role_rewards, role_weights)
                role_advantages[unique_role] = advantages

            # Reconstruct advantages in original order
            chunk_advantages = []
            role_counters = {role: 0 for role in set(chunk_role_list)}
            for role in chunk_role_list:
                chunk_advantages.append(role_advantages[role][role_counters[role]])
                role_counters[role] += 1

            # Compute policy loss using GRPO objective
            # Backward each role separately to avoid cross-GPU tensor operations
            for unique_role in set(chunk_role_list):
                role_indices = [i for i, r in enumerate(chunk_role_list) if r == unique_role]
                role_log_probs = torch.stack([chunk_log_probs[i] for i in role_indices])
                role_advantages_tensor = torch.tensor(
                    [chunk_advantages[i] for i in role_indices],
                    device=role_log_probs.device,
                    dtype=role_log_probs.dtype
                )

                # GRPO loss: -mean(log_prob * advantage)
                role_loss = -(role_log_probs * role_advantages_tensor).mean()

                # Store loss value (detached) for metrics
                all_role_losses[unique_role].append(role_loss.item())

                # Backward pass for this role (gradients accumulate across chunks)
                # Scale loss by number of chunks to maintain effective learning rate
                scaled_loss = role_loss / num_chunks
                scaled_loss.backward()

            total_samples_processed += len(chunk_prompts) * self.config.group_size

        # After processing all chunks, update parameters
        total_loss = 0.0
        role_losses = {}

        for role in set(player_roles):
            optimizer = self.optimizers[role]
            scheduler = self.schedulers[role]

            # Gradient clipping
            player = self.players[role]
            torch.nn.utils.clip_grad_norm_(player.get_training_model().parameters(), 1.0)

            # Optimizer step (gradients already accumulated from all chunks)
            optimizer.step()
            scheduler.step()

            # Compute average loss for this role across chunks
            if all_role_losses[role]:
                role_losses[role] = np.mean(all_role_losses[role])
                total_loss += role_losses[role]

        # Compute metrics
        metrics = {
            "total_loss": total_loss,
            "samples_processed": total_samples_processed,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "global_step": self.global_step
        }

        # Add per-role loss metrics
        for role, loss_value in role_losses.items():
            metrics[f"{role}_loss"] = loss_value

        self.global_step += 1
        self.training_history.append(metrics)

        return metrics
    
    def _compute_log_probability(self, player, prompt: str, output: str) -> torch.Tensor:
        """Compute log probability of generated output given prompt."""
        # Tokenize input and output
        full_text = prompt + output
        inputs = player.tokenizer(prompt, return_tensors="pt", truncation=True)
        full_inputs = player.tokenizer(full_text, return_tensors="pt", truncation=True)

        # Get model first to determine its device
        model = player.get_training_model()
        model_device = next(model.parameters()).device

        # Move inputs to the same device as the model
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        full_inputs = {k: v.to(model_device) for k, v in full_inputs.items()}

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