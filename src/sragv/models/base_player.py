"""
Base player class for SRAG-V 4-player architecture.
Implements common functionality for all players.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)

logger = logging.getLogger(__name__)

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel
    )
    PEFT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"PEFT not available: {e}. LoRA functionality will be disabled.")
    PEFT_AVAILABLE = False


class BasePlayer(ABC):
    """Base class for all SRAG-V players."""
    
    def __init__(
        self,
        model_name: str,
        max_length: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        device: Union[str, int] = "auto",
        quantization: Optional[str] = None,
        lora_config: Optional[Dict] = None
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.device = device
        self.quantization = quantization
        self.lora_config = lora_config
        
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
        logger.info(f"Initializing {self.__class__.__name__} with model {model_name}")
        
    def load_model(self):
        """Load the model and tokenizer."""
        logger.info(f"Loading model {self.model_name}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Add pad token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configure quantization if specified and GPU available
        quantization_config = None
        if self.quantization in ["4bit", "8bit"]:
            try:
                import bitsandbytes as bnb
                # Check if bitsandbytes has GPU support
                if hasattr(bnb.functional, 'cadam32bit_grad_fp32'):
                    if self.quantization == "4bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                        )
                    elif self.quantization == "8bit":
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                        )
                    logger.info(f"Using {self.quantization} quantization")
                else:
                    logger.warning(f"Quantization requested but bitsandbytes has no GPU support, loading in fp32")
            except (ImportError, AttributeError):
                logger.warning(f"Quantization requested but bitsandbytes not available, loading in fp32")
        
        # Load model WITHOUT device_map (we'll move it explicitly after)
        # device_map adds Accelerate hooks that prevent model movement
        # Use fp16 for memory efficiency (2x reduction vs fp32) unless quantized
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Always use fp16 for memory efficiency
            low_cpu_mem_usage=True
        )

        # Setup LoRA if configured and available
        if self.lora_config and PEFT_AVAILABLE:
            logger.info("Setting up LoRA adapters...")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_config.get("rank", 32),
                lora_alpha=self.lora_config.get("alpha", 64),
                target_modules=self.lora_config.get("target_modules", ["q_proj", "v_proj"]),
                lora_dropout=self.lora_config.get("dropout", 0.1),
                bias="none",
            )
            self.peft_model = get_peft_model(self.model, lora_config)
            logger.info(f"LoRA setup complete. Trainable parameters: {self.peft_model.print_trainable_parameters()}")
        elif self.lora_config and not PEFT_AVAILABLE:
            logger.warning("LoRA configuration provided but PEFT not available. Using base model only.")

        # Move model to target device (standard PyTorch approach for multi-GPU)
        # This must happen AFTER LoRA setup so both base model and adapters move together
        target_device = None
        if self.device != "auto":
            target_device = f"cuda:{self.device}" if isinstance(self.device, int) else self.device
            logger.info(f"Moving model to device: {target_device}")
            if self.peft_model:
                self.peft_model = self.peft_model.to(target_device)
            else:
                self.model = self.model.to(target_device)
        elif torch.cuda.is_available():
            # If device is "auto" and CUDA is available, use cuda:0
            target_device = "cuda:0"
            logger.info(f"Moving model to device: {target_device} (auto)")
            if self.peft_model:
                self.peft_model = self.peft_model.to(target_device)
            else:
                self.model = self.model.to(target_device)

        final_device = target_device if target_device else "cpu"
        logger.info(f"Model {self.model_name} loaded successfully on {final_device}")
    
    def generate_text(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None
    ) -> List[str]:
        """Generate text from a prompt or chat messages (SOTA Qwen2.5-Coder format)."""
        if self.model is None:
            self.load_model()
        
        # Use instance defaults if not specified
        temperature = temperature if temperature is not None else self.temperature
        top_p = top_p if top_p is not None else self.top_p
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_length // 2
        
        # Handle both prompt and messages format
        if messages is not None:
            # Use chat template for messages (SOTA approach for Qwen2.5-Coder)
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template failed, falling back to simple format: {e}")
                # Fallback: simple concatenation
                formatted_prompt = ""
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        formatted_prompt += f"System: {content}\n\n"
                    elif role == "user":
                        formatted_prompt += f"User: {content}\n\nAssistant: "
        elif prompt is not None:
            formatted_prompt = prompt
        else:
            raise ValueError("Either prompt or messages must be provided")
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True
        )

        # Get model and move inputs to its device
        model_to_use = self.peft_model if self.peft_model else self.model
        model_device = next(model_to_use.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": num_return_sequences,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.05,  # Reduced for better quality
            "length_penalty": 1.0,
        }
        
        # Remove None values to avoid errors
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        
        with torch.no_grad():
            try:
                outputs = model_to_use.generate(**generation_kwargs)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return []
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Remove the input prompt from the output
            input_length = inputs["input_ids"].shape[1] 
            generated_tokens = output[input_length:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text.strip())
        
        return generated_texts
    
    @abstractmethod
    def process_input(self, **kwargs) -> str:
        """Process input and create a prompt for the model."""
        pass
    
    @abstractmethod
    def parse_output(self, output: str) -> Any:
        """Parse model output into structured format."""
        pass
    
    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """Main generation method for this player."""
        pass
    
    def get_training_model(self):
        """Get the model to use for training (PEFT model if available)."""
        return self.peft_model if self.peft_model else self.model
    
    def save_adapter(self, save_path: str):
        """Save LoRA adapter if available."""
        if self.peft_model:
            self.peft_model.save_pretrained(save_path)
            logger.info(f"Saved LoRA adapter to {save_path}")
        else:
            logger.warning("No LoRA adapter to save")
    
    def load_adapter(self, adapter_path: str):
        """Load LoRA adapter."""
        if not PEFT_AVAILABLE:
            logger.warning("Cannot load LoRA adapter: PEFT not available")
            return
            
        if self.model is None:
            self.load_model()
        
        self.peft_model = PeftModel.from_pretrained(self.model, adapter_path)
        logger.info(f"Loaded LoRA adapter from {adapter_path}")
    
    def unload_model(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            self.model = None
        if self.peft_model:
            del self.peft_model
            self.peft_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Unloaded model {self.model_name}")


class PlayerConfig:
    """Configuration class for players."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        quantization: Optional[str] = None,
        lora_rank: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        device: Union[str, int] = "auto"
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.quantization = quantization
        self.device = device

        # LoRA configuration
        self.lora_config = {
            "rank": lora_rank,
            "alpha": lora_alpha,
            "dropout": lora_dropout,
            "target_modules": target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        } if lora_rank > 0 else None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "quantization": self.quantization,
            "lora_config": self.lora_config,
            "device": self.device
        }