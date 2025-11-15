"""
Configuration module for LLM training with QLoRA.
Contains shared constants and configurations across all training methods.
"""
from dataclasses import dataclass, field
import warnings
import torch


def _is_mps_available() -> bool:
    """Return True if MPS backend is available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _detect_default_device() -> str:
    """Detect the best default device for the current host."""
    if torch.cuda.is_available():
        return "cuda"
    if _is_mps_available():
        return "mps"
    return "cpu"


def _default_compute_dtype() -> torch.dtype:
    """Select an appropriate compute dtype for the detected hardware."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if _is_mps_available():
        return torch.float16
    return torch.float32


def _default_quantization_flag() -> bool:
    """Enable 4-bit quantization only when CUDA is available."""
    return torch.cuda.is_available()


def _default_optimizer() -> str:
    """Pick a compatible optimizer for the detected hardware."""
    return "paged_adamw_8bit" if torch.cuda.is_available() else "adamw_torch"


@dataclass
class ModelConfig:
    """Base model configuration."""
    base_model_id: str = "gpt2-medium"
    base_model_path: str | None = None
    max_length: int = 256
    device: str = field(default_factory=_detect_default_device)


@dataclass
class LoRAConfig:
    """LoRA-specific configuration."""
    r: int = 4 # 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | None = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["c_attn", "c_proj"]  # GPT-2 default


@dataclass
class QuantizationConfig:
    """4-bit quantization configuration."""
    load_in_4bit: bool = field(default_factory=_default_quantization_flag)
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: torch.dtype = field(default_factory=_default_compute_dtype)
    
    def __post_init__(self):
        # BitsAndBytes 4-bit requires CUDA hardware.
        # Protects against invalid user overrides
        if self.load_in_4bit and not torch.cuda.is_available():
            warnings.warn(
                "4-bit quantization is only supported on CUDA devices. "
                "Falling back to full-precision weights.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.load_in_4bit = False


@dataclass
class TrainingConfig:
    """Base training configuration."""
    output_dir: str = "./results"
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    batch_size: int = 1 # 8
    gradient_accumulation_steps: int = 1 # 4
    num_epochs: int = 1
    warmup_ratio: float = 0.05
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    save_total_limit: int = 2
    seed: int = 42
    report_to: str = "none"
    
    def __post_init__(self):
        # Auto-detect precision based on hardware capability
        self.fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
        self.bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()


@dataclass
class SFTConfig(TrainingConfig):
    """Supervised Fine-Tuning specific configuration."""
    dataset_name: str = "vblagoje/cc_news"
    dataset_split: str = "train[:2%]"
    test_size: float = 0.1
    optim: str = field(default_factory=_default_optimizer)


@dataclass
class InstructionConfig(TrainingConfig):
    """Instruction tuning specific configuration."""
    dataset_name: str = "databricks/databricks-dolly-15k"
    num_samples: int = 12000
    test_size: float = 0.02
    instruction_template: str = (
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{inp}\n\n"
        "### Response:\n{out}\n"
    )
    optim: str = field(default_factory=_default_optimizer)


@dataclass
class RewardModelConfig(TrainingConfig):
    """Reward model training configuration."""
    dataset_name: str = "Anthropic/hh-rlhf"
    num_train_samples: int = 4000
    num_val_samples: int = 400
    learning_rate: float = 1e-4  # Lower LR for reward model
    optim: str = field(default_factory=_default_optimizer)


@dataclass
class DPOConfig(TrainingConfig):
    """Direct Preference Optimization configuration."""
    dataset_name: str = "Anthropic/hh-rlhf"
    num_train_samples: int = 4000
    num_val_samples: int = 400
    learning_rate: float = 1e-5  
    batch_size: int = 1  # 4
    beta: float = 0.1  # DPO regularization parameter
    optim: str = field(default_factory=_default_optimizer)


# Global constants
RANDOM_SEED = 42
DEFAULT_OUTPUT_BASE = "./results"
SUPPORTED_MODELS = ["gpt2", "gpt2-medium", "gpt2-large"]
