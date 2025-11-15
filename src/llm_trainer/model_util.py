"""
Model utilities for loading and configuring models with QLoRA.
"""
from typing import Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2ForSequenceClassification,
    GPT2Tokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType

from .config import ModelConfig, LoRAConfig, QuantizationConfig


def _resolve_model_id(model_config: ModelConfig) -> str:
    """Return the identifier/path to load the model from."""
    return model_config.base_model_path or model_config.base_model_id


def _is_mps_available() -> bool:
    """Return True if Apple's Metal backend can be used."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _maybe_move_model_to_device(model: Any, device: str) -> None:
    """Move the model to the requested device when quantization is disabled."""
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")
    elif device == "mps" and _is_mps_available():
        model.to("mps")
    elif device == "cpu":
        model.to("cpu")


def create_bnb_config(quant_config: QuantizationConfig) -> BitsAndBytesConfig | None:
    """Create BitsAndBytesConfig for 4-bit quantization, if enabled."""
    if not quant_config.load_in_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=quant_config.load_in_4bit,
        bnb_4bit_use_double_quant=quant_config.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=quant_config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=quant_config.bnb_4bit_compute_dtype,
    )


def create_lora_config(
    lora_config: LoRAConfig, 
    task_type: TaskType = TaskType.CAUSAL_LM
) -> LoraConfig:
    """Create LoRA configuration."""
    return LoraConfig(
        r=lora_config.r,
        lora_alpha=lora_config.lora_alpha,
        lora_dropout=lora_config.lora_dropout,
        target_modules=lora_config.target_modules,
        task_type=task_type,
        bias="none"
    )


def load_base_model_and_tokenizer(
    model_config: ModelConfig,
    quant_config: QuantizationConfig,
    for_sequence_classification: bool = False,
    num_labels: int = 1
) -> tuple[Any, AutoTokenizer]:
    """Load base model and tokenizer with quantization."""
    bnb_config = create_bnb_config(quant_config)
    model_kwargs: dict[str, Any] = {}
    model_source = _resolve_model_id(model_config)
    
    if bnb_config:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = quant_config.bnb_4bit_compute_dtype
    
    if for_sequence_classification:
        model = GPT2ForSequenceClassification.from_pretrained(
            model_source,
            num_labels=num_labels,
            **model_kwargs,
        )
        tokenizer = GPT2Tokenizer.from_pretrained(model_source)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            **model_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_source)
    
    if not bnb_config:
        _maybe_move_model_to_device(model, model_config.device)
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    
    return model, tokenizer


def setup_lora_model(
    base_model: Any,
    lora_config: LoRAConfig,
    task_type: TaskType = TaskType.CAUSAL_LM
) -> Any:
    """Apply LoRA to base model."""
    peft_config = create_lora_config(lora_config, task_type)
    model = get_peft_model(base_model, peft_config)
    
    # For reward models, initialize small weights
    if task_type == TaskType.SEQ_CLS and hasattr(model, 'score'):
        try:
            import torch
            with torch.no_grad():
                torch.nn.init.normal_(model.score.weight, std=1e-3)
                if model.score.bias is not None:
                    torch.nn.init.zeros_(model.score.bias)
        except ImportError:
            pass
    
    return model


def load_model_for_sft(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    quant_config: QuantizationConfig,
) -> tuple[Any, AutoTokenizer]:
    """Load model and tokenizer for supervised fine-tuning."""
    base_model, tokenizer = load_base_model_and_tokenizer(
        model_config, quant_config, for_sequence_classification=False
    )
    
    model = setup_lora_model(base_model, lora_config, TaskType.CAUSAL_LM)
    
    return model, tokenizer


def load_model_for_reward(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    quant_config: QuantizationConfig,
) -> tuple[Any, AutoTokenizer]:
    """Load model and tokenizer for reward modeling."""
    base_model, tokenizer = load_base_model_and_tokenizer(
        model_config, quant_config, for_sequence_classification=True, num_labels=1
    )
    
    model = setup_lora_model(base_model, lora_config, TaskType.SEQ_CLS)
    
    return model, tokenizer


def load_models_for_dpo(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    quant_config: QuantizationConfig,
) -> tuple[Any, Any, AutoTokenizer]:
    """Load policy and reference models for DPO training."""
    # Load policy model (will be trained)
    policy_model, tokenizer = load_base_model_and_tokenizer(
        model_config, quant_config, for_sequence_classification=False
    )
    policy_model = setup_lora_model(policy_model, lora_config, TaskType.CAUSAL_LM)
    
    # Load reference model (frozen)
    ref_model, _ = load_base_model_and_tokenizer(
        model_config, quant_config, for_sequence_classification=False
    )
    
    return policy_model, ref_model, tokenizer


def print_trainable_parameters(model: Any) -> None:
    """Print trainable parameters information."""
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable_params:,}")
        print(f"All params: {total_params:,}")
        print(f"Trainable %: {100 * trainable_params / total_params:.2f}")
