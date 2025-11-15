"""
LLM Trainer: A modular framework for fine-tuning language models with QLoRA.

This package provides a clean, modular approach to:
- Supervised Fine-Tuning (SFT)
- Instruction Tuning
- Reward Model Training
- Direct Preference Optimization (DPO)

All methods use QLoRA for parameter-efficient fine-tuning.
"""

from .sft_trainer import SFTTrainer
from .instruction_trainer import InstructionTrainer
from .reward_trainer import RewardModelTrainer
from .dpo_trainer import DPOTrainerCustom

from .config import (
    ModelConfig,
    LoRAConfig,
    QuantizationConfig,
    SFTConfig,
    InstructionConfig,
    RewardModelConfig,
    DPOConfig,
)

__version__ = "0.1.0"
__author__ = "W.J."

__all__ = [
    # Trainer classes
    "SFTTrainer",
    "InstructionTrainer", 
    "RewardModelTrainer",
    "DPOTrainerCustom",
    
    # Configuration classes
    "ModelConfig",
    "LoRAConfig", 
    "QuantizationConfig",
    "SFTConfig",
    "InstructionConfig",
    "RewardModelConfig",
    "DPOConfig",
]