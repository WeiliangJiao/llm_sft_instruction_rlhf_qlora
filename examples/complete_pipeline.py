#!/usr/bin/env python3
"""Minimal SFT → Instruction → Reward → DPO pipeline test run."""

import argparse
import sys
from pathlib import Path

from llm_trainer import (
    SFTTrainer,
    InstructionTrainer,
    RewardModelTrainer,
    DPOTrainerCustom,
)
from llm_trainer.config import (
    ModelConfig,
    LoRAConfig,
    QuantizationConfig,
    SFTConfig,
    InstructionConfig,
    RewardModelConfig,
    DPOConfig,
)


BASE_MODEL = "gpt2"
MAX_LENGTH = 64
OUTPUT_DIR = Path("./results/minimal_pipeline").resolve()

# Settings for minimal pipeline run
MINI_SETTINGS = {
    "sft": {
        "dataset_split": "train[:100]",
        "learning_rate": 1e-4,
        "batch_size": 1,
        "num_epochs": 1,
    },
    "instruction": {
        "num_samples": 100,
        "learning_rate": 1e-4,
        "batch_size": 1,
        "num_epochs": 1,
    },
    "reward": {
        "num_train_samples": 100,
        "num_val_samples": 20,
        "learning_rate": 5e-5,
        "batch_size": 1,
        "num_epochs": 1,
    },
    "dpo": {
        "num_train_samples": 100,
        "num_val_samples": 20,
        "learning_rate": 5e-6,
        "batch_size": 1,
        "num_epochs": 1,
        "beta": 0.1,
    },
}


def _print_stage_header(title: str) -> None:
    print(f"\n{title}")
    print("-" * 40)


def run_minimal_pipeline() -> None:
    """Run the minimal end-to-end pipeline with gpt2 on small datasets
    with minimal training settings."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stage_dirs = {
        "sft": OUTPUT_DIR / "sft",
        "instruction": OUTPUT_DIR / "instruction",
        "reward": OUTPUT_DIR / "reward",
        "dpo": OUTPUT_DIR / "dpo",
    }
    checkpoints = {
        "sft": stage_dirs["sft"] / "sft_qlora",
        "instruction": stage_dirs["instruction"] / "instruction_qlora",
        "reward": stage_dirs["reward"] / "reward_model_qlora",
        "dpo": stage_dirs["dpo"] / "dpo_rlhf_qlora",
    }

    quant_config = QuantizationConfig()
    stage_results: dict[str, dict] = {}

    print("Running minimal pipeline (gpt2, tiny datasets, minimal epochs)...")
    print("=" * 60)

    try:
        _print_stage_header("Step 1: Supervised Fine-Tuning")
        sft_model_config = ModelConfig(base_model_id=BASE_MODEL, max_length=MAX_LENGTH)
        lora_config = LoRAConfig()
        sft_config = SFTConfig(
            dataset_split=MINI_SETTINGS["sft"]["dataset_split"],
            output_dir=str(stage_dirs["sft"]),
            learning_rate=MINI_SETTINGS["sft"]["learning_rate"],
            batch_size=MINI_SETTINGS["sft"]["batch_size"],
            num_epochs=MINI_SETTINGS["sft"]["num_epochs"],
        )
        sft_trainer = SFTTrainer(sft_model_config, lora_config, quant_config, sft_config)
        stage_results["sft"] = sft_trainer.run_full_training()

        _print_stage_header("Step 2: Instruction Tuning")
        lora_config_inst = LoRAConfig(target_modules=["c_attn", "c_proj", "c_fc"])
        instruction_model_config = ModelConfig(
            base_model_id=BASE_MODEL,
            base_model_path=str(checkpoints["sft"]),
            max_length=MAX_LENGTH,
        )
        instruction_config = InstructionConfig(
            num_samples=MINI_SETTINGS["instruction"]["num_samples"],
            output_dir=str(stage_dirs["instruction"]),
            learning_rate=MINI_SETTINGS["instruction"]["learning_rate"],
            batch_size=MINI_SETTINGS["instruction"]["batch_size"],
            num_epochs=MINI_SETTINGS["instruction"]["num_epochs"],
        )
        inst_trainer = InstructionTrainer(
            instruction_model_config, lora_config_inst, quant_config, instruction_config
        )
        stage_results["instruction"] = inst_trainer.run_full_training()

        _print_stage_header("Step 3: Reward Model Training")
        lora_config_reward = LoRAConfig(target_modules=["c_attn", "c_proj"])
        reward_model_config = ModelConfig(
            base_model_id=BASE_MODEL,
            base_model_path=str(checkpoints["instruction"]),
            max_length=MAX_LENGTH,
        )
        reward_config = RewardModelConfig(
            num_train_samples=MINI_SETTINGS["reward"]["num_train_samples"],
            num_val_samples=MINI_SETTINGS["reward"]["num_val_samples"],
            output_dir=str(stage_dirs["reward"]),
            learning_rate=MINI_SETTINGS["reward"]["learning_rate"],
            batch_size=MINI_SETTINGS["reward"]["batch_size"],
            num_epochs=MINI_SETTINGS["reward"]["num_epochs"],
        )
        reward_trainer = RewardModelTrainer(
            reward_model_config, lora_config_reward, quant_config, reward_config
        )
        stage_results["reward"] = reward_trainer.run_full_training()

        _print_stage_header("Step 4: Direct Preference Optimization")
        dpo_model_config = ModelConfig(
            base_model_id=BASE_MODEL,
            base_model_path=str(checkpoints["instruction"]),
            max_length=MAX_LENGTH,
        )
        dpo_config = DPOConfig(
            num_train_samples=MINI_SETTINGS["dpo"]["num_train_samples"],
            num_val_samples=MINI_SETTINGS["dpo"]["num_val_samples"],
            output_dir=str(stage_dirs["dpo"]),
            learning_rate=MINI_SETTINGS["dpo"]["learning_rate"],
            batch_size=MINI_SETTINGS["dpo"]["batch_size"],
            num_epochs=MINI_SETTINGS["dpo"]["num_epochs"],
            beta=MINI_SETTINGS["dpo"]["beta"],
        )
        dpo_trainer = DPOTrainerCustom(dpo_model_config, lora_config_reward, quant_config, dpo_config)
        stage_results["dpo"] = dpo_trainer.run_full_training()

        print("\nPipeline finished successfully!")
        print("=" * 60)
        print("Artifacts saved at:")
        print(f"  - SFT: {checkpoints['sft']}")
        print(f"  - Instruction: {checkpoints['instruction']}")
        print(f"  - Reward: {checkpoints['reward']}")
        print(f"  - DPO: {checkpoints['dpo']}")

        print("\nTraining Summary:")
        print(
            f"  - SFT perplexity: {stage_results['sft']['eval_results'].get('perplexity', 'N/A')}"
        )
        print(
            "  - Instruction perplexity: "
            f"{stage_results['instruction']['eval_results'].get('perplexity', 'N/A')}"
        )
        print("  - Reward eval:", stage_results["reward"]["eval_results"])
        print("  - DPO eval:", stage_results["dpo"]["eval_results"])

    except Exception as e:  
        print(f"\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal GPT-2 pipeline for Mac M1 (SFT → Instruction → Reward → DPO)."
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR),
        help="Directory to store stage outputs (defaults to ./results/minimal_pipeline).",
    )
    args = parser.parse_args()

    OUTPUT_DIR = Path(args.output).resolve()
    run_minimal_pipeline()
