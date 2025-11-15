"""
Reward Model trainer with QLoRA.
"""
import os
from typing import Any

from trl import RewardTrainer, RewardConfig

from .config import ModelConfig, LoRAConfig, QuantizationConfig, RewardModelConfig
from .data_util import load_preference_dataset, set_seed
from .model_util import load_model_for_reward, print_trainable_parameters
from .training_util import (
    plot_training_curve,
    save_model_and_tokenizer,
    setup_output_directory,
    print_training_info,
    log_gpu_memory,
)


class RewardModelTrainer:
    """Reward model trainer with QLoRA using TRL."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        quant_config: QuantizationConfig,
        reward_config: RewardModelConfig
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.reward_config = reward_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup(self) -> None:
        """Setup model, tokenizer, and datasets."""
        set_seed(self.reward_config.seed)
        
        # Setup output directory
        self.reward_config.output_dir = setup_output_directory(
            self.reward_config.output_dir, "reward_model_qlora"
        )
        
        print("Loading reward model and tokenizer...")
        if self.model_config.base_model_path:
            print(f"Loading from previous checkpoint: {self.model_config.base_model_path}")
        self.model, self.tokenizer = load_model_for_reward(
            self.model_config,
            self.lora_config,
            self.quant_config,
        )
        print_trainable_parameters(self.model)
        
        print("Loading preference dataset...")
        datasets = load_preference_dataset(
            dataset_name=self.reward_config.dataset_name,
            num_train_samples=self.reward_config.num_train_samples,
            num_val_samples=self.reward_config.num_val_samples,
            seed=self.reward_config.seed
        )
        
        self.train_dataset = datasets["train"]
        self.eval_dataset = datasets["validation"]
        
        # Print setup info
        print_training_info(
            "Reward Model Training",
            {"train": self.train_dataset, "validation": self.eval_dataset},
            self.reward_config
        )
    
    def create_trainer(self) -> None:
        """Create the TRL RewardTrainer."""        
        trl_config = RewardConfig(
            output_dir=self.reward_config.output_dir,
            per_device_train_batch_size=self.reward_config.batch_size,
            per_device_eval_batch_size=self.reward_config.batch_size,
            gradient_accumulation_steps=self.reward_config.gradient_accumulation_steps,
            learning_rate=self.reward_config.learning_rate,
            num_train_epochs=self.reward_config.num_epochs,
            logging_steps=self.reward_config.logging_steps,
            save_steps=self.reward_config.save_steps,
            eval_strategy=self.reward_config.eval_strategy,
            save_strategy=self.reward_config.save_strategy,
            eval_steps=self.reward_config.eval_steps,
            warmup_ratio=self.reward_config.warmup_ratio,
            fp16=self.reward_config.fp16,
            bf16=self.reward_config.bf16,
            report_to=self.reward_config.report_to,
            optim=self.reward_config.optim,
            save_total_limit=self.reward_config.save_total_limit,
        )
        
        self.trainer = RewardTrainer(
            model=self.model,
            args=trl_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )


    def train(self) -> dict[str, Any]:
        """Run the training process."""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call create_trainer() first.")
        
        print("\nStarting reward model training...")
        log_gpu_memory()
        
        # Train the model
        train_result = self.trainer.train()
        
        print("\nReward model training completed!")
        log_gpu_memory()
        
        # Save model and tokenizer
        save_model_and_tokenizer(
            self.trainer, self.tokenizer, self.reward_config.output_dir
        )
        print(f"Model saved to: {self.reward_config.output_dir}")
        
        # Evaluate the model
        eval_results = self.trainer.evaluate()
        print("Evaluation results:", eval_results)
        
        # Plot training curve
        plot_path = os.path.join(self.reward_config.output_dir, "training_curve.png")
        plot_training_curve(
            self.trainer,
            title="Reward Model Training Curve",
            save_path=plot_path
        )
        
        return {
            "train_result": train_result,
            "eval_results": eval_results,
        }
    
    def run_full_training(self) -> dict[str, Any]:
        """Run the complete reward model training pipeline."""
        self.setup()
        self.create_trainer()
        return self.train()
