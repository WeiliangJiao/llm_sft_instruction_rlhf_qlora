"""
Direct Preference Optimization (DPO) trainer with QLoRA.
"""
import os
from typing import Any

from trl import DPOTrainer, DPOConfig

from .config import ModelConfig, LoRAConfig, QuantizationConfig, DPOConfig as DPOTrainingConfig
from .data_util import load_preference_dataset, set_seed
from .model_util import load_models_for_dpo, print_trainable_parameters
from .training_util import (
    plot_training_curve,
    save_model_and_tokenizer,
    setup_output_directory,
    print_training_info,
    log_gpu_memory,
)


class DPOTrainerCustom:
    """Direct Preference Optimization trainer with QLoRA using TRL."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        quant_config: QuantizationConfig,
        dpo_config: DPOTrainingConfig
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.dpo_config = dpo_config
        
        self.policy_model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup(self) -> None:
        """Setup models, tokenizer, and datasets."""
        set_seed(self.dpo_config.seed)
        
        # Setup output directory
        self.dpo_config.output_dir = setup_output_directory(
            self.dpo_config.output_dir, "dpo_rlhf_qlora"
        )
        
        print("Loading policy and reference models...")
        if self.model_config.base_model_path:
            print(f"Loading from previous checkpoint: {self.model_config.base_model_path}")
        self.policy_model, self.ref_model, self.tokenizer = load_models_for_dpo(
            self.model_config,
            self.lora_config,
            self.quant_config,
        )
        print_trainable_parameters(self.policy_model)
        
        print("Loading preference dataset...")
        datasets = load_preference_dataset(
            dataset_name=self.dpo_config.dataset_name,
            num_train_samples=self.dpo_config.num_train_samples,
            num_val_samples=self.dpo_config.num_val_samples,
            seed=self.dpo_config.seed
        )
        
        self.train_dataset = datasets["train"]
        self.eval_dataset = datasets["validation"]
        
        # Print setup info
        print_training_info(
            "Direct Preference Optimization (DPO)",
            {"train": self.train_dataset, "validation": self.eval_dataset},
            self.dpo_config
        )
    
    def create_trainer(self) -> None:
        """Create the TRL DPOTrainer."""
        trl_config = DPOConfig(
            output_dir=self.dpo_config.output_dir,
            per_device_train_batch_size=self.dpo_config.batch_size,
            per_device_eval_batch_size=self.dpo_config.batch_size,
            gradient_accumulation_steps=self.dpo_config.gradient_accumulation_steps,
            learning_rate=self.dpo_config.learning_rate,
            num_train_epochs=self.dpo_config.num_epochs,
            logging_steps=self.dpo_config.logging_steps,
            save_steps=self.dpo_config.save_steps,
            eval_strategy=self.dpo_config.eval_strategy,
            save_strategy=self.dpo_config.save_strategy,
            eval_steps=self.dpo_config.eval_steps,
            warmup_ratio=self.dpo_config.warmup_ratio,
            remove_unused_columns=False,
            beta=self.dpo_config.beta,
            report_to=self.dpo_config.report_to,
            fp16=self.dpo_config.fp16,
            bf16=self.dpo_config.bf16,
            optim=self.dpo_config.optim,
            save_total_limit=self.dpo_config.save_total_limit,
        )

        self.trainer = DPOTrainer(
            model=self.policy_model,
            ref_model=self.ref_model,
            args=trl_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )
    
    def train(self) -> dict[str, Any]:
        """Run the training process."""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call create_trainer() first.")
        
        print("\nStarting DPO training...")
        log_gpu_memory()
        
        # Train the model
        train_result = self.trainer.train()
        
        print("\nDPO training completed!")
        log_gpu_memory()
        
        # Save model and tokenizer
        save_model_and_tokenizer(
            self.trainer, self.tokenizer, self.dpo_config.output_dir
        )
        print(f"Model saved to: {self.dpo_config.output_dir}")
        
        # Evaluate the model
        eval_results = self.trainer.evaluate()
        print("Evaluation results:", eval_results)
        
        # Plot training curve
        plot_path = os.path.join(self.dpo_config.output_dir, "training_curve.png")
        plot_training_curve(
            self.trainer,
            title="DPO Training Curve",
            save_path=plot_path
        )
        
        return {
            "train_result": train_result,
            "eval_results": eval_results,
        }
    
    def run_full_training(self) -> dict[str, Any]:
        """Run the complete DPO training pipeline."""
        self.setup()
        self.create_trainer()
        return self.train()
