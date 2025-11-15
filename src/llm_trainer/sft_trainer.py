"""
Supervised Fine-Tuning (SFT) trainer with QLoRA.
"""
import os
from typing import Any

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from .config import ModelConfig, LoRAConfig, QuantizationConfig, SFTConfig
from .data_util import load_sft_dataset, preprocess_sft_dataset, set_seed
from .model_util import load_model_for_sft, print_trainable_parameters
from .training_util import (
    plot_training_curve,
    evaluate_perplexity,
    save_model_and_tokenizer,
    setup_output_directory,
    print_training_info,
    log_gpu_memory,
)


class SFTTrainer:
    """Supervised Fine-Tuning trainer with QLoRA."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        quant_config: QuantizationConfig,
        sft_config: SFTConfig
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.sft_config = sft_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup(self) -> None:
        """Setup model, tokenizer, and datasets."""
        set_seed(self.sft_config.seed)
        
        # Setup output directory
        self.sft_config.output_dir = setup_output_directory(
            self.sft_config.output_dir, "sft_qlora"
        )
        
        print("Loading model and tokenizer...")
        if self.model_config.base_model_path:
            print(f"Loading from previous checkpoint: {self.model_config.base_model_path}")
        self.model, self.tokenizer = load_model_for_sft(
            self.model_config,
            self.lora_config,
            self.quant_config,
        )
        print_trainable_parameters(self.model)
        
        print("Loading and preprocessing dataset...")
        dataset = load_sft_dataset(
            dataset_name=self.sft_config.dataset_name,
            dataset_split=self.sft_config.dataset_split,
            test_size=self.sft_config.test_size,
            seed=self.sft_config.seed
        )
        
        # Preprocess datasets
        self.train_dataset = preprocess_sft_dataset(
            dataset["train"], self.tokenizer, self.model_config.max_length
        )
        self.eval_dataset = preprocess_sft_dataset(
            dataset["test"], self.tokenizer, self.model_config.max_length
        )
        
        # Print setup info
        print_training_info(
            "Supervised Fine-Tuning (SFT)",
            {"train": self.train_dataset, "test": self.eval_dataset},
            self.sft_config
        )
    
    def create_trainer(self) -> None:
        """Create the Hugging Face trainer."""
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir=self.sft_config.output_dir,
            eval_strategy=self.sft_config.eval_strategy,
            save_strategy=self.sft_config.save_strategy,
            save_steps=self.sft_config.save_steps,
            eval_steps=self.sft_config.eval_steps,
            logging_steps=self.sft_config.logging_steps,
            learning_rate=self.sft_config.learning_rate,
            per_device_train_batch_size=self.sft_config.batch_size,
            per_device_eval_batch_size=self.sft_config.batch_size,
            gradient_accumulation_steps=self.sft_config.gradient_accumulation_steps,
            num_train_epochs=self.sft_config.num_epochs,
            weight_decay=self.sft_config.weight_decay,
            warmup_ratio=self.sft_config.warmup_ratio,
            fp16=self.sft_config.fp16,
            bf16=self.sft_config.bf16,
            report_to=self.sft_config.report_to,
            optim=self.sft_config.optim,
            save_total_limit=self.sft_config.save_total_limit,
            dataloader_drop_last=True,
        )
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
    
    def train(self) -> dict[str, Any]:
        """Run the training process."""
        if not self.trainer:
            raise ValueError("Trainer not initialized. Call create_trainer() first.")
        
        print("\nStarting supervised fine-tuning...")
        log_gpu_memory()
        
        # Train the model
        train_result = self.trainer.train()
        
        print("\nSupervised fine-tuning completed!")
        log_gpu_memory()
        
        # Save model and tokenizer
        save_model_and_tokenizer(
            self.trainer, self.tokenizer, self.sft_config.output_dir
        )
        print(f"Model saved to: {self.sft_config.output_dir}")
        
        # Evaluate and calculate perplexity
        eval_results = evaluate_perplexity(self.trainer)
        
        # Plot training curve
        plot_path = os.path.join(self.sft_config.output_dir, "training_curve.png")
        plot_training_curve(
            self.trainer,
            title="SFT Training Curve",
            save_path=plot_path
        )
        
        return {
            "train_result": train_result,
            "eval_results": eval_results,
        }
    
    def run_full_training(self) -> dict[str, Any]:
        """Run the complete SFT training pipeline."""
        self.setup()
        self.create_trainer()
        return self.train()
