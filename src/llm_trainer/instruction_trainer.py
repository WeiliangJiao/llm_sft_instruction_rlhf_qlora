"""
Instruction Tuning trainer with QLoRA.
"""
import os
from typing import Any

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from .config import ModelConfig, LoRAConfig, QuantizationConfig, InstructionConfig
from .data_util import load_instruction_dataset, preprocess_instruction_dataset, set_seed
from .model_util import load_model_for_sft, print_trainable_parameters
from .training_util import (
    plot_training_curve,
    evaluate_perplexity,
    save_model_and_tokenizer,
    setup_output_directory,
    print_training_info,
    log_gpu_memory,
)


class InstructionTrainer:
    """Instruction tuning trainer with QLoRA."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        lora_config: LoRAConfig,
        quant_config: QuantizationConfig,
        instruction_config: InstructionConfig
    ):
        self.model_config = model_config
        self.lora_config = lora_config
        self.quant_config = quant_config
        self.instruction_config = instruction_config
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
    
    def setup(self) -> None:
        """Setup model, tokenizer, and datasets."""
        set_seed(self.instruction_config.seed)
        
        # Setup output directory
        self.instruction_config.output_dir = setup_output_directory(
            self.instruction_config.output_dir, "instruction_qlora"
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
        
        print("Loading and preprocessing instruction dataset...")
        dataset = load_instruction_dataset(
            dataset_name=self.instruction_config.dataset_name,
            num_samples=self.instruction_config.num_samples,
            test_size=self.instruction_config.test_size,
            seed=self.instruction_config.seed
        )
        
        # Preprocess datasets
        self.train_dataset = preprocess_instruction_dataset(
            dataset["train"], 
            self.tokenizer,
            self.instruction_config.instruction_template,
            self.model_config.max_length
        )
        self.eval_dataset = preprocess_instruction_dataset(
            dataset["test"],
            self.tokenizer, 
            self.instruction_config.instruction_template,
            self.model_config.max_length
        )
        
        # Print setup info
        print_training_info(
            "Instruction Tuning",
            {"train": self.train_dataset, "test": self.eval_dataset},
            self.instruction_config
        )
    
    def create_trainer(self) -> None:
        """Create the Hugging Face trainer."""
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        training_args = TrainingArguments(
            output_dir=self.instruction_config.output_dir,
            eval_strategy=self.instruction_config.eval_strategy,
            save_strategy=self.instruction_config.save_strategy,
            save_steps=self.instruction_config.save_steps,
            eval_steps=self.instruction_config.eval_steps,
            logging_steps=self.instruction_config.logging_steps,
            learning_rate=self.instruction_config.learning_rate,
            per_device_train_batch_size=self.instruction_config.batch_size,
            per_device_eval_batch_size=self.instruction_config.batch_size,
            gradient_accumulation_steps=self.instruction_config.gradient_accumulation_steps,
            num_train_epochs=self.instruction_config.num_epochs,
            weight_decay=self.instruction_config.weight_decay,
            warmup_ratio=self.instruction_config.warmup_ratio,
            fp16=self.instruction_config.fp16,
            bf16=self.instruction_config.bf16,
            report_to=self.instruction_config.report_to,
            optim=self.instruction_config.optim,
            save_total_limit=self.instruction_config.save_total_limit,
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
        
        print("\nStarting instruction tuning...")
        log_gpu_memory()
        
        # Train the model
        train_result = self.trainer.train()
        
        print("\nInstruction tuning completed!")
        log_gpu_memory()
        
        # Save model and tokenizer
        save_model_and_tokenizer(
            self.trainer, self.tokenizer, self.instruction_config.output_dir
        )
        print(f"Model saved to: {self.instruction_config.output_dir}")
        
        # Evaluate and calculate perplexity
        eval_results = evaluate_perplexity(self.trainer)
        
        # Plot training curve
        plot_path = os.path.join(self.instruction_config.output_dir, "training_curve.png")
        plot_training_curve(
            self.trainer,
            title="Instruction Tuning Training Curve",
            save_path=plot_path
        )
        
        return {
            "train_result": train_result,
            "eval_results": eval_results,
        }
    
    def run_full_training(self) -> dict[str, Any]:
        """Run the complete instruction tuning pipeline."""
        self.setup()
        self.create_trainer()
        return self.train()
