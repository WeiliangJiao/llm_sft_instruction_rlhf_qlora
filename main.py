#!/usr/bin/env python3
"""
Main CLI interface for LLM training with QLoRA.
"""
import argparse
import sys

try:
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
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import `llm_trainer`. Install the project (e.g., `pip install -e .`) before running CLI commands."
    ) from exc


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="LLM Training with QLoRA - Modular framework for fine-tuning language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Supervised Fine-Tuning
  python main.py sft --model gpt2-medium --output ./results/sft
  
  # Instruction Tuning
  python main.py instruction --model gpt2-medium --samples 10000
  
  # Reward Model Training (after instruction tuning)
  python main.py reward --base-model-path ./results/instruction_qlora
  
  # DPO Training (after reward model training)
  python main.py dpo --base-model-path ./results/instruction_qlora
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Training method")
    
    # Initialize config objects for default values
    model_config = ModelConfig()
    sft_config = SFTConfig()
    inst_config = InstructionConfig()
    reward_config = RewardModelConfig()
    dpo_config = DPOConfig()
    
    # SFT subcommand
    sft_parser = subparsers.add_parser("sft", help="Supervised Fine-Tuning")
    sft_parser.add_argument("--model", default=model_config.base_model_id, help="Base model ID")
    sft_parser.add_argument("--base-model-path", help="Path to previously fine-tuned model (optional)")
    sft_parser.add_argument("--dataset", default=sft_config.dataset_name, help="Dataset name")
    sft_parser.add_argument("--dataset-split", default=sft_config.dataset_split, help="Dataset split")
    sft_parser.add_argument("--output", default=sft_config.output_dir, help="Output directory")
    sft_parser.add_argument("--max-length", type=int, default=model_config.max_length, help="Max sequence length")
    sft_parser.add_argument("--batch-size", type=int, default=sft_config.batch_size, help="Batch size")
    sft_parser.add_argument("--learning-rate", type=float, default=sft_config.learning_rate, help="Learning rate")
    sft_parser.add_argument("--epochs", type=int, default=sft_config.num_epochs, help="Number of epochs")
    
    # Instruction tuning subcommand
    inst_parser = subparsers.add_parser("instruction", help="Instruction Tuning")
    inst_parser.add_argument("--model", default=model_config.base_model_id, help="Base model ID")
    inst_parser.add_argument("--base-model-path", help="Path to previously fine-tuned model (optional)")
    inst_parser.add_argument("--dataset", default=inst_config.dataset_name, help="Dataset name")
    inst_parser.add_argument("--samples", type=int, default=inst_config.num_samples, help="Number of samples")
    inst_parser.add_argument("--output", default=inst_config.output_dir, help="Output directory")
    inst_parser.add_argument("--max-length", type=int, default=model_config.max_length, help="Max sequence length")
    inst_parser.add_argument("--batch-size", type=int, default=inst_config.batch_size, help="Batch size")
    inst_parser.add_argument("--learning-rate", type=float, default=inst_config.learning_rate, help="Learning rate")
    inst_parser.add_argument("--epochs", type=int, default=inst_config.num_epochs, help="Number of epochs")
    
    # Reward model subcommand
    reward_parser = subparsers.add_parser("reward", help="Reward Model Training")
    reward_parser.add_argument("--model", default=model_config.base_model_id, help="Base model ID")
    reward_parser.add_argument("--base-model-path", help="Path to fine-tuned model (optional)")
    reward_parser.add_argument("--dataset", default=reward_config.dataset_name, help="Dataset name")
    reward_parser.add_argument("--train-samples", type=int, default=reward_config.num_train_samples, help="Training samples")
    reward_parser.add_argument("--val-samples", type=int, default=reward_config.num_val_samples, help="Validation samples")
    reward_parser.add_argument("--output", default=reward_config.output_dir, help="Output directory")
    reward_parser.add_argument("--max-length", type=int, default=model_config.max_length, help="Max sequence length")
    reward_parser.add_argument("--batch-size", type=int, default=reward_config.batch_size, help="Batch size")
    reward_parser.add_argument("--learning-rate", type=float, default=reward_config.learning_rate, help="Learning rate")
    reward_parser.add_argument("--epochs", type=int, default=reward_config.num_epochs, help="Number of epochs")
    
    # DPO subcommand
    dpo_parser = subparsers.add_parser("dpo", help="Direct Preference Optimization")
    dpo_parser.add_argument("--model", default=model_config.base_model_id, help="Base model ID")
    dpo_parser.add_argument("--base-model-path", help="Path to fine-tuned model (optional)")
    dpo_parser.add_argument("--dataset", default=dpo_config.dataset_name, help="Dataset name")
    dpo_parser.add_argument("--train-samples", type=int, default=dpo_config.num_train_samples, help="Training samples")
    dpo_parser.add_argument("--val-samples", type=int, default=dpo_config.num_val_samples, help="Validation samples")
    dpo_parser.add_argument("--output", default=dpo_config.output_dir, help="Output directory")
    dpo_parser.add_argument("--max-length", type=int, default=model_config.max_length, help="Max sequence length")
    dpo_parser.add_argument("--batch-size", type=int, default=dpo_config.batch_size, help="Batch size")
    dpo_parser.add_argument("--learning-rate", type=float, default=dpo_config.learning_rate, help="Learning rate")
    dpo_parser.add_argument("--epochs", type=int, default=dpo_config.num_epochs, help="Number of epochs")
    dpo_parser.add_argument("--beta", type=float, default=dpo_config.beta, help="DPO beta parameter")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "sft":
            print("Starting Supervised Fine-Tuning...")
            
            # Create configurations
            model_config = ModelConfig(
                base_model_id=args.model,
                base_model_path=args.base_model_path,
                max_length=args.max_length
            )
            lora_config = LoRAConfig()
            quant_config = QuantizationConfig()
            sft_config = SFTConfig(
                dataset_name=args.dataset,
                dataset_split=args.dataset_split,
                output_dir=args.output,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
            )
            
            # Run training
            trainer = SFTTrainer(model_config, lora_config, quant_config, sft_config)
            results = trainer.run_full_training()
            
        elif args.command == "instruction":
            print("Starting Instruction Tuning...")
            
            # Create configurations
            model_config = ModelConfig(
                base_model_id=args.model,
                base_model_path=args.base_model_path,
                max_length=args.max_length
            )
            lora_config = LoRAConfig(
                target_modules=["c_attn", "c_proj", "c_fc"]  # More modules for instruction tuning
            )
            quant_config = QuantizationConfig()
            instruction_config = InstructionConfig(
                dataset_name=args.dataset,
                num_samples=args.samples,
                output_dir=args.output,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
            )
            
            # Run training
            trainer = InstructionTrainer(model_config, lora_config, quant_config, instruction_config)
            results = trainer.run_full_training()
            
        elif args.command == "reward":
            print("Starting Reward Model Training...")
            
            # Create configurations
            model_config = ModelConfig(
                base_model_id=args.model,
                base_model_path=args.base_model_path,
                max_length=args.max_length
            )
            lora_config = LoRAConfig(
                target_modules=["c_attn", "c_proj"]  # Focused on attention for reward modeling
            )
            quant_config = QuantizationConfig()
            reward_config = RewardModelConfig(
                dataset_name=args.dataset,
                num_train_samples=args.train_samples,
                num_val_samples=args.val_samples,
                output_dir=args.output,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
            )
            
            # Run training
            trainer = RewardModelTrainer(model_config, lora_config, quant_config, reward_config)
            results = trainer.run_full_training()
            
        elif args.command == "dpo":
            print("Starting DPO Training...")
            
            # Create configurations
            model_config = ModelConfig(
                base_model_id=args.model,
                base_model_path=args.base_model_path,
                max_length=args.max_length
            )
            lora_config = LoRAConfig(
                target_modules=["c_attn", "c_proj"]  # Standard attention modules for DPO
            )
            quant_config = QuantizationConfig()
            dpo_config = DPOConfig(
                dataset_name=args.dataset,
                num_train_samples=args.train_samples,
                num_val_samples=args.val_samples,
                output_dir=args.output,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                num_epochs=args.epochs,
                beta=args.beta,
            )
            
            # Run training
            trainer = DPOTrainerCustom(model_config, lora_config, quant_config, dpo_config)
            results = trainer.run_full_training()
        
        print("\nTraining completed successfully!")
        print(f"Results: {results}")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
