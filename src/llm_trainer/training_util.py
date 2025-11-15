"""
Training utilities and common functions for all training methods.
"""
import os
import math
from typing import Any
from transformers import Trainer


def plot_training_curve(
    trainer: Trainer,
    title: str = "Training Curve",
    save_path: str | None = None
) -> None:
    """Plot training and evaluation loss curves."""
    try:
        import matplotlib.pyplot as plt
        
        logs = trainer.state.log_history
        
        # Extract training losses
        steps = [x["step"] for x in logs if "step" in x]
        losses = [x["loss"] for x in logs if "loss" in x]
        
        # Extract evaluation losses
        eval_steps = [x["step"] for x in logs if "step" in x]
        eval_losses = [x["eval_loss"] for x in logs if "eval_loss" in x]
        
        plt.figure(figsize=(10, 6))
        
        if steps and losses:
            plt.plot(steps, losses, label="Train Loss", alpha=0.8)
        
        if eval_steps and eval_losses:
            plt.plot(eval_steps, eval_losses, label="Eval Loss", alpha=0.8)
        
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curve saved to: {save_path}")
        
        plt.show()
        
    except ImportError:
        print("matplotlib not available, skipping plot generation")
    except Exception as e:
        print(f"Error plotting training curve: {e}")


def evaluate_perplexity(trainer: Trainer) -> dict[str, float]:
    """Evaluate model and calculate perplexity."""
    eval_results = trainer.evaluate()
    eval_loss = eval_results.get('eval_loss', 0.0)
    perplexity = math.exp(eval_loss) if eval_loss > 0 else float('inf')
    
    results = {
        'eval_loss': eval_loss,
        'perplexity': perplexity
    }
    
    print(f"Evaluation Loss: {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    return results


def save_model_and_tokenizer(
    trainer: Trainer,
    tokenizer: Any,
    output_dir: str
) -> None:
    """Save trained model and tokenizer."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    trainer.save_model(output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    print(f"Model and tokenizer saved to: {output_dir}")


def setup_output_directory(base_dir: str, task_name: str) -> str:
    """Setup output directory for a specific training task."""
    output_dir = os.path.join(base_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def print_training_info(
    model_name: str,
    dataset_info: dict[str, Any],
    config: Any
) -> None:
    """Print training setup information."""
    print("=" * 60)
    print(f"Training Setup: {model_name}")
    print("=" * 60)
    print(f"Base Model: {getattr(config, 'base_model_id', 'Unknown')}")
    print(f"Dataset: {getattr(config, 'dataset_name', 'Unknown')}")
    
    if 'train' in dataset_info and 'validation' in dataset_info:
        print(f"Train samples: {len(dataset_info['train'])}")
        print(f"Validation samples: {len(dataset_info['validation'])}")
    elif isinstance(dataset_info, dict) and 'train' in dataset_info:
        print(f"Train samples: {len(dataset_info['train'])}")
        if 'test' in dataset_info:
            print(f"Test samples: {len(dataset_info['test'])}")
    
    print(f"Learning Rate: {getattr(config, 'learning_rate', 'Unknown')}")
    print(f"Batch Size: {getattr(config, 'batch_size', 'Unknown')}")
    print(f"Epochs: {getattr(config, 'num_epochs', 'Unknown')}")
    print(f"Output Directory: {getattr(config, 'output_dir', 'Unknown')}")
    print("=" * 60)


def log_gpu_memory() -> None:
    """Log GPU memory usage if available."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3  # GB
            print(f"GPU Memory - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps"):
            allocated = torch.mps.current_allocated_memory() / 1024**3  # type: ignore[attr-defined]
            cached = torch.mps.driver_allocated_memory() / 1024**3  # type: ignore[attr-defined]
            print(f"MPS Memory - Allocated: {allocated:.2f} GB, Driver: {cached:.2f} GB")
    except ImportError:
        pass


def create_training_summary(
    trainer: Trainer,
    config: Any,
    results: dict[str, float]
) -> dict[str, Any]:
    """Create a summary of training results."""
    summary = {
        'config': {
            'base_model': getattr(config, 'base_model_id', 'Unknown'),
            'learning_rate': getattr(config, 'learning_rate', 'Unknown'),
            'batch_size': getattr(config, 'batch_size', 'Unknown'),
            'num_epochs': getattr(config, 'num_epochs', 'Unknown'),
        },
        'results': results,
        'final_train_loss': None,
        'total_steps': trainer.state.global_step if trainer.state else 0,
    }
    
    # Get final training loss if available
    if trainer.state and trainer.state.log_history:
        train_losses = [x.get("loss") for x in trainer.state.log_history if "loss" in x]
        if train_losses:
            summary['final_train_loss'] = train_losses[-1]
    
    return summary


def compare_model_sizes(original_model: Any, peft_model: Any) -> None:
    """Compare original and PEFT model parameter counts."""
    try:
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        def count_trainable_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        original_params = count_parameters(original_model)
        peft_total_params = count_parameters(peft_model)
        peft_trainable_params = count_trainable_parameters(peft_model)
        
        print(f"\nModel Size Comparison:")
        print(f"Original model parameters: {original_params:,}")
        print(f"PEFT model total parameters: {peft_total_params:,}")
        print(f"PEFT model trainable parameters: {peft_trainable_params:,}")
        print(f"Trainable parameter ratio: {peft_trainable_params/peft_total_params:.4f}")
        print(f"Parameter reduction: {(original_params - peft_trainable_params)/original_params:.4f}")
        
    except Exception as e:
        print(f"Error comparing model sizes: {e}")


def format_time(seconds: float) -> str:
    """Format training time in human readable format."""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"
