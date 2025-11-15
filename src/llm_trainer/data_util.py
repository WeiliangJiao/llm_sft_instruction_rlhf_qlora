"""
Data utilities for loading and preprocessing datasets for LLM training.
"""
import random
from typing import Dict, Any, Tuple, Optional
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

from .config import RANDOM_SEED
from functools import partial


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_sft_dataset(
    dataset_name: str = "vblagoje/cc_news",
    dataset_split: str = "train[:2%]",
    test_size: float = 0.1,
    seed: int = RANDOM_SEED
) -> DatasetDict:
    """Load and split dataset for supervised fine-tuning."""
    set_seed(seed)
    
    dataset = load_dataset(dataset_name, split=dataset_split)
    dataset = dataset.shuffle(seed=seed)
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    
    return split_dataset


def load_instruction_dataset(
    dataset_name: str = "databricks/databricks-dolly-15k",
    num_samples: int = 12000,
    test_size: float = 0.02,
    seed: int = RANDOM_SEED
) -> DatasetDict:
    """Load and split Dolly dataset for instruction tuning."""
    set_seed(seed)
    
    dataset = load_dataset(dataset_name, split="train")
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    split_dataset = dataset.train_test_split(test_size=test_size, seed=seed)
    
    return split_dataset


def load_preference_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    num_train_samples: int = 4000,
    num_val_samples: int = 400,
    seed: int = RANDOM_SEED
) -> Dict[str, Dataset]:
    """Load preference dataset for reward modeling and DPO."""
    set_seed(seed)
    
    raw_dataset = load_dataset(dataset_name)
    
    train_data = raw_dataset["train"].shuffle(seed=seed).select(range(num_train_samples))
    val_data = raw_dataset["test"].shuffle(seed=seed).select(range(num_val_samples))
    
    return {"train": train_data, "validation": val_data}


def tokenize(batch, tokenizer, max_length):
    '''Tokenize a batch of texts.'''
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=max_length,
        padding=False
    )


def preprocess_sft_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    max_length: int = 256
) -> Dataset:
    """Preprocess SFT dataset by tokenizing text."""
    tokenize_fn = partial(tokenize, tokenizer=tokenizer, max_length=max_length)
    
    processed = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )
    
    processed.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )
    
    return processed


def preprocess_instruction_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    instruction_template: str,
    max_length: int = 256
) -> Dataset:
    """Preprocess instruction dataset by formatting and tokenizing."""
    # Format examples using the template (mirrors notebook logic)
    def format_example(ex):
        instr = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        out = ex.get("output", "").strip()

        text = instruction_template.format(
            instruction=instr,
            inp=inp,
            out=out,
        )
        return {"text": text}

    formatted_dataset = dataset.map(
        format_example,
        batched=False,
        remove_columns=dataset.column_names,
    )
    
    # Tokenize
    tokenize_fn = partial(tokenize, tokenizer=tokenizer, max_length=max_length)
    
    tokenized = formatted_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )
    
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"]
    )
    
    return tokenized
