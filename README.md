# LLM Fine-Tuning with QLoRA

An LLM fine-tuning project showcasing different LLM fine-tuning techniques using QLoRA (parameter-efficient training). Includes modular implementations of:

- **Supervised Fine-Tuning (SFT)** - learning domain knowledge
- **Instruction Tuning** - teaching models to follow instructions
- **Reward Model Training** - learning human preferences
- **Direct Preference Optimization (DPO)** - alignment without reward models

Built with PyTorch, Transformers, PEFT, and TRL.

## Quick Start

Clone, create a virtual environment, and install in editable mode:

```bash
git clone <repository-url>
cd llm_sft_rlhf
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
```

Or, if you just need the dependencies without installing the package:

```bash
pip install -r requirements.txt
```

## Usage Examples

### Run training via CLI:

```bash
# SFT training
python main.py sft --model gpt2-medium --output ./results/sft

# Instruction tuning
python main.py instruction --model gpt2-medium --samples 10000

# Reward model
python main.py reward --base-model-path ./results/instruction_qlora

# DPO training
python main.py dpo --base-model-path ./results/instruction_qlora
```

### Minimal pipeline script

This example script runs a tiny SFT → Instruction → Reward → DPO chain, loading the checkpoint from the previous step each time (all based on `gpt2`):

```bash
# writes to ./results/minimal_pipeline by default
python examples/complete_pipeline.py

# specify a custom output directory
python examples/complete_pipeline.py --output ./results/my_minimal_run
```

Artifacts stored in `<output>/<stage>/<stage_name>` (e.g., `results/minimal_pipeline/sft/sft_qlora`), and instruction/reward/DPO automatically load the prior stage’s checkpoint.

### Or use as a Python library:

```python
from llm_trainer import SFTTrainer, ModelConfig, LoRAConfig, SFTConfig

# Configure training
model_config = ModelConfig(base_model_id="gpt2-medium", max_length=256)
lora_config = LoRAConfig(r=16, lora_alpha=32)
sft_config = SFTConfig(dataset_name="vblagoje/cc_news", batch_size=8, num_epochs=1)

# Train
trainer = SFTTrainer(model_config, lora_config, QuantizationConfig(), sft_config)
results = trainer.run_full_training()
```

## Project Structure

```
llm_sft_rlhf/
├── src/llm_trainer/          # Main package
│   ├── config.py             # Config dataclasses
│   ├── data.py               # Dataset utilities
│   ├── models.py             # Model loading + QLoRA setup
│   ├── training.py           # Training helpers
│   ├── sft_trainer.py        # Supervised fine-tuning
│   ├── instruction_trainer.py # Instruction tuning
│   ├── reward_trainer.py     # Reward model
│   └── dpo_trainer.py        # DPO implementation
├── main.py                   # CLI interface
├── setup.py                  # Setuptools for packaging/install
├── requirements.txt          # Dependencies
├── examples/                 # Example scripts
└── Notebooks/                # Jupyter notebooks (dev)
```

## Key Features

**QLoRA Training**: 4-bit quantization + LoRA for efficient fine-tuning on consumer GPUs

**Modular Design**: Each training method is self-contained and reusable

**Full Pipeline**: Base Model -> SFT -> Instruction Tuning -> Reward Model -> DPO

**Datasets Used**: CC News (SFT), Dolly-15k (Instruction), HH-RLHF (Reward & DPO)

## Requirements & Compatibility

- Python 3.10+
- Works on Linux, macOS (Apple Silicon with MPS), and Windows; training on CPU/MPS is slower but supported
- For best performance: CUDA-capable GPU with ≥8GB VRAM and ≥16GB system RAM

Key dependencies: `torch`, `transformers`, `peft`, `trl`, `datasets`, `bitsandbytes` _(CUDA-only; automatically disabled on MPS/CPU)_.

> **Note:** When running on Apple Silicon or CPU-only Windows, 4-bit QLoRA quantization from `bitsandbytes` is skipped automatically; everything else still runs.

## Troubleshooting

**CUDA Out of Memory**: Reduce batch size or reduce sequence length

**Import Errors**: Ensure the virtual environment is active and run `pip install -e .`

**bitsandbytes on non-CUDA hardware**: The package may emit warnings or fail to load on macOS/Windows without CUDA; this is expected, and the code falls back to full-precision weights automatically.
