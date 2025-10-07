# Evaluation Pipeline Guide

This guide explains how to use the `evaluate_model_pipeline` function to comprehensively evaluate and fine-tune models for inductive reasoning tasks.

## Overview

The `evaluate_model_pipeline` function provides an end-to-end workflow for:

1. **Baseline Evaluation**: Load and evaluate a pre-trained model on your datasets
2. **Fine-tuning**: Apply LoRA (Low-Rank Adaptation) to fine-tune the model
3. **Fine-tuned Evaluation**: Evaluate the fine-tuned model
4. **Results Saving**: Save all predictions, metrics, and model adapters

## Quick Start

### Basic Usage

```python
from src.models.evaluate import evaluate_model_pipeline
from datasets import DatasetDict

# Prepare your datasets (DatasetDict with 'train' and 'validation' splits)
results = evaluate_model_pipeline(
    model_id="Qwen/Qwen2.5-0.5B",
    id_dataset=id_dataset,  # In-distribution dataset
    od_dataset=od_dataset,  # Out-of-distribution dataset (optional)
    output_dir="./my_evaluation_results",
    hf_token="your_hf_token"
)
```

### Complete Example

See `notebooks/evaluation_demo.ipynb` for a complete working example.

```python
import os
from datasets import Dataset, DatasetDict
from src.models.evaluate import evaluate_model_pipeline
import src.config as cfg

# Set up HuggingFace token
HF_TOKEN = os.environ["HUGGINGFACE_HUB_TOKEN"]

# Run complete pipeline
results = evaluate_model_pipeline(
    model_id=cfg.MODEL_ID,
    id_dataset=id_dataset,
    od_dataset=od_dataset,
    output_dir=cfg.OUTPUT_DIR,
    lora_r=cfg.LORA_R,
    lora_alpha=cfg.LORA_ALPHA,
    lora_dropout=cfg.LORA_DROPOUT,
    target_modules=cfg.TARGET_MODULES,
    batch_size=cfg.BATCH_SIZE,
    grad_accum=cfg.GRAD_ACCUM,
    learning_rate=cfg.LR,
    epochs=cfg.EPOCHS,
    max_seq_len=cfg.MAX_SEQ_LEN,
    log_steps=cfg.LOG_STEPS,
    save_steps=cfg.SAVE_STEPS,
    hf_token=HF_TOKEN
)
```

## Function Parameters

### Required Parameters

- **`model_id`** (str): HuggingFace model identifier (e.g., "Qwen/Qwen2.5-0.5B")
- **`id_dataset`** (DatasetDict): In-distribution dataset with 'train' and 'validation' splits

### Optional Parameters

- **`od_dataset`** (DatasetDict, optional): Out-of-distribution dataset for generalization testing
- **`output_dir`** (str): Directory to save all results (default: "./evaluation_output")
- **`train_dataset`** (Dataset, optional): Custom training dataset (default: uses id_dataset['train'])
- **`hf_token`** (str, optional): HuggingFace authentication token
- **`skip_finetuning`** (bool): If True, only runs baseline evaluation (default: False)

### LoRA Configuration

- **`lora_r`** (int): LoRA rank (default: 16)
- **`lora_alpha`** (int): LoRA alpha scaling (default: 32)
- **`lora_dropout`** (float): LoRA dropout rate (default: 0.1)
- **`target_modules`** (List[str]): Modules to apply LoRA to (default: None)

### Training Configuration

- **`batch_size`** (int): Training batch size (default: 4)
- **`grad_accum`** (int): Gradient accumulation steps (default: 1)
- **`learning_rate`** (float): Learning rate (default: 2e-4)
- **`epochs`** (int): Number of training epochs (default: 3)
- **`max_seq_len`** (int): Maximum sequence length (default: 512)
- **`log_steps`** (int): Logging frequency (default: 10)
- **`save_steps`** (int): Model checkpoint frequency (default: 500)

## Dataset Format

Your datasets should be `DatasetDict` objects with specific splits and fields:

```python
from datasets import Dataset, DatasetDict

# Example dataset creation
train_data = [
    {"prompt": "Training Observations:\n...\n\nQuestion:\n...\n\nAnswer:\n",
     "response": "answer text"}
]
val_data = [...]

train_ds = Dataset.from_list(train_data)
val_ds = Dataset.from_list(val_data)

dataset = DatasetDict({
    "train": train_ds,
    "validation": val_ds
})
```

### Required Fields

Each dataset entry must have:
- **`prompt`**: The input prompt containing observations and question
- **`response`**: The expected answer/response

### Prompt Format

```
Training Observations:
<observation text>

Question:
<question text>

Answer:
```

## Output Files

The pipeline creates the following structure in the output directory:

```
output_dir/
├── id_val_predictions_baseline.jsonl      # Baseline predictions (in-distribution)
├── id_val_predictions_finetuned.jsonl     # Fine-tuned predictions (in-distribution)
├── od_val_predictions_baseline.jsonl      # Baseline predictions (out-of-distribution)
├── od_val_predictions_finetuned.jsonl     # Fine-tuned predictions (out-of-distribution)
├── metrics_summary.json                   # Complete metrics with improvements
├── adapter/                               # Fine-tuned LoRA adapters
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer_config.json
│   └── ...
└── checkpoint-*/                          # Training checkpoints
```

### Prediction Files

Each prediction file (`.jsonl`) contains one JSON object per line:

```json
{
  "Training Observations": "observation text",
  "Question": "question text",
  "Reference": "reference answer",
  "Prediction": "model prediction"
}
```

### Metrics Summary

The `metrics_summary.json` file contains:

```json
{
  "timestamp": "2025-10-06T20:30:00",
  "in_distribution": {
    "baseline": {
      "rouge1": 0.027,
      "rouge2": 0.002,
      "rougeL": 0.023,
      "rougeLsum": 0.023,
      "bleu": 0.021
    },
    "finetuned": {
      "rouge1": 0.035,
      "rouge2": 0.005,
      "rougeL": 0.030,
      "rougeLsum": 0.030,
      "bleu": 0.028
    },
    "improvements": {
      "rouge1": 0.008,
      "rouge2": 0.003,
      "rougeL": 0.007,
      "rougeLsum": 0.007,
      "bleu": 0.007
    }
  },
  "out_of_distribution": { ... }
}
```

## Return Value

The function returns a dictionary with:

```python
{
    "id_baseline_metrics": dict,      # In-distribution baseline metrics
    "id_finetuned_metrics": dict,     # In-distribution fine-tuned metrics
    "od_baseline_metrics": dict,      # Out-of-distribution baseline metrics (if provided)
    "od_finetuned_metrics": dict,     # Out-of-distribution fine-tuned metrics (if provided)
    "output_dir": str,                # Path to output directory
    "adapter_dir": str                # Path to saved adapters
}
```

## Use Cases

### 1. Complete Evaluation Pipeline

Run baseline evaluation, fine-tuning, and fine-tuned evaluation:

```python
results = evaluate_model_pipeline(
    model_id="Qwen/Qwen2.5-0.5B",
    id_dataset=id_dataset,
    od_dataset=od_dataset,
    output_dir="./full_evaluation"
)
```

### 2. Baseline Evaluation Only

Evaluate a model without fine-tuning:

```python
results = evaluate_model_pipeline(
    model_id="Qwen/Qwen2.5-0.5B",
    id_dataset=id_dataset,
    od_dataset=od_dataset,
    output_dir="./baseline_only",
    skip_finetuning=True
)
```

### 3. Custom LoRA Configuration

Fine-tune with custom LoRA parameters:

```python
results = evaluate_model_pipeline(
    model_id="Qwen/Qwen2.5-0.5B",
    id_dataset=id_dataset,
    lora_r=32,           # Larger rank
    lora_alpha=64,       # Larger alpha
    lora_dropout=0.05,   # Lower dropout
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    output_dir="./custom_lora"
)
```

### 4. In-Distribution Only

Evaluate without out-of-distribution data:

```python
results = evaluate_model_pipeline(
    model_id="Qwen/Qwen2.5-0.5B",
    id_dataset=id_dataset,
    od_dataset=None,  # No OD evaluation
    output_dir="./id_only"
)
```

## Evaluation Metrics

The pipeline computes the following metrics:

- **ROUGE-1**: Unigram overlap between prediction and reference
- **ROUGE-2**: Bigram overlap between prediction and reference
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: ROUGE-L with summary-level aggregation
- **BLEU**: Bilingual Evaluation Understudy score

## Loading Saved Adapters

To load the fine-tuned model later:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Load LoRA adapters
model = PeftModel.from_pretrained(base_model, "./output_dir/adapter")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./output_dir/adapter")
```

## Memory Management

The pipeline includes automatic memory management:

- Clears GPU memory after baseline evaluation
- Uses garbage collection between model loads
- Employs memory-efficient LoRA fine-tuning

For large models, consider:
- Reducing `batch_size`
- Increasing `grad_accum` to maintain effective batch size
- Using smaller `lora_r` values

## Best Practices

1. **Use Version Control**: Track your `output_dir` with git or similar
2. **Set Random Seeds**: Use `transformers.set_seed()` for reproducibility
3. **Monitor GPU Memory**: Watch memory usage during long runs
4. **Save Configurations**: Keep track of hyperparameters used
5. **Validate Datasets**: Ensure your datasets follow the required format

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size and increase gradient accumulation
results = evaluate_model_pipeline(
    ...,
    batch_size=2,
    grad_accum=4
)
```

### Slow Generation

```python
# Adjust generation parameters in src/models/inference/inference.py
# Or reduce validation set size for faster iteration
```

### Missing HuggingFace Token

```python
# Set token explicitly or via environment variable
os.environ["HUGGINGFACE_HUB_TOKEN"] = "your_token"
```

## Related Files

- **Source**: `src/models/evaluate.py`
- **Demo Notebook**: `notebooks/evaluation_demo.ipynb`
- **Configuration**: `src/config.py`
- **Fine-tuning**: `src/models/sft/lora.py`
- **Inference**: `src/models/inference/inference.py`
