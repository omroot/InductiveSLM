# Quick Reference - Evaluation Pipeline

## One-Line Commands

```bash
# Single model evaluation (recommended)
python run_evaluation.py

# Batch evaluation - all models
python run_batch_evaluation.py

# Selective evaluation - choose models
python run_selective_evaluation.py --small        # Small models only
python run_selective_evaluation.py 0 1 2          # Specific indices
python run_selective_evaluation.py                # Interactive mode

# Show all output directories
python show_output_dirs.py

# Open demo notebook
jupyter lab notebooks/evaluation_demo.ipynb

# View results (using predefined directory names from config.py)
cat cache/models/qwen2_0.5b_instruct/metrics_summary.json | python -m json.tool

# View batch summary
cat cache/models/batch_evaluation_summary.json | python -m json.tool
```

## Basic Usage

```python
from src.models.evaluate import evaluate_model_pipeline

results = evaluate_model_pipeline(
    model_id="Qwen/Qwen2-0.5B-Instruct",
    id_dataset=id_dataset,
    od_dataset=od_dataset,
    output_dir="cache/models/my_test"
)
```

## Common Patterns

### Full Pipeline
```python
results = evaluate_model_pipeline(
    model_id=MODEL_ID,
    id_dataset=id_dataset,
    od_dataset=od_dataset,
    output_dir=OUTPUT_DIR,
    hf_token=HF_TOKEN
)
```

### Baseline Only
```python
results = evaluate_model_pipeline(
    model_id=MODEL_ID,
    id_dataset=id_dataset,
    output_dir=OUTPUT_DIR,
    skip_finetuning=True
)
```

### Custom LoRA
```python
results = evaluate_model_pipeline(
    model_id=MODEL_ID,
    id_dataset=id_dataset,
    lora_r=32,
    lora_alpha=64,
    target_modules=["q_proj", "v_proj"],
    output_dir=OUTPUT_DIR
)
```

### Small GPU
```python
results = evaluate_model_pipeline(
    model_id=MODEL_ID,
    id_dataset=id_dataset,
    batch_size=2,
    grad_accum=4,
    output_dir=OUTPUT_DIR
)
```

## Output Files

```
output_dir/
├── id_val_predictions_baseline.jsonl      # Baseline ID predictions
├── id_val_predictions_finetuned.jsonl     # Fine-tuned ID predictions
├── od_val_predictions_baseline.jsonl      # Baseline OD predictions
├── od_val_predictions_finetuned.jsonl     # Fine-tuned OD predictions
├── metrics_summary.json                   # All metrics + improvements
└── adapter/                               # LoRA adapters + tokenizer
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | Required | HF model identifier |
| `id_dataset` | Required | In-distribution DatasetDict |
| `od_dataset` | None | Out-of-distribution DatasetDict |
| `output_dir` | "./evaluation_output" | Results directory |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha |
| `batch_size` | 4 | Training batch size |
| `epochs` | 3 | Training epochs |
| `learning_rate` | 2e-4 | Learning rate |
| `skip_finetuning` | False | Skip fine-tuning |

## Loading Results

```python
import json

# Load metrics
with open("cache/models/*/metrics_summary.json") as f:
    metrics = json.load(f)

# Access results
results = evaluate_model_pipeline(...)
print(results["id_baseline_metrics"])
print(results["id_finetuned_metrics"])
```

## Loading Saved Adapters

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base + adapters
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "cache/models/*/adapter")
tokenizer = AutoTokenizer.from_pretrained("cache/models/*/adapter")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `batch_size`, increase `grad_accum` |
| Slow | Reduce validation set size |
| Token error | Set `HUGGINGFACE_HUB_TOKEN` in `.env` |
| Import error | Run from project root: `python -m src.models.evaluate` |

## Configuration Files

- **Settings**: `src/config.py`
- **Paths**: `src/settings.py`
- **Environment**: `.env`

## Documentation

- 📘 Complete Guide: `EVALUATION_GUIDE.md`
- 🧪 Testing: `TESTING.md`
- 📝 Changes: `CHANGES_SUMMARY.md`
- 📓 Demo: `notebooks/evaluation_demo.ipynb`

## Quick Checks

```bash
# Check if directory created
ls cache/models/

# Count predictions
wc -l cache/models/*/*.jsonl

# View metrics
cat cache/models/*/metrics_summary.json

# Check adapters
ls cache/models/*/adapter/

# Verify imports
python -c "from src.models.evaluate import evaluate_model_pipeline; print('OK')"
```
