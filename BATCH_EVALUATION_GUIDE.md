# Batch Evaluation Guide

This guide explains how to evaluate multiple models automatically using the batch evaluation scripts.

## Overview

Three scripts are available for different evaluation scenarios:

1. **`run_evaluation.py`** - Evaluate a single model (from config.py)
2. **`run_batch_evaluation.py`** - Evaluate ALL models in MODEL_ID_LIST
3. **`run_selective_evaluation.py`** - Choose specific models to evaluate

## Quick Start

### Single Model Evaluation

```bash
# Edit src/config.py to set MODEL_ID
python run_evaluation.py
```

### Batch Evaluation (All Models)

```bash
# Evaluates all 14 models in MODEL_ID_LIST
python run_batch_evaluation.py
```

### Selective Evaluation

```bash
# Interactive mode - choose models
python run_selective_evaluation.py

# Command line - evaluate specific indices
python run_selective_evaluation.py 0 1 2

# Evaluate by size category
python run_selective_evaluation.py --small    # <2B models
python run_selective_evaluation.py --medium   # 2-8B models
python run_selective_evaluation.py --large    # >8B models
```

## Model List

Current models in `MODEL_ID_LIST` (src/config.py):

| Index | Model ID | Size | Category |
|-------|----------|------|----------|
| 0 | Qwen/Qwen2-0.5B-Instruct | 500M | small |
| 1 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | small |
| 2 | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | small |
| 3 | google/gemma-2-2b-it | 2B | medium |
| 4 | microsoft/Phi-3-mini-4k-instruct | 3.8B | medium |
| 5 | google/gemma-3-4b-it | 4B | medium |
| 6 | deepseek-ai/deepseek-llm-7b-base | 7B | medium |
| 7 | allenai/OLMo-7B-0424-hf | 7B | medium |
| 8 | google/gemma-7b-it | 7B | medium |
| 9 | Qwen/Qwen3-8B | 8B | medium |
| 10 | unsloth/Meta-Llama-3.1-8B | 8B | medium |
| 11 | swiss-ai/Apertus-8B-2509 | 8B | medium |
| 12 | 01-ai/Yi-9B | 9B | large |
| 13 | google/gemma-2-9b | 9B | large |

## Batch Evaluation Features

### 1. Automatic Dataset Reuse
- Loads datasets once at the beginning
- Reuses across all models
- Saves significant time

### 2. Error Handling
- Continues evaluation even if one model fails
- Saves error details for debugging
- Doesn't stop the entire batch

### 3. Progress Tracking
- Shows current model number (e.g., "3/14")
- Displays time taken per model
- Saves intermediate results after each model

### 4. Summary Generation
- Creates comprehensive JSON summary
- Compares all models side-by-side
- Shows improvements for each metric

## Output Structure

### Individual Model Outputs

Each model gets its own directory:

```
cache/models/
├── Qwen2-0.5B-Instruct_evaluation/
│   ├── id_val_predictions_baseline.jsonl
│   ├── id_val_predictions_finetuned.jsonl
│   ├── od_val_predictions_baseline.jsonl
│   ├── od_val_predictions_finetuned.jsonl
│   ├── metrics_summary.json
│   └── adapter/
├── TinyLlama-1.1B-Chat-v1.0_evaluation/
│   └── ...
└── batch_evaluation_summary.json  # Overall summary
```

### Batch Summary File

The `batch_evaluation_summary.json` contains:

```json
{
  "timestamp": "2025-10-06T20:30:00",
  "total_models": 14,
  "successful": 12,
  "failed": 2,
  "results": [
    {
      "model_id": "Qwen/Qwen2-0.5B-Instruct",
      "model_name": "Qwen2-0.5B-Instruct",
      "status": "success",
      "duration_seconds": 857.3,
      "output_dir": "cache/models/Qwen2-0.5B-Instruct_evaluation",
      "metrics": {
        "in_distribution": {
          "baseline": { "rouge1": 0.027, "bleu": 0.021, ... },
          "finetuned": { "rouge1": 0.035, "bleu": 0.028, ... },
          "improvements": { "rouge1": 0.008, "bleu": 0.007, ... }
        },
        "out_of_distribution": { ... }
      }
    },
    ...
  ]
}
```

## Usage Examples

### Example 1: Evaluate Small Models Only

```bash
python run_selective_evaluation.py --small
```

This will evaluate:
- Qwen2-0.5B-Instruct
- TinyLlama-1.1B-Chat-v1.0
- Qwen2.5-1.5B-Instruct

### Example 2: Evaluate Specific Models

```bash
# Evaluate Qwen 500M, Gemma 2B, and Phi-3
python run_selective_evaluation.py 0 3 4
```

### Example 3: Interactive Selection

```bash
python run_selective_evaluation.py
```

Then follow the prompts:
```
AVAILABLE MODELS
================================================================================
  [0] Qwen/Qwen2-0.5B-Instruct                          (small)
  [1] TinyLlama/TinyLlama-1.1B-Chat-v1.0                (small)
  [2] Qwen/Qwen2.5-1.5B-Instruct                        (small)
  ...

Your selection: 0,1,2
```

### Example 4: Evaluate Range of Models

```bash
# Interactive mode, then enter:
# 0-3  (evaluates indices 0, 1, 2, 3)
python run_selective_evaluation.py
```

### Example 5: All Models Overnight

```bash
# Run all models (may take several hours)
nohup python run_batch_evaluation.py > batch_log.txt 2>&1 &

# Check progress
tail -f batch_log.txt
```

## Time Estimates

Approximate times per model (on typical GPU):

| Model Size | Time |
|------------|------|
| 500M-1.5B | 10-20 min |
| 2B-4B | 20-40 min |
| 7B-8B | 40-80 min |
| 9B+ | 80-120 min |

**Total time for all 14 models**: ~10-15 hours on GPU

## Memory Requirements

| Model Size | GPU Memory | Recommendation |
|------------|------------|----------------|
| <2B | 4-8 GB | Any modern GPU |
| 2-4B | 8-12 GB | RTX 3060 or better |
| 7-8B | 16-24 GB | RTX 3090/4090 |
| 9B+ | 24+ GB | A100 or reduce batch_size |

### Memory-Constrained Setups

For limited GPU memory, edit `src/config.py`:

```python
BATCH_SIZE = 2      # Reduce from 8
GRAD_ACCUM = 4      # Increase from 2
```

## Monitoring Progress

### During Execution

The scripts print:
- Current model being evaluated
- Progress (e.g., "Model 3/14")
- Metrics after each model

### After Each Model

Check intermediate results:
```bash
# View latest summary
cat cache/models/batch_evaluation_summary.json | python -m json.tool

# Count completed models
ls -d cache/models/*_evaluation/ | wc -l
```

### Background Execution

Run in background and monitor:
```bash
# Start batch evaluation
nohup python run_batch_evaluation.py > batch.log 2>&1 &

# Monitor progress
tail -f batch.log

# Check summary periodically
watch -n 60 'cat cache/models/batch_evaluation_summary.json | jq ".successful, .failed"'
```

## Resuming Failed Evaluations

If a batch run fails partway through:

1. Check `batch_evaluation_summary.json` to see which models succeeded
2. Note which models failed or weren't evaluated
3. Use selective evaluation to complete:

```bash
# If models 5, 6, 7 weren't evaluated
python run_selective_evaluation.py 5 6 7
```

## Analyzing Results

### Compare All Models

```python
import json

with open('cache/models/batch_evaluation_summary.json') as f:
    data = json.load(f)

# Sort by in-distribution BLEU improvement
results = sorted(
    [r for r in data['results'] if r['status'] == 'success'],
    key=lambda x: x['metrics']['in_distribution']['improvements']['bleu'],
    reverse=True
)

# Print top 5
for r in results[:5]:
    print(f"{r['model_name']}: {r['metrics']['in_distribution']['improvements']['bleu']:.4f}")
```

### Extract Best Model

```bash
# Find model with best improvements
cat cache/models/batch_evaluation_summary.json | \
  jq -r '.results[] | select(.status=="success") |
    "\(.metrics.in_distribution.improvements.bleu) \(.model_name)"' | \
  sort -rn | head -5
```

## Best Practices

1. **Start Small**: Test with 1-2 models first
2. **Monitor First Model**: Watch the first model complete to ensure everything works
3. **Use Selective Mode**: For experiments, use selective evaluation
4. **Save Logs**: Always redirect output to a log file for long runs
5. **Check Disk Space**: Ensure enough space (~5-10 GB per model)
6. **Background Execution**: Use `nohup` or `screen` for long runs

## Troubleshooting

### Out of Memory

```python
# In src/config.py
BATCH_SIZE = 1
GRAD_ACCUM = 8
```

### Model Download Fails

Ensure HuggingFace token is set:
```bash
# Check .env file
cat .env | grep HUGGINGFACE_HUB_TOKEN
```

### Slow Evaluation

- Reduce validation set size in code
- Use smaller models first
- Skip OD evaluation if not needed

### Script Stops Unexpectedly

```bash
# Use nohup for resilience
nohup python run_batch_evaluation.py > batch.log 2>&1 &

# Or use screen
screen -S batch_eval
python run_batch_evaluation.py
# Press Ctrl+A, D to detach
```

## Configuration

All models use the same configuration from `src/config.py`:

```python
# Shared settings
LORA_R = 16
LORA_ALPHA = 32
EPOCHS = 3
BATCH_SIZE = 8
LR = 2e-4
```

To use different settings per model, modify the config dict in the batch script.

## Summary Table Output

After batch completion, you'll see:

```
BATCH EVALUATION SUMMARY
================================================================================
Total models: 14
Successful: 12
Failed: 2

SUCCESSFUL MODELS - IN-DISTRIBUTION IMPROVEMENTS
--------------------------------------------------------------------------------
Model                                    ROUGE-1    ROUGE-L    BLEU
--------------------------------------------------------------------------------
Qwen2.5-1.5B-Instruct                    +0.0123    +0.0098    +0.0156
TinyLlama-1.1B-Chat-v1.0                 +0.0087    +0.0071    +0.0092
...
```

## Related Files

- **Batch script**: `run_batch_evaluation.py`
- **Selective script**: `run_selective_evaluation.py`
- **Single model**: `run_evaluation.py`
- **Config**: `src/config.py`
- **Evaluation function**: `src/models/evaluate.py`
