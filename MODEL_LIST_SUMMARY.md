# Model List Summary

## Current Model Selection (10 Models)

The model list has been refined to focus on 10 high-quality small language models across different sizes.

### Model Distribution

| Index | Model | Size | Category | Output Directory |
|-------|-------|------|----------|------------------|
| 0 | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | Small | `tinyllama_1.1b_chat` |
| 1 | Qwen/Qwen2.5-1.5B-Instruct | 1.5B | Small | `qwen2.5_1.5b_instruct` |
| 2 | google/gemma-2-2b-it | 2B | Small | `gemma2_2b_it` |
| 3 | deepseek-ai/deepseek-llm-7b-base | 7B | Medium | `deepseek_7b_base` |
| 4 | allenai/OLMo-7B-0424-hf | 7B | Medium | `olmo_7b` |
| 5 | google/gemma-7b-it | 7B | Medium | `gemma_7b_it` |
| 6 | unsloth/Meta-Llama-3.1-8B | 8B | Medium | `llama3.1_8b` |
| 7 | swiss-ai/Apertus-8B-2509 | 8B | Medium | `apertus_8b` |
| 8 | google/gemma-2-9b | 9B | Large | `gemma2_9b` |
| 9 | 01-ai/Yi-9B | 9B | Large | `yi_9b` |

### By Size Category

#### Small Models (<2B) - 3 models
- **TinyLlama 1.1B** - Compact, efficient
- **Qwen 2.5 1.5B** - Latest Qwen small model
- **Gemma 2 2B** - Google's small model

#### Medium Models (7-8B) - 5 models
- **DeepSeek 7B** - Base model for reasoning
- **OLMo 7B** - Open source research model
- **Gemma 7B** - Google's instruction-tuned
- **Llama 3.1 8B** - Meta's latest
- **Apertus 8B** - Swiss AI model

#### Large Models (9B) - 2 models
- **Gemma 2 9B** - Google's largest in series
- **Yi 9B** - 01.ai's model

## Changes from Original List

### Removed Models (4 models)
- ❌ Qwen/Qwen2-0.5B-Instruct (500M) - Too small
- ❌ microsoft/Phi-3-mini-4k-instruct (3.8B) - Coverage overlap
- ❌ google/gemma-3-4b-it (4B) - Redundant with Gemma-2-2b
- ❌ Qwen/Qwen3-8B - Keeping Qwen2.5 instead

### Why These 10 Models?

✅ **Size Diversity**: 1.1B to 9B range
✅ **Architecture Variety**: Different model families
✅ **Research Value**: Mix of base and instruction-tuned
✅ **Feasibility**: All runnable on typical GPU setups
✅ **Recency**: Recent models (2023-2024)

## Evaluation Coverage

### By Model Family
- **Qwen**: 1 model (1.5B)
- **Gemma**: 3 models (2B, 7B, 9B)
- **Llama**: 1 model (8B)
- **DeepSeek**: 1 model (7B)
- **OLMo**: 1 model (7B)
- **TinyLlama**: 1 model (1.1B)
- **Apertus**: 1 model (8B)
- **Yi**: 1 model (9B)

### By Training Type
- **Base Models**: 1 (DeepSeek)
- **Instruction-Tuned**: 9 (all others)

## Expected Batch Run Time

| Setup | Estimated Total Time |
|-------|---------------------|
| GPU (RTX 3090+) | 6-8 hours |
| GPU (RTX 3060) | 10-12 hours |
| CPU | 50-60 hours |

### Per-Model Estimates (GPU)
- Small (1-2B): 15-25 min each = ~1 hour total
- Medium (7-8B): 45-80 min each = ~5 hours total
- Large (9B): 80-100 min each = ~3 hours total

## Running Evaluations

### All 10 Models
```bash
python run_batch_evaluation.py
```

### By Size Category
```bash
# Small models only (3 models, ~1 hour)
python run_selective_evaluation.py --small

# Medium models (5 models, ~5 hours)
python run_selective_evaluation.py --medium

# Large models (2 models, ~3 hours)
python run_selective_evaluation.py --large
```

### Specific Models
```bash
# First 3 models (small)
python run_selective_evaluation.py 0 1 2

# Medium models
python run_selective_evaluation.py 3 4 5 6 7

# Large models
python run_selective_evaluation.py 8 9
```

## Output Organization

Each model's results will be saved to:
```
cache/models/
├── tinyllama_1.1b_chat/
├── qwen2.5_1.5b_instruct/
├── gemma2_2b_it/
├── deepseek_7b_base/
├── olmo_7b/
├── gemma_7b_it/
├── llama3.1_8b/
├── apertus_8b/
├── gemma2_9b/
└── yi_9b/
```

## Memory Requirements

| Model Size | GPU Memory | Recommended GPU |
|------------|-----------|-----------------|
| 1-2B | 4-8 GB | GTX 1080 Ti, RTX 3060 |
| 7-8B | 16-24 GB | RTX 3090, RTX 4090 |
| 9B | 20-28 GB | RTX 3090, RTX 4090 |

### If Memory Limited

Edit `src/config.py`:
```python
BATCH_SIZE = 2      # Reduce from 8
GRAD_ACCUM = 4      # Increase from 2
```

## Expected Results

### Metrics Collected Per Model
- ✅ Baseline performance (ID + OD)
- ✅ Fine-tuned performance (ID + OD)
- ✅ Improvements (ROUGE, BLEU)
- ✅ WeightWatcher quality metrics
- ✅ Training loss/checkpoints
- ✅ LoRA adapters

### Comparison Metrics
After batch run completes:
```bash
# View summary
cat cache/models/batch_evaluation_summary.json | python -m json.tool

# Find best model
cat cache/models/batch_evaluation_summary.json | \
  jq -r '.results[] | select(.status=="success") |
    "\(.metrics.in_distribution.improvements.bleu) \(.model_name)"' | \
  sort -rn
```

## Configuration

All settings in `src/config.py`:

```python
MODEL_ID_LIST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "deepseek-ai/deepseek-llm-7b-base",
    "allenai/OLMo-7B-0424-hf",
    "google/gemma-7b-it",
    "unsloth/Meta-Llama-3.1-8B",
    "swiss-ai/Apertus-8B-2509",
    "google/gemma-2-9b",
    "01-ai/Yi-9B"
]

OUTPUT_DIRS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama_1.1b_chat",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5_1.5b_instruct",
    "google/gemma-2-2b-it": "gemma2_2b_it",
    "deepseek-ai/deepseek-llm-7b-base": "deepseek_7b_base",
    "allenai/OLMo-7B-0424-hf": "olmo_7b",
    "google/gemma-7b-it": "gemma_7b_it",
    "unsloth/Meta-Llama-3.1-8B": "llama3.1_8b",
    "swiss-ai/Apertus-8B-2509": "apertus_8b",
    "google/gemma-2-9b": "gemma2_9b",
    "01-ai/Yi-9B": "yi_9b"
}
```

## Verification

Check current configuration:
```bash
python show_output_dirs.py
```

Expected output:
```
================================================================================
MODEL OUTPUT DIRECTORIES (from src/config.py)
================================================================================
...
Total models: 10
Defined output dirs: 10

✅ All models have predefined output directories!
```

## Adding/Removing Models

To modify the list:

1. Edit `src/config.py`:
```python
MODEL_ID_LIST = [
    # Add or remove model IDs here
    "organization/model-name",
]
```

2. Update `OUTPUT_DIRS`:
```python
OUTPUT_DIRS = {
    "organization/model-name": "custom_output_dir",
}
```

3. Verify:
```bash
python show_output_dirs.py
```

## Related Documentation

- **Batch Evaluation**: [BATCH_EVALUATION_GUIDE.md](BATCH_EVALUATION_GUIDE.md)
- **Output Directories**: [OUTPUT_DIRECTORIES.md](OUTPUT_DIRECTORIES.md)
- **WeightWatcher**: [WEIGHTWATCHER_METRICS.md](WEIGHTWATCHER_METRICS.md)
- **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
