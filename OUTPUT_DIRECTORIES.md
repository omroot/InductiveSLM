# Output Directory Configuration

## Overview

All model output directories are predefined in `src/config.py` for consistent, clean naming across evaluations.

## Configuration

### Location
All output directories are defined in `src/config.py`:

```python
# Base directory for all model outputs
BASE_OUTPUT_DIR = "cache/models"

# Predefined output directories for each model
OUTPUT_DIRS = {
    "Qwen/Qwen2-0.5B-Instruct": "qwen2_0.5b_instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama_1.1b_chat",
    # ... etc
}
```

## Output Directory Mapping

| Model | Output Directory | Full Path |
|-------|-----------------|-----------|
| Qwen/Qwen2-0.5B-Instruct | `qwen2_0.5b_instruct` | `cache/models/qwen2_0.5b_instruct` |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | `tinyllama_1.1b_chat` | `cache/models/tinyllama_1.1b_chat` |
| Qwen/Qwen2.5-1.5B-Instruct | `qwen2.5_1.5b_instruct` | `cache/models/qwen2.5_1.5b_instruct` |
| google/gemma-2-2b-it | `gemma2_2b_it` | `cache/models/gemma2_2b_it` |
| microsoft/Phi-3-mini-4k-instruct | `phi3_mini_4k` | `cache/models/phi3_mini_4k` |
| google/gemma-3-4b-it | `gemma3_4b_it` | `cache/models/gemma3_4b_it` |
| deepseek-ai/deepseek-llm-7b-base | `deepseek_7b_base` | `cache/models/deepseek_7b_base` |
| allenai/OLMo-7B-0424-hf | `olmo_7b` | `cache/models/olmo_7b` |
| google/gemma-7b-it | `gemma_7b_it` | `cache/models/gemma_7b_it` |
| Qwen/Qwen3-8B | `qwen3_8b` | `cache/models/qwen3_8b` |
| unsloth/Meta-Llama-3.1-8B | `llama3.1_8b` | `cache/models/llama3.1_8b` |
| swiss-ai/Apertus-8B-2509 | `apertus_8b` | `cache/models/apertus_8b` |
| 01-ai/Yi-9B | `yi_9b` | `cache/models/yi_9b` |
| google/gemma-2-9b | `gemma2_9b` | `cache/models/gemma2_9b` |

## Directory Structure

Each model's output directory contains:

```
cache/models/qwen2_0.5b_instruct/
├── id_val_predictions_baseline.jsonl      # In-distribution baseline predictions
├── id_val_predictions_finetuned.jsonl     # In-distribution fine-tuned predictions
├── od_val_predictions_baseline.jsonl      # Out-of-distribution baseline predictions
├── od_val_predictions_finetuned.jsonl     # Out-of-distribution fine-tuned predictions
├── metrics_summary.json                   # Complete metrics and improvements
├── adapter/                               # Fine-tuned LoRA adapters
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── tokenizer_config.json
│   └── ...
└── checkpoint-*/                          # Training checkpoints
```

## View All Directories

To see all predefined output directories:

```bash
python show_output_dirs.py
```

This displays:
- Base directory
- Model-to-directory mapping
- Full paths
- Verification that all models have directories defined

## Benefits of Predefined Directories

### ✅ Consistency
- Clean, readable names across all runs
- No auto-generated names with random suffixes
- Easy to reference in papers/reports

### ✅ Organization
- Logical grouping by model family
- Version numbers preserved (e.g., `qwen2.5` vs `qwen2`)
- Size information retained where relevant

### ✅ Easy Navigation
- Short, memorable directory names
- Tab-completion friendly
- Searchable and scriptable

### ✅ Version Control
- Can track which models were evaluated
- Easy to compare across different runs
- Reproducible paths in documentation

## Customization

### Change a Directory Name

Edit `src/config.py`:

```python
OUTPUT_DIRS = {
    "Qwen/Qwen2-0.5B-Instruct": "my_custom_qwen_dir",  # Change here
    # ...
}
```

### Change Base Directory

Edit `src/config.py`:

```python
BASE_OUTPUT_DIR = "my_results/models"  # Change here
```

### Add a New Model

1. Add model to `MODEL_ID_LIST`:
```python
MODEL_ID_LIST = [
    "Qwen/Qwen2-0.5B-Instruct",
    # ... existing models
    "organization/new-model-name",  # Add new model
]
```

2. Add output directory to `OUTPUT_DIRS`:
```python
OUTPUT_DIRS = {
    "Qwen/Qwen2-0.5B-Instruct": "qwen2_0.5b_instruct",
    # ... existing mappings
    "organization/new-model-name": "new_model_dir",  # Add mapping
}
```

3. Verify:
```bash
python show_output_dirs.py
```

## Naming Convention

The predefined names follow this pattern:

- **Lowercase**: All directories use lowercase
- **Underscores**: Use `_` instead of hyphens or spaces
- **Version info**: Include version numbers (e.g., `2.5`, `3.1`)
- **Size info**: Include model size where it's part of the name
- **Family first**: Start with model family (e.g., `qwen`, `gemma`, `llama`)
- **Concise**: Keep names short but descriptive

Examples:
- ✅ `qwen2_0.5b_instruct` - Good
- ✅ `llama3.1_8b` - Good
- ❌ `Qwen2-0.5B-Instruct` - Bad (mixed case)
- ❌ `qwen-2-0.5b-instruct` - Bad (too verbose)
- ❌ `model1` - Bad (not descriptive)

## Usage in Scripts

All evaluation scripts automatically use these predefined directories:

```python
# run_evaluation.py
if cfg.MODEL_ID in cfg.OUTPUT_DIRS:
    output_dir = f"{cfg.BASE_OUTPUT_DIR}/{cfg.OUTPUT_DIRS[cfg.MODEL_ID]}"

# run_batch_evaluation.py
output_dir = f"{BASE_OUTPUT_DIR}/{OUTPUT_DIRS[model_id]}"

# run_selective_evaluation.py
# Same pattern
```

## Accessing Results

### By Model Name
```bash
# View Qwen 500M results
cat cache/models/qwen2_0.5b_instruct/metrics_summary.json

# View Llama 3.1 8B results
cat cache/models/llama3.1_8b/metrics_summary.json
```

### By Pattern
```bash
# All Qwen models
ls -d cache/models/qwen*/

# All 7B models
ls -d cache/models/*_7b*/

# All Gemma models
ls -d cache/models/gemma*/
```

### Programmatically
```python
import src.config as cfg

# Get output directory for a specific model
model_id = "Qwen/Qwen2-0.5B-Instruct"
output_dir = f"{cfg.BASE_OUTPUT_DIR}/{cfg.OUTPUT_DIRS[model_id]}"

# Iterate over all models
for model_id in cfg.MODEL_ID_LIST:
    output_dir = f"{cfg.BASE_OUTPUT_DIR}/{cfg.OUTPUT_DIRS[model_id]}"
    print(f"{model_id} -> {output_dir}")
```

## Migration from Auto-Generated Names

If you have existing results with auto-generated names:

```bash
# Rename old directories to match config
mv cache/models/Qwen2-0.5B-Instruct_evaluation cache/models/qwen2_0.5b_instruct
mv cache/models/TinyLlama-1.1B-Chat-v1.0_evaluation cache/models/tinyllama_1.1b_chat
# ... etc
```

Or use a script:
```python
import os
import shutil
import src.config as cfg

old_pattern = "{}_evaluation"
for model_id in cfg.MODEL_ID_LIST:
    old_name = old_pattern.format(model_id.split('/')[-1])
    new_name = cfg.OUTPUT_DIRS[model_id]

    old_path = f"{cfg.BASE_OUTPUT_DIR}/{old_name}"
    new_path = f"{cfg.BASE_OUTPUT_DIR}/{new_name}"

    if os.path.exists(old_path) and not os.path.exists(new_path):
        shutil.move(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")
```

## Best Practices

1. ✅ **Keep names consistent** - Use the same pattern for all models
2. ✅ **Update config first** - Add to `OUTPUT_DIRS` before running evaluation
3. ✅ **Verify before batch runs** - Run `show_output_dirs.py` to check
4. ✅ **Document custom names** - If you change names, note why
5. ✅ **Use descriptive names** - Make it clear what model it is

## Troubleshooting

### Directory Not Found

```bash
# Check if directory is defined
python show_output_dirs.py | grep "model-name"

# Check config
grep "model-name" src/config.py
```

### Model Not in OUTPUT_DIRS

If a model isn't in `OUTPUT_DIRS`, it falls back to auto-generated name:
```
{model_id.split('/')[-1]}_evaluation
```

Add it to config to use a custom name.

### Wrong Directory Used

Verify the model ID matches exactly:
```python
# In config.py
MODEL_ID_LIST = [
    "Qwen/Qwen2-0.5B-Instruct",  # Must match exactly
]

OUTPUT_DIRS = {
    "Qwen/Qwen2-0.5B-Instruct": "qwen2_0.5b_instruct",  # Same string
}
```

## Related Files

- **Config**: `src/config.py`
- **Display script**: `show_output_dirs.py`
- **Evaluation scripts**: `run_*.py`
- **Documentation**: This file
