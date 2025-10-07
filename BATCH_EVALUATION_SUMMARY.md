# Batch Evaluation Scripts - Summary

## 🎯 Purpose

Automatically evaluate multiple language models on inductive reasoning tasks with LoRA fine-tuning, comparing baseline vs fine-tuned performance.

## 📋 Available Scripts

| Script | Purpose | Use Case |
|--------|---------|----------|
| `run_evaluation.py` | Single model | Test one model from config |
| `run_batch_evaluation.py` | All models | Evaluate entire model list |
| `run_selective_evaluation.py` | Choose models | Interactive or CLI selection |

## 🚀 Quick Start

### Evaluate One Model
```bash
python run_evaluation.py
```

### Evaluate All Models (14 total)
```bash
python run_batch_evaluation.py
```

### Choose Specific Models
```bash
# Interactive selection
python run_selective_evaluation.py

# Small models only (<2B)
python run_selective_evaluation.py --small

# Specific indices
python run_selective_evaluation.py 0 1 2

# Range
python run_selective_evaluation.py
# Then enter: 0-5
```

## 📊 Model List (from config.py)

### Small Models (<2B)
- [0] Qwen/Qwen2-0.5B-Instruct (500M)
- [1] TinyLlama/TinyLlama-1.1B-Chat-v1.0 (1.1B)
- [2] Qwen/Qwen2.5-1.5B-Instruct (1.5B)

### Medium Models (2-8B)
- [3] google/gemma-2-2b-it (2B)
- [4] microsoft/Phi-3-mini-4k-instruct (3.8B)
- [5] google/gemma-3-4b-it (4B)
- [6] deepseek-ai/deepseek-llm-7b-base (7B)
- [7] allenai/OLMo-7B-0424-hf (7B)
- [8] google/gemma-7b-it (7B)
- [9] Qwen/Qwen3-8B (8B)
- [10] unsloth/Meta-Llama-3.1-8B (8B)
- [11] swiss-ai/Apertus-8B-2509 (8B)

### Large Models (>8B)
- [12] 01-ai/Yi-9B (9B)
- [13] google/gemma-2-9b (9B)

## ⚙️ Common Configuration (all models)

From `src/config.py`:
```python
LORA_R = 16              # LoRA rank
LORA_ALPHA = 32          # LoRA alpha
LORA_DROPOUT = 0.05      # LoRA dropout
EPOCHS = 3               # Training epochs
BATCH_SIZE = 8           # Batch size
LR = 2e-4                # Learning rate
```

## 📁 Output Structure

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
├── batch_evaluation_summary.json           # Overall comparison
└── selective_evaluation_summary.json       # Selective run results
```

## 📈 What Gets Evaluated

For each model:
1. ✅ **Baseline Performance** (no fine-tuning)
   - In-distribution (ID) test set
   - Out-of-distribution (OD) test set (DEER)

2. ✅ **Fine-tuning** with LoRA
   - 3 epochs on training set
   - LoRA adapters saved

3. ✅ **Fine-tuned Performance**
   - Same ID and OD test sets
   - Improvements calculated automatically

## 📊 Metrics Computed

- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- **ROUGE-Lsum**: Summary-level ROUGE-L
- **BLEU**: Translation quality

## ⏱️ Time Estimates

| Model Size | Time (GPU) | Time (CPU) |
|------------|-----------|------------|
| 500M-1.5B | 10-20 min | 1-2 hours |
| 2B-4B | 20-40 min | 2-4 hours |
| 7B-8B | 40-80 min | 4-8 hours |
| 9B+ | 80-120 min | 8+ hours |

**Total for all 14 models**: ~10-15 hours on GPU

## 💾 Disk Space

- ~2-5 GB per model
- ~50-70 GB for all 14 models

## 🎯 Use Cases

### Research Comparison
```bash
# Evaluate all models, compare results
python run_batch_evaluation.py

# Analyze summary
cat cache/models/batch_evaluation_summary.json | python -m json.tool
```

### Quick Testing
```bash
# Test on smallest model first
python run_selective_evaluation.py 0

# Then test on a few more
python run_selective_evaluation.py --small
```

### Overnight Run
```bash
# Run all models in background
nohup python run_batch_evaluation.py > batch.log 2>&1 &

# Check progress next day
tail -f batch.log
cat cache/models/batch_evaluation_summary.json
```

### Specific Model Comparison
```bash
# Compare Qwen, Llama, and Gemma at similar sizes
python run_selective_evaluation.py 2 10 3
```

## 🔧 Key Features

### ✅ Automatic
- Loads datasets once
- Runs all evaluations sequentially
- No manual intervention needed

### ✅ Robust
- Continues on errors
- Saves intermediate results
- Won't lose progress if one model fails

### ✅ Comprehensive
- Baseline + fine-tuned for each model
- ID + OD evaluation
- Complete metrics + improvements

### ✅ Organized
- Separate directory per model
- Summary files for comparison
- Predictions saved for inspection

## 📝 Example Workflow

### Day 1: Test Small Models
```bash
python run_selective_evaluation.py --small
# ~1 hour, verifies everything works
```

### Day 2: Medium Models
```bash
python run_selective_evaluation.py --medium
# ~6-8 hours, run overnight
```

### Day 3: Large Models
```bash
python run_selective_evaluation.py --large
# ~3-4 hours
```

### Analysis
```python
import json
import pandas as pd

# Load batch summary
with open('cache/models/batch_evaluation_summary.json') as f:
    data = json.load(f)

# Create comparison DataFrame
results = []
for r in data['results']:
    if r['status'] == 'success':
        results.append({
            'model': r['model_name'],
            'bleu_improvement': r['metrics']['in_distribution']['improvements']['bleu'],
            'rouge1_improvement': r['metrics']['in_distribution']['improvements']['rouge1'],
        })

df = pd.DataFrame(results).sort_values('bleu_improvement', ascending=False)
print(df)
```

## 🐛 Troubleshooting

### Out of Memory
```python
# Edit src/config.py
BATCH_SIZE = 2
GRAD_ACCUM = 4
```

### Model Download Fails
```bash
# Check HuggingFace token
cat .env | grep HUGGINGFACE_HUB_TOKEN
```

### Resume After Failure
```bash
# Check which models succeeded
cat cache/models/batch_evaluation_summary.json | jq '.successful, .failed'

# Continue with remaining models
python run_selective_evaluation.py 7 8 9 10 11 12 13
```

## 📚 Documentation

- **Detailed Guide**: [BATCH_EVALUATION_GUIDE.md](BATCH_EVALUATION_GUIDE.md)
- **Quick Reference**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Single Model**: [TESTING.md](TESTING.md)
- **Main README**: [README.md](README.md)

## 🎓 Best Practices

1. ✅ **Start small**: Test with 1-2 models first
2. ✅ **Monitor first**: Watch first model complete
3. ✅ **Save logs**: Use `nohup` for long runs
4. ✅ **Check disk**: Ensure 50+ GB free
5. ✅ **Use selective**: For iterative experiments

## 🔍 Monitoring Progress

### During Run
```bash
# Watch live output
tail -f batch.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Check Status
```bash
# Count completed models
ls -d cache/models/*_evaluation/ | wc -l

# View latest summary
cat cache/models/batch_evaluation_summary.json | \
  jq '{successful, failed, total_models}'
```

### Find Best Model
```bash
cat cache/models/batch_evaluation_summary.json | \
  jq -r '.results[] | select(.status=="success") |
    "\(.metrics.in_distribution.improvements.bleu) \(.model_name)"' | \
  sort -rn | head -5
```

## 🎉 Expected Output

After successful batch run:

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
swiss-ai/Apertus-8B-2509                 +0.0098    +0.0087    +0.0134
TinyLlama-1.1B-Chat-v1.0                 +0.0087    +0.0071    +0.0092
...
```

## 🚦 Status Codes

- ✅ `success` - Model evaluated successfully
- ❌ `failed` - Model evaluation failed (error logged)

## 🔄 Updating Model List

To add/remove models:

1. Edit `src/config.py`:
```python
MODEL_ID_LIST = [
    "Qwen/Qwen2-0.5B-Instruct",
    "your/new-model",  # Add here
    # ...
]
```

2. Run batch evaluation:
```bash
python run_batch_evaluation.py
```

---

**Ready to start?** Try evaluating small models first:
```bash
python run_selective_evaluation.py --small
```
