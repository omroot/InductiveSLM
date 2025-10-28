# Quick Start After Cloning

## TL;DR - 3 Steps to Running Your First Evaluation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env_template .env
# Edit .env: add your paths and tokens

# 3. Run evaluation
python run_evaluation.py
```

## What's Already Included (No Downloads Needed!)

✅ **IR-Triplet Dataset** (1,807 examples) - Your main training data
✅ **DEER Dataset** (3 Excel files) - For out-of-distribution testing
✅ **All preprocessing code** - Converts data to training format
✅ **Evaluation scripts** - Complete training & evaluation pipeline

## Setup Checklist

### 1. Install Dependencies ⚙️

```bash
cd InductiveSLM
pip install -r requirements.txt
```

**What this installs**:
- PyTorch & Transformers (for model training)
- PEFT (for LoRA fine-tuning)
- Evaluation metrics (ROUGE, BLEU)
- Data processing (pandas, openpyxl)
- Optional: WeightWatcher (model quality analysis)

### 2. Configure Environment 🔧

```bash
cp .env_template .env
```

**Edit `.env` with**:

```bash
DEBUG = True
DEBUG_ROOT_DIR = /path/to/your/InductiveSLM  # CHANGE THIS
PROD_ROOT_DIR = /path/to/your/InductiveSLM   # CHANGE THIS

# Get from https://huggingface.co/settings/tokens
HUGGINGFACE_HUB_TOKEN = hf_your_token_here

# Get from https://platform.openai.com/api-keys
# (Required only for semantic evaluation)
OPENAI_API_KEY = sk-your_key_here
```

### 3. Choose a Model 🤖

**Edit `src/config.py`**:

For your first test, use a small model:

```python
# Line 81 - Change MODEL_ID to a small model
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"  # Fast, good for testing

# Other small options:
# MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
```

### 4. Run Your First Evaluation 🚀

```bash
python run_evaluation.py
```

**What happens**:
- Loads 1,807 IR-Triplet examples ✓
- Splits into 70% train (1,265) / 30% val (542) ✓
- Loads DEER dataset for out-of-distribution testing ✓
- Evaluates baseline model (no training) ✓
- Fine-tunes with LoRA (parameter-efficient) ✓
- Evaluates fine-tuned model ✓
- Computes metrics (ROUGE, BLEU, semantic similarity) ✓
- Saves all results to `cache/models/` ✓

**Expected time** (for small models):
- Loading data: ~10 seconds
- Baseline evaluation: 2-5 minutes
- Fine-tuning: 10-30 minutes (depends on GPU)
- Fine-tuned evaluation: 2-5 minutes
- **Total**: ~20-40 minutes

## Understanding the Output

Results are saved to `cache/models/{model_name}/`:

```
cache/models/qwen_500m_ft_inductive_slm/
│
├─ 📊 metrics_summary.json              # All metrics in one file
│
├─ 📝 Predictions:
│  ├─ id_val_predictions_baseline.jsonl
│  ├─ id_val_predictions_finetuned.jsonl
│  ├─ od_val_predictions_baseline.jsonl
│  └─ od_val_predictions_finetuned.jsonl
│
├─ 🎯 Semantic Evaluation:
│  ├─ id_baseline_semantic_eval_summary.json
│  ├─ id_finetuned_semantic_eval_summary.json
│  └─ [detailed .jsonl files]
│
└─ 💾 Model:
   └─ adapter/                          # LoRA weights (small!)
      ├─ adapter_model.safetensors
      └─ adapter_config.json
```

### Reading metrics_summary.json

```json
{
  "timestamp": "2025-01-15T10:30:00",
  "in_distribution": {
    "baseline": {
      "rouge1": 0.2341,
      "rougeL": 0.1876,
      "bleu": 8.45,
      "semantic_accuracy": 0.3250
    },
    "finetuned": {
      "rouge1": 0.4521,   // ⬆️ Higher is better
      "rougeL": 0.3987,   // ⬆️ Higher is better
      "bleu": 22.31,      // ⬆️ Higher is better
      "semantic_accuracy": 0.6750  // ⬆️ Higher is better
    },
    "improvements": {
      "rouge1": +0.2180,   // 🎉 Big improvement!
      "rougeL": +0.2111,
      "bleu": +13.86,
      "semantic_accuracy": +0.3500
    }
  }
}
```

## What If You Want To...

### Evaluate Multiple Models

```bash
# Option 1: All small models
python run_selective_evaluation.py --small

# Option 2: Specific models by index
python run_selective_evaluation.py 0 1 2

# Option 3: Interactive selection
python run_selective_evaluation.py
```

### Explore the Dataset

```bash
streamlit run dashboard_ir_triplets.py
```

Opens an interactive web interface to:
- Browse all 1,807 triplets
- Filter by reasoning form
- View statistics
- Export subsets

### Use Jupyter Notebook

```bash
jupyter lab
# Open notebooks/main.ipynb
```

Interactive walkthrough of:
- Data loading
- Preprocessing
- Training
- Evaluation
- Analysis

### Change Hyperparameters

Edit `src/config.py`:

```python
# LoRA settings (affects model capacity)
LORA_R = 16          # Rank (higher = more parameters)
LORA_ALPHA = 32      # Scaling factor

# Training settings
BATCH_SIZE = 8       # Reduce if out of memory
EPOCHS = 3           # More epochs = longer training
LR = 2e-4            # Learning rate

# Sequence length
MAX_SEQ_LEN = 512    # Reduce if out of memory
```

## Troubleshooting

### "CUDA out of memory"
```python
# In src/config.py, reduce:
BATCH_SIZE = 2      # or even 1
MAX_SEQ_LEN = 256   # from 512
```

### "ModuleNotFoundError: No module named 'X'"
```bash
pip install -r requirements.txt
# Check if you activated the right environment
```

### "FileNotFoundError" for datasets
```bash
# Check your .env file:
# DEBUG_ROOT_DIR should point to InductiveSLM folder
# Use absolute paths, not relative
```

### "Missing HUGGINGFACE_HUB_TOKEN"
```bash
# Edit .env and add token from:
# https://huggingface.co/settings/tokens
```

### Semantic evaluation fails
```bash
# Check OPENAI_API_KEY in .env
# Or disable semantic eval by commenting out lines 449-450, 472-474
# in src/models/evaluate.py
```

## Understanding the Data Pipeline

```
Raw Data (cache/raw_data/)
         │
         ├─ ir_triplets.json (1,807 triplets)
         │  └─ {"Training Observations": "...", "Question": "...", "Answer": "..."}
         │
         └─ deer/*.xlsx (Excel files)
            └─ Converted to triplet format
                    ↓
            [Shuffle & Split]
                    ↓
         ┌──────────┴──────────┐
         │                     │
    Train (70%)          Validation (30%)
    1,265 examples       542 examples (ID)
                         + DEER examples (OD)
                    ↓
            [Convert Format]
                    ↓
         {"prompt": "Training Observations:\n...\n\nQuestion:\n...\n\nAnswer:\n",
          "response": "..."}
                    ↓
            [HuggingFace Dataset]
                    ↓
         Ready for training!
```

**Key insight**:
- During training, only the "response" part contributes to loss
- The "prompt" is masked with -100 labels
- This teaches the model to generate answers, not memorize prompts

## Next Steps

1. ✅ Run your first evaluation on a small model
2. 📊 Check the results in `cache/models/`
3. 🔍 Explore the dataset with Streamlit dashboard
4. 📓 Try the Jupyter notebook for interactive learning
5. 🚀 Experiment with different models and hyperparameters
6. 📈 Compare results across models using batch evaluation

## Need More Details?

- **Data pipeline**: See `DATA_PIPELINE_GUIDE.md`
- **Architecture**: See `CLAUDE.md`
- **Paper**: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5529459
- **README**: `README.md`
