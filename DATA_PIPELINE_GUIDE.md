# Data Pipeline Guide

## Quick Answer: Data is Already Ready!

After cloning the repository, **all datasets are already included** in `cache/raw_data/`. You don't need to download anything - just configure your environment and run the evaluation scripts.

## Complete Setup Steps (New Clone)

### 1. Install Dependencies

```bash
cd InductiveSLM
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template
cp .env_template .env

# Edit .env file with your settings:
# - Set DEBUG_ROOT_DIR to your InductiveSLM path (e.g., /home/user/InductiveSLM)
# - Add your HUGGINGFACE_HUB_TOKEN (get from https://huggingface.co/settings/tokens)
# - Add your OPENAI_API_KEY (get from https://platform.openai.com/api-keys)
```

Example `.env`:
```bash
DEBUG = True
DEBUG_ROOT_DIR = /home/jack/code/InductiveSLM
PROD_ROOT_DIR = /home/jack/code/InductiveSLM
OPENAI_API_KEY = sk-your-key-here
HUGGINGFACE_HUB_TOKEN = hf_your-token-here
```

### 3. Verify Data is Present

```bash
# Check IR-Triplets
ls -lh cache/raw_data/ir_triplets/ir_triplets.json
# Should show: ~764K file

# Check DEER
ls -lh cache/raw_data/deer/
# Should show: 3 Excel files (train, val, test)
```

### 4. Run Your First Evaluation

```bash
# Option A: Edit config.py to choose a small model for testing
# Change MODEL_ID to something small like "Qwen/Qwen2-0.5B-Instruct"

python run_evaluation.py
```

That's it! The script will:
- Load the data from cache/raw_data/
- Split it into train/validation
- Fine-tune the model
- Evaluate and save results

## Understanding the Data Pipeline

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    YOUR LOCAL MACHINE                        │
│                                                              │
│  InductiveSLM/                                              │
│  ├─ cache/raw_data/                 [Already included]     │
│  │  ├─ ir_triplets/                                        │
│  │  │  └─ ir_triplets.json          (1,807 triplets)      │
│  │  └─ deer/                                               │
│  │     ├─ Hypothetical_Induction_train.xlsx                │
│  │     ├─ Hypothetical_Induction_val.xlsx                  │
│  │     └─ Hypothetical_Induction_test.xlsx                 │
│  │                                                          │
│  └─ src/                                                    │
│     ├─ utils/io/read.py           [Loads data]            │
│     ├─ preprocess/                 [Converts formats]      │
│     └─ models/                     [Trains & evaluates]    │
└─────────────────────────────────────────────────────────────┘
```

### What Happens When You Run `python run_evaluation.py`

```
Step 1: Load Raw Data
├─ RawDataReader.read_ir_triplets()
│  └─ Returns: List of 1,807 dicts
│     Example: {"Training Observations": "...", "Question": "...", "Answer": "..."}
│
└─ RawDataReader.read_deer()
   └─ Returns: Pandas DataFrame with Excel data

Step 2: Convert DEER to Triplets
├─ DeerToTriplets().process(deer_df)
│  └─ Combines fact columns into observations
│  └─ Creates questions about rule types
│  └─ Uses rule column as answer
│
└─ Result: DEER triplets in same format as IR-Triplets

Step 3: Split IR-Triplets
├─ Shuffle with seed=42 (reproducible)
├─ Split: 70% train (1,265 triplets)
│         30% validation (542 triplets)
│
└─ DEER becomes out-of-distribution validation (not in training)

Step 4: Convert to HuggingFace Format
├─ to_text() converts each triplet:
│  From: {"Training Observations": "...", "Question": "...", "Answer": "..."}
│  To:   {"prompt": "Training Observations:\n...\n\nQuestion:\n...\n\nAnswer:\n",
│         "response": "...",
│         "text": "{prompt + response}"}
│
└─ Dataset.from_list() creates HuggingFace Dataset

Step 5: Train Model
├─ PromptAnswerCollator prepares batches
│  ├─ Tokenizes prompt and response separately
│  ├─ Sets prompt tokens to -100 (excluded from loss)
│  └─ Only response tokens contribute to training loss
│
└─ LoRA fine-tunes only adapter weights (not full model)

Step 6: Evaluate
├─ Generate predictions on validation sets
├─ Compute metrics:
│  ├─ ROUGE (1, 2, L, Lsum)
│  ├─ BLEU
│  └─ Semantic similarity (via OpenAI API)
│
└─ Save results to cache/models/{model_name}/
```

## Data Formats Explained

### 1. Raw IR-Triplet (as stored in JSON)

```json
{
  "form": "Enumerative induction",
  "Training Observations": "From 1980-01 to 1999-12, CPI rose from ~78.5 to 168.2 (+114.4%) with a near-linear trend (R²≈0.995), no significant outliers, weak seasonality, and very high lag-1 autocorrelation (~0.987).",
  "Question": "Do U.S. consumer prices generally compound upward over multi-year spans?",
  "Answer": "Yes, defeasibly: across many months in 1980–1999 the level steadily rose, suggesting prices typically drift upward over long horizons, though the pace can change."
}
```

**What it means**:
- `form`: Type of inductive reasoning (classification, not used in training)
- `Training Observations`: Historical data or facts the model observes
- `Question`: What the model needs to infer from observations
- `Answer`: The correct inductive inference

### 2. After to_text() (ready for model training)

```python
{
  "prompt": """Training Observations:
From 1980-01 to 1999-12, CPI rose from ~78.5 to 168.2 (+114.4%)...

Question:
Do U.S. consumer prices generally compound upward over multi-year spans?

Answer:
""",
  "response": "Yes, defeasibly: across many months in 1980–1999...",
  "text": "{full prompt + response concatenated}"
}
```

**What it means**:
- `prompt`: The input given to the model (observations + question + "Answer:\n")
- `response`: The target output (what the model should generate)
- `text`: Full text for reference (not used directly in training)

### 3. During Training (inside PromptAnswerCollator)

```
Input IDs:    [token_1, token_2, ..., token_n, answer_1, answer_2, ...]
                ^^^^^^^^^ prompt tokens ^^^^^^^^^ ^^^^ response tokens ^^^^

Labels:       [-100,    -100,    ..., -100,     answer_1, answer_2, ...]
               ^^^^^ masked (no loss) ^^^^^     ^^^^ supervised ^^^^
```

**Why mask the prompt?**
- We want the model to learn to *generate* answers
- Not to predict the observations/question (we already have those)
- Loss is only computed on answer tokens

## Typical Data Sizes

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| IR-Triplet Train | Fine-tuning | ~1,265 triplets | 70% of IR-Triplet |
| IR-Triplet Val | In-distribution eval | ~542 triplets | 30% of IR-Triplet |
| DEER | Out-of-distribution eval | ~450-500 triplets | All DEER files combined |

## Where Results Are Saved

After running evaluation, check `cache/models/{model_name}/`:

```
cache/models/qwen2.5_1.5b_instruct/
├── adapter/                                    [LoRA weights]
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
│
├── id_val_predictions_baseline.jsonl          [Baseline predictions]
├── id_val_predictions_finetuned.jsonl         [Fine-tuned predictions]
├── od_val_predictions_baseline.jsonl
├── od_val_predictions_finetuned.jsonl
│
├── id_baseline_semantic_eval_summary.json     [Semantic accuracy]
├── id_baseline_semantic_eval_detailed.jsonl
├── id_finetuned_semantic_eval_summary.json
├── id_finetuned_semantic_eval_detailed.jsonl
│
├── metrics_summary.json                        [All metrics combined]
│
└── checkpoint-XXX/                             [Training checkpoints]
```

## Common Questions

### Q: Do I need to download any datasets?
**A**: No! Everything is included in the repository.

### Q: How do I use my own data?
**A**: Create triplets in the same format as IR-Triplets:
```python
my_triplets = [
    {
        "Training Observations": "your observations here",
        "Question": "your question here",
        "Answer": "expected answer here"
    },
    # ... more triplets
]
# Then use the same pipeline: to_text() → Dataset.from_list()
```

### Q: Can I skip the DEER out-of-distribution evaluation?
**A**: Yes! In `evaluate_model_pipeline()`, pass `od_dataset=None`:
```python
results = evaluate_model_pipeline(
    model_id=cfg.MODEL_ID,
    id_dataset=id_dataset,
    od_dataset=None,  # Skip OD evaluation
    # ... other params
)
```

### Q: What if I only want to evaluate without fine-tuning?
**A**: Set `skip_finetuning=True`:
```python
results = evaluate_model_pipeline(
    model_id=cfg.MODEL_ID,
    id_dataset=id_dataset,
    skip_finetuning=True,  # Only baseline evaluation
    # ... other params
)
```

### Q: How do I change the train/validation split?
**A**: Edit `VAL_FRACTION` in `src/config.py`:
```python
VAL_FRACTION = 0.30  # 30% validation (default)
# or
VAL_FRACTION = 0.20  # 20% validation, 80% train
```

### Q: Can I use this pipeline in a Jupyter notebook?
**A**: Yes! See `notebooks/main.ipynb` for a complete example. The notebook walks through each step interactively.

## Next Steps

1. **Explore the data**: `streamlit run dashboard_ir_triplets.py`
2. **Run a quick test**: `python run_evaluation.py` (edit config.py first to choose a small model)
3. **Batch evaluation**: `python run_selective_evaluation.py --small`
4. **Customize**: Edit `src/config.py` to change hyperparameters

## Troubleshooting

### Error: "File not found" for datasets
- Check that `DEBUG_ROOT_DIR` in `.env` points to your InductiveSLM directory
- Verify files exist: `ls cache/raw_data/ir_triplets/ir_triplets.json`

### Error: "Missing HUGGINGFACE_HUB_TOKEN"
- Edit `.env` and add your token from https://huggingface.co/settings/tokens

### Error: "Missing OPENAI_API_KEY" during semantic evaluation
- Edit `.env` and add your OpenAI API key
- Or comment out the semantic evaluation calls in `src/models/evaluate.py`

### Out of memory during training
- Reduce `BATCH_SIZE` in `src/config.py` (try 2 or 1)
- Choose a smaller model
- Reduce `MAX_SEQ_LEN` (try 256 instead of 512)
