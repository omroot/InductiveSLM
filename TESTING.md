# Testing the Evaluation Pipeline

This document explains how to test the evaluation pipeline.

## Quick Test

To run a complete test of the evaluation pipeline with the Qwen 500M model:

```bash
python run_evaluation.py
```

Or alternatively, run as a module:
```bash
python -m src.models.evaluate
```

This will:
1. Load the IR-Triplets and DEER datasets
2. Evaluate the baseline Qwen 500M model
3. Fine-tune the model with LoRA
4. Evaluate the fine-tuned model
5. Save all results to `cache/models/Qwen2-0.5B-Instruct_evaluation/`

## Expected Output Structure

After running the test, you'll find the following in `cache/models/Qwen2-0.5B-Instruct_evaluation/`:

```
cache/models/Qwen2-0.5B-Instruct_evaluation/
├── id_val_predictions_baseline.jsonl       # Baseline predictions (ID)
├── id_val_predictions_finetuned.jsonl      # Fine-tuned predictions (ID)
├── od_val_predictions_baseline.jsonl       # Baseline predictions (OD)
├── od_val_predictions_finetuned.jsonl      # Fine-tuned predictions (OD)
├── metrics_summary.json                    # Complete metrics comparison
├── adapter/                                # Fine-tuned LoRA adapters
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── tokenizer files...
└── checkpoint-*/                           # Training checkpoints
```

## Configuration

The test uses settings from `src/config.py`:

- **Model**: `Qwen/Qwen2-0.5B-Instruct` (500M parameters)
- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Batch Size**: 8
- **Max Sequence Length**: 512

## Modifying Test Parameters

To test with different parameters, edit `src/config.py` before running the test:

```python
# src/config.py
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"  # Change model
LORA_R = 16                             # Change LoRA rank
EPOCHS = 3                              # Change number of epochs
BATCH_SIZE = 8                          # Change batch size
```

## Running with Different Models

To test with a different model from the available list in `config.py`:

1. Edit `src/config.py` and change `MODEL_ID`:
   ```python
   MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
   ```

2. Run the test:
   ```bash
   python -m src.models.evaluate
   ```

Available models in `config.py`:
- `Qwen/Qwen2-0.5B-Instruct` (500M)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (1.1B)
- `Qwen/Qwen2.5-1.5B-Instruct` (1.5B)
- `google/gemma-2-2b-it` (2B)
- And others...

## Memory Requirements

Expected memory usage for Qwen 500M:
- **GPU Memory**: ~4-6 GB
- **System RAM**: ~8-10 GB

For larger models, adjust `BATCH_SIZE` and `GRAD_ACCUM` in `config.py`:

```python
# For 2GB+ models on limited GPU
BATCH_SIZE = 2
GRAD_ACCUM = 4  # Maintains effective batch size of 8
```

## Test Duration

Expected runtime on typical hardware:
- **Qwen 500M on GPU**: ~10-15 minutes
- **Qwen 500M on CPU**: ~1-2 hours
- **Larger models**: Scale accordingly

## Verifying Results

After the test completes, check:

1. **Console Output**: Should show metrics improvements
2. **Metrics File**: `cache/models/*/metrics_summary.json`
3. **Predictions**: `cache/models/*/*.jsonl` files
4. **Adapters**: `cache/models/*/adapter/` directory

### Sample Metrics Check

```bash
# View metrics summary
cat cache/models/Qwen2-0.5B-Instruct_evaluation/metrics_summary.json | python -m json.tool

# Count predictions
wc -l cache/models/Qwen2-0.5B-Instruct_evaluation/*_predictions_*.jsonl
```

## Troubleshooting

### CUDA Out of Memory

```python
# Reduce batch size in config.py
BATCH_SIZE = 2
GRAD_ACCUM = 4
```

### HuggingFace Token Error

Make sure your `.env` file contains:
```
HUGGINGFACE_HUB_TOKEN=your_token_here
```

### Dataset Not Found

Ensure your raw data is in the correct location specified in `src/settings.py`.

## Baseline Only Test

To test only baseline evaluation (skip fine-tuning):

1. Modify the `__main__` section in `src/models/evaluate.py`:
   ```python
   skip_finetuning=True  # Change False to True
   ```

2. Run:
   ```bash
   python -m src.models.evaluate
   ```

This is useful for quick testing or comparing baseline models.

## Next Steps

After successful testing:

1. Review the metrics in `metrics_summary.json`
2. Examine predictions in the `.jsonl` files
3. Use the saved adapters for inference
4. Try different models or hyperparameters
5. Use the evaluation function in your own scripts (see `EVALUATION_GUIDE.md`)
