# WeightWatcher Model Quality Metrics

## Overview

The evaluation pipeline now includes **WeightWatcher** analysis for both baseline and fine-tuned models, providing insights into model quality and fine-tuning effects.

## What is WeightWatcher?

WeightWatcher is a tool that analyzes the weight matrices of neural networks to predict model quality without requiring test data. It provides metrics that correlate with generalization performance.

## Metrics Provided

### Key WeightWatcher Metrics

| Metric | Description | Good Values |
|--------|-------------|-------------|
| `log_norm` | Log of weight matrix norms | Lower is better |
| `alpha` | Power law exponent | 2-6 typical |
| `alpha_weighted` | Weighted power law | Lower indicates better training |
| `log_alpha_norm` | Normalized alpha metric | Lower is better |
| `log_spectral_norm` | Log of spectral norms | Lower is better |
| `stable_rank` | Effective rank of layers | Higher indicates more learning |

## Integration in Evaluation Pipeline

### Automatic Analysis

WeightWatcher analysis runs automatically during evaluation:

1. **After loading baseline model** - Analyzes pretrained weights
2. **After fine-tuning** - Analyzes adapted weights
3. **Comparison** - Calculates changes from baseline to fine-tuned

### Where to Find Results

#### Console Output

During evaluation, you'll see:

```
[4/7] Loading baseline model...
  - Analyzing baseline model quality...
    Baseline WeightWatcher metrics: {...}

[6/7] Fine-tuning model with LoRA...
  - Analyzing fine-tuned model quality...
    Fine-tuned WeightWatcher metrics: {...}

================================================================================
WEIGHTWATCHER MODEL QUALITY ANALYSIS
================================================================================

Baseline Model:
  log_norm            : 2.5570
  alpha               : 6.3327
  alpha_weighted      : 3.8719
  ...

Fine-tuned Model:
  log_norm            : 1.2555
  alpha               : 6.1077
  alpha_weighted      : 0.4408
  ...

Changes (Fine-tuned - Baseline):
  log_norm            : -1.3015
  alpha               : -0.2250
  alpha_weighted      : -3.4311
  ...
```

#### metrics_summary.json

WeightWatcher metrics are saved in the metrics summary:

```json
{
  "timestamp": "2025-10-06T20:30:00",
  "in_distribution": { ... },
  "out_of_distribution": { ... },
  "weightwatcher": {
    "baseline": {
      "log_norm": 2.557,
      "alpha": 6.333,
      "alpha_weighted": 3.872,
      "log_alpha_norm": 4.251,
      "log_spectral_norm": 0.748,
      "stable_rank": 84.571
    },
    "finetuned": {
      "log_norm": 1.255,
      "alpha": 6.108,
      "alpha_weighted": 0.441,
      "log_alpha_norm": 0.797,
      "log_spectral_norm": -0.057,
      "stable_rank": 44.860
    },
    "changes": {
      "log_norm": -1.302,
      "alpha": -0.225,
      "alpha_weighted": -3.431,
      "log_alpha_norm": -3.454,
      "log_spectral_norm": -0.805,
      "stable_rank": -39.711
    }
  }
}
```

## Interpreting Results

### After Fine-tuning (Typical)

✅ **Good Signs:**
- `alpha_weighted` decreases significantly
- `log_norm` decreases
- `stable_rank` changes (varies by model)

❌ **Warning Signs:**
- `alpha_weighted` increases substantially
- Extreme changes in any metric
- `alpha` outside 2-6 range

### Example Interpretation

```
Changes (Fine-tuned - Baseline):
  log_norm            : -1.3015  ✅ Decreased (good)
  alpha               : -0.2250  ✅ Slight decrease (good)
  alpha_weighted      : -3.4311  ✅ Large decrease (excellent)
  stable_rank         : -39.711  ⚠️  Decreased (LoRA effect)
```

**Interpretation:**
- Fine-tuning improved model quality (`alpha_weighted` decreased significantly)
- Weight matrices are more organized (`log_norm` decreased)
- `stable_rank` decrease is expected with LoRA (only adapts small subspace)

## Use Cases

### 1. Model Selection

Compare WeightWatcher metrics across different models:

```python
import json

# Load batch summary
with open('cache/models/batch_evaluation_summary.json') as f:
    data = json.load(f)

# Compare alpha_weighted improvements
for r in data['results']:
    if r['status'] == 'success' and 'weightwatcher' in r:
        model = r['model_name']
        change = r['weightwatcher']['changes']['alpha_weighted']
        print(f"{model}: {change:.4f}")
```

### 2. Fine-tuning Quality

Assess whether fine-tuning improved model quality:

```python
ww_metrics = results['baseline_ww_metrics']
ww_ft_metrics = results['finetuned_ww_metrics']

alpha_change = ww_ft_metrics['alpha_weighted'] - ww_metrics['alpha_weighted']

if alpha_change < -1.0:
    print("Excellent fine-tuning quality")
elif alpha_change < 0:
    print("Good fine-tuning quality")
else:
    print("Fine-tuning may need adjustment")
```

### 3. Monitoring Training

Track WeightWatcher metrics across epochs to detect:
- Overtraining
- Underfitting
- Optimal stopping point

## Dependencies

WeightWatcher is included in `requirements.txt`:

```
weightwatcher
```

If not available, the evaluation pipeline will:
- Print a warning
- Continue without WeightWatcher analysis
- Not include metrics in output

## Disabling WeightWatcher

WeightWatcher analysis is automatic but can be skipped by:

1. **Not installing weightwatcher**:
   ```bash
   pip install -r requirements.txt --no-deps weightwatcher
   ```

2. **Comment out in code** (not recommended):
   ```python
   # In src/models/evaluate.py
   WEIGHTWATCHER_AVAILABLE = False
   ```

## Performance Impact

- **Analysis time**: ~30-60 seconds per model
- **Memory**: Minimal additional memory
- **Does not affect training**: Analysis runs after training

## References

### Key Papers

1. **Predicting trends in the quality of state-of-the-art neural networks without access to training or testing data**
   - Martin et al., Nature Communications 2021

2. **WeightWatcher: Learning from weight matrices in neural networks**
   - GitHub: https://github.com/CalculatedContent/WeightWatcher

### Understanding Metrics

- **Power Law Alpha**: Indicates heavy-tailed distributions in weight matrices
  - α < 2: Over-trained
  - 2 < α < 6: Well-trained
  - α > 6: Under-trained

- **Alpha Weighted**: Combines alpha with layer size
  - Lower = Better generalization
  - Large decreases after fine-tuning = Successful adaptation

- **Stable Rank**: Effective dimensionality
  - Higher = More expressive
  - With LoRA: Expected to decrease (by design)

## Example Output

### Complete Pipeline with WeightWatcher

```bash
python run_evaluation.py
```

Output:
```
================================================================================
STARTING EVALUATION PIPELINE
================================================================================

[1/7] Loading tokenizer...
[2/7] Loading evaluation metrics...
[3/7] Extracting validation data...
[4/7] Loading baseline model...
  - Analyzing baseline model quality...
WARNING:weightwatcher:PyTorch is available but CUDA is not. Defaulting to NumPy for SVD
    Baseline WeightWatcher metrics: {
      'log_norm': 2.557,
      'alpha': 6.333,
      'alpha_weighted': 3.872,
      'log_alpha_norm': 4.251,
      'log_spectral_norm': 0.748,
      'stable_rank': 84.571
    }

[5/7] Evaluating baseline model...
  - Generating in-distribution baseline predictions...
  - In-distribution baseline metrics: {...}

[6/7] Fine-tuning model with LoRA...
Trainable parameters: 2,162,688 || all params: 496,195,456 || trainable%: 0.4359
Starting fine-tuning...
[Training progress bars...]

  - Analyzing fine-tuned model quality...
    Fine-tuned WeightWatcher metrics: {
      'log_norm': 1.255,
      'alpha': 6.108,
      'alpha_weighted': 0.441,
      ...
    }

[7/7] Evaluating fine-tuned model...

================================================================================
WEIGHTWATCHER MODEL QUALITY ANALYSIS
================================================================================

Baseline Model:
  log_norm            : 2.5570
  alpha               : 6.3327
  alpha_weighted      : 3.8719
  log_alpha_norm      : 4.2511
  log_spectral_norm   : 0.7483
  stable_rank         : 84.5712

Fine-tuned Model:
  log_norm            : 1.2555
  alpha               : 6.1077
  alpha_weighted      : 0.4408
  log_alpha_norm      : 0.7971
  log_spectral_norm   : -0.0566
  stable_rank         : 44.8605

Changes (Fine-tuned - Baseline):
  log_norm            : -1.3015
  alpha               : -0.2250
  alpha_weighted      : -3.4311
  log_alpha_norm      : -3.4540
  log_spectral_norm   : -0.8049
  stable_rank         : -39.7107

================================================================================
EVALUATION PIPELINE COMPLETE
All results saved to: cache/models/qwen2_0.5b_instruct
================================================================================
```

## Troubleshooting

### "weightwatcher not available" Warning

Install weightwatcher:
```bash
pip install weightwatcher
```

### CUDA Warnings

These are normal:
```
WARNING:weightwatcher:PyTorch is available but CUDA is not. Defaulting to NumPy for SVD
```

WeightWatcher works on CPU; warning can be ignored.

### Analysis Fails

If WeightWatcher analysis fails for a model:
- Pipeline continues without it
- Check model architecture compatibility
- Verify model loaded correctly

## Advanced Usage

### Custom Analysis

```python
from src.models.evaluate import analyze_model_quality

# Analyze any model
metrics = analyze_model_quality(my_model, "my_model_name")
if metrics:
    print(f"Alpha weighted: {metrics['alpha_weighted']}")
```

### Batch Comparison

```python
import json
import pandas as pd

# Load all results
with open('cache/models/batch_evaluation_summary.json') as f:
    data = json.load(f)

# Create DataFrame
ww_results = []
for r in data['results']:
    if r['status'] == 'success' and 'weightwatcher' in r.get('metrics', {}):
        ww = r['metrics'].get('weightwatcher', {})
        if 'changes' in ww:
            ww_results.append({
                'model': r['model_name'],
                **ww['changes']
            })

df = pd.DataFrame(ww_results)
print(df.sort_values('alpha_weighted'))
```

## Related Documentation

- **Evaluation Guide**: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- **Batch Evaluation**: [BATCH_EVALUATION_GUIDE.md](BATCH_EVALUATION_GUIDE.md)
- **Main README**: [README.md](README.md)
