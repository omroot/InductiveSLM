import os
import json
import gc
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import torch
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.models.inference.inference import generate_answers_with
from src.models.sft.lora import finetune_model_with_lora

try:
    import weightwatcher as ww
    WEIGHTWATCHER_AVAILABLE = True
except ImportError:
    WEIGHTWATCHER_AVAILABLE = False
    print("Warning: weightwatcher not available. Model quality analysis will be skipped.")


def eval_metrics(preds: list[str],
                 refs: list[str],
                 rouge: evaluate,
                 bleu: evaluate) -> dict[str, float]:
    """
    Compute evaluation metrics (ROUGE and BLEU) for predictions vs references.

    Args:
        preds: List of predicted answers
        refs: List of reference answers
        rouge: Loaded ROUGE metric
        bleu: Loaded BLEU metric

    Returns:
        Dictionary of metric scores
    """
    try:
        r = rouge.compute(predictions=preds, references=refs)
        b = bleu.compute(predictions=preds, references=[[x] for x in refs])
        return {
            "rouge1": r.get("rouge1", 0.0),
            "rouge2": r.get("rouge2", 0.0),
            "rougeL": r.get("rougeL", 0.0),
            "rougeLsum": r.get("rougeLsum", 0.0),
            "bleu": b["score"]
        }
    except Exception as e:
        print(f"Error computing metrics: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0, "bleu": 0.0}


def print_compare(baseline: dict[str, float], finetuned: dict[str, float]):
    """
    Print a comparison table of baseline vs fine-tuned metrics.

    Args:
        baseline: Baseline model metrics
        finetuned: Fine-tuned model metrics
    """
    print("\n===== Metrics (Baseline vs Fine-tuned) =====")
    keys = ["rouge1", "rouge2", "rougeL", "rougeLsum", "bleu"]
    for k in keys:
        print(f"{k:10s}  base: {baseline[k]:6.3f}   ft: {finetuned[k]:6.3f}   Δ: {finetuned[k]-baseline[k]:+6.3f}")


def extract_triplets_from_dataset(ds: Dataset) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Extract prompts, observations, questions, and references from a dataset.

    Args:
        ds: Dataset containing validation data

    Returns:
        Tuple of (prompts, observations, questions, references)
    """
    prompts = ds["validation"]["prompt"]
    obs = [p.split("Question:\n")[0].replace("Training Observations:\n", "").strip() for p in prompts]
    qs = [p.split("Question:\n")[1].split("\n\nAnswer:\n")[0].strip() for p in prompts]
    refs = [x.strip() for x in ds["validation"]["response"]]
    return prompts, obs, qs, refs


def save_predictions(output_dir: str,
                     filename: str,
                     observations: List[str],
                     questions: List[str],
                     references: List[str],
                     predictions: List[str]):
    """
    Save predictions to a JSONL file.

    Args:
        output_dir: Output directory path
        filename: Output filename
        observations: List of observations
        questions: List of questions
        references: List of reference answers
        predictions: List of predicted answers
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        for obs, q, ref, pred in zip(observations, questions, references, predictions):
            f.write(json.dumps({
                "Training Observations": obs,
                "Question": q,
                "Reference": ref,
                "Prediction": pred
            }, ensure_ascii=False) + "\n")

    print(f"Saved predictions to {filepath}")


def analyze_model_quality(model, model_name: str = "model") -> Optional[Dict]:
    """
    Analyze model quality using WeightWatcher.

    Args:
        model: The model to analyze
        model_name: Name for logging purposes

    Returns:
        Dictionary of WeightWatcher metrics or None if not available
    """
    if not WEIGHTWATCHER_AVAILABLE:
        return None

    try:
        print(f"  - Running WeightWatcher analysis on {model_name}...")
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze()
        summary = watcher.get_summary(details)

        # Convert numpy types to native Python types for JSON serialization
        summary_clean = {k: float(v) if hasattr(v, 'item') else v for k, v in summary.items()}

        return summary_clean
    except Exception as e:
        print(f"  - Warning: WeightWatcher analysis failed for {model_name}: {e}")
        return None


def save_metrics(output_dir: str,
                 id_baseline_metrics: Dict[str, float],
                 id_finetuned_metrics: Dict[str, float],
                 od_baseline_metrics: Optional[Dict[str, float]] = None,
                 od_finetuned_metrics: Optional[Dict[str, float]] = None,
                 baseline_ww_metrics: Optional[Dict[str, float]] = None,
                 finetuned_ww_metrics: Optional[Dict[str, float]] = None):
    """
    Save all metrics to a JSON file.

    Args:
        output_dir: Output directory path
        id_baseline_metrics: In-distribution baseline metrics
        id_finetuned_metrics: In-distribution fine-tuned metrics
        od_baseline_metrics: Out-of-distribution baseline metrics (optional)
        od_finetuned_metrics: Out-of-distribution fine-tuned metrics (optional)
        baseline_ww_metrics: WeightWatcher metrics for baseline model (optional)
        finetuned_ww_metrics: WeightWatcher metrics for fine-tuned model (optional)
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics_summary = {
        "timestamp": datetime.now().isoformat(),
        "in_distribution": {
            "baseline": id_baseline_metrics,
            "finetuned": id_finetuned_metrics,
            "improvements": {
                k: id_finetuned_metrics[k] - id_baseline_metrics[k]
                for k in id_baseline_metrics.keys()
            }
        }
    }

    if od_baseline_metrics and od_finetuned_metrics:
        metrics_summary["out_of_distribution"] = {
            "baseline": od_baseline_metrics,
            "finetuned": od_finetuned_metrics,
            "improvements": {
                k: od_finetuned_metrics[k] - od_baseline_metrics[k]
                for k in od_baseline_metrics.keys()
            }
        }

    # Add WeightWatcher metrics if available
    if baseline_ww_metrics or finetuned_ww_metrics:
        metrics_summary["weightwatcher"] = {}

        if baseline_ww_metrics:
            metrics_summary["weightwatcher"]["baseline"] = baseline_ww_metrics

        if finetuned_ww_metrics:
            metrics_summary["weightwatcher"]["finetuned"] = finetuned_ww_metrics

        # Calculate changes if both are available
        if baseline_ww_metrics and finetuned_ww_metrics:
            changes = {}
            for key in baseline_ww_metrics.keys():
                if key in finetuned_ww_metrics:
                    changes[key] = finetuned_ww_metrics[key] - baseline_ww_metrics[key]
            metrics_summary["weightwatcher"]["changes"] = changes

    filepath = os.path.join(output_dir, "metrics_summary.json")
    with open(filepath, "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"Saved metrics summary to {filepath}")


def evaluate_model_pipeline(
    model_id: str,
    id_dataset: Dataset,
    od_dataset: Optional[Dataset] = None,
    output_dir: str = "./evaluation_output",
    train_dataset: Optional[Dataset] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: Optional[List[str]] = None,
    batch_size: int = 4,
    grad_accum: int = 1,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    max_seq_len: int = 512,
    log_steps: int = 10,
    save_steps: int = 500,
    hf_token: Optional[str] = None,
    skip_finetuning: bool = False
) -> Dict:
    """
    Complete evaluation pipeline: baseline evaluation, fine-tuning, and fine-tuned evaluation.

    Args:
        model_id: HuggingFace model ID
        id_dataset: In-distribution dataset (DatasetDict with 'train' and 'validation')
        od_dataset: Out-of-distribution dataset (DatasetDict with 'validation') - optional
        output_dir: Directory to save results
        train_dataset: Training dataset (if None, uses id_dataset['train'])
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: Target modules for LoRA
        batch_size: Training batch size
        grad_accum: Gradient accumulation steps
        learning_rate: Learning rate
        epochs: Number of training epochs
        max_seq_len: Maximum sequence length
        log_steps: Logging frequency
        save_steps: Save frequency
        hf_token: HuggingFace token
        skip_finetuning: If True, skip fine-tuning and only run baseline evaluation

    Returns:
        Dictionary containing all metrics and file paths
    """
    print("=" * 80)
    print("STARTING EVALUATION PIPELINE")
    print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    # Initialize tokenizer
    print("\n[1/7] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation metrics
    print("[2/7] Loading evaluation metrics...")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("sacrebleu")

    # Extract validation data
    print("[3/7] Extracting validation data...")
    id_prompts, id_obs, id_qs, id_refs = extract_triplets_from_dataset(id_dataset)
    print(f"  - In-distribution validation examples: {len(id_refs)}")

    od_prompts, od_obs, od_qs, od_refs = None, None, None, None
    if od_dataset:
        od_prompts, od_obs, od_qs, od_refs = extract_triplets_from_dataset(od_dataset)
        print(f"  - Out-of-distribution validation examples: {len(od_refs)}")

    # Load baseline model
    print("\n[4/7] Loading baseline model...")
    baseline_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        token=hf_token
    )

    # Analyze baseline model quality with WeightWatcher
    print("  - Analyzing baseline model quality...")
    baseline_ww_metrics = analyze_model_quality(baseline_model, "baseline model")
    if baseline_ww_metrics:
        print(f"    Baseline WeightWatcher metrics: {baseline_ww_metrics}")

    # Evaluate baseline on in-distribution
    print("[5/7] Evaluating baseline model...")
    print("  - Generating in-distribution baseline predictions...")
    id_baseline_preds = generate_answers_with(baseline_model, tokenizer, id_obs, id_qs)
    id_baseline_metrics = eval_metrics(id_baseline_preds, id_refs, rouge, bleu)
    print(f"  - In-distribution baseline metrics: {id_baseline_metrics}")

    # Save in-distribution baseline predictions
    save_predictions(output_dir, "id_val_predictions_baseline.jsonl",
                    id_obs, id_qs, id_refs, id_baseline_preds)

    # Evaluate baseline on out-of-distribution (if provided)
    od_baseline_metrics = None
    if od_dataset:
        print("  - Generating out-of-distribution baseline predictions...")
        od_baseline_preds = generate_answers_with(baseline_model, tokenizer, od_obs, od_qs)
        od_baseline_metrics = eval_metrics(od_baseline_preds, od_refs, rouge, bleu)
        print(f"  - Out-of-distribution baseline metrics: {od_baseline_metrics}")

        # Save out-of-distribution baseline predictions
        save_predictions(output_dir, "od_val_predictions_baseline.jsonl",
                        od_obs, od_qs, od_refs, od_baseline_preds)

    # Free baseline model memory
    del baseline_model
    gc.collect()
    torch.cuda.empty_cache()

    if skip_finetuning:
        print("\n[INFO] Skipping fine-tuning as requested.")
        return {
            "id_baseline_metrics": id_baseline_metrics,
            "od_baseline_metrics": od_baseline_metrics,
            "output_dir": output_dir
        }

    # Fine-tune model
    print("\n[6/7] Fine-tuning model with LoRA...")
    if train_dataset is None:
        train_dataset = id_dataset["train"]

    print(f"  - Training examples: {len(train_dataset)}")

    finetuned_model = finetune_model_with_lora(
        model_id=model_id,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        output_dir=output_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        epochs=epochs,
        max_seq_len=max_seq_len,
        log_steps=log_steps,
        save_steps=save_steps,
        hf_token=hf_token
    )

    # Analyze fine-tuned model quality with WeightWatcher
    print("  - Analyzing fine-tuned model quality...")
    finetuned_ww_metrics = analyze_model_quality(finetuned_model, "fine-tuned model")
    if finetuned_ww_metrics:
        print(f"    Fine-tuned WeightWatcher metrics: {finetuned_ww_metrics}")

    # Evaluate fine-tuned model
    print("\n[7/7] Evaluating fine-tuned model...")
    print("  - Generating in-distribution fine-tuned predictions...")
    id_finetuned_preds = generate_answers_with(finetuned_model, tokenizer, id_obs, id_qs)
    id_finetuned_metrics = eval_metrics(id_finetuned_preds, id_refs, rouge, bleu)
    print(f"  - In-distribution fine-tuned metrics: {id_finetuned_metrics}")

    # Save in-distribution fine-tuned predictions
    save_predictions(output_dir, "id_val_predictions_finetuned.jsonl",
                    id_obs, id_qs, id_refs, id_finetuned_preds)

    # Evaluate fine-tuned on out-of-distribution (if provided)
    od_finetuned_metrics = None
    if od_dataset:
        print("  - Generating out-of-distribution fine-tuned predictions...")
        od_finetuned_preds = generate_answers_with(finetuned_model, tokenizer, od_obs, od_qs)
        od_finetuned_metrics = eval_metrics(od_finetuned_preds, od_refs, rouge, bleu)
        print(f"  - Out-of-distribution fine-tuned metrics: {od_finetuned_metrics}")

        # Save out-of-distribution fine-tuned predictions
        save_predictions(output_dir, "od_val_predictions_finetuned.jsonl",
                        od_obs, od_qs, od_refs, od_finetuned_preds)

    # Save model and tokenizer
    print("\n  - Saving fine-tuned model adapters...")
    adapter_dir = os.path.join(output_dir, "adapter")
    finetuned_model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"  - Saved adapters to {adapter_dir}")

    # Save metrics summary (including WeightWatcher if available)
    print("\n  - Saving metrics summary...")
    save_metrics(output_dir, id_baseline_metrics, id_finetuned_metrics,
                od_baseline_metrics, od_finetuned_metrics,
                baseline_ww_metrics, finetuned_ww_metrics)

    # Print comparison
    print("\n" + "=" * 80)
    print("IN-DISTRIBUTION RESULTS")
    print_compare(id_baseline_metrics, id_finetuned_metrics)

    if od_dataset:
        print("\n" + "=" * 80)
        print("OUT-OF-DISTRIBUTION RESULTS")
        print_compare(od_baseline_metrics, od_finetuned_metrics)

    # Print WeightWatcher comparison if available
    if baseline_ww_metrics and finetuned_ww_metrics:
        print("\n" + "=" * 80)
        print("WEIGHTWATCHER MODEL QUALITY ANALYSIS")
        print("=" * 80)
        print("\nBaseline Model:")
        for key, value in baseline_ww_metrics.items():
            print(f"  {key:20s}: {value:.4f}")
        print("\nFine-tuned Model:")
        for key, value in finetuned_ww_metrics.items():
            print(f"  {key:20s}: {value:.4f}")
        print("\nChanges (Fine-tuned - Baseline):")
        for key in baseline_ww_metrics.keys():
            if key in finetuned_ww_metrics:
                change = finetuned_ww_metrics[key] - baseline_ww_metrics[key]
                print(f"  {key:20s}: {change:+.4f}")

    print("\n" + "=" * 80)
    print("EVALUATION PIPELINE COMPLETE")
    print(f"All results saved to: {output_dir}")
    print("=" * 80)

    # Return results
    return {
        "id_baseline_metrics": id_baseline_metrics,
        "id_finetuned_metrics": id_finetuned_metrics,
        "od_baseline_metrics": od_baseline_metrics,
        "od_finetuned_metrics": od_finetuned_metrics,
        "baseline_ww_metrics": baseline_ww_metrics,
        "finetuned_ww_metrics": finetuned_ww_metrics,
        "output_dir": output_dir,
        "adapter_dir": adapter_dir
    }
