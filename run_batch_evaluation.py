#!/usr/bin/env python
"""
Batch evaluation script to run the evaluation pipeline on all models.
Run from project root: python run_batch_evaluation.py

Features:
- Runs evaluation on all models in MODEL_ID_LIST
- Saves results for each model separately
- Continues on error (doesn't stop the whole batch)
- Generates a summary comparison at the end
- Progress tracking and time estimation
"""

import os
import sys
import json
import random
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from transformers import set_seed
from datasets import Dataset, DatasetDict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.preprocess.deer import DeerToTriplets
from src.preprocess.utils import to_text
from src.utils.io.read import RawDataReader
from src.settings import Settings
from src.models.evaluate import evaluate_model_pipeline
import src.config as cfg


def load_and_prepare_datasets(seed: int, val_fraction: float):
    """Load and prepare datasets once to reuse across all models."""
    print("\n" + "=" * 80)
    print("LOADING AND PREPARING DATASETS")
    print("=" * 80)

    # Load raw data
    print("\n[1/4] Loading datasets...")
    rdr = RawDataReader(Settings.paths.RAW_DATA_PATH)
    ir_triplets_dataset = rdr.read_ir_triplets()
    deer_dataset = rdr.read_deer()

    # Convert DEER to triplets
    print("[2/4] Converting DEER dataset to triplets...")
    deer_to_triplets_converter = DeerToTriplets()
    deer_to_triplets_converter.process(deer_dataset)
    od_val_data = deer_to_triplets_converter.triplets

    # Set seed for reproducibility
    set_seed(seed)

    # Split data into train and in-distribution validation
    print("[3/4] Splitting data...")
    data = ir_triplets_dataset
    random.Random(seed).shuffle(data)
    split_idx = int(len(data) * (1 - val_fraction))
    train_raw, id_val_raw = data[:split_idx], data[split_idx:]
    od_val_raw = od_val_data

    print(f"  - Training examples: {len(train_raw)}")
    print(f"  - In-distribution validation examples: {len(id_val_raw)}")
    print(f"  - Out-of-distribution validation examples: {len(od_val_raw)}")

    # Create datasets
    print("[4/4] Creating HuggingFace datasets...")
    train_ds = Dataset.from_list([to_text(x) for x in train_raw])
    id_val_ds = Dataset.from_list([to_text(x) for x in id_val_raw])
    id_dataset = DatasetDict({"train": train_ds, "validation": id_val_ds})

    od_val_ds = Dataset.from_list([to_text(x) for x in od_val_raw])
    od_dataset = DatasetDict({"train": train_ds, "validation": od_val_ds})

    print("✓ Datasets ready!")
    return id_dataset, od_dataset


def evaluate_single_model(
    model_id: str,
    id_dataset: DatasetDict,
    od_dataset: DatasetDict,
    base_output_dir: str,
    hf_token: str,
    config: dict,
    output_dirs: dict = None
) -> Dict:
    """Evaluate a single model and return results."""

    # Use predefined output directory if available, otherwise use model name
    if output_dirs and model_id in output_dirs:
        output_dir = f"{base_output_dir}/{output_dirs[model_id]}"
    else:
        model_name = model_id.split('/')[-1]
        output_dir = f"{base_output_dir}/{model_name}_evaluation"

    print("\n" + "=" * 80)
    print(f"EVALUATING MODEL: {model_id}")
    print("=" * 80)
    print(f"Output directory: {output_dir}")

    try:
        start_time = datetime.now()

        results = evaluate_model_pipeline(
            model_id=model_id,
            id_dataset=id_dataset,
            od_dataset=od_dataset,
            output_dir=output_dir,
            lora_r=config['lora_r'],
            lora_alpha=config['lora_alpha'],
            lora_dropout=config['lora_dropout'],
            target_modules=config['target_modules'],
            batch_size=config['batch_size'],
            grad_accum=config['grad_accum'],
            learning_rate=config['learning_rate'],
            epochs=config['epochs'],
            max_seq_len=config['max_seq_len'],
            log_steps=config['log_steps'],
            save_steps=config['save_steps'],
            hf_token=hf_token,
            skip_finetuning=False
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        results['model_id'] = model_id
        results['model_name'] = model_name
        results['duration_seconds'] = duration
        results['status'] = 'success'
        results['error'] = None

        print(f"\n✓ Model {model_id} completed in {duration/60:.1f} minutes")
        return results

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()

        print(f"\n✗ Error evaluating {model_id}:")
        print(f"  {error_msg}")
        print(f"\nFull traceback:\n{error_trace}")

        return {
            'model_id': model_id,
            'model_name': model_name,
            'status': 'failed',
            'error': error_msg,
            'error_trace': error_trace,
            'output_dir': output_dir
        }


def save_batch_summary(results_list: List[Dict], output_file: str):
    """Save a summary of all batch results."""

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(results_list),
        'successful': sum(1 for r in results_list if r['status'] == 'success'),
        'failed': sum(1 for r in results_list if r['status'] == 'failed'),
        'results': []
    }

    for result in results_list:
        if result['status'] == 'success':
            summary['results'].append({
                'model_id': result['model_id'],
                'model_name': result['model_name'],
                'status': result['status'],
                'duration_seconds': result['duration_seconds'],
                'output_dir': result['output_dir'],
                'metrics': {
                    'in_distribution': {
                        'baseline': result['id_baseline_metrics'],
                        'finetuned': result['id_finetuned_metrics'],
                        'improvements': {
                            k: result['id_finetuned_metrics'][k] - result['id_baseline_metrics'][k]
                            for k in result['id_baseline_metrics'].keys()
                        }
                    },
                    'out_of_distribution': {
                        'baseline': result['od_baseline_metrics'],
                        'finetuned': result['od_finetuned_metrics'],
                        'improvements': {
                            k: result['od_finetuned_metrics'][k] - result['od_baseline_metrics'][k]
                            for k in result['od_baseline_metrics'].keys()
                        } if result['od_baseline_metrics'] else None
                    } if result['od_baseline_metrics'] else None
                }
            })
        else:
            summary['results'].append({
                'model_id': result['model_id'],
                'model_name': result['model_name'],
                'status': result['status'],
                'error': result['error'],
                'output_dir': result.get('output_dir', 'N/A')
            })

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Batch summary saved to: {output_file}")


def print_batch_summary(results_list: List[Dict]):
    """Print a summary table of all results."""

    print("\n" + "=" * 80)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 80)

    successful = [r for r in results_list if r['status'] == 'success']
    failed = [r for r in results_list if r['status'] == 'failed']

    print(f"\nTotal models: {len(results_list)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        print("\n" + "-" * 80)
        print("SUCCESSFUL MODELS - IN-DISTRIBUTION IMPROVEMENTS")
        print("-" * 80)
        print(f"{'Model':<40} {'ROUGE-1':<10} {'ROUGE-L':<10} {'BLEU':<10}")
        print("-" * 80)

        for r in successful:
            improvements = {
                k: r['id_finetuned_metrics'][k] - r['id_baseline_metrics'][k]
                for k in r['id_baseline_metrics'].keys()
            }
            print(f"{r['model_name']:<40} {improvements['rouge1']:>+9.4f} {improvements['rougeL']:>+9.4f} {improvements['bleu']:>+9.4f}")

    if failed:
        print("\n" + "-" * 80)
        print("FAILED MODELS")
        print("-" * 80)
        for r in failed:
            print(f"✗ {r['model_id']}")
            print(f"  Error: {r['error'][:100]}...")
            print()


if __name__ == "__main__":
    print("=" * 80)
    print("BATCH EVALUATION PIPELINE")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    HF_TOKEN = cfg.HUGGINGFACE_HUB_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

    BASE_OUTPUT_DIR = cfg.BASE_OUTPUT_DIR
    OUTPUT_DIRS = cfg.OUTPUT_DIRS
    BATCH_SUMMARY_FILE = f"{BASE_OUTPUT_DIR}/batch_evaluation_summary.json"

    # Get model list
    MODEL_LIST = cfg.MODEL_ID_LIST

    print(f"\nOutput directories (from config.py):")
    for model_id in MODEL_LIST:
        output_name = OUTPUT_DIRS.get(model_id, f"{model_id.split('/')[-1]}_evaluation")
        print(f"  {model_id:<50} -> {output_name}")

    print(f"\nModels to evaluate: {len(MODEL_LIST)}")
    for i, model_id in enumerate(MODEL_LIST, 1):
        print(f"  {i}. {model_id}")

    # Configuration for all models
    config = {
        'lora_r': cfg.LORA_R,
        'lora_alpha': cfg.LORA_ALPHA,
        'lora_dropout': cfg.LORA_DROPOUT,
        'target_modules': cfg.TARGET_MODULES,
        'batch_size': cfg.BATCH_SIZE,
        'grad_accum': cfg.GRAD_ACCUM,
        'learning_rate': cfg.LR,
        'epochs': cfg.EPOCHS,
        'max_seq_len': cfg.MAX_SEQ_LEN,
        'log_steps': cfg.LOG_STEPS,
        'save_steps': cfg.SAVE_STEPS
    }

    print("\nConfiguration:")
    print(f"  LoRA rank: {config['lora_r']}")
    print(f"  LoRA alpha: {config['lora_alpha']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")

    # Load datasets once
    id_dataset, od_dataset = load_and_prepare_datasets(
        seed=cfg.SEED,
        val_fraction=cfg.VAL_FRACTION
    )

    # Evaluate all models
    results_list = []

    for i, model_id in enumerate(MODEL_LIST, 1):
        print(f"\n{'='*80}")
        print(f"PROGRESS: Model {i}/{len(MODEL_LIST)}")
        print(f"{'='*80}")

        result = evaluate_single_model(
            model_id=model_id,
            id_dataset=id_dataset,
            od_dataset=od_dataset,
            base_output_dir=BASE_OUTPUT_DIR,
            hf_token=HF_TOKEN,
            config=config,
            output_dirs=OUTPUT_DIRS
        )

        results_list.append(result)

        # Save intermediate results after each model
        save_batch_summary(results_list, BATCH_SUMMARY_FILE)

    # Final summary
    print_batch_summary(results_list)
    save_batch_summary(results_list, BATCH_SUMMARY_FILE)

    print("\n" + "=" * 80)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {BASE_OUTPUT_DIR}/")
    print(f"Summary file: {BATCH_SUMMARY_FILE}")
