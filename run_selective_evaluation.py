#!/usr/bin/env python
"""
Selective evaluation script - choose which models to evaluate.
Run from project root: python run_selective_evaluation.py

Usage:
    python run_selective_evaluation.py              # Interactive mode
    python run_selective_evaluation.py 0 2 4        # Evaluate models at indices 0, 2, 4
    python run_selective_evaluation.py --small      # Only small models (<2B)
    python run_selective_evaluation.py --medium     # Only medium models (2-8B)
    python run_selective_evaluation.py --large      # Only large models (>8B)
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import src.config as cfg
from run_batch_evaluation import (
    load_and_prepare_datasets,
    evaluate_single_model,
    save_batch_summary,
    print_batch_summary
)


def get_model_size_category(model_id: str) -> str:
    """Estimate model size category from model ID."""
    model_lower = model_id.lower()

    # Small models (<2B)
    if any(x in model_lower for x in ['0.5b', '500m', '1b', '1.1b', '1.5b']):
        return 'small'
    # Medium models (2-8B)
    elif any(x in model_lower for x in ['2b', '3b', '4b', '7b', '8b']):
        return 'medium'
    # Large models (>8B)
    elif any(x in model_lower for x in ['9b', '13b', '70b']):
        return 'large'
    else:
        return 'unknown'


def filter_models_by_size(model_list, category):
    """Filter models by size category."""
    return [m for m in model_list if get_model_size_category(m) == category]


def interactive_selection(model_list):
    """Interactive model selection."""
    print("\n" + "=" * 80)
    print("AVAILABLE MODELS")
    print("=" * 80)

    for i, model_id in enumerate(model_list):
        size_cat = get_model_size_category(model_id)
        print(f"  [{i}] {model_id:<50} ({size_cat})")

    print("\n" + "=" * 80)
    print("Enter model indices to evaluate (comma or space separated)")
    print("Examples:")
    print("  0,1,2   - Evaluate first three models")
    print("  0-3     - Evaluate models 0 through 3")
    print("  all     - Evaluate all models")
    print("  small   - Evaluate only small models")
    print("  q       - Quit")
    print("=" * 80)

    while True:
        user_input = input("\nYour selection: ").strip().lower()

        if user_input == 'q':
            print("Exiting...")
            sys.exit(0)

        if user_input == 'all':
            return list(range(len(model_list)))

        if user_input in ['small', 'medium', 'large']:
            filtered = filter_models_by_size(model_list, user_input)
            indices = [model_list.index(m) for m in filtered]
            print(f"\nSelected {len(indices)} {user_input} models:")
            for idx in indices:
                print(f"  [{idx}] {model_list[idx]}")
            confirm = input("\nProceed? (y/n): ").strip().lower()
            if confirm == 'y':
                return indices
            continue

        # Parse indices
        try:
            indices = []

            # Handle ranges (e.g., "0-3")
            if '-' in user_input:
                parts = user_input.split('-')
                if len(parts) == 2:
                    start, end = int(parts[0]), int(parts[1])
                    indices = list(range(start, end + 1))
            else:
                # Handle comma or space separated
                if ',' in user_input:
                    indices = [int(x.strip()) for x in user_input.split(',')]
                else:
                    indices = [int(x.strip()) for x in user_input.split()]

            # Validate indices
            if all(0 <= idx < len(model_list) for idx in indices):
                print(f"\nSelected {len(indices)} model(s):")
                for idx in indices:
                    print(f"  [{idx}] {model_list[idx]}")
                confirm = input("\nProceed? (y/n): ").strip().lower()
                if confirm == 'y':
                    return indices
            else:
                print(f"Error: Indices must be between 0 and {len(model_list)-1}")

        except ValueError:
            print("Error: Invalid input. Please enter numbers, ranges, or keywords.")


if __name__ == "__main__":
    print("=" * 80)
    print("SELECTIVE MODEL EVALUATION")
    print("=" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Configuration
    HF_TOKEN = cfg.HUGGINGFACE_HUB_TOKEN
    os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

    BASE_OUTPUT_DIR = cfg.BASE_OUTPUT_DIR
    OUTPUT_DIRS = cfg.OUTPUT_DIRS
    BATCH_SUMMARY_FILE = f"{BASE_OUTPUT_DIR}/selective_evaluation_summary.json"

    # Get model list
    MODEL_LIST = cfg.MODEL_ID_LIST

    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--small':
            selected_models = filter_models_by_size(MODEL_LIST, 'small')
        elif arg == '--medium':
            selected_models = filter_models_by_size(MODEL_LIST, 'medium')
        elif arg == '--large':
            selected_models = filter_models_by_size(MODEL_LIST, 'large')
        else:
            # Assume indices provided
            try:
                indices = [int(x) for x in sys.argv[1:]]
                selected_models = [MODEL_LIST[i] for i in indices]
            except (ValueError, IndexError):
                print("Error: Invalid arguments")
                print(__doc__)
                sys.exit(1)

        print(f"\nSelected {len(selected_models)} model(s):")
        for model_id in selected_models:
            print(f"  - {model_id}")

    else:
        # Interactive mode
        indices = interactive_selection(MODEL_LIST)
        selected_models = [MODEL_LIST[i] for i in indices]

    if not selected_models:
        print("No models selected. Exiting.")
        sys.exit(0)

    # Configuration
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

    # Evaluate selected models
    results_list = []

    for i, model_id in enumerate(selected_models, 1):
        print(f"\n{'='*80}")
        print(f"PROGRESS: Model {i}/{len(selected_models)}")
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
    print("SELECTIVE EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nResults saved to: {BASE_OUTPUT_DIR}/")
    print(f"Summary file: {BATCH_SUMMARY_FILE}")
