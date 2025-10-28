#!/usr/bin/env python
"""
Hyperparameter grid search for LoRA fine-tuning.

This script traverses a grid of hyperparameters, training a model for each
combination and saving results to separate directories.

Usage:
    python run_hyperparam_search.py
"""

import os
import sys
import json
import random
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Iterator
from datetime import datetime

from transformers import AutoTokenizer, set_seed
from datasets import Dataset

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import project modules
from src.preprocess.utils import to_text
from src.utils.io.read import RawDataReader
from src.settings import Settings
from src.models.sft.lora import finetune_model_with_lora
import src.config as cfg


@dataclass
class HyperparamConfig:
    """Container for tunable hyperparameters."""
    # Model
    model_id: str

    # LoRA hyperparameters
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    # Training hyperparameters
    lr: float
    epochs: int
    batch_size: int
    grad_accum: int

    # Sequence settings
    max_seq_len: int
    gen_max_new_tokens: int

    # Fixed parameters (not part of grid search)
    target_modules: List[str] = None
    log_steps: int = 10
    save_steps: int = 50
    seed: int = 42
    val_fraction: float = 0.30

    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    def get_signature(self) -> str:
        """
        Generate hyperparameter signature for directory naming.

        Example: r16_a32_lr2e-04_e3_bs8_ga2_sl512
        """
        return (f"r{self.lora_r}_a{self.lora_alpha}_"
                f"lr{self.lr:.0e}_e{self.epochs}_"
                f"bs{self.batch_size}_ga{self.grad_accum}_"
                f"sl{self.max_seq_len}")

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class HyperparamGrid:
    """
    Defines the hyperparameter search space.

    Each parameter can have multiple possible values. The grid generator
    will create all valid combinations.
    """
    def __init__(self):
        # Model(s) to train
        self.model_ids = [
            cfg.MODEL_ID  # Default from config
            # Add more models here if you want to search across models
        ]

        # LoRA hyperparameters
        self.lora_r_values = [8, 16, 32]
        self.lora_alpha_values = [16, 32, 64]
        self.lora_dropout_values = [0.05, 0.1]

        # Training hyperparameters
        self.lr_values = [1e-4, 2e-4, 5e-4]
        self.epochs_values = [3, 5]
        self.batch_size_values = [4, 8]
        self.grad_accum_values = [2, 4]

        # Sequence settings
        self.max_seq_len_values = [256, 512]
        self.gen_max_new_tokens_values = [64, 128]

    def generate_configs(self) -> Iterator[HyperparamConfig]:
        """
        Generate all valid hyperparameter combinations.

        You can add filtering logic here to skip invalid combinations.
        For example, skip configs where batch_size * grad_accum is too large.

        Yields:
            HyperparamConfig for each valid combination
        """
        # Create cartesian product of all hyperparameter values
        combinations = itertools.product(
            self.model_ids,
            self.lora_r_values,
            self.lora_alpha_values,
            self.lora_dropout_values,
            self.lr_values,
            self.epochs_values,
            self.batch_size_values,
            self.grad_accum_values,
            self.max_seq_len_values,
            self.gen_max_new_tokens_values
        )

        for (model_id, lora_r, lora_alpha, lora_dropout,
             lr, epochs, batch_size, grad_accum,
             max_seq_len, gen_max_new_tokens) in combinations:

            # Filter: lora_alpha should be >= lora_r (common practice)
            if lora_alpha < lora_r:
                continue

            # Filter: effective batch size shouldn't be too large
            effective_batch_size = batch_size * grad_accum
            if effective_batch_size > 32:
                continue

            # Create config
            config = HyperparamConfig(
                model_id=model_id,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lr=lr,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum=grad_accum,
                max_seq_len=max_seq_len,
                gen_max_new_tokens=gen_max_new_tokens
            )

            yield config

    def count_configs(self) -> int:
        """Count total number of configs that will be generated."""
        return sum(1 for _ in self.generate_configs())


def create_output_dir(config: HyperparamConfig, base_dir: str = None) -> str:
    """
    Create output directory with model name and hyperparameter signature.

    Example: cache/models/qwen2.5_1.5b_instruct/r16_a32_lr2e-04_e3_bs8_ga2_sl512/
    """
    if base_dir is None:
        base_dir = cfg.BASE_OUTPUT_DIR

    # Get clean model name
    if config.model_id in cfg.OUTPUT_DIRS:
        model_name = cfg.OUTPUT_DIRS[config.model_id]
    else:
        model_name = config.model_id.split('/')[-1].lower().replace('-', '_').replace('.', '_')

    # Create path: base/model_name/hyperparam_signature/
    model_dir = os.path.join(base_dir, model_name)
    run_dir = os.path.join(model_dir, config.get_signature())

    return run_dir


def train_with_config(config: HyperparamConfig,
                     train_ds: Dataset,
                     val_ds: Dataset,
                     hf_token: str) -> Dict:
    """
    Train a model with the given hyperparameter configuration.

    Args:
        config: HyperparamConfig with all hyperparameters
        train_ds: Training dataset
        val_ds: Validation dataset
        hf_token: HuggingFace token

    Returns:
        Dict with training results and metadata
    """
    # Create output directory
    output_dir = create_output_dir(config)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"TRAINING: {config.get_signature()}")
    print("=" * 80)
    print(f"Model: {config.model_id}")
    print(f"Output: {output_dir}")
    print(f"\nHyperparameters:")
    print(f"  LoRA: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"  Training: lr={config.lr}, epochs={config.epochs}")
    print(f"  Batch: size={config.batch_size}, grad_accum={config.grad_accum}, effective={config.batch_size * config.grad_accum}")
    print(f"  Sequence: max_len={config.max_seq_len}, gen_tokens={config.gen_max_new_tokens}")
    print("=" * 80)

    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True, token=hf_token)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Record start time
        start_time = datetime.now()

        # Fine-tune model
        finetuned_model = finetune_model_with_lora(
            model_id=config.model_id,
            train_dataset=train_ds,
            tokenizer=tokenizer,
            output_dir=output_dir,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            batch_size=config.batch_size,
            grad_accum=config.grad_accum,
            learning_rate=config.lr,
            epochs=config.epochs,
            max_seq_len=config.max_seq_len,
            log_steps=config.log_steps,
            save_steps=config.save_steps,
            hf_token=hf_token
        )

        # Record end time
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # Save model
        adapter_dir = os.path.join(output_dir, "adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        finetuned_model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        # Save configuration and metadata
        result = {
            "status": "success",
            "config": config.to_dict(),
            "signature": config.get_signature(),
            "output_dir": output_dir,
            "adapter_dir": adapter_dir,
            "training_examples": len(train_ds),
            "validation_examples": len(val_ds),
            "training_time_seconds": training_time,
            "training_time_minutes": training_time / 60,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
        }

        # Save result
        result_path = os.path.join(output_dir, "training_result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n✓ Training completed in {training_time/60:.1f} minutes")
        print(f"✓ Saved to: {output_dir}")

        return result

    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")

        # Save error info
        result = {
            "status": "failed",
            "config": config.to_dict(),
            "signature": config.get_signature(),
            "output_dir": output_dir,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

        result_path = os.path.join(output_dir, "training_result.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)

        return result


def main():
    """Main entry point for hyperparameter grid search."""
    print("=" * 80)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 80)

    # Set HuggingFace token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = cfg.HUGGINGFACE_HUB_TOKEN
    hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]

    # Define hyperparameter grid
    grid = HyperparamGrid()
    total_configs = grid.count_configs()

    print(f"\nTotal configurations to train: {total_configs}")
    print("\nPress Enter to continue or Ctrl+C to cancel...")
    input()

    # Load and prepare data (once for all configs)
    print("\n" + "=" * 80)
    print("LOADING DATA")
    print("=" * 80)

    rdr = RawDataReader(Settings.paths.RAW_DATA_PATH)
    ir_triplets = rdr.read_ir_triplets()
    print(f"✓ Loaded {len(ir_triplets)} IR-Triplet examples")

    # Split data (using default seed for consistency across all runs)
    set_seed(cfg.SEED)
    random.Random(cfg.SEED).shuffle(ir_triplets)
    split_idx = int(len(ir_triplets) * (1 - cfg.VAL_FRACTION))
    train_raw, val_raw = ir_triplets[:split_idx], ir_triplets[split_idx:]

    print(f"✓ Train: {len(train_raw)} examples")
    print(f"✓ Validation: {len(val_raw)} examples")

    # Convert to datasets
    train_ds = Dataset.from_list([to_text(x) for x in train_raw])
    val_ds = Dataset.from_list([to_text(x) for x in val_raw])

    # Train all configurations
    print("\n" + "=" * 80)
    print("STARTING GRID SEARCH")
    print("=" * 80)

    all_results = []
    successful = 0
    failed = 0

    for i, config in enumerate(grid.generate_configs(), 1):
        print(f"\n{'='*80}")
        print(f"CONFIG {i}/{total_configs}")
        print(f"{'='*80}")

        result = train_with_config(config, train_ds, val_ds, hf_token)
        all_results.append(result)

        if result["status"] == "success":
            successful += 1
        else:
            failed += 1

        # Save intermediate summary
        summary_path = os.path.join(cfg.BASE_OUTPUT_DIR, "hyperparam_search_summary.json")
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_configs": total_configs,
            "completed": i,
            "successful": successful,
            "failed": failed,
            "results": all_results
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

    # Final summary
    print("\n" + "=" * 80)
    print("GRID SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nTotal configurations: {total_configs}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {summary_path}")

    # Print successful configs sorted by training time
    if successful > 0:
        print("\n" + "-" * 80)
        print("SUCCESSFUL CONFIGURATIONS (sorted by training time)")
        print("-" * 80)

        success_results = [r for r in all_results if r["status"] == "success"]
        success_results.sort(key=lambda x: x["training_time_seconds"])

        for r in success_results:
            time_min = r["training_time_minutes"]
            sig = r["signature"]
            print(f"{time_min:6.1f} min  |  {sig}")


if __name__ == "__main__":
    main()
