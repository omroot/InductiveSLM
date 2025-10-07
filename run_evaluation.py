#!/usr/bin/env python
"""
Standalone script to run the evaluation pipeline.
Run from project root: python run_evaluation.py
"""

import os
import sys
import random
from pathlib import Path
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

if __name__ == "__main__":
    print("=" * 80)
    print("EVALUATION PIPELINE TEST")
    print("=" * 80)
    print(f"\nModel: {cfg.MODEL_ID}")
    print(f"Output directory: cache/models/{cfg.MODEL_ID.split('/')[-1]}_evaluation")

    # Set HuggingFace token
    os.environ["HUGGINGFACE_HUB_TOKEN"] = cfg.HUGGINGFACE_HUB_TOKEN
    HF_TOKEN = os.environ["HUGGINGFACE_HUB_TOKEN"]

    # Load raw data
    print("\n[Data Loading] Loading datasets...")
    rdr = RawDataReader(Settings.paths.RAW_DATA_PATH)
    ir_triplets_dataset = rdr.read_ir_triplets()
    deer_dataset = rdr.read_deer()

    # Convert DEER to triplets
    print("[Data Loading] Converting DEER dataset to triplets...")
    deer_to_triplets_converter = DeerToTriplets()
    deer_to_triplets_converter.process(deer_dataset)
    od_val_data = deer_to_triplets_converter.triplets

    # Set seed for reproducibility
    set_seed(cfg.SEED)

    # Split data into train and in-distribution validation
    print("[Data Loading] Splitting data...")
    data = ir_triplets_dataset
    random.Random(cfg.SEED).shuffle(data)
    split_idx = int(len(data) * (1 - cfg.VAL_FRACTION))
    train_raw, id_val_raw = data[:split_idx], data[split_idx:]
    od_val_raw = od_val_data

    print(f"  - Training examples: {len(train_raw)}")
    print(f"  - In-distribution validation examples: {len(id_val_raw)}")
    print(f"  - Out-of-distribution validation examples: {len(od_val_raw)}")

    # Create datasets
    print("[Data Loading] Creating HuggingFace datasets...")
    train_ds = Dataset.from_list([to_text(x) for x in train_raw])
    id_val_ds = Dataset.from_list([to_text(x) for x in id_val_raw])
    id_dataset = DatasetDict({"train": train_ds, "validation": id_val_ds})

    od_val_ds = Dataset.from_list([to_text(x) for x in od_val_raw])
    od_dataset = DatasetDict({"train": train_ds, "validation": od_val_ds})

    # Set output directory - use predefined directory from config if available
    if cfg.MODEL_ID in cfg.OUTPUT_DIRS:
        output_dir = f"{cfg.BASE_OUTPUT_DIR}/{cfg.OUTPUT_DIRS[cfg.MODEL_ID]}"
    else:
        model_name = cfg.MODEL_ID.split('/')[-1]
        output_dir = f"{cfg.BASE_OUTPUT_DIR}/{model_name}_evaluation"

    print(f"\n[Configuration] Starting evaluation pipeline...")
    print(f"  - Model ID: {cfg.MODEL_ID}")
    print(f"  - Output directory: {output_dir}")
    print(f"  - LoRA rank: {cfg.LORA_R}")
    print(f"  - LoRA alpha: {cfg.LORA_ALPHA}")
    print(f"  - Learning rate: {cfg.LR}")
    print(f"  - Epochs: {cfg.EPOCHS}")
    print(f"  - Batch size: {cfg.BATCH_SIZE}")

    # Run complete evaluation pipeline
    results = evaluate_model_pipeline(
        model_id=cfg.MODEL_ID,
        id_dataset=id_dataset,
        od_dataset=od_dataset,
        output_dir=output_dir,
        lora_r=cfg.LORA_R,
        lora_alpha=cfg.LORA_ALPHA,
        lora_dropout=cfg.LORA_DROPOUT,
        target_modules=cfg.TARGET_MODULES,
        batch_size=cfg.BATCH_SIZE,
        grad_accum=cfg.GRAD_ACCUM,
        learning_rate=cfg.LR,
        epochs=cfg.EPOCHS,
        max_seq_len=cfg.MAX_SEQ_LEN,
        log_steps=cfg.LOG_STEPS,
        save_steps=cfg.SAVE_STEPS,
        hf_token=HF_TOKEN,
        skip_finetuning=False
    )

    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nResults saved to: {results['output_dir']}")
    print(f"Adapters saved to: {results['adapter_dir']}")

    # Print final metrics summary
    print("\n" + "=" * 80)
    print("FINAL METRICS SUMMARY")
    print("=" * 80)

    print("\nIn-Distribution:")
    print("  Baseline:")
    for k, v in results["id_baseline_metrics"].items():
        print(f"    {k}: {v:.4f}")
    print("  Fine-tuned:")
    for k, v in results["id_finetuned_metrics"].items():
        print(f"    {k}: {v:.4f}")

    if results["od_baseline_metrics"]:
        print("\nOut-of-Distribution:")
        print("  Baseline:")
        for k, v in results["od_baseline_metrics"].items():
            print(f"    {k}: {v:.4f}")
        print("  Fine-tuned:")
        for k, v in results["od_finetuned_metrics"].items():
            print(f"    {k}: {v:.4f}")
