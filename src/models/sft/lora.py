
import os
import json
import pandas as pd
import warnings
import random
import gc
from typing import Dict, List, Tuple

import torch
from torch.nn import Module
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
    GenerationConfig
)
from peft import LoraConfig, get_peft_model
import evaluate
import torch
import torch, sys, platform
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.models.inference.inference import build_inference_prompt, generate_answers_with, PromptAnswerCollator

import src.config as cfg


def check_model_parameters(model: Module) -> Tuple[int, int]:
    """
    Debug function to check which parameters require gradients.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        tuple: (trainable_params, total_params) counts
    """
    print("Checking gradient requirements:")
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # print(f"  Trainable: {name} - shape: {param.shape}")
        # else:
            # print(f"  Frozen: {name} - shape: {param.shape}")

    print(f"Trainable: {trainable_params}, Total: {total_params}")
    return trainable_params, total_params


def finetune_model_with_lora(
    model_id: str,
    train_dataset: Dataset,
    tokenizer,
    output_dir: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: List[str] = None,
    batch_size: int = 4,
    grad_accum: int = 1,
    learning_rate: float = 2e-4,
    epochs: int = 3,
    max_seq_len: int = 512,
    log_steps: int = 10,
    save_steps: int = 500,
    hf_token: str = None
) -> Module:
    """
    Fine-tune a model using LoRA (Low-Rank Adaptation).
    
    Args:
        model_id: Hugging Face model ID to load
        train_dataset: Training dataset
        tokenizer: Tokenizer for the model
        output_dir: Directory to save the fine-tuned model
        lora_r: LoRA rank parameter
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        target_modules: List of modules to apply LoRA to
        batch_size: Training batch size
        grad_accum: Gradient accumulation steps
        learning_rate: Learning rate
        epochs: Number of training epochs
        max_seq_len: Maximum sequence length
        log_steps: Logging frequency
        save_steps: Save frequency
        hf_token: Hugging Face token (if needed)
        
    Returns:
        Module: The fine-tuned model with LoRA adapters
    """
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        token=hf_token
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)

    # Explicitly enable gradients for LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            param.requires_grad_(True)

    print("Trainable parameters:")
    model.print_trainable_parameters()

    # Training arguments - GPU optimized
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=2,  # Reduced for float32 memory usage
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        logging_steps=log_steps,
        save_steps=save_steps,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        load_best_model_at_end=False,
        gradient_checkpointing=False,  # Disable to avoid potential LoRA conflicts
        report_to="none",
        # Disable mixed precision for now to avoid LoRA compatibility issues
        dataloader_pin_memory=True,  # Enable for GPU
        dataloader_num_workers=0,  # Set to 0 to avoid potential issues
        remove_unused_columns=False,
        prediction_loss_only=True,
        eval_strategy="no",  # Disable evaluation during training for speed
    )

    # Data collator
    data_collator = PromptAnswerCollator(tokenizer, max_seq_len=max_seq_len)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train the model
    print("Starting fine-tuning...")
    trainer.train()
    
    return model


