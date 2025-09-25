
import os
import json
import pandas as pd
import warnings
import random
import gc
from typing import Dict, List

import torch
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

import src.config as cfg
def build_inference_prompt(obs: str, 
                           q: str) -> str:
    return f"Training Observations:\n{obs.strip()}\n\nQuestion:\n{q.strip()}\n\nAnswer:\n"



# Generation function - GPU optimized
@torch.no_grad()
def generate_answers_with(model, 
                          tokenizer: AutoTokenizer,
                          obs_list: list[str], 
                          q_list: list[str]) -> list[str]:
    model.eval()  # Ensure model is in eval mode
    prompts = [build_inference_prompt(o, q) for o, q in zip(obs_list, q_list)]

    # Set tokenizer to left padding for generation
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    # Process in batches for GPU
    batch_size = 4  # Larger batch size for GPU
    all_outputs = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.MAX_SEQ_LEN
        ).to(model.device)  # Move to GPU device

        gen = model.generate(
            **inputs,
            max_new_tokens=cfg.GEN_MAX_NEW_TOKENS,
            do_sample=False,  # Deterministic generation
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )

        texts = tokenizer.batch_decode(gen, skip_special_tokens=True)

        # Extract answers
        for prompt, full in zip(batch_prompts, texts):
            if full.startswith(prompt):
                answer = full[len(prompt):].strip()
            else:
                # Fallback: split on "Answer:"
                parts = full.split("Answer:")
                answer = parts[-1].strip() if len(parts) > 1 else full.strip()

            # Clean up answer (remove extra newlines, etc.)
            answer = answer.split('\n')[0].strip()  # Take first line only
            all_outputs.append(answer)

    # Reset tokenizer padding
    tokenizer.padding_side = original_padding_side
    return all_outputs


class PromptAnswerCollator:
    def __init__(self, tokenizer:AutoTokenizer, max_seq_len:int=512):
        self.tok = tokenizer
        self.max_len = max_seq_len

    def __call__(self, features: List[Dict]):
        prompts = [f["prompt"] for f in features]
        answers = [f["response"] for f in features]

        # Tokenize separately
        enc_p = self.tok(prompts, add_special_tokens=False)
        enc_a = self.tok(answers, add_special_tokens=False)

        input_ids, labels, attn = [], [], []
        for p_ids, a_ids in zip(enc_p["input_ids"], enc_a["input_ids"]):
            # Concatenate prompt + answer
            ids = p_ids + a_ids
            if len(ids) > self.max_len:
                # Truncate from the left, keeping the answer
                ids = ids[-self.max_len:]
                cut = max(0, len(p_ids) - (len(p_ids) + len(a_ids) - self.max_len))
                p_len = max(0, len(p_ids) - cut)
            else:
                p_len = len(p_ids)

            # Create labels: mask prompt (-100), supervise answer
            lab = [-100] * p_len + ids[p_len:]
            am = [1] * len(ids)

            input_ids.append(ids)
            labels.append(lab)
            attn.append(am)

        # Pad to batch max length
        pad_id = self.tok.pad_token_id or self.tok.eos_token_id
        maxlen = max(len(x) for x in input_ids) if input_ids else 1

        def pad(seq, fill):
            return seq + [fill] * (maxlen - len(seq))

        input_ids = torch.tensor([pad(x, pad_id) for x in input_ids], dtype=torch.long)
        labels = torch.tensor([pad(x, -100) for x in labels], dtype=torch.long)
        attn = torch.tensor([pad(x, 0) for x in attn], dtype=torch.long)

        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
