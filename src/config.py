import os
from pathlib import Path
import datetime

from dotenv import load_dotenv

try:
    load_dotenv()
except Exception as e:
    print(f"Error loading .env file : {e}")

# Env setup configs


HUGGINGFACE_HUB_TOKEN =  os.getenv("HUGGINGFACE_HUB_TOKEN")





DEBUG = str(os.getenv("DEBUG")).lower() in ['true']

DEBUG_ROOT_DIR = Path(os.getenv("DEBUG_ROOT_DIR"))
PROD_ROOT_DIR = Path(os.getenv("PROD_ROOT_DIR"))




if DEBUG:
    ROOT_DIR = DEBUG_ROOT_DIR
else:
    ROOT_DIR = PROD_ROOT_DIR

# MODEL_ID_LIST = ["Qwen/Qwen2-0.5B-Instruct" ,
#                 "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#                 "Qwen/Qwen2.5-1.5B-Instruct",
#                 "google/gemma-2-2b-it",
#                 "microsoft/Phi-3-mini-4k-instruct",
#                 "google/gemma-3-4b-it" ,
#                 "deepseek-ai/deepseek-llm-7b-base" ,
#                 "allenai/OLMo-7B-0424-hf" ,
#                 "google/gemma-7b-it" ,
#                 "Qwen/Qwen3-8B" ,
#                 "unsloth/Meta-Llama-3.1-8B" ,
#                 "swiss-ai/Apertus-8B-2509",
#                 "01-ai/Yi-9B" ,
#                 "google/gemma-2-9b" ]


MODEL_ID_LIST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "google/gemma-2-2b-it",
    "deepseek-ai/deepseek-llm-7b-base",
    "allenai/OLMo-7B-0424-hf",
    "google/gemma-7b-it",
    "unsloth/Meta-Llama-3.1-8B",
    "swiss-ai/Apertus-8B-2509",
    "google/gemma-2-9b",
    "01-ai/Yi-9B"
]

# Output directories for each model (under cache/models/)
OUTPUT_DIRS = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "tinyllama_1.1b_chat",
    "Qwen/Qwen2.5-1.5B-Instruct": "qwen2.5_1.5b_instruct",
    "google/gemma-2-2b-it": "gemma2_2b_it",
    "deepseek-ai/deepseek-llm-7b-base": "deepseek_7b_base",
    "allenai/OLMo-7B-0424-hf": "olmo_7b",
    "google/gemma-7b-it": "gemma_7b_it",
    "unsloth/Meta-Llama-3.1-8B": "llama3.1_8b",
    "swiss-ai/Apertus-8B-2509": "apertus_8b",
    "google/gemma-2-9b": "gemma2_9b",
    "01-ai/Yi-9B": "yi_9b"
}

# Base directory for model outputs
BASE_OUTPUT_DIR = "cache/models"

# Configuration - GPU Optimized
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "qwen_500m_ft_inductive_slm"  # Legacy single model output
SEED = 42
VAL_FRACTION = 0.30
MAX_SEQ_LEN = 512  # Increased for GPU performance

# LoRA Configuration - Larger for GPU
LORA_R = 16  # Increased from 8
LORA_ALPHA = 32  # Increased from 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training Configuration - GPU Optimized
LR = 2e-4
EPOCHS = 3
BATCH_SIZE = 8  # Larger batch size for GPU
GRAD_ACCUM = 2  # Reduced since we have larger batch size
LOG_STEPS = 10
EVAL_STEPS = 50
SAVE_STEPS = 50
GEN_MAX_NEW_TOKENS = 64


