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




# Configuration - GPU Optimized
MODEL_ID = "Qwen/Qwen2-0.5B-Instruct" 
OUTPUT_DIR = "swiss_apertus_8b_ft_inductive_smalllm_compare"
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


