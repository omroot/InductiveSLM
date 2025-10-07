#!/usr/bin/env python
"""
Display all model output directories from config.py
Run from project root: python show_output_dirs.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import src.config as cfg

print("=" * 80)
print("MODEL OUTPUT DIRECTORIES (from src/config.py)")
print("=" * 80)
print(f"\nBase directory: {cfg.BASE_OUTPUT_DIR}")
print("\nModel -> Output Directory Mapping:")
print("-" * 80)

for i, model_id in enumerate(cfg.MODEL_ID_LIST):
    output_name = cfg.OUTPUT_DIRS.get(model_id, "NOT DEFINED")
    full_path = f"{cfg.BASE_OUTPUT_DIR}/{output_name}"
    print(f"[{i:2d}] {model_id:<50}")
    print(f"     -> {output_name}")
    print(f"     Full path: {full_path}")
    print()

print("=" * 80)
print(f"Total models: {len(cfg.MODEL_ID_LIST)}")
print(f"Defined output dirs: {len(cfg.OUTPUT_DIRS)}")

# Check for any models without defined output directories
missing = [m for m in cfg.MODEL_ID_LIST if m not in cfg.OUTPUT_DIRS]
if missing:
    print(f"\n⚠️  WARNING: {len(missing)} model(s) without predefined output directory:")
    for m in missing:
        print(f"  - {m}")
else:
    print("\n✅ All models have predefined output directories!")
