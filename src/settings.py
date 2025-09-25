from pathlib import Path
import src.config as cfg


ROOT_DIR = cfg.ROOT_DIR

class Settings:
    class paths:
        LOGS_DIR = ROOT_DIR / "logs"
        RAW_DATA_PATH = ROOT_DIR / 'cache' / 'raw_data'
        OUTPUT_DIR = ROOT_DIR / "cache" / "output"  
        MODELS_DIR = ROOT_DIR / "cache" / "models"  
    class loggers:
        DAILY = "daily"
