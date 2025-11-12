#!/usr/bin/env python3
"""
Constants and file paths for the project.
Centralizes all file paths to avoid inconsistencies.
"""

from pathlib import Path


# Directory paths
DATA_DIR = Path("data")
CHECKPOINT_DIR = Path("checkpoints")
LOG_DIR = Path("logs")

# Data files
RAW_DATA = DATA_DIR / "raw_wikipedia.txt"
TRAIN_TEXT = DATA_DIR / "train.txt"
VAL_TEXT = DATA_DIR / "val.txt"

# Tokenization files
VOCAB_FILE = DATA_DIR / "vocab.json"
TRAIN_TOKENS = DATA_DIR / "train_tokens.npy"
VAL_TOKENS = DATA_DIR / "val_tokens.npy"

# Model checkpoints
BEST_MODEL = CHECKPOINT_DIR / "model_best.pt"

# Logs
TRAINING_LOG = LOG_DIR / "training_log.csv"


def ensure_directories():
    """Create necessary directories if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)