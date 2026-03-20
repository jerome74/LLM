"""
Central configuration module.

All URLs and tunable constants are read from environment variables so the
project can be run without modifying source files.  Copy `.env.example` to
`.env`, adjust values as needed, and load it before running:

    export $(cat .env | xargs)   # bash
    # or use python-dotenv in your scripts / notebooks
"""

import os

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------

# Plain-text corpus used by LLM.py for GPT-2 pre-training
GENESIS_URL: str = os.environ.get(
    "GENESIS_URL",
    "https://people.sc.fsu.edu/~jburkardt/datasets/text/genesis.txt",
)

# Instruction-tuning dataset used by instruction_file_tuning.ipynb
INSTRUCTION_DATA_URL: str = os.environ.get(
    "INSTRUCTION_DATA_URL",
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json",
)

# SMS Spam Collection used by classifier_fine_tuning.ipynb
SPAM_DATASET_URL: str = os.environ.get(
    "SPAM_DATASET_URL",
    "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip",
)

SPAM_DATASET_BACKUP_URL: str = os.environ.get(
    "SPAM_DATASET_BACKUP_URL",
    "https://f001.backblazeb2.com/file/LLMs-from-scratch/sms%2Bspam%2Bcollection.zip",
)

# ---------------------------------------------------------------------------
# GPT-2 model weights
# ---------------------------------------------------------------------------

GPT2_BASE_URL: str = os.environ.get(
    "GPT2_BASE_URL",
    "https://openaipublic.blob.core.windows.net/gpt-2/models",
)

GPT2_BACKUP_BASE_URL: str = os.environ.get(
    "GPT2_BACKUP_BASE_URL",
    "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2",
)

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------

GENESIS_FILE_PATH: str = os.environ.get("GENESIS_FILE_PATH", "genesis.txt")
MODELS_DIR: str = os.environ.get("MODELS_DIR", "gpt2")
