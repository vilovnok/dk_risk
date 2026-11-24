import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "evaluated"
RUNS_DIR = OUTPUT_DIR / "runs"

DATABASE_PATH = OUTPUT_DIR / "experiments.db"

ENDPOINTS = {
    "direct": "https://tagme.sberdevices.ru/ai/llm/v1/chat/completions",
    "aihf": "https://tagme.sberdevices.ru/x/aihf/llm/v1/chat/completions",
}

LLM_API_KEY = os.getenv("LLM_API_KEY", "")

DEFAULT_MODEL = "DeepSeek-V4.1-Flash"

TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 8000

TIMEOUT = (60, 180)

MAX_RETRIES = 3
RETRY_DELAYS = (2, 4, 8)

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DEVICE = "auto"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_MAX_LENGTH = 8192

MENTION_THRESHOLD = 0.80

DEFAULT_WORKERS = 8