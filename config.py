import os

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