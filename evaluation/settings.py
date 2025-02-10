import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# General settings
SHOW_COMPLETION = False
SYSTEM_PROMPT_PATH = "evaluation/system_prompt.txt"
DATA_PATH = "data_v2/nestful_data.jsonl"

# Provider settings
LM_STUDIO_URL = "http://localhost:1234/v1"
OLLAMA_URL = "http://localhost:11434/v1"
VLLM_URL = "http://localhost:8000/v1"
OPENROUTER_URL = "https://openrouter.ai/api/v1"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

PROVIDER_URLS = {
    "lm_studio": (LM_STUDIO_URL, "api_key"),
    "ollama": (OLLAMA_URL, "api_key"),
    "vllm": (VLLM_URL, "api_key"),
    "openrouter": (OPENROUTER_URL, OPENROUTER_API_KEY),
}