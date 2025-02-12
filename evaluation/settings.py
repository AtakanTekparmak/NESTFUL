import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# General settings
SHOW_COMPLETION = False
SAVE_RESULTS = False
SYSTEM_PROMPT_PATH = "evaluation/system_prompt.txt"
DATA_PATH = "data_v2/toy_data.jsonl"
TOY_DATA_PATH = "data_v2/toy_data.jsonl"
BATCH_SIZE = 64
ROW_TIMEOUT = 30  # Timeout in seconds for each row evaluation

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

# Function loading settings
EXECUTABLE_FUNCTIONS_DIR = "data_v2/executable_functions"
FUNCTION_MAP_PATH = os.path.join(EXECUTABLE_FUNCTIONS_DIR, "func_file_map.json")
BASIC_FUNCTIONS_PATH = os.path.join(EXECUTABLE_FUNCTIONS_DIR, "basic_functions.py")