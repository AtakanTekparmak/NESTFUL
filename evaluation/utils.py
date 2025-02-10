from typing import List
import json

from evaluation.schemas import NestfulRow, ToolCall, Tool
from evaluation.settings import SYSTEM_PROMPT_PATH, DATA_PATH

def load_system_prompt(tools: List[Tool], system_prompt_path: str = SYSTEM_PROMPT_PATH) -> str:
    """
    Load the system prompt from the file and replace the {{tools}} placeholder with the tools.

    Args:
        file_path: The path to the file containing the system prompt.
        tools: A list of tools.

    Returns:
        The system prompt with the tools.
    """
    try:
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read()
        return system_prompt.replace('{{tools}}', str(tools))
    except FileNotFoundError:
        raise FileNotFoundError(f"System prompt file not found at: {system_prompt_path}")
    except IOError:
        raise IOError(f"Error reading system prompt file at: {system_prompt_path}")
    
def load_data(data_path: str = DATA_PATH) -> List[NestfulRow]:
    """
    Load the data from the file and return a list of NestfulRow objects.

    Args:
        data_path: The path to the file containing the data.

    Returns:
        A list of NestfulRow objects.
    """
    try:
        with open(data_path, 'r') as f:
            return [NestfulRow(**json.loads(line)) for line in f]
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {data_path}")
    except IOError:
        raise IOError(f"Error reading data file at: {data_path}")