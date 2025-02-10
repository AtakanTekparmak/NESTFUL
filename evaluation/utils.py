from typing import List, Dict, Any
import json
import importlib.util
import os
import logging

from evaluation.schemas import NestfulRow, ToolCall, Tool
from evaluation.settings import (
    SYSTEM_PROMPT_PATH, 
    DATA_PATH, 
    FUNCTION_MAP_PATH,
    EXECUTABLE_FUNCTIONS_DIR
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def load_function_map(map_path: str = FUNCTION_MAP_PATH) -> Dict[str, str]:
    """
    Load the function file mapping from the JSON file.

    Args:
        map_path: Path to the function map JSON file.

    Returns:
        A dictionary mapping function names to their file paths.
    """
    try:
        with open(map_path, 'r') as f:
            func_map = json.load(f)
            logger.info(f"Loaded function map with {len(func_map)} functions")
            return func_map
    except FileNotFoundError:
        raise FileNotFoundError(f"Function map file not found at: {map_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in function map file at: {map_path}")
    except IOError:
        raise IOError(f"Error reading function map file at: {map_path}")

def load_module_from_file(file_path: str) -> Any:
    """
    Dynamically load a Python module from a file path.

    Args:
        file_path: Path to the Python file to load.

    Returns:
        The loaded module.
    """
    try:
        spec = importlib.util.spec_from_file_location("module", file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {file_path}")
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"Error loading module from {file_path}: {str(e)}")
        raise

def resolve_function(func_name: str, func_map: Dict[str, str]) -> Any:
    """
    Resolve a function by its name using the function map.

    Args:
        func_name: Name of the function to resolve.
        func_map: Dictionary mapping function names to file paths.

    Returns:
        The resolved function object.
    """
    try:
        if func_name not in func_map:
            raise ValueError(f"Unknown function: {func_name}")
        
        func_file = func_map[func_name]
        module_path = os.path.join(EXECUTABLE_FUNCTIONS_DIR, func_file)
        
        module = load_module_from_file(module_path)
        if not hasattr(module, func_name):
            raise AttributeError(f"Function {func_name} not found in module {module_path}")
        
        return getattr(module, func_name)
    except Exception as e:
        logger.error(f"Error resolving function {func_name}: {str(e)}")
        raise