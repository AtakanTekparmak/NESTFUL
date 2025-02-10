import pytest
import os
from typing import Dict, Any

from evaluation.utils import load_function_map, load_module_from_file, resolve_function
from evaluation.settings import FUNCTION_MAP_PATH, EXECUTABLE_FUNCTIONS_DIR, BASIC_FUNCTIONS_PATH

def test_load_function_map():
    """Test loading the function map file"""
    func_map = load_function_map()
    assert isinstance(func_map, dict)
    assert len(func_map) > 0
    # We don't test for specific functions here as the map content may vary

def test_load_function_map_invalid_path():
    """Test loading function map with invalid path"""
    with pytest.raises(FileNotFoundError):
        load_function_map("invalid/path/to/map.json")

def test_load_basic_functions():
    """Test loading basic arithmetic functions module"""
    module = load_module_from_file(BASIC_FUNCTIONS_PATH)
    assert module is not None
    # Test if basic arithmetic functions exist
    assert hasattr(module, "add")
    assert hasattr(module, "subtract")
    assert hasattr(module, "multiply")
    assert hasattr(module, "divide")
    
    # Test basic arithmetic operations
    assert module.add(2, 3) == 5
    assert module.subtract(5, 3) == 2
    assert module.multiply(2, 3) == 6
    assert module.divide(6, 2) == 3

def test_load_module_from_file_invalid_path():
    """Test loading module with invalid path"""
    with pytest.raises(FileNotFoundError):
        load_module_from_file("invalid/path/to/module.py")

def test_resolve_function():
    """Test resolving functions from the map"""
    func_map = load_function_map()
    
    # Get a function that exists in the map
    # We'll get the first function from the map for testing
    test_func_name = next(iter(func_map))
    func = resolve_function(test_func_name, func_map)
    assert callable(func)

def test_resolve_function_unknown():
    """Test resolving unknown function"""
    func_map = load_function_map()
    with pytest.raises(ValueError, match="Unknown function"):
        resolve_function("nonexistent_function", func_map)

def test_resolve_function_missing_in_module():
    """Test resolving a function that exists in map but not in module"""
    # Create a temporary function map with invalid function
    invalid_map = {"invalid_func": "basic_functions.py"}
    with pytest.raises(AttributeError):
        resolve_function("invalid_func", invalid_map) 