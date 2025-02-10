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
    
    # Check first two functions from the map
    assert "find_strings_with_pattern" in func_map
    assert func_map["find_strings_with_pattern"] == "py_code_file_1.py"
    assert "find_max_for_each_row" in func_map
    assert func_map["find_max_for_each_row"] == "py_code_file_2.py"

def test_load_function_map_invalid_path():
    """Test loading function map with invalid path"""
    with pytest.raises(FileNotFoundError):
        load_function_map("invalid/path/to/map.json")

def test_load_basic_functions():
    """Test loading and executing basic arithmetic functions module"""
    module = load_module_from_file(BASIC_FUNCTIONS_PATH)
    assert module is not None
    
    # Test if basic arithmetic functions exist
    assert hasattr(module, "add")
    assert hasattr(module, "subtract")
    assert hasattr(module, "multiply")
    assert hasattr(module, "divide")
    
    # Test basic arithmetic operations with integers
    assert module.add(2, 3) == 5
    assert module.subtract(5, 3) == 2
    assert module.multiply(2, 3) == 6
    assert module.divide(6, 2) == 3
    
    # Test basic arithmetic operations with floats
    assert abs(module.add(2.5, 3.7) - 6.2) < 1e-10
    assert abs(module.subtract(5.5, 3.2) - 2.3) < 1e-10
    assert abs(module.multiply(2.5, 3.0) - 7.5) < 1e-10
    assert abs(module.divide(7.5, 2.5) - 3.0) < 1e-10
    
    # Test division by zero
    with pytest.raises(ZeroDivisionError):
        module.divide(5, 0)

def test_load_first_two_map_functions():
    """Test loading and basic validation of first two functions from the map"""
    func_map = load_function_map()
    
    # Test find_strings_with_pattern function
    pattern_func = resolve_function("find_strings_with_pattern", func_map)
    assert callable(pattern_func)
    
    # Test find_max_for_each_row function
    max_row_func = resolve_function("find_max_for_each_row", func_map)
    assert callable(max_row_func)
    
    # Verify the functions are loaded from correct files
    pattern_module = load_module_from_file(os.path.join(EXECUTABLE_FUNCTIONS_DIR, "py_code_file_1.py"))
    max_row_module = load_module_from_file(os.path.join(EXECUTABLE_FUNCTIONS_DIR, "py_code_file_2.py"))
    
    assert hasattr(pattern_module, "find_strings_with_pattern")
    assert hasattr(max_row_module, "find_max_for_each_row")

def test_load_module_from_file_invalid_path():
    """Test loading module with invalid path"""
    with pytest.raises(FileNotFoundError):
        load_module_from_file("invalid/path/to/module.py")

def test_resolve_function():
    """Test resolving functions from the map"""
    func_map = load_function_map()
    
    # Test resolving first function from the map
    func = resolve_function("find_strings_with_pattern", func_map)
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