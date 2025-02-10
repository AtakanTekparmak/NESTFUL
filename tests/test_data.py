import pytest
from pathlib import Path
import json

from evaluation.utils import load_data, load_system_prompt
from evaluation.schemas import NestfulRow, Tool
from evaluation.settings import DATA_PATH, SYSTEM_PROMPT_PATH, TOY_DATA_PATH

def test_load_data_exists():
    """Test that the data files exist at the expected locations"""
    assert Path(DATA_PATH).exists(), f"Data file not found at {DATA_PATH}"
    assert Path(TOY_DATA_PATH).exists(), f"Toy data file not found at {TOY_DATA_PATH}"

def test_load_toy_data_content():
    """Test that toy data contains expected content and structure"""
    data = load_data(TOY_DATA_PATH)
    assert len(data) == 3, "Toy data should contain exactly 3 samples"
    
    # Test first sample (population decrease problem)
    first_sample = data[0]
    assert first_sample.sample_id == "a5ae1249-7f33-4d40-8ad1-914a0623e3bd"
    assert "population of a town is 10000" in first_sample.input
    assert len(first_sample.output) == 9  # Should have 9 function calls
    assert first_sample.gold_answer == 6400.0
    
    # Test second sample (URL content problem)
    second_sample = data[1]
    assert second_sample.sample_id == "fa57c28a-e9d8-44fb-a358-a9404a557d5b"
    assert "https://www.example.com" in second_sample.input
    assert len(second_sample.output) == 2  # Should have 2 function calls
    assert second_sample.gold_answer == [-1, 0, 0]
    
    # Test third sample (arithmetic mean problem)
    third_sample = data[2]
    assert third_sample.sample_id == "4c457bfb-c6e5-48ce-890d-b6ac4b2f307b"
    assert "21 different numbers" in third_sample.input
    assert len(third_sample.output) == 8  # Should have 8 function calls
    assert abs(third_sample.gold_answer - 0.16666666666666666) < 1e-10

def test_load_data_is_jsonl():
    """Test that the data file is in valid JSONL format"""
    with open(DATA_PATH, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0, "Data file is empty"
        # Verify each line is valid JSON and matches our schema
        for i, line in enumerate(lines, 1):
            try:
                data = json.loads(line)
                NestfulRow(**data)  # This will raise ValidationError if schema doesn't match
            except json.JSONDecodeError:
                pytest.fail(f"Invalid JSON on line {i}")
            except Exception as e:
                pytest.fail(f"Schema validation failed on line {i}: {str(e)}")

def test_load_data_content():
    """Test that loaded data contains expected fields and structure"""
    data = load_data()
    assert len(data) > 0, "No data was loaded"
    
    # Test first row has all required fields
    first_row = data[0]
    assert isinstance(first_row, NestfulRow), "Data row is not a NestfulRow instance"
    assert first_row.sample_id, "Missing sample_id"
    assert first_row.input, "Missing input"
    assert isinstance(first_row.output, list), "Output is not a list"
    assert isinstance(first_row.tools, list), "Tools is not a list"
    assert first_row.gold_answer is not None, "Missing gold_answer"

def test_load_data_tools_structure():
    """Test that tools in the data have correct structure"""
    data = load_data(TOY_DATA_PATH)  # Use toy data for consistent tool checking
    first_sample = data[0]
    
    # Check specific tools from toy data
    tool_names = {tool.name for tool in first_sample.tools}
    expected_tools = {"add", "subtract", "multiply", "divide"}
    assert expected_tools.issubset(tool_names), f"Missing basic tools: {expected_tools - tool_names}"
    
    # Check tool structure
    for tool in first_sample.tools:
        assert isinstance(tool, Tool), "Tool is not a Tool instance"
        assert tool.name, "Tool missing name"
        assert tool.description, "Tool missing description"
        assert isinstance(tool.parameters, dict), "Tool parameters is not a dict"
        assert "arg_0" in tool.parameters, "Tool missing arg_0 parameter"
        if tool.output_parameter:
            assert "result" in tool.output_parameter, "Tool output missing result parameter"

def test_load_data_invalid_path():
    """Test that loading data from invalid path raises appropriate error"""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.jsonl")

def test_system_prompt_loading():
    """Test that system prompt can be loaded and formatted with tools"""
    # Use toy data for consistent tool testing
    data = load_data(TOY_DATA_PATH)
    tools = data[0].tools
    
    # Test loading and formatting system prompt
    prompt = load_system_prompt(tools)
    assert prompt, "System prompt is empty"
    assert "{{tools}}" not in prompt, "Tools placeholder not replaced"
    
    # Verify basic tools are in the prompt
    basic_tools = ["add", "subtract", "multiply", "divide"]
    for tool_name in basic_tools:
        assert tool_name in prompt, f"Basic tool {tool_name} not found in formatted prompt" 