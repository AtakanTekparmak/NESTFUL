import pytest
from pathlib import Path
import json

from evaluation.utils import load_data, load_system_prompt
from evaluation.schemas import NestfulRow, Tool
from evaluation.settings import DATA_PATH, SYSTEM_PROMPT_PATH

def test_load_data_exists():
    """Test that the data file exists at the expected location"""
    assert Path(DATA_PATH).exists(), f"Data file not found at {DATA_PATH}"

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
    data = load_data()
    for row in data:
        for tool in row.tools:
            assert isinstance(tool, Tool), "Tool is not a Tool instance"
            assert tool.name, "Tool missing name"
            assert tool.description, "Tool missing description"
            assert isinstance(tool.parameters, dict), "Tool parameters is not a dict"
            # output_parameter is optional, so only check type if it exists
            if tool.output_parameter is not None:
                assert isinstance(tool.output_parameter, dict), "Tool output_parameter is not a dict"

def test_load_data_invalid_path():
    """Test that loading data from invalid path raises appropriate error"""
    with pytest.raises(FileNotFoundError):
        load_data("nonexistent_file.jsonl")

def test_system_prompt_loading():
    """Test that system prompt can be loaded and formatted with tools"""
    # First get some tools from the data
    data = load_data()
    tools = data[0].tools
    
    # Test loading and formatting system prompt
    prompt = load_system_prompt(tools)
    assert prompt, "System prompt is empty"
    assert "{{tools}}" not in prompt, "Tools placeholder not replaced"
    
    # Verify tools are actually in the prompt
    for tool in tools:
        assert tool.name in prompt, f"Tool {tool.name} not found in formatted prompt" 