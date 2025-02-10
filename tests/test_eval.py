import pytest
import json
from typing import List, Dict

from evaluation.eval import NESTFULEvaluator
from evaluation.schemas import NestfulRow

@pytest.fixture
def toy_data() -> List[NestfulRow]:
    """Load toy data for testing"""
    with open("data_v2/toy_data.jsonl", "r") as f:
        return [NestfulRow(**json.loads(line)) for line in f]

@pytest.fixture
def evaluator():
    """Create an evaluator instance for testing"""
    return NESTFULEvaluator(
        model_name="test-model",
        provider="test-provider",
        debug=True
    )

def test_parse_function_sequence(evaluator):
    """Test parsing of function sequence from model output"""
    # Test valid sequence
    valid_output = '''Here's the sequence of functions:
    [
        {"name": "divide", "label": "$var_1", "arguments": {"arg_0": 20, "arg_1": 100}},
        {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": 10000, "arg_1": "$var_1.result$"}}
    ]'''
    sequence = evaluator.parse_function_sequence(valid_output)
    assert len(sequence) == 2
    assert sequence[0]["name"] == "divide"
    assert sequence[1]["name"] == "multiply"

    # Test invalid sequence
    invalid_output = "No JSON here"
    sequence = evaluator.parse_function_sequence(invalid_output)
    assert sequence == []

def test_calculate_f1(evaluator):
    """Test F1 score calculation"""
    pred = ["divide", "multiply"]
    gold = ["divide", "multiply", "add"]
    f1 = evaluator._calculate_f1(pred, gold)
    assert 0 <= f1 <= 1

    # Test empty lists
    assert evaluator._calculate_f1([], []) == 0
    assert evaluator._calculate_f1(pred, []) == 0
    assert evaluator._calculate_f1([], gold) == 0

def test_calculate_metrics(evaluator):
    """Test metrics calculation"""
    predicted = [
        {"name": "divide", "label": "$var_1", "arguments": {"arg_0": 20, "arg_1": 100}},
        {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": 10000, "arg_1": "$var_1.result$"}}
    ]
    gold = [
        {"name": "divide", "label": "$var_1", "arguments": {"arg_0": 20, "arg_1": 100}},
        {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": 10000, "arg_1": "$var_1.result$"}}
    ]
    
    metrics = evaluator.calculate_metrics(predicted, gold, 2000.0, 2000.0)
    assert all(0 <= v <= 1 for v in metrics.values())
    assert metrics["full_match"] == 1.0
    assert metrics["win_rate"] == 1.0

def test_evaluate_sample(evaluator, toy_data):
    """Test evaluation of a single sample"""
    # Mock the get_completion function to return a known response
    def mock_get_completion(*args, **kwargs):
        return '''[
            {"name": "divide", "label": "$var_1", "arguments": {"arg_0": 20, "arg_1": 100}},
            {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": 10000, "arg_1": "$var_1.result$"}}
        ]'''
    
    # Replace the real get_completion with our mock
    import evaluation.model
    original_get_completion = evaluation.model.get_completion
    evaluation.model.get_completion = mock_get_completion
    
    try:
        # Test with first sample from toy data
        metrics = evaluator.evaluate_sample(toy_data[0])
        assert all(isinstance(v, float) for v in metrics.values())
        assert all(0 <= v <= 1 for v in metrics.values())
    finally:
        # Restore the original function
        evaluation.model.get_completion = original_get_completion

def test_evaluate(evaluator, toy_data):
    """Test evaluation of multiple samples"""
    # Mock the get_completion function
    def mock_get_completion(*args, **kwargs):
        return '''[
            {"name": "divide", "label": "$var_1", "arguments": {"arg_0": 20, "arg_1": 100}},
            {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": 10000, "arg_1": "$var_1.result$"}}
        ]'''
    
    # Replace the real get_completion with our mock
    import evaluation.model
    original_get_completion = evaluation.model.get_completion
    evaluation.model.get_completion = mock_get_completion
    
    try:
        # Test full evaluation
        metrics = evaluator.evaluate(batch_size=2)
        assert all(isinstance(v, float) for v in metrics.values())
        assert all(0 <= v <= 1 for v in metrics.values())
    finally:
        # Restore the original function
        evaluation.model.get_completion = original_get_completion 