import pytest
from evaluation.model import get_completion

def test_openrouter_gemini_completion():
    """Test completion with OpenRouter using google/gemini-2.0-flash-001."""
    # Test parameters
    model_name = "google/gemini-2.0-flash-001"
    provider = "openrouter"
    system_prompt = "You are a helpful AI assistant."
    user_query = "What is 2+2?"
    
    # Get completion
    response = get_completion(
        model_name=model_name,
        provider=provider,
        system_prompt=system_prompt,
        user_query=user_query
    )
    
    # Basic assertions
    assert isinstance(response, str)
    assert len(response) > 0
    assert "4" in response.lower()  # The response should contain "4" since 2+2=4 