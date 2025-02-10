from openai import OpenAI
import asyncio

from evaluation.settings import PROVIDER_URLS

# Initialize OpenAI clients for each provider
CLIENTS = {}
for provider, (url, api_key) in PROVIDER_URLS.items():
    CLIENTS[provider] = OpenAI(api_key=api_key, base_url=url)


def get_completion(
    model_name: str, provider: str, system_prompt: str, user_query: str
) -> str:
    """
    Get a completion from a model for a given provider.

    Args:
        model_name: The name of the model to use
        provider: The provider to use
        system_prompt: The system prompt to use
        user_query: The user query to use

    Returns:
        The completion from the model
    """
    client = CLIENTS.get(provider)
    if not client:
        raise ValueError(f"Provider '{provider}' not recognized.")

    # Create the messages for the chat completion
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ]

    # Make the API call to get the completion
    response = client.chat.completions.create(
        model=model_name, messages=messages, temperature=0.0
    )

    # Extract and return the assistant's reply
    return response.choices[0].message.content


async def get_completions_batch(request):
    """
    Get completions for a batch of requests.
    
    Args:
        request: A dictionary containing:
            - model: The model name
            - provider: The provider name
            - system_prompts: List of system prompts
            - user_prompts: List of user prompts
    
    Returns:
        List of completions
    """
    loop = asyncio.get_running_loop()
    tasks = []
    
    for system_prompt, user_prompt in zip(request["system_prompts"], request["user_prompts"]):
        tasks.append(
            loop.run_in_executor(
                None,
                get_completion,
                request["model"],
                request["provider"],
                system_prompt,
                user_prompt,
            )
        )
    return await asyncio.gather(*tasks)