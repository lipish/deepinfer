"""
API Server Example

This example demonstrates how to start the DeepInfer API server
and make requests to it using the requests library.
"""

import requests
import json
import time

# Server URL
BASE_URL = "http://localhost:8000"


def check_health():
    """Check server health."""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())


def get_gpu_info():
    """Get GPU information."""
    response = requests.get(f"{BASE_URL}/v1/gpu/info")
    print("\nGPU Info:", json.dumps(response.json(), indent=2))


def completion_example():
    """Text completion example."""
    request_data = {
        "prompt": "The future of artificial intelligence is",
        "max_tokens": 100,
        "temperature": 0.7,
        "top_p": 0.95,
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/completions",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    result = response.json()
    print("\nCompletion Response:")
    print(f"Generated: {result['choices'][0]['text']}")
    print(f"Usage: {result['usage']}")


def chat_completion_example():
    """Chat completion example."""
    request_data = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the speed of light?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
    
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json=request_data,
        headers={"Content-Type": "application/json"}
    )
    
    result = response.json()
    print("\nChat Completion Response:")
    print(f"Assistant: {result['choices'][0]['message']['content']}")
    print(f"Usage: {result['usage']}")


if __name__ == "__main__":
    print("DeepInfer API Server Example")
    print("="*60)
    
    # Wait for server to start
    print("\nWaiting for server to start...")
    time.sleep(2)
    
    try:
        # Run examples
        check_health()
        get_gpu_info()
        completion_example()
        chat_completion_example()
        
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to server.")
        print("Please start the server first:")
        print("  python -m deepinfer.server --model meta-llama/Llama-3.1-8B-Instruct")
