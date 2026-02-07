"""
Basic Inference Example

This example demonstrates how to use DeepInfer for simple text generation.
"""

from deepinfer import InferenceEngine, Config

# Option 1: Use default configuration with auto GPU detection
engine = InferenceEngine(model_name="meta-llama/Llama-3.1-8B-Instruct")

# Generate text
prompt = "What is the capital of France?"
outputs = engine.generate(
    prompts=prompt,
    max_tokens=100,
    temperature=0.7,
)

# Print results
print(f"Prompt: {prompt}")
print(f"Generated: {outputs[0].outputs[0].text}")
print(f"Tokens: {len(outputs[0].outputs[0].token_ids)}")
