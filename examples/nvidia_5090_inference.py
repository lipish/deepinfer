"""
NVIDIA RTX 5090 Inference Example

This example demonstrates how to use DeepInfer with NVIDIA RTX 5090 optimizations.
"""

from deepinfer import InferenceEngine, Config

# Load RTX 5090 optimized configuration
config = Config.from_yaml("configs/nvidia_5090.yaml")

# Initialize engine (will auto-detect RTX 5090 and apply optimizations)
engine = InferenceEngine(config=config)

# Example prompts
prompts = [
    "Explain quantum computing in simple terms.",
    "Write a short poem about artificial intelligence.",
    "What are the main differences between Python and JavaScript?",
]

# Batch generation (RTX 5090 can handle multiple sequences efficiently)
outputs = engine.generate(
    prompts=prompts,
    max_tokens=150,
    temperature=0.8,
    top_p=0.95,
)

# Print results
for i, output in enumerate(outputs):
    print(f"\n{'='*60}")
    print(f"Prompt {i+1}: {prompts[i]}")
    print(f"{'='*60}")
    print(f"Generated: {output.outputs[0].text}")
    print(f"Tokens: {len(output.outputs[0].token_ids)}")
