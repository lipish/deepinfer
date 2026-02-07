"""
Custom Configuration Example

This example shows how to create and use custom configurations.
"""

from deepinfer import InferenceEngine, Config
from deepinfer.gpu_utils import GPUDetector

# Detect GPUs
gpu_detector = GPUDetector()
gpu_detector.print_gpu_info()

# Create custom configuration
config = Config()

# Model settings
config.model.name = "meta-llama/Llama-3.1-8B-Instruct"
config.model.dtype = "float16"  # Use FP16 for faster inference

# GPU settings
if gpu_detector.has_nvidia_5090():
    print("\nUsing NVIDIA RTX 5090 optimizations")
    config.gpu.gpu_memory_utilization = 0.95
    config.gpu.max_num_seqs = 512
    config.gpu.nvidia_5090_optimizations = True
    config.gpu.enable_chunked_prefill = True
    config.gpu.enable_prefix_caching = True
else:
    # Use conservative settings for other GPUs
    config.gpu.gpu_memory_utilization = 0.90
    config.gpu.max_num_seqs = 256

# Sampling settings
config.sampling.temperature = 0.8
config.sampling.top_p = 0.95
config.sampling.max_tokens = 256

# Save configuration for future use
config.to_yaml("my_config.yaml")
print("\nConfiguration saved to my_config.yaml")

# Initialize engine with custom config
engine = InferenceEngine(config=config)

# Test generation
prompt = "Tell me a fun fact about space exploration."
outputs = engine.generate(prompts=prompt)

print(f"\nPrompt: {prompt}")
print(f"Generated: {outputs[0].outputs[0].text}")
