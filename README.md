# DeepInfer

A high-performance inference engine based on vLLM, optimized for NVIDIA GPUs including the latest RTX 5090.

## Features

- 🚀 Built on vLLM for state-of-the-art inference performance
- 🎮 Native support for NVIDIA 5090 GPU with automatic detection
- 📊 Efficient memory management with PagedAttention
- 🔧 Easy configuration and deployment
- 🌐 RESTful API compatible with OpenAI format
- 📈 Real-time monitoring and metrics

## Requirements

### Hardware
- NVIDIA GPU (GTX 1080 or newer)
- Optimized for RTX 4090/5090
- Minimum 8GB GPU VRAM (16GB+ recommended)

### Software
- Python 3.9+
- CUDA 12.1+ (for RTX 5090)
- Linux OS (recommended)

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/lipish/deepinfer.git
cd deepinfer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install vLLM (with CUDA 12.1 for RTX 5090):
```bash
pip install vllm>=0.6.0
```

### Basic Usage

1. Start the inference server:
```bash
python -m deepinfer.server --model meta-llama/Llama-3.1-8B-Instruct
```

2. Make an inference request:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 100
  }'
```

### NVIDIA 5090 Optimization

For RTX 5090, use the optimized configuration:
```bash
python -m deepinfer.server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --config configs/nvidia_5090.yaml
```

## Configuration

See the `configs/` directory for example configurations:
- `default.yaml`: Default configuration
- `nvidia_5090.yaml`: Optimized for RTX 5090
- `nvidia_4090.yaml`: Optimized for RTX 4090

## Documentation

- [Installation Guide](docs/installation.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
- [GPU Optimization](docs/gpu_optimization.md)

## Architecture

DeepInfer is built on vLLM and provides:
- Automatic GPU detection and configuration
- Model loading and caching
- Request batching and scheduling
- Streaming and non-streaming inference
- Multi-GPU support (tensor parallelism)

## Examples

See the `examples/` directory for various usage examples:
- Basic inference
- Streaming responses
- Batch processing
- Custom sampling parameters
- Multi-GPU deployment

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
