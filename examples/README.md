# DeepInfer Examples

This directory contains examples demonstrating various features of DeepInfer.

## Basic Examples

### 1. Basic Inference (`basic_inference.py`)

Simple text generation with automatic GPU detection.

```bash
python examples/basic_inference.py
```

### 2. NVIDIA 5090 Inference (`nvidia_5090_inference.py`)

Demonstrates RTX 5090 optimizations with batch processing.

```bash
python examples/nvidia_5090_inference.py
```

### 3. Custom Configuration (`custom_config.py`)

Shows how to create and use custom configurations.

```bash
python examples/custom_config.py
```

### 4. API Server (`api_server.py`)

Client examples for the REST API.

First, start the server:
```bash
deepinfer serve --model meta-llama/Llama-3.1-8B-Instruct
```

Then run the client:
```bash
python examples/api_server.py
```

## Deployment Examples

### NVIDIA 5090 Deployment (`deployment_5090.md`)

Complete deployment guide for RTX 5090 systems.

## Running Examples

### Prerequisites

1. Install DeepInfer:
```bash
pip install -e .
```

2. Ensure you have access to a model (models will be downloaded automatically on first run)

### Notes

- Models are cached in `~/.cache/huggingface/`
- First run will download the model (may take time)
- GPU is required for most examples (some can run on CPU)
- Examples use `meta-llama/Llama-3.1-8B-Instruct` by default

## Modifying Examples

Feel free to modify these examples:
- Change the model name
- Adjust sampling parameters
- Try different prompts
- Experiment with batch sizes

## Need Help?

- Check the [Documentation](../docs/)
- Review [Configuration Guide](../docs/configuration.md)
- See [API Reference](../docs/api.md)
