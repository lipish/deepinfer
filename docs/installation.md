# Installation Guide

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support (GTX 1080 or newer)
  - Recommended: RTX 4090 or RTX 5090
  - Minimum: 8GB VRAM (16GB+ recommended for larger models)
- **CPU**: Modern multi-core processor
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: 50GB+ free space for models

### Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended), Windows with WSL2, or macOS
- **Python**: 3.9 or higher (3.10+ recommended)
- **CUDA**: 12.1 or higher (for RTX 5090, recommended for vLLM 0.6.0+)
- **Driver**: Latest NVIDIA drivers
  - RTX 5090: Driver version 550.54.14 or higher
  - RTX 4090: Driver version 520.61.05 or higher

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/lipish/deepinfer.git
cd deepinfer
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install DeepInfer

#### Option A: Install from Source (Recommended)

```bash
pip install -e .
```

#### Option B: Install Dependencies Only

```bash
pip install -r requirements.txt
```

### 4. Install vLLM

For CUDA 12.1+ (RTX 5090):
```bash
pip install vllm>=0.6.0
```

For CUDA 11.8:
```bash
pip install vllm>=0.6.0 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 5. Verify Installation

Check GPU detection:
```bash
deepinfer gpu-info
```

Expected output:
```
============================================================
GPU Information
============================================================

GPU 0: NVIDIA GeForce RTX 5090
  Total Memory: 32.00 GB
  Free Memory:  31.50 GB
  Compute Capability: 8.9
  Driver Version: 550.54.14
  CUDA Version: 12.1
  ✓ NVIDIA RTX 5090 - Optimizations Available
============================================================
```

## NVIDIA 5090 Specific Setup

### 1. Install CUDA 12.1+

Download and install CUDA Toolkit 12.1 or higher from:
https://developer.nvidia.com/cuda-downloads

### 2. Update NVIDIA Drivers

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-550

# Verify installation
nvidia-smi
```

### 3. Install PyTorch with CUDA 12.1

```bash
pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify CUDA Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### GPU Not Detected

If GPUs are not detected:

1. Check NVIDIA driver installation:
   ```bash
   nvidia-smi
   ```

2. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

3. Check PyTorch CUDA support:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

### vLLM Installation Issues

If you encounter issues installing vLLM:

1. Ensure you have the correct CUDA version
2. Try installing with specific CUDA version:
   ```bash
   pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121
   ```

3. Check vLLM documentation: https://docs.vllm.ai/

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce `gpu_memory_utilization` in config:
   ```yaml
   gpu:
     gpu_memory_utilization: 0.85  # Reduce from 0.90
   ```

2. Reduce `max_num_seqs`:
   ```yaml
   gpu:
     max_num_seqs: 128  # Reduce from 256
   ```

3. Use quantization:
   ```yaml
   model:
     quantization: "awq"  # or "gptq"
   ```

## Next Steps

- Read the [Configuration Guide](configuration.md)
- Check out [Examples](../examples/)
- Review [API Documentation](api.md)
- Learn about [GPU Optimization](gpu_optimization.md)
