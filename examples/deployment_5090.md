# Example: Basic NVIDIA 5090 Deployment

This example demonstrates deploying DeepInfer on a machine with NVIDIA RTX 5090 GPU.

## Prerequisites

1. NVIDIA RTX 5090 GPU
2. CUDA 12.1+ installed (required for vLLM 0.6.0+ with RTX 5090)
3. Python 3.9+
4. At least 50GB free disk space

## Step 1: Installation

```bash
# Clone repository
git clone https://github.com/lipish/deepinfer.git
cd deepinfer

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e .
```

## Step 2: Verify GPU

```bash
# Check GPU detection
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

## Step 3: Start Server

```bash
# Start with RTX 5090 optimized config
deepinfer serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --config configs/nvidia_5090.yaml \
  --port 8000
```

## Step 4: Test Inference

In another terminal:

```bash
# Health check
curl http://localhost:8000/health

# GPU info
curl http://localhost:8000/v1/gpu/info

# Test completion
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The NVIDIA RTX 5090 is",
    "max_tokens": 100,
    "temperature": 0.7
  }'

# Test chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What makes RTX 5090 special?"}
    ],
    "max_tokens": 150
  }'
```

## Step 5: Monitor Performance

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f deepinfer.log
```

## Performance Expectations

With RTX 5090 and Llama-3.1-8B-Instruct:

- **Throughput**: ~2,500 tokens/second (estimated, adjust based on actual testing)
- **Concurrent Users**: Up to 512
- **Latency**: <100ms for first token (estimated)
- **Memory Usage**: ~18GB VRAM (model dependent)

## Production Deployment

For production, use systemd service:

```ini
# /etc/systemd/system/deepinfer.service
[Unit]
Description=DeepInfer Server
After=network.target

[Service]
Type=simple
User=deepinfer
WorkingDirectory=/opt/deepinfer
Environment="PATH=/opt/deepinfer/venv/bin"
ExecStart=/opt/deepinfer/venv/bin/deepinfer serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --config /opt/deepinfer/configs/nvidia_5090.yaml \
  --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable deepinfer
sudo systemctl start deepinfer
sudo systemctl status deepinfer
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce memory utilization in config:
```yaml
gpu:
  gpu_memory_utilization: 0.90  # Reduce from 0.95
```

### Issue: Low Throughput

**Solution**: Check for bottlenecks:
1. Verify CUDA version: `nvcc --version`
2. Check PCIe speed: `nvidia-smi -q | grep "Link Speed"`
3. Monitor CPU usage: `htop`

### Issue: Model Download Slow

**Solution**: Pre-download model:
```bash
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model = 'meta-llama/Llama-3.1-8B-Instruct'
AutoTokenizer.from_pretrained(model)
AutoModelForCausalLM.from_pretrained(model)
"
```
