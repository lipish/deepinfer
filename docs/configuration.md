# Configuration Guide

This guide explains how to configure DeepInfer for your specific needs.

## Configuration File Format

DeepInfer uses YAML configuration files. Here's the complete structure:

```yaml
model:
  name: "model-name-or-path"
  trust_remote_code: false
  revision: null
  tokenizer: null
  tokenizer_mode: "auto"
  dtype: "auto"
  max_model_len: null
  quantization: null

gpu:
  device: "cuda"
  gpu_memory_utilization: 0.90
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  max_num_batched_tokens: null
  max_num_seqs: 256
  max_paddings: 256
  enable_chunked_prefill: false
  enable_prefix_caching: true
  disable_log_stats: false
  nvidia_5090_optimizations: false
  kv_cache_dtype: "auto"
  enforce_eager: false

server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
  api_key: null
  allowed_origins:
    - "*"
  timeout: 600

sampling:
  temperature: 0.7
  top_p: 0.95
  top_k: -1
  max_tokens: 512
  presence_penalty: 0.0
  frequency_penalty: 0.0
  repetition_penalty: 1.0
  stop: null
```

## Configuration Sections

### Model Configuration

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  trust_remote_code: false
  revision: null
  tokenizer: null
  tokenizer_mode: "auto"
  dtype: "auto"
  max_model_len: null
  quantization: null
```

**Parameters**:

- **name** (required): Model name from Hugging Face or local path
  - Examples: `"meta-llama/Llama-3.1-8B-Instruct"`, `"/path/to/local/model"`

- **trust_remote_code**: Whether to trust remote code in model
  - Default: `false`
  - Set to `true` for models that require custom code

- **revision**: Specific model revision/branch
  - Default: `null` (uses main/default)
  - Example: `"v1.0"`

- **tokenizer**: Custom tokenizer name or path
  - Default: `null` (uses model's tokenizer)
  - Example: `"meta-llama/Llama-3.1-8B-Instruct"`

- **tokenizer_mode**: Tokenizer loading mode
  - Options: `"auto"`, `"slow"`
  - Default: `"auto"`

- **dtype**: Model data type
  - Options: `"auto"`, `"float16"`, `"bfloat16"`, `"float32"`
  - Default: `"auto"` (uses model's default)
  - Recommendation: `"float16"` for inference, `"bfloat16"` for RTX 5090

- **max_model_len**: Maximum sequence length
  - Default: `null` (uses model's default)
  - Example: `4096`, `8192`

- **quantization**: Quantization method
  - Options: `null`, `"awq"`, `"gptq"`, `"squeezellm"`, `"fp8"`
  - Default: `null` (no quantization)

### GPU Configuration

```yaml
gpu:
  device: "cuda"
  gpu_memory_utilization: 0.90
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  max_num_seqs: 256
```

**Parameters**:

- **device**: Compute device
  - Options: `"cuda"`, `"cpu"`
  - Default: `"cuda"`

- **gpu_memory_utilization**: Fraction of GPU memory to use
  - Range: 0.0 - 1.0
  - Default: 0.90
  - RTX 5090: 0.90 (can increase to 0.95 after testing)
  - RTX 4090: 0.92
  - 24GB GPU: 0.90
  - 16GB GPU: 0.88

- **tensor_parallel_size**: Number of GPUs for tensor parallelism
  - Default: 1
  - Example: 2 (for 2 GPUs), 4 (for 4 GPUs)

- **pipeline_parallel_size**: Number of GPUs for pipeline parallelism
  - Default: 1
  - Advanced feature, typically use tensor parallelism instead

- **max_num_seqs**: Maximum concurrent sequences
  - Default: 256
  - RTX 5090: 512
  - RTX 4090: 384
  - Lower for smaller GPUs: 128, 64

- **enable_chunked_prefill**: Enable chunked prefill for long contexts
  - Default: false
  - Recommended for RTX 5090: true

- **enable_prefix_caching**: Cache common prefixes
  - Default: true
  - Improves performance when prompts share prefixes

- **nvidia_5090_optimizations**: Enable RTX 5090 specific optimizations
  - Default: false
  - Set to true for RTX 5090

### Server Configuration

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"
  api_key: null
  allowed_origins:
    - "*"
  timeout: 600
```

**Parameters**:

- **host**: Server bind address
  - Default: `"0.0.0.0"` (all interfaces)
  - Local only: `"127.0.0.1"`

- **port**: Server port
  - Default: 8000

- **log_level**: Logging level
  - Options: `"debug"`, `"info"`, `"warning"`, `"error"`
  - Default: `"info"`

- **api_key**: Optional API key for authentication
  - Default: `null` (no authentication)
  - Example: `"sk-your-secret-key"`

- **allowed_origins**: CORS allowed origins
  - Default: `["*"]` (all origins)
  - Production: `["https://yourdomain.com"]`

- **timeout**: Request timeout in seconds
  - Default: 600 (10 minutes)

### Sampling Configuration

```yaml
sampling:
  temperature: 0.7
  top_p: 0.95
  top_k: -1
  max_tokens: 512
  presence_penalty: 0.0
  frequency_penalty: 0.0
  repetition_penalty: 1.0
  stop: null
```

**Parameters**:

- **temperature**: Sampling temperature
  - Range: 0.0 - 2.0
  - 0.0: Deterministic (greedy)
  - 0.7: Balanced (default)
  - 1.0: Creative
  - 2.0: Very creative

- **top_p**: Nucleus sampling probability
  - Range: 0.0 - 1.0
  - Default: 0.95

- **top_k**: Top-k sampling
  - -1: Disabled (default)
  - 50: Only consider top 50 tokens

- **max_tokens**: Maximum tokens to generate
  - Default: 512

- **presence_penalty**: Penalty for token presence
  - Range: -2.0 - 2.0
  - Default: 0.0
  - Positive: Encourage new topics

- **frequency_penalty**: Penalty for token frequency
  - Range: -2.0 - 2.0
  - Default: 0.0
  - Positive: Reduce repetition

- **repetition_penalty**: Repetition penalty
  - Range: 1.0 - 2.0
  - Default: 1.0
  - >1.0: Penalize repetition

- **stop**: Stop sequences
  - Default: null
  - Example: `["\n", "END"]`

## Loading Configuration

### From YAML File

```python
from deepinfer import Config

config = Config.from_yaml("configs/nvidia_5090.yaml")
```

### From Python

```python
from deepinfer import Config

config = Config()
config.model.name = "meta-llama/Llama-3.1-8B-Instruct"
config.gpu.gpu_memory_utilization = 0.95
config.gpu.max_num_seqs = 512
```

### From Environment Variables

```bash
export DEEPINFER_MODEL__NAME="meta-llama/Llama-3.1-8B-Instruct"
export DEEPINFER_GPU__GPU_MEMORY_UTILIZATION=0.95
export DEEPINFER_SERVER__PORT=8080
```

### Command Line

```bash
deepinfer serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --config configs/nvidia_5090.yaml \
  --host 0.0.0.0 \
  --port 8000
```

## Pre-configured Files

DeepInfer includes several pre-configured files:

### configs/default.yaml
General-purpose configuration for most GPUs

### configs/nvidia_5090.yaml
Optimized for NVIDIA RTX 5090
- 95% GPU memory utilization
- 512 concurrent sequences
- Chunked prefill enabled
- Prefix caching enabled

### configs/nvidia_4090.yaml
Optimized for NVIDIA RTX 4090
- 92% GPU memory utilization
- 384 concurrent sequences
- Chunked prefill enabled

## Creating Custom Configuration

1. Start with a template:
```bash
cp configs/default.yaml my_config.yaml
```

2. Edit settings:
```yaml
model:
  name: "your-model-name"

gpu:
  gpu_memory_utilization: 0.90
  max_num_seqs: 256
```

3. Use your configuration:
```bash
deepinfer serve --config my_config.yaml
```

## Configuration Best Practices

1. **Start with defaults** for your GPU type
2. **Test incrementally** - adjust one parameter at a time
3. **Monitor GPU usage** with `nvidia-smi`
4. **Lower utilization** if you encounter OOM errors
5. **Use quantization** for larger models on smaller GPUs
6. **Enable caching** for production workloads
7. **Adjust max_num_seqs** based on your concurrency needs

## Troubleshooting

### Configuration Not Found

```bash
FileNotFoundError: Config file not found: configs/my_config.yaml
```

**Solution**: Check file path and ensure file exists

### Invalid Configuration

```bash
ValidationError: Invalid configuration
```

**Solution**: Check YAML syntax and parameter values

### Out of Memory

```bash
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Reduce `gpu_memory_utilization`
2. Reduce `max_num_seqs`
3. Enable quantization
4. Use smaller model
