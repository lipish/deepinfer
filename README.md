# deepinfer - Rust-Native Inference Platform

A high-performance, declarative inference platform with pure Rust control plane.

## Architecture

**deepinfer** is a Rust-native inference orchestration platform:

- **Rust Control Plane**: Gateway (Axum HTTP), Scheduler, Router, Worker Agent, MetaStore, Device Abstraction, CLI
- **Docker Data Plane**: vLLM runs in Docker containers with OpenAI-compatible API

### Core Design Principles

- **Declarative Reconcile**: Scheduler writes intent to MetaStore, Agents autonomously execute via Watch
- **Non-Invasive Engines**: vLLM runs in Docker containers, communicates via HTTP
- **Unified Hardware Abstraction**: Rust FFI calls NVML for device discovery and monitoring

## Features

- ✅ vLLM Docker support with OpenAI-compatible API
- ✅ NVIDIA GPU support (including RTX 5090 / Blackwell SM 12.0)
- ✅ Multi-GPU tensor parallelism
- ✅ Declarative scheduling with reconciliation loops
- ✅ KV-cache aware routing with session affinity
- ✅ etcd-based distributed metadata storage
- ✅ Multi-accelerator support (CUDA, ROCm, XPU, NPU, etc.)

## Supported Hardware

### Accelerator Types
- NVIDIA CUDA GPUs (including Blackwell/RTX 5090)
- AMD ROCm GPUs
- Intel XPU
- Apple Metal (MPS)
- Ascend NPU
- Cambricon MLU
- Enflame GCU
- Moore Threads MUSA
- Hygon DCU
- Kunlunxin XPU
- CPU fallback

### Data Type Support
FP32, FP16, BF16, INT8, INT4, FP8 (E4M3/E5M2), FP4

## Quick Start

### Prerequisites
- Rust toolchain (1.70+)
- CUDA 12.0+ (for NVIDIA GPUs)
- etcd 3.5+
- Docker

### Start Services

```bash
# Start etcd (if not running)
./start-etcd.sh

# Start the gateway and scheduler
deepinfer serve -c configs/default.toml

# Start a worker agent on this node
deepinfer worker -n node-1
```

### Deploy Model

```bash
# Launch a model using Docker vLLM backend
curl -X POST http://localhost:8082/v1/deployments \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "backend": "docker",
    "docker_image": "vllm/vllm-openai:v0.11.0",
    "model_path": "/path/to/model",
    "device": "0,1",
    "tensor_parallel_size": 2,
    "gpu_memory_utilization": 0.8
  }'
```

### Inference

```bash
# Chat completion
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-1.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Project Structure

```
.
├── crates/                    # Rust control plane
│   ├── deepinfer-common/      # Shared types and utilities
│   ├── deepinfer-device/      # Hardware abstraction layer
│   ├── deepinfer-meta/        # Metadata storage (etcd)
│   ├── deepinfer-scheduler/   # Placement scheduler
│   ├── deepinfer-router/      # Request routing
│   ├── deepinfer-agent/       # Worker agent & engine launcher
│   ├── deepinfer-gateway/     # API gateway
│   └── deepinfer-cli/         # Command-line interface
└── configs/                   # Configuration files
```

## API Reference

### POST /v1/deployments

Deploy a new model engine.

| Parameter | Type | Description |
|-----------|------|-------------|
| model | string | Model name (required) |
| backend | string | "docker" (required) |
| docker_image | string | Docker image (e.g. "vllm/vllm-openai:v0.11.0") |
| model_path | string | Path to model files |
| device | string | Device spec, e.g. "0,1" |
| tensor_parallel_size | int | Number of GPUs for tensor parallelism |
| gpu_memory_utilization | float | GPU memory fraction (0.0-1.0) |

### POST /v1/chat/completions

OpenAI-compatible chat completion endpoint.

## Development

### Build
```bash
cargo build --release
```

## Configuration

Main configuration file: `configs/default.toml`

## License

Apache-2.0
