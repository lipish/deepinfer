# xinf - Rust-Native Inference Platform

A high-performance, declarative inference platform with Rust control plane and Python data plane.

## Architecture

**xinf** adopts a hybrid architecture:

- **Rust Control Plane**: Gateway (Axum HTTP), Scheduler, Router, Worker Agent, MetaStore, Device Abstraction, CLI
- **Python Data Plane**: Engine Shim (gRPC wrapper for vLLM), Model Registry, Chat Business Logic

### Core Design Principles

- **Declarative Reconcile**: Scheduler writes intent to MetaStore, Agents autonomously execute via Watch
- **Non-Invasive Engines**: vLLM runs as independent process, communicates via gRPC
- **Unified Hardware Abstraction**: Rust FFI calls NVML for device discovery and monitoring

## Features

- ✅ vLLM engine support via gRPC Engine Shim
- ✅ NVIDIA GPU support (including RTX 5090 / Blackwell SM 10.0)
- ✅ Declarative scheduling with reconciliation loops
- ✅ KV-cache aware routing with session affinity
- ✅ Single-machine mode with embedded sled storage
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
- Python 3.9+
- CUDA 12.0+ (for NVIDIA GPUs)

### Single Machine Mode

```bash
# Start the gateway and scheduler
xinf serve

# Start a worker agent on this node
xinf worker

# Launch a model
xinf launch --model Qwen/Qwen2.5-7B-Instruct --engine vllm --device cuda:0

# Test inference
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Project Structure

```
.
├── crates/              # Rust control plane
│   ├── xinf-common/     # Shared types and utilities
│   ├── xinf-device/     # Hardware abstraction layer
│   ├── xinf-meta/       # Metadata storage
│   ├── xinf-scheduler/  # Placement scheduler
│   ├── xinf-router/     # Request routing
│   ├── xinf-agent/      # Worker agent
│   ├── xinf-gateway/    # API gateway
│   └── xinf-cli/        # Command-line interface
├── python/              # Python data plane
│   ├── xinf_engine/     # Engine Shim (vLLM wrapper)
│   ├── xinf_registry/   # Model registry
│   └── xinf_chat/       # Chat template logic
├── protos/              # gRPC protocol definitions
└── configs/             # Configuration files
```

## Development

### Build Rust Components
```bash
cargo build --release
```

### Install Python Dependencies
```bash
cd python
pip install -e xinf_engine -e xinf_registry -e xinf_chat
```

### Generate gRPC Code
```bash
python -m grpc_tools.protoc -I protos --python_out=python/xinf_engine/generated --grpc_python_out=python/xinf_engine/generated protos/engine_service.proto
```

## Configuration

Main configuration file: `configs/default.toml`

Engine compatibility matrix: `configs/engine_compatibility.toml`

## License

Apache-2.0
