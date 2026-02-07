#!/usr/bin/env python3
"""
Unified Engine Server Entry Point

Supports multiple engine types via --engine parameter:
- vllm: vLLM engine (default)
- TODO: Add more engines
"""
import argparse
import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="xinf Engine Server")
    parser.add_argument("--engine", type=str, default="vllm", 
                       help="Engine type (vllm, etc.)")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name or path")
    parser.add_argument("--port", type=int, default=50051,
                       help="gRPC server port")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="gRPC server host")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="Tensor parallel size")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                       help="Pipeline parallel size")
    parser.add_argument("--dtype", type=str, default="auto",
                       help="Data type (auto, fp16, bf16, etc.)")
    parser.add_argument("--max-model-len", type=int, default=None,
                       help="Max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                       help="GPU memory utilization")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda:0, cpu, etc.)")
    
    args = parser.parse_args()
    
    logger.info(f"Starting {args.engine} engine for model: {args.model}")
    
    # Setup device
    from xinf_engine.device import setup_device
    setup_device(args.device)
    
    # Select and start engine
    if args.engine == "vllm":
        from xinf_engine.vllm_shim import serve_vllm
        asyncio.run(serve_vllm(args))
    else:
        logger.error(f"Unsupported engine: {args.engine}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
