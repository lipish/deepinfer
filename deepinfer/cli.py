"""Command-line interface for DeepInfer."""

import logging
import sys
from typing import Optional
import click

from deepinfer import __version__
from deepinfer.config import Config
from deepinfer.engine import InferenceEngine
from deepinfer.gpu_utils import GPUDetector


@click.group()
@click.version_option(version=__version__)
def main():
    """DeepInfer - High-performance inference engine based on vLLM."""
    pass


@main.command()
@click.option("--config", type=str, help="Path to configuration file")
@click.option("--model", type=str, required=True, help="Model name or path")
@click.option("--host", type=str, default="0.0.0.0", help="Server host")
@click.option("--port", type=int, default=8000, help="Server port")
@click.option("--log-level", type=str, default="info", help="Log level")
def serve(config: Optional[str], model: str, host: str, port: int, log_level: str):
    """Start the inference server."""
    # Setup logging
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load configuration
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config()
    
    # Override with command line arguments
    cfg.model.name = model
    cfg.server.host = host
    cfg.server.port = port
    cfg.server.log_level = log_level
    
    # Import here to avoid circular import
    from deepinfer.server import create_app
    import uvicorn
    
    # Create and run app
    app = create_app(cfg)
    
    uvicorn.run(
        app,
        host=cfg.server.host,
        port=cfg.server.port,
        log_level=cfg.server.log_level,
    )


@main.command()
@click.option("--config", type=str, help="Path to configuration file")
@click.option("--model", type=str, required=True, help="Model name or path")
@click.option("--prompt", type=str, required=True, help="Input prompt")
@click.option("--max-tokens", type=int, default=512, help="Maximum tokens to generate")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.option("--top-p", type=float, default=0.95, help="Top-p sampling")
def generate(
    config: Optional[str],
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
):
    """Generate text from a prompt."""
    # Setup logging
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    # Load configuration
    if config:
        cfg = Config.from_yaml(config)
    else:
        cfg = Config()
    
    # Override model
    cfg.model.name = model
    
    # Initialize engine
    engine = InferenceEngine(config=cfg)
    
    # Generate
    click.echo(f"\nPrompt: {prompt}")
    click.echo("\nGenerating...\n")
    
    outputs = engine.generate(
        prompts=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    
    # Print output
    output = outputs[0]
    generated_text = output.outputs[0].text
    
    click.echo(f"Generated: {generated_text}")
    click.echo(f"\nTokens: {len(output.outputs[0].token_ids)}")


@main.command()
def gpu_info():
    """Display GPU information."""
    detector = GPUDetector()
    detector.print_gpu_info()
    
    if detector.has_nvidia_5090():
        click.echo("✓ NVIDIA RTX 5090 detected - optimizations available!")
    elif detector.has_nvidia_4090():
        click.echo("✓ NVIDIA RTX 4090 detected - optimizations available!")


@main.command()
@click.option("--output", type=str, required=True, help="Output configuration file path")
@click.option("--model", type=str, required=True, help="Model name or path")
@click.option("--gpu-type", type=click.Choice(["auto", "5090", "4090", "default"]), default="auto", help="GPU type")
def init_config(output: str, model: str, gpu_type: str):
    """Initialize a configuration file."""
    config = Config()
    config.model.name = model
    
    # Configure for specific GPU
    if gpu_type == "auto":
        detector = GPUDetector()
        optimal_config = detector.get_optimal_config()
        for key, value in optimal_config.items():
            if hasattr(config.gpu, key):
                setattr(config.gpu, key, value)
    elif gpu_type == "5090":
        config.gpu.gpu_memory_utilization = 0.95
        config.gpu.max_num_seqs = 512
        config.gpu.nvidia_5090_optimizations = True
        config.gpu.enable_chunked_prefill = True
        config.gpu.enable_prefix_caching = True
    elif gpu_type == "4090":
        config.gpu.gpu_memory_utilization = 0.92
        config.gpu.max_num_seqs = 384
        config.gpu.enable_chunked_prefill = True
        config.gpu.enable_prefix_caching = True
    
    # Save configuration
    config.to_yaml(output)
    click.echo(f"Configuration saved to: {output}")


if __name__ == "__main__":
    main()
