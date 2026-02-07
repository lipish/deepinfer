"""Configuration management for DeepInfer."""

import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Model configuration."""
    name: str = Field(description="Model name or path")
    trust_remote_code: bool = Field(default=False, description="Trust remote code")
    revision: Optional[str] = Field(default=None, description="Model revision")
    tokenizer: Optional[str] = Field(default=None, description="Tokenizer name or path")
    tokenizer_mode: str = Field(default="auto", description="Tokenizer mode")
    dtype: str = Field(default="auto", description="Data type (auto, float16, bfloat16, float32)")
    max_model_len: Optional[int] = Field(default=None, description="Maximum model context length")
    quantization: Optional[str] = Field(default=None, description="Quantization method (awq, gptq, squeezellm, fp8)")


class GPUConfig(BaseModel):
    """GPU configuration."""
    device: str = Field(default="cuda", description="Device to use (cuda, cpu)")
    gpu_memory_utilization: float = Field(default=0.90, ge=0.0, le=1.0, description="GPU memory utilization")
    tensor_parallel_size: int = Field(default=1, ge=1, description="Tensor parallel size")
    pipeline_parallel_size: int = Field(default=1, ge=1, description="Pipeline parallel size")
    max_num_batched_tokens: Optional[int] = Field(default=None, description="Maximum number of batched tokens")
    max_num_seqs: int = Field(default=256, description="Maximum number of sequences")
    max_paddings: int = Field(default=256, description="Maximum padding length")
    enable_chunked_prefill: bool = Field(default=False, description="Enable chunked prefill")
    enable_prefix_caching: bool = Field(default=True, description="Enable prefix caching")
    disable_log_stats: bool = Field(default=False, description="Disable logging stats")
    
    # NVIDIA 5090 specific optimizations
    nvidia_5090_optimizations: bool = Field(default=False, description="Enable NVIDIA 5090 specific optimizations")
    kv_cache_dtype: str = Field(default="auto", description="KV cache data type")
    enforce_eager: bool = Field(default=False, description="Enforce eager mode (disable CUDA graphs)")


class ServerConfig(BaseModel):
    """Server configuration."""
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="info", description="Log level")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    allowed_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    timeout: int = Field(default=600, description="Request timeout in seconds")
    

class SamplingConfig(BaseModel):
    """Default sampling configuration."""
    temperature: float = Field(default=0.7, ge=0.0, description="Sampling temperature")
    top_p: float = Field(default=0.95, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(default=-1, description="Top-k sampling")
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    presence_penalty: float = Field(default=0.0, description="Presence penalty")
    frequency_penalty: float = Field(default=0.0, description="Frequency penalty")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    stop: Optional[List[str]] = Field(default=None, description="Stop sequences")


class Config(BaseSettings):
    """Main configuration for DeepInfer."""
    
    model: ModelConfig = Field(default_factory=ModelConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    
    class Config:
        env_prefix = "DEEPINFER_"
        case_sensitive = False
        
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
            
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
            
        return cls(**config_dict)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.model_dump()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def update(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
