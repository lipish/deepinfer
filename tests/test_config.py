"""
Tests for configuration.
"""

import pytest
import yaml
import tempfile
import os
from pathlib import Path

from deepinfer.config import Config, ModelConfig, GPUConfig, ServerConfig, SamplingConfig


def test_config_defaults():
    """Test default configuration values."""
    config = Config()
    
    # Check model defaults
    assert config.model is not None
    assert isinstance(config.model, ModelConfig)
    
    # Check GPU defaults
    assert config.gpu.device == "cuda"
    assert 0.0 < config.gpu.gpu_memory_utilization <= 1.0
    assert config.gpu.tensor_parallel_size >= 1
    
    # Check server defaults
    assert config.server.host == "0.0.0.0"
    assert config.server.port == 8000
    
    # Check sampling defaults
    assert config.sampling.temperature >= 0.0
    assert 0.0 <= config.sampling.top_p <= 1.0


def test_config_from_yaml():
    """Test loading configuration from YAML."""
    config_data = {
        "model": {
            "name": "test-model",
            "dtype": "float16",
        },
        "gpu": {
            "device": "cuda",
            "gpu_memory_utilization": 0.95,
            "max_num_seqs": 512,
        },
        "server": {
            "port": 9000,
        },
    }
    
    # Create temporary YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        temp_file = f.name
    
    try:
        # Load config
        config = Config.from_yaml(temp_file)
        
        # Verify loaded values
        assert config.model.name == "test-model"
        assert config.model.dtype == "float16"
        assert config.gpu.gpu_memory_utilization == 0.95
        assert config.gpu.max_num_seqs == 512
        assert config.server.port == 9000
    finally:
        # Cleanup
        os.unlink(temp_file)


def test_config_to_yaml():
    """Test saving configuration to YAML."""
    config = Config()
    config.model.name = "test-model"
    config.gpu.max_num_seqs = 512
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        temp_file = f.name
    
    try:
        # Save config
        config.to_yaml(temp_file)
        
        # Load and verify
        with open(temp_file, "r") as f:
            loaded_data = yaml.safe_load(f)
        
        assert loaded_data["model"]["name"] == "test-model"
        assert loaded_data["gpu"]["max_num_seqs"] == 512
    finally:
        # Cleanup
        os.unlink(temp_file)


def test_model_config():
    """Test ModelConfig."""
    model_config = ModelConfig(name="test-model")
    
    assert model_config.name == "test-model"
    assert model_config.trust_remote_code is False
    assert model_config.dtype == "auto"


def test_gpu_config():
    """Test GPUConfig."""
    gpu_config = GPUConfig()
    
    assert gpu_config.device == "cuda"
    assert 0.0 < gpu_config.gpu_memory_utilization <= 1.0
    assert gpu_config.tensor_parallel_size >= 1


def test_server_config():
    """Test ServerConfig."""
    server_config = ServerConfig()
    
    assert server_config.host == "0.0.0.0"
    assert server_config.port == 8000
    assert server_config.log_level == "info"


def test_sampling_config():
    """Test SamplingConfig."""
    sampling_config = SamplingConfig()
    
    assert sampling_config.temperature >= 0.0
    assert 0.0 <= sampling_config.top_p <= 1.0
    assert sampling_config.max_tokens > 0
