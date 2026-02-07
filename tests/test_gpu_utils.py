"""
Tests for GPU utilities.
"""

import pytest
from deepinfer.gpu_utils import GPUDetector


def test_gpu_detector_initialization():
    """Test GPU detector initialization."""
    detector = GPUDetector()
    assert detector is not None
    assert isinstance(detector.gpus, list)


def test_gpu_detector_methods():
    """Test GPU detector methods."""
    detector = GPUDetector()
    
    # Test methods exist and return correct types
    assert isinstance(detector.has_nvidia_5090(), bool)
    assert isinstance(detector.has_nvidia_4090(), bool)
    assert isinstance(detector.get_nvidia_5090_gpus(), list)
    assert isinstance(detector.get_optimal_config(), dict)


def test_optimal_config_structure():
    """Test optimal config returns correct structure."""
    detector = GPUDetector()
    config = detector.get_optimal_config()
    
    # Verify required keys exist
    assert "device" in config
    assert "gpu_memory_utilization" in config
    assert "tensor_parallel_size" in config


def test_gpu_info():
    """Test GPU info retrieval."""
    detector = GPUDetector()
    
    if len(detector.gpus) > 0:
        gpu_info = detector.get_gpu_info(0)
        assert gpu_info is not None
        assert hasattr(gpu_info, "id")
        assert hasattr(gpu_info, "name")
        assert hasattr(gpu_info, "total_memory")
    else:
        # No GPUs available
        gpu_info = detector.get_gpu_info(0)
        assert gpu_info is None
