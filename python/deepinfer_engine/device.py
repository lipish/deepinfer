"""
Device setup utilities
"""
import logging
import os

logger = logging.getLogger(__name__)


def setup_device(device: str = None):
    """
    Setup PyTorch device based on specification
    
    Args:
        device: Device specification (cuda:0, cpu, etc.)
                If None, auto-detect best available device
    """
    if device is None:
        # Auto-detect
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Auto-detected CUDA device: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("No CUDA device found, using CPU")
        except ImportError:
            logger.warning("PyTorch not installed, skipping device setup")
            return
    
    logger.info(f"Setting up device: {device}")
    
    # Set CUDA_VISIBLE_DEVICES if specific GPU specified
    if device.startswith("cuda:"):
        gpu_id = device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        logger.info(f"Set CUDA_VISIBLE_DEVICES={gpu_id}")
