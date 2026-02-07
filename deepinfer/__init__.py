"""DeepInfer - High-performance inference engine based on vLLM."""

__version__ = "0.1.0"
__author__ = "DeepInfer Team"
__description__ = "High-performance inference engine based on vLLM"

# Import Config always (doesn't require vllm)
from deepinfer.config import Config

# Conditionally import InferenceEngine (requires vllm)
try:
    from deepinfer.engine import InferenceEngine
    _has_vllm = True
except ImportError:
    _has_vllm = False
    InferenceEngine = None

__all__ = [
    "Config",
    "InferenceEngine",
    "__version__",
]

