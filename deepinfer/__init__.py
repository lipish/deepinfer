"""DeepInfer - High-performance inference engine based on vLLM."""

__version__ = "0.1.0"
__author__ = "DeepInfer Team"
__description__ = "High-performance inference engine based on vLLM"

from deepinfer.engine import InferenceEngine
from deepinfer.config import Config

__all__ = [
    "InferenceEngine",
    "Config",
    "__version__",
]
