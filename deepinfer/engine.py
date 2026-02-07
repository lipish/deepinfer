"""Main inference engine based on vLLM."""

import logging
from typing import Optional, List, Dict, Any, AsyncIterator, Union
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput

from deepinfer.config import Config
from deepinfer.gpu_utils import GPUDetector

logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-performance inference engine based on vLLM."""
    
    def __init__(self, config: Optional[Config] = None, model_name: Optional[str] = None):
        """
        Initialize the inference engine.
        
        Args:
            config: Configuration object
            model_name: Model name or path (overrides config)
        """
        self.config = config or Config()
        
        # Override model name if provided
        if model_name:
            self.config.model.name = model_name
        
        # Detect and configure GPU
        self.gpu_detector = GPUDetector()
        self._configure_gpu()
        
        # Initialize vLLM engine
        self.llm = self._initialize_llm()
        
        logger.info(f"InferenceEngine initialized with model: {self.config.model.name}")
    
    def _configure_gpu(self) -> None:
        """Configure GPU settings based on detected hardware."""
        if not self.gpu_detector.gpus:
            logger.warning("No GPUs detected - using CPU mode")
            self.config.gpu.device = "cpu"
            return
        
        # Get optimal configuration for detected GPUs
        optimal_config = self.gpu_detector.get_optimal_config()
        
        # Update GPU configuration
        for key, value in optimal_config.items():
            if hasattr(self.config.gpu, key):
                setattr(self.config.gpu, key, value)
        
        # Log GPU configuration
        if self.gpu_detector.has_nvidia_5090():
            logger.info("NVIDIA RTX 5090 detected - using optimized settings")
            logger.info(f"GPU Memory Utilization: {self.config.gpu.gpu_memory_utilization}")
            logger.info(f"Tensor Parallel Size: {self.config.gpu.tensor_parallel_size}")
            logger.info(f"Max Sequences: {self.config.gpu.max_num_seqs}")
    
    def _initialize_llm(self) -> LLM:
        """Initialize vLLM engine."""
        logger.info(f"Loading model: {self.config.model.name}")
        
        # Prepare vLLM initialization parameters
        llm_kwargs = {
            "model": self.config.model.name,
            "tokenizer": self.config.model.tokenizer or self.config.model.name,
            "tokenizer_mode": self.config.model.tokenizer_mode,
            "trust_remote_code": self.config.model.trust_remote_code,
            "dtype": self.config.model.dtype,
            "gpu_memory_utilization": self.config.gpu.gpu_memory_utilization,
            "tensor_parallel_size": self.config.gpu.tensor_parallel_size,
            "pipeline_parallel_size": self.config.gpu.pipeline_parallel_size,
            "max_num_seqs": self.config.gpu.max_num_seqs,
            "disable_log_stats": self.config.gpu.disable_log_stats,
        }
        
        # Add optional parameters
        if self.config.model.max_model_len:
            llm_kwargs["max_model_len"] = self.config.model.max_model_len
        
        if self.config.model.quantization:
            llm_kwargs["quantization"] = self.config.model.quantization
        
        if self.config.model.revision:
            llm_kwargs["revision"] = self.config.model.revision
        
        if self.config.gpu.max_num_batched_tokens:
            llm_kwargs["max_num_batched_tokens"] = self.config.gpu.max_num_batched_tokens
        
        # NVIDIA 5090 specific optimizations
        if self.config.gpu.nvidia_5090_optimizations:
            logger.info("Applying NVIDIA RTX 5090 optimizations")
            llm_kwargs["enable_chunked_prefill"] = self.config.gpu.enable_chunked_prefill
            llm_kwargs["enable_prefix_caching"] = self.config.gpu.enable_prefix_caching
            llm_kwargs["kv_cache_dtype"] = self.config.gpu.kv_cache_dtype
            llm_kwargs["enforce_eager"] = self.config.gpu.enforce_eager
        
        try:
            llm = LLM(**llm_kwargs)
            logger.info("vLLM engine initialized successfully")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize vLLM engine: {e}")
            raise
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        **kwargs
    ) -> List[RequestOutput]:
        """
        Generate text from prompts.
        
        Args:
            prompts: Single prompt or list of prompts
            sampling_params: vLLM sampling parameters
            **kwargs: Additional sampling parameters
            
        Returns:
            List of generated outputs
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Use provided sampling params or create from config/kwargs
        if sampling_params is None:
            sampling_params = self._create_sampling_params(**kwargs)
        
        logger.debug(f"Generating for {len(prompts)} prompt(s)")
        
        try:
            outputs = self.llm.generate(prompts, sampling_params)
            return outputs
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    def _create_sampling_params(self, **kwargs) -> SamplingParams:
        """Create SamplingParams from config and kwargs."""
        # Start with config defaults
        params = {
            "temperature": kwargs.get("temperature", self.config.sampling.temperature),
            "top_p": kwargs.get("top_p", self.config.sampling.top_p),
            "top_k": kwargs.get("top_k", self.config.sampling.top_k),
            "max_tokens": kwargs.get("max_tokens", self.config.sampling.max_tokens),
            "presence_penalty": kwargs.get("presence_penalty", self.config.sampling.presence_penalty),
            "frequency_penalty": kwargs.get("frequency_penalty", self.config.sampling.frequency_penalty),
            "repetition_penalty": kwargs.get("repetition_penalty", self.config.sampling.repetition_penalty),
        }
        
        # Add stop sequences if provided
        if "stop" in kwargs:
            params["stop"] = kwargs["stop"]
        elif self.config.sampling.stop:
            params["stop"] = self.config.sampling.stop
        
        return SamplingParams(**params)
    
    def get_tokenizer(self):
        """Get the tokenizer from vLLM engine."""
        return self.llm.get_tokenizer()
    
    def get_model_config(self):
        """Get the model config from vLLM engine."""
        return self.llm.llm_engine.model_config
    
    def __repr__(self) -> str:
        """String representation."""
        return f"InferenceEngine(model={self.config.model.name}, device={self.config.gpu.device})"
