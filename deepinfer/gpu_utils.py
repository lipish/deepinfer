"""GPU utilities for detection and configuration."""

import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """GPU information."""
    id: int
    name: str
    total_memory: int  # in bytes
    free_memory: int   # in bytes
    compute_capability: tuple
    driver_version: str
    cuda_version: str
    is_nvidia_5090: bool = False
    is_nvidia_4090: bool = False


class GPUDetector:
    """GPU detection and configuration utility."""
    
    def __init__(self):
        self.gpus: List[GPUInfo] = []
        self._detect_gpus()
    
    def _detect_gpus(self) -> None:
        """Detect available GPUs."""
        try:
            import pynvml
            pynvml.nvmlInit()
            
            device_count = pynvml.nvmlDeviceGetCount()
            logger.info(f"Detected {device_count} GPU(s)")
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get GPU information
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                # Get compute capability
                try:
                    major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                    minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                    compute_capability = (major, minor)
                except:
                    compute_capability = (0, 0)
                
                # Get driver and CUDA version
                try:
                    driver_version = pynvml.nvmlSystemGetDriverVersion()
                    if isinstance(driver_version, bytes):
                        driver_version = driver_version.decode('utf-8')
                except:
                    driver_version = "unknown"
                
                try:
                    cuda_version = pynvml.nvmlSystemGetCudaDriverVersion()
                    cuda_version_str = f"{cuda_version // 1000}.{(cuda_version % 1000) // 10}"
                except:
                    cuda_version_str = "unknown"
                
                # Detect specific GPU models - use precise matching
                # Only match official NVIDIA naming patterns
                is_5090 = False
                is_4090 = False
                
                # Check for RTX 5090 - match official naming
                if "RTX 5090" in name or "GeForce RTX 5090" in name:
                    is_5090 = True
                elif "RTX 4090" in name or "GeForce RTX 4090" in name:
                    is_4090 = True
                
                gpu_info = GPUInfo(
                    id=i,
                    name=name,
                    total_memory=memory_info.total,
                    free_memory=memory_info.free,
                    compute_capability=compute_capability,
                    driver_version=driver_version,
                    cuda_version=cuda_version_str,
                    is_nvidia_5090=is_5090,
                    is_nvidia_4090=is_4090,
                )
                
                self.gpus.append(gpu_info)
                
                logger.info(
                    f"GPU {i}: {name} | "
                    f"Memory: {memory_info.total / 1024**3:.2f} GB | "
                    f"Compute: {compute_capability[0]}.{compute_capability[1]} | "
                    f"Driver: {driver_version} | "
                    f"CUDA: {cuda_version_str}"
                )
                
                if is_5090:
                    logger.info(f"NVIDIA RTX 5090 detected on GPU {i} - enabling optimizations")
                
            pynvml.nvmlShutdown()
            
        except ImportError:
            logger.warning("pynvml not available - GPU detection disabled")
        except Exception as e:
            logger.error(f"Error detecting GPUs: {e}")
    
    def has_nvidia_5090(self) -> bool:
        """Check if any NVIDIA 5090 GPU is available."""
        return any(gpu.is_nvidia_5090 for gpu in self.gpus)
    
    def has_nvidia_4090(self) -> bool:
        """Check if any NVIDIA 4090 GPU is available."""
        return any(gpu.is_nvidia_4090 for gpu in self.gpus)
    
    def get_nvidia_5090_gpus(self) -> List[GPUInfo]:
        """Get list of NVIDIA 5090 GPUs."""
        return [gpu for gpu in self.gpus if gpu.is_nvidia_5090]
    
    def get_optimal_config(self) -> Dict[str, Any]:
        """Get optimal configuration based on detected GPUs."""
        if not self.gpus:
            logger.warning("No GPUs detected - using default CPU configuration")
            return {
                "device": "cpu",
                "gpu_memory_utilization": 0.0,
                "tensor_parallel_size": 1,
            }
        
        # Check for NVIDIA 5090
        if self.has_nvidia_5090():
            logger.info("Optimizing for NVIDIA RTX 5090")
            return {
                "device": "cuda",
                "gpu_memory_utilization": 0.90,  # Conservative default, can increase to 0.95 after testing
                "tensor_parallel_size": len(self.get_nvidia_5090_gpus()),
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "nvidia_5090_optimizations": True,
                "kv_cache_dtype": "auto",
                "max_num_seqs": 512,  # 5090 can handle more sequences
            }
        
        # Check for NVIDIA 4090
        if self.has_nvidia_4090():
            logger.info("Optimizing for NVIDIA RTX 4090")
            return {
                "device": "cuda",
                "gpu_memory_utilization": 0.92,
                "tensor_parallel_size": 1,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": True,
                "max_num_seqs": 384,
            }
        
        # Default CUDA configuration for other GPUs
        total_memory_gb = self.gpus[0].total_memory / 1024**3
        if total_memory_gb >= 40:
            gpu_memory_utilization = 0.92
            max_num_seqs = 384
        elif total_memory_gb >= 24:
            gpu_memory_utilization = 0.90
            max_num_seqs = 256
        elif total_memory_gb >= 16:
            gpu_memory_utilization = 0.88
            max_num_seqs = 128
        else:
            gpu_memory_utilization = 0.85
            max_num_seqs = 64
        
        return {
            "device": "cuda",
            "gpu_memory_utilization": gpu_memory_utilization,
            "tensor_parallel_size": 1,
            "max_num_seqs": max_num_seqs,
        }
    
    def get_gpu_info(self, gpu_id: int = 0) -> Optional[GPUInfo]:
        """Get information about a specific GPU."""
        if 0 <= gpu_id < len(self.gpus):
            return self.gpus[gpu_id]
        return None
    
    def print_gpu_info(self) -> None:
        """Print detailed GPU information."""
        if not self.gpus:
            print("No GPUs detected")
            return
        
        print(f"\n{'='*60}")
        print("GPU Information")
        print(f"{'='*60}")
        
        for gpu in self.gpus:
            print(f"\nGPU {gpu.id}: {gpu.name}")
            print(f"  Total Memory: {gpu.total_memory / 1024**3:.2f} GB")
            print(f"  Free Memory:  {gpu.free_memory / 1024**3:.2f} GB")
            print(f"  Compute Capability: {gpu.compute_capability[0]}.{gpu.compute_capability[1]}")
            print(f"  Driver Version: {gpu.driver_version}")
            print(f"  CUDA Version: {gpu.cuda_version}")
            
            if gpu.is_nvidia_5090:
                print(f"  ✓ NVIDIA RTX 5090 - Optimizations Available")
            elif gpu.is_nvidia_4090:
                print(f"  ✓ NVIDIA RTX 4090 - Optimizations Available")
        
        print(f"{'='*60}\n")
