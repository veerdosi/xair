"""
Device detection utilities for the XAIR system.
"""

import logging
import torch
from typing import Tuple, Union, Optional

logger = logging.getLogger(__name__)

def get_optimal_device(
    requested_device: str = "auto",
    mps_fallback_to_cpu: bool = False,
    warn_on_fallback: bool = True
) -> Tuple[str, torch.dtype]:
    """
    Determine the optimal device and dtype for the current environment.
    
    Args:
        requested_device: Requested device ('cuda', 'mps', 'cpu', or 'auto')
        mps_fallback_to_cpu: Whether to fallback to CPU if MPS encountered issues
        warn_on_fallback: Whether to log a warning when falling back to CPU
        
    Returns:
        Tuple of (device, dtype)
    """
    device = "cpu"
    dtype = torch.float32
    
    # Auto-detect device if requested
    if requested_device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16  # Default to half precision on CUDA
            logger.info("Auto-selected CUDA device with float16")
        elif hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Test MPS availability
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                device = "mps"
                dtype = torch.float16  # Half precision on MPS
                logger.info("Auto-selected MPS device with float16")
            except Exception as e:
                if warn_on_fallback:
                    logger.warning(f"MPS is available but encountered an error: {e}")
                if mps_fallback_to_cpu:
                    device = "cpu"
                    dtype = torch.float32
                    if warn_on_fallback:
                        logger.warning("Falling back to CPU with float32")
                else:
                    device = "mps"
                    dtype = torch.float32
                    logger.info("Using MPS with float32 due to error")
        else:
            device = "cpu"
            dtype = torch.float32
            logger.info("Auto-selected CPU device with float32")
    else:
        # Use the requested device
        device = requested_device
        
        # Set dtype based on device
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.float16
            logger.info(f"Using requested CUDA device with float16")
        elif device == "mps" and hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Test MPS availability
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                dtype = torch.float16
                logger.info(f"Using requested MPS device with float16")
            except Exception as e:
                if warn_on_fallback:
                    logger.warning(f"MPS is available but encountered an error: {e}")
                if mps_fallback_to_cpu:
                    device = "cpu"
                    dtype = torch.float32
                    if warn_on_fallback:
                        logger.warning("Falling back to CPU with float32")
                else:
                    dtype = torch.float32
                    logger.info("Using MPS with float32 due to error")
        else:
            # For CPU or if requested device is not available
            if device != "cpu" and warn_on_fallback:
                logger.warning(f"Requested device '{device}' is not available, using CPU")
            device = "cpu"
            dtype = torch.float32
            logger.info(f"Using CPU device with float32")
    
    return device, dtype

def get_memory_info(device: str = "auto") -> dict:
    """
    Get memory information for the current device.
    
    Args:
        device: Device to get memory info for
        
    Returns:
        Dictionary with memory information
    """
    if device == "auto":
        device, _ = get_optimal_device()
    
    memory_info = {
        "device": device,
        "total_memory_mb": 0,
        "used_memory_mb": 0,
        "free_memory_mb": 0,
        "platform": "unknown"
    }
    
    try:
        import platform
        import psutil
        
        memory_info["platform"] = platform.system()
        
        if device == "cuda" and torch.cuda.is_available():
            # CUDA memory info
            current_device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(current_device).total_memory
            reserved_memory = torch.cuda.memory_reserved(current_device)
            allocated_memory = torch.cuda.memory_allocated(current_device)
            free_memory = total_memory - reserved_memory
            
            memory_info["total_memory_mb"] = total_memory / (1024 ** 2)
            memory_info["used_memory_mb"] = allocated_memory / (1024 ** 2)
            memory_info["free_memory_mb"] = free_memory / (1024 ** 2)
        elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't provide memory info, use system memory as proxy
            vm = psutil.virtual_memory()
            memory_info["total_memory_mb"] = vm.total / (1024 ** 2)
            memory_info["used_memory_mb"] = vm.used / (1024 ** 2)
            memory_info["free_memory_mb"] = vm.available / (1024 ** 2)
        else:
            # CPU memory info
            vm = psutil.virtual_memory()
            memory_info["total_memory_mb"] = vm.total / (1024 ** 2)
            memory_info["used_memory_mb"] = vm.used / (1024 ** 2)
            memory_info["free_memory_mb"] = vm.available / (1024 ** 2)
    except ImportError:
        logger.warning("psutil not available, memory info will be limited")
    except Exception as e:
        logger.warning(f"Error getting memory info: {e}")
    
    return memory_info

def optimize_for_device(device: str) -> dict:
    """
    Get device-specific optimization settings.
    
    Args:
        device: Device type ('cuda', 'mps', 'cpu')
        
    Returns:
        Dictionary with optimization settings
    """
    # Default settings
    settings = {
        "use_fp16": False,
        "use_bettertransformer": False,
        "use_cpu_offloading": False,
        "max_batch_size": 1,
        "recommended_max_tokens": 512
    }
    
    if device == "cuda":
        # CUDA optimizations
        settings["use_fp16"] = True
        settings["use_bettertransformer"] = True
        settings["max_batch_size"] = 4
        settings["recommended_max_tokens"] = 1024
        
        # Check CUDA memory and adjust settings
        try:
            current_device = torch.cuda.current_device()
            total_memory_gb = torch.cuda.get_device_properties(current_device).total_memory / (1024 ** 3)
            
            if total_memory_gb < 8:
                # Low VRAM settings
                settings["max_batch_size"] = 1
                settings["recommended_max_tokens"] = 512
            elif total_memory_gb < 16:
                # Medium VRAM settings
                settings["max_batch_size"] = 2
                settings["recommended_max_tokens"] = 768
            # High VRAM settings are the defaults
        except Exception:
            pass
    
    elif device == "mps":
        # MPS optimizations for Mac
        settings["use_fp16"] = True
        settings["use_bettertransformer"] = False  # BetterTransformer may have issues on MPS
        settings["max_batch_size"] = 1
        settings["recommended_max_tokens"] = 256
    
    elif device == "cpu":
        # CPU optimizations
        settings["use_fp16"] = False
        settings["use_bettertransformer"] = False
        settings["max_batch_size"] = 1
        settings["recommended_max_tokens"] = 256
        
        # Check CPU memory
        try:
            import psutil
            total_memory_gb = psutil.virtual_memory().total / (1024 ** 3)
            
            if total_memory_gb > 32:
                # High memory CPU
                settings["recommended_max_tokens"] = 512
        except ImportError:
            pass
    
    return settings