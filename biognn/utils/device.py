"""
Device management utilities for CPU/CUDA/MPS support.

Automatically detects and selects the best available device for training and inference.
Supports CUDA (NVIDIA GPUs), MPS (Apple Silicon), and CPU fallback.
"""

import torch
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


def get_device(device: Optional[Union[str, torch.device]] = None, verbose: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch operations.

    Priority order:
    1. User-specified device (if provided and available)
    2. CUDA (NVIDIA GPU)
    3. MPS (Apple Silicon GPU - M1/M2/M3)
    4. CPU

    Args:
        device: Optional device specification ('cuda', 'mps', 'cpu', 'cuda:0', etc.)
                If None, automatically selects best available device
        verbose: If True, print device information

    Returns:
        torch.device: Selected device

    Examples:
        >>> # Auto-detect best device
        >>> device = get_device()
        Using device: cuda (NVIDIA GeForce RTX 3090)

        >>> # Force CPU
        >>> device = get_device('cpu')
        Using device: cpu

        >>> # Specific GPU
        >>> device = get_device('cuda:1')
        Using device: cuda:1 (NVIDIA GeForce RTX 3080)
    """
    # If device explicitly specified, try to use it
    if device is not None:
        if isinstance(device, torch.device):
            return device

        device_str = str(device).lower()

        # Check CUDA
        if device_str.startswith('cuda'):
            if torch.cuda.is_available():
                selected_device = torch.device(device_str)
                if verbose:
                    gpu_name = torch.cuda.get_device_name(selected_device)
                    logger.info(f"Using device: {device_str} ({gpu_name})")
                return selected_device
            else:
                logger.warning(f"CUDA requested but not available. Falling back to CPU.")
                return torch.device('cpu')

        # Check MPS
        elif device_str == 'mps':
            if torch.backends.mps.is_available():
                if verbose:
                    logger.info("Using device: mps (Apple Silicon GPU)")
                return torch.device('mps')
            else:
                logger.warning("MPS requested but not available. Falling back to CPU.")
                return torch.device('cpu')

        # CPU
        elif device_str == 'cpu':
            if verbose:
                logger.info("Using device: cpu")
            return torch.device('cpu')

        else:
            logger.warning(f"Unknown device '{device}'. Falling back to auto-detection.")

    # Auto-detect best device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_count = torch.cuda.device_count()
            logger.info(f"Using device: cuda ({gpu_name})")
            if gpu_count > 1:
                logger.info(f"  {gpu_count} GPUs available. Use device='cuda:N' for specific GPU.")
        return device

    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            logger.info("Using device: mps (Apple Silicon GPU)")
            logger.info("  Note: MPS support is experimental in PyTorch")
        return device

    else:
        device = torch.device('cpu')
        if verbose:
            logger.info("Using device: cpu")
            logger.info("  For better performance, consider using a machine with GPU")
        return device


def get_device_info(device: Optional[torch.device] = None) -> dict:
    """
    Get detailed information about a device.

    Args:
        device: Device to query. If None, uses current default device.

    Returns:
        dict: Device information including type, name, memory, etc.
    """
    if device is None:
        device = get_device(verbose=False)

    info = {
        'type': device.type,
        'index': device.index if device.index is not None else 0,
    }

    if device.type == 'cuda':
        info['name'] = torch.cuda.get_device_name(device)
        info['capability'] = torch.cuda.get_device_capability(device)
        info['total_memory'] = torch.cuda.get_device_properties(device).total_memory
        info['total_memory_gb'] = torch.cuda.get_device_properties(device).total_memory / 1e9
        info['available'] = True
        info['count'] = torch.cuda.device_count()

    elif device.type == 'mps':
        info['name'] = 'Apple Silicon GPU'
        info['available'] = torch.backends.mps.is_available()
        info['built'] = torch.backends.mps.is_built()

    elif device.type == 'cpu':
        import platform
        info['name'] = platform.processor() or 'CPU'
        info['cores'] = torch.get_num_threads()
        info['interop_threads'] = torch.get_num_interop_threads()

    return info


def print_device_info(device: Optional[torch.device] = None):
    """
    Print detailed device information.

    Args:
        device: Device to print info for. If None, uses current default device.
    """
    if device is None:
        device = get_device(verbose=False)

    info = get_device_info(device)

    print("=" * 60)
    print("Device Information")
    print("=" * 60)
    print(f"Type: {info['type'].upper()}")

    if info['type'] == 'cuda':
        print(f"Name: {info['name']}")
        print(f"Index: {info['index']}")
        print(f"Compute Capability: {info['capability'][0]}.{info['capability'][1]}")
        print(f"Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"Total GPUs: {info['count']}")

    elif info['type'] == 'mps':
        print(f"Name: {info['name']}")
        print(f"Available: {info['available']}")
        print(f"Built: {info['built']}")

    elif info['type'] == 'cpu':
        print(f"Processor: {info['name']}")
        print(f"Threads: {info['cores']}")
        print(f"Interop Threads: {info['interop_threads']}")

    print("=" * 60)


def optimize_for_device(device: torch.device) -> dict:
    """
    Get recommended settings for a given device.

    Args:
        device: Target device

    Returns:
        dict: Recommended settings (batch_size, num_workers, etc.)
    """
    settings = {}

    if device.type == 'cuda':
        # NVIDIA GPU settings
        info = get_device_info(device)
        memory_gb = info['total_memory_gb']

        if memory_gb >= 24:  # High-end GPU (RTX 3090, A100, etc.)
            settings['batch_size'] = 64
            settings['num_workers'] = 8
            settings['pin_memory'] = True
            settings['enable_amp'] = True
        elif memory_gb >= 11:  # Mid-range GPU (RTX 2080Ti, 3080, etc.)
            settings['batch_size'] = 32
            settings['num_workers'] = 4
            settings['pin_memory'] = True
            settings['enable_amp'] = True
        else:  # Lower-end GPU
            settings['batch_size'] = 16
            settings['num_workers'] = 2
            settings['pin_memory'] = True
            settings['enable_amp'] = True

    elif device.type == 'mps':
        # Apple Silicon settings
        settings['batch_size'] = 16
        settings['num_workers'] = 4
        settings['pin_memory'] = False  # Not needed for MPS
        settings['enable_amp'] = False  # AMP support limited on MPS

    else:  # CPU
        # CPU settings - more conservative
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()

        settings['batch_size'] = 8  # Smaller batch for CPU
        settings['num_workers'] = min(4, cpu_count // 2)  # Don't use all cores
        settings['pin_memory'] = False
        settings['enable_amp'] = False
        settings['gradient_accumulation_steps'] = 4  # Simulate larger batch

    return settings


def move_to_device(obj, device: torch.device, non_blocking: bool = False):
    """
    Recursively move tensors/models to device.

    Args:
        obj: Object to move (tensor, model, dict, list, etc.)
        device: Target device
        non_blocking: Whether to use non-blocking transfers

    Returns:
        Object moved to device
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device, non_blocking) for item in obj)
    else:
        return obj


# Compatibility check
def check_compatibility():
    """
    Check PyTorch installation and device compatibility.
    Prints warnings if there are potential issues.
    """
    print("=" * 60)
    print("PyTorch Compatibility Check")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    print(f"MPS available: {torch.backends.mps.is_available()}")
    if hasattr(torch.backends.mps, 'is_built'):
        print(f"MPS built: {torch.backends.mps.is_built()}")

    # Check for common issues
    warnings = []

    if not torch.cuda.is_available() and not torch.backends.mps.is_available():
        warnings.append("⚠️  No GPU detected. Training will use CPU (slower).")

    if torch.backends.mps.is_available():
        warnings.append("ℹ️  MPS (Apple Silicon) detected. Note: Some operations may fall back to CPU.")

    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  {warning}")

    print("=" * 60)


if __name__ == '__main__':
    # Test device detection
    check_compatibility()
    print()
    device = get_device()
    print()
    print_device_info(device)
    print()
    settings = optimize_for_device(device)
    print("Recommended settings:")
    for key, value in settings.items():
        print(f"  {key}: {value}")
