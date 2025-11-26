#!/usr/bin/env python3
"""
Device compatibility checker for BioGNN.

Checks PyTorch installation, available devices, and provides
optimized configuration recommendations.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from biognn.utils import check_compatibility, get_device, get_device_info, optimize_for_device, print_device_info


def main():
    print()
    print("=" * 70)
    print("BioGNN Device Compatibility Checker")
    print("=" * 70)
    print()

    # 1. Check PyTorch installation
    check_compatibility()

    print()

    # 2. Detect and display best device
    device = get_device(verbose=True)

    print()

    # 3. Show detailed device info
    print_device_info(device)

    print()

    # 4. Get recommended settings
    settings = optimize_for_device(device)

    print("=" * 70)
    print("Recommended Training Settings")
    print("=" * 70)
    print()

    for key, value in settings.items():
        print(f"  {key:30s}: {value}")

    print()

    # 5. Provide config recommendation
    print("=" * 70)
    print("Configuration Recommendation")
    print("=" * 70)
    print()

    if device.type == 'cuda':
        info = get_device_info(device)
        memory_gb = info.get('total_memory_gb', 0)

        print("✓ NVIDIA GPU detected!")
        print()
        print("Recommended config files:")
        if memory_gb >= 24:
            print("  • configs/default_config.yaml (high-end GPU)")
            print("  • configs/ensemble_config.yaml (for best performance)")
        elif memory_gb >= 11:
            print("  • configs/gcn_config.yaml (balanced)")
            print("  • configs/default_config.yaml (may need batch_size adjustment)")
        else:
            print("  • configs/gcn_config.yaml (reduce batch_size to 16)")
            print("  • configs/cpu_config.yaml (if out of memory)")

        print()
        print("Training command:")
        print("  python train.py --config configs/gcn_config.yaml --gpu 0")

    elif device.type == 'mps':
        print("✓ Apple Silicon GPU detected!")
        print()
        print("Note: MPS support is experimental in PyTorch.")
        print("Some operations may fall back to CPU.")
        print()
        print("Recommended config files:")
        print("  • configs/cpu_config.yaml (optimized for MPS as well)")
        print()
        print("Training command:")
        print("  python train.py --config configs/cpu_config.yaml --device mps")
        print()
        print("Alternative (auto-detect):")
        print("  python train.py --config configs/cpu_config.yaml")

    else:  # CPU
        import platform
        import multiprocessing

        print("✓ Using CPU")
        print()
        print(f"Processor: {platform.processor() or 'Unknown'}")
        print(f"CPU cores: {multiprocessing.cpu_count()}")
        print(f"PyTorch threads: {torch.get_num_threads()}")
        print()
        print("Recommended config files:")
        print("  • configs/cpu_config.yaml (optimized for CPU)")
        print()
        print("Training command:")
        print("  python train.py --config configs/cpu_config.yaml")
        print()
        print("Performance tips:")
        print("  • Use smaller batch sizes (4-8)")
        print("  • Enable gradient accumulation")
        print("  • Use ResNet18 instead of ResNet50")
        print("  • Reduce image resolutions")
        print("  • Training will be slower, but fully functional")

    print()
    print("=" * 70)
    print("Quick Test")
    print("=" * 70)
    print()

    # Quick tensor test
    print("Testing tensor operations...")
    try:
        x = torch.randn(2, 3, 224, 224).to(device)
        conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)
        y = conv(x)
        print(f"✓ Tensor operations successful on {device}")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
    except Exception as e:
        print(f"✗ Tensor operation failed: {e}")

    print()
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
