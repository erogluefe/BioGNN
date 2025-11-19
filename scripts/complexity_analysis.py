#!/usr/bin/env python3
"""
Complexity analysis: time, memory, FLOPs

Analyzes computational requirements of models
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
import numpy as np
from thop import profile, clever_format


def measure_inference_time(model, input_data, num_runs=100, warmup=10):
    """Measure average inference time"""
    model.eval()
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_data)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.time()
            _ = model(input_data)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end = time.time()
            times.append((end - start) * 1000)  # ms

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'fps': 1000 / np.mean(times)
    }


def measure_memory_usage(model, input_data):
    """Measure GPU memory usage"""
    if not torch.cuda.is_available():
        return {'peak_mb': 0}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():
        _ = model(input_data)

    torch.cuda.synchronize()
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

    return {'peak_mb': peak_memory}


def count_parameters(model):
    """Count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total': total,
        'trainable': trainable,
        'total_mb': total * 4 / 1024**2,  # Assuming float32
        'trainable_mb': trainable * 4 / 1024**2
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    print("Complexity Analysis")
    print("=" * 60)

    # Load model and create dummy input
    # model = load_model(args.checkpoint)
    # input_data = create_dummy_input(args.batch_size)

    # params = count_parameters(model)
    # print(f"Parameters: {params['total']:,} ({params['total_mb']:.2f} MB)")

    # timing = measure_inference_time(model, input_data)
    # print(f"Inference time: {timing['mean_ms']:.2f} ± {timing['std_ms']:.2f} ms")
    # print(f"FPS: {timing['fps']:.2f}")

    # memory = measure_memory_usage(model, input_data)
    # print(f"Peak memory: {memory['peak_mb']:.2f} MB")


if __name__ == '__main__':
    main()
