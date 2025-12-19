#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Local GPU test runner for Stone Soup.

This script runs GPU-accelerated tests and benchmarks locally.
It automatically detects GPU availability and provides detailed output.

Usage:
    python scripts/run_gpu_tests.py           # Run all GPU tests
    python scripts/run_gpu_tests.py --bench   # Run benchmarks
    python scripts/run_gpu_tests.py --info    # Show GPU info only
"""

import argparse
import subprocess
import sys
import time


def check_gpu():
    """Check if GPU is available and return info."""
    try:
        from stonesoup.backend import get_device_info, is_gpu_available

        if not is_gpu_available():
            return None
        return get_device_info()
    except ImportError:
        return None


def print_gpu_info(info):
    """Print GPU information."""
    print("=" * 60)
    print("GPU Information")
    print("=" * 60)
    print(f"Backend: {info['backend']}")
    print(f"GPU Available: {info['gpu_available']}")
    for gpu in info["gpu_devices"]:
        print(f"  Device {gpu['id']}: {gpu['name']}")
        print(f"    Memory: {gpu['total_memory_mb']} MB")
    print()


def run_tests():
    """Run GPU tests."""
    print("=" * 60)
    print("Running GPU Tests")
    print("=" * 60)

    result = subprocess.run(  # noqa: S603 - safe: only runs pytest with fixed args
        [
            sys.executable,
            "-m",
            "pytest",
            "stonesoup/backend/tests/test_backend.py",
            "-v",
            "--tb=short",
        ],
        cwd="/home/e/Development/Stone-Soup",
        check=False,
    )
    return result.returncode


def run_benchmarks():
    """Run GPU benchmarks."""
    print("=" * 60)
    print("GPU vs CPU Benchmarks")
    print("=" * 60)

    try:
        import cupy as cp
        import numpy as np

        from stonesoup.backend import cupy_backend, numpy_backend
    except ImportError as e:
        print(f"Error: {e}")
        return 1

    def benchmark(func, *args, warmup=3, runs=10):
        """Run benchmark with warmup."""
        for _ in range(warmup):
            result = func(*args)
            if hasattr(result[0], "get"):
                cp.cuda.Stream.null.synchronize()

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            result = func(*args)
            if hasattr(result[0], "get"):
                cp.cuda.Stream.null.synchronize()
            times.append(time.perf_counter() - start)

        return np.mean(times) * 1000, np.std(times) * 1000

    print("\nBatch Kalman Predict (6D state)")
    print("-" * 60)
    print(f"{'Batch Size':>12} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Speedup':>10}")
    print("-" * 60)

    for batch_size in [100, 1000, 10000, 50000]:
        state_dim = 6

        # CPU data
        x_cpu = np.random.randn(batch_size, state_dim)
        P_cpu = np.tile(np.eye(state_dim) * 0.1, (batch_size, 1, 1))
        F_cpu = np.eye(state_dim)
        F_cpu[0, 3] = F_cpu[1, 4] = F_cpu[2, 5] = 1.0
        Q_cpu = np.eye(state_dim) * 0.01

        # GPU data
        x_gpu = cp.asarray(x_cpu)
        P_gpu = cp.asarray(P_cpu)
        F_gpu = cp.asarray(F_cpu)
        Q_gpu = cp.asarray(Q_cpu)

        cpu_time, _ = benchmark(numpy_backend.batch_kalman_predict, x_cpu, P_cpu, F_cpu, Q_cpu)
        gpu_time, _ = benchmark(cupy_backend.batch_kalman_predict, x_gpu, P_gpu, F_gpu, Q_gpu)

        speedup = cpu_time / gpu_time
        print(f"{batch_size:>12,} {cpu_time:>12.2f} {gpu_time:>12.2f} {speedup:>9.1f}x")

    print()
    print("Particle Filter Resampling")
    print("-" * 60)
    print(f"{'Particles':>12} {'CPU (ms)':>12} {'GPU (ms)':>12} {'Speedup':>10}")
    print("-" * 60)

    for n_particles in [1000, 10000, 100000]:
        weights_cpu = np.ones(n_particles) / n_particles
        weights_gpu = cp.asarray(weights_cpu)

        cpu_time, _ = benchmark(numpy_backend.systematic_resample, weights_cpu)
        gpu_time, _ = benchmark(cupy_backend.systematic_resample, weights_gpu)

        speedup = cpu_time / gpu_time
        print(f"{n_particles:>12,} {cpu_time:>12.3f} {gpu_time:>12.3f} {speedup:>9.1f}x")

    print()
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Stone Soup GPU Test Runner")
    parser.add_argument("--info", action="store_true", help="Show GPU info only")
    parser.add_argument("--bench", action="store_true", help="Run benchmarks")
    parser.add_argument("--test", action="store_true", help="Run tests (default)")
    args = parser.parse_args()

    # Check GPU
    info = check_gpu()
    if info is None:
        print("ERROR: GPU not available or CuPy not installed")
        print("Install CuPy with: pip install cupy-cuda12x")
        return 1

    print_gpu_info(info)

    if args.info:
        return 0

    if args.bench:
        return run_benchmarks()

    # Default: run tests
    return run_tests()


if __name__ == "__main__":
    sys.exit(main())
