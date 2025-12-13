#!/usr/bin/env python3
"""
Benchmark: PyO3 Bindings vs Pure Python Implementation

This script compares the performance of Stone Soup's PyO3 native bindings
against the pure Python implementation for common tracking operations.

Usage:
    python benchmark_pyo3.py [--iterations N] [--warmup W]

Results are output in a markdown table format suitable for documentation.
"""

import argparse
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# Try to import PyO3 bindings
try:
    import stonesoup_native  # PyO3 bindings

    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    print("Warning: Native bindings not available. Install with: pip install stonesoup-native")

# Import pure Python Stone Soup
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.models.transition.linear import (
    CombinedLinearGaussianTransitionModel,
    ConstantVelocity,
)
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.types.array import CovarianceMatrix, StateVector
from stonesoup.types.state import GaussianState
from stonesoup.updater.kalman import KalmanUpdater


@dataclass
class BenchmarkResult:
    """Results from a single benchmark"""

    name: str
    pure_python_ms: float
    native_ms: float | None
    speedup: float | None
    iterations: int


def time_function(func: Callable, iterations: int, warmup: int = 10) -> float:
    """Time a function over multiple iterations, return average in milliseconds"""
    # Warmup
    for _ in range(warmup):
        func()

    # Actual timing
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append((end - start) / 1e6)  # Convert to ms

    return statistics.mean(times)


class PythonKalmanBenchmark:
    """Pure Python Kalman filter implementation for benchmarking"""

    def __init__(self, state_dim: int = 4, meas_dim: int = 2):
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # Create constant velocity model (2D)
        self.transition_model = CombinedLinearGaussianTransitionModel(
            [ConstantVelocity(0.1), ConstantVelocity(0.1)]
        )

        # Create measurement model (observe position only)
        self.measurement_model = LinearGaussian(
            ndim_state=state_dim, mapping=(0, 2), noise_covar=np.eye(meas_dim) * 0.5
        )

        # Create predictor and updater
        self.predictor = KalmanPredictor(self.transition_model)
        self.updater = KalmanUpdater(self.measurement_model)

        # Create initial state
        self.prior = GaussianState(
            state_vector=StateVector([0, 1, 0, 1]), covar=CovarianceMatrix(np.eye(state_dim))
        )

    def predict(self):
        """Run prediction step"""
        return self.predictor.predict(self.prior, timestamp=1.0)

    def update(self, predicted):
        """Run update step"""
        from stonesoup.types.detection import Detection
        from stonesoup.types.hypothesis import SingleHypothesis

        detection = Detection(StateVector([1.0, 1.0]))
        hypothesis = SingleHypothesis(predicted, detection)
        return self.updater.update(hypothesis)

    def full_cycle(self):
        """Run full predict-update cycle"""
        predicted = self.predict()
        return self.update(predicted)


class NativeKalmanBenchmark:
    """Native PyO3 Kalman filter benchmark"""

    def __init__(self, state_dim: int = 4, meas_dim: int = 2):
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # Initialize native state
        self.state_vector = np.array([0.0, 1.0, 0.0, 1.0])
        self.covariance = np.eye(state_dim)

        # Transition matrix for constant velocity
        dt = 1.0
        self.F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        # Process noise
        self.Q = np.eye(state_dim) * 0.1

        # Measurement matrix (observe x and y)
        self.H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

        # Measurement noise
        self.R = np.eye(meas_dim) * 0.5

    def predict(self):
        """Run native prediction step"""
        if not HAS_NATIVE:
            return None

        return stonesoup_native.kalman_predict(self.state_vector, self.covariance, self.F, self.Q)

    def update(self, predicted_state, predicted_cov):
        """Run native update step"""
        if not HAS_NATIVE:
            return None

        measurement = np.array([1.0, 1.0])
        return stonesoup_native.kalman_update(
            predicted_state, predicted_cov, measurement, self.H, self.R
        )

    def full_cycle(self):
        """Run full predict-update cycle"""
        if not HAS_NATIVE:
            return None

        pred_state, pred_cov = self.predict()
        return self.update(pred_state, pred_cov)


class StateVectorBenchmark:
    """Benchmark StateVector operations"""

    def __init__(self, dim: int = 100):
        self.dim = dim
        self.data1 = np.random.randn(dim)
        self.data2 = np.random.randn(dim)

    def python_add(self):
        sv1 = StateVector(self.data1)
        sv2 = StateVector(self.data2)
        return sv1 + sv2

    def python_norm(self):
        sv = StateVector(self.data1)
        return np.linalg.norm(sv)

    def native_add(self):
        if not HAS_NATIVE:
            return None
        return stonesoup_native.state_vector_add(self.data1, self.data2)

    def native_norm(self):
        if not HAS_NATIVE:
            return None
        return stonesoup_native.state_vector_norm(self.data1)


class MatrixBenchmark:
    """Benchmark matrix operations"""

    def __init__(self, dim: int = 50):
        self.dim = dim
        # Create positive definite matrices
        A = np.random.randn(dim, dim)
        B = np.random.randn(dim, dim)
        self.mat1 = A @ A.T + np.eye(dim)
        self.mat2 = B @ B.T + np.eye(dim)

    def python_multiply(self):
        return self.mat1 @ self.mat2

    def python_inverse(self):
        return np.linalg.inv(self.mat1)

    def python_cholesky(self):
        return np.linalg.cholesky(self.mat1)

    def native_multiply(self):
        if not HAS_NATIVE:
            return None
        return stonesoup_native.matrix_multiply(self.mat1, self.mat2)

    def native_inverse(self):
        if not HAS_NATIVE:
            return None
        return stonesoup_native.matrix_inverse(self.mat1)

    def native_cholesky(self):
        if not HAS_NATIVE:
            return None
        return stonesoup_native.matrix_cholesky(self.mat1)


class BatchProcessingBenchmark:
    """Benchmark batch processing of large arrays - where PyO3 excels"""

    def __init__(self, batch_size: int = 1000, state_dim: int = 6):
        self.batch_size = batch_size
        self.state_dim = state_dim
        # Pre-generate batch data
        self.state_vectors = [np.random.randn(state_dim) for _ in range(batch_size)]
        self.covariances = [np.eye(state_dim) * (i + 1) for i in range(batch_size)]
        # Single large array for vectorized operations
        self.large_array = np.random.randn(batch_size, state_dim)
        self.weights = np.random.rand(batch_size)
        self.weights /= self.weights.sum()

    def python_batch_norms(self):
        """Compute norms of batch of state vectors using pure Python loop"""
        return [np.linalg.norm(sv) for sv in self.state_vectors]

    def python_weighted_mean(self):
        """Compute weighted mean of state vectors"""
        result = np.zeros(self.state_dim)
        for i, sv in enumerate(self.state_vectors):
            result += self.weights[i] * sv
        return result

    def python_vectorized_norms(self):
        """Compute norms using NumPy vectorized operations"""
        return np.linalg.norm(self.large_array, axis=1)

    def python_covariance_traces(self):
        """Compute traces of batch of covariance matrices"""
        return [np.trace(cov) for cov in self.covariances]

    def native_batch_norms(self):
        """Native batch norm computation"""
        if not HAS_NATIVE:
            return None
        return stonesoup_native.batch_norms(self.state_vectors)

    def native_weighted_mean(self):
        """Native weighted mean computation"""
        if not HAS_NATIVE:
            return None
        return stonesoup_native.weighted_mean(self.state_vectors, self.weights)

    def native_vectorized_norms(self):
        """Native vectorized norm computation"""
        if not HAS_NATIVE:
            return None
        return stonesoup_native.matrix_row_norms(self.large_array)

    def native_covariance_traces(self):
        """Native batch trace computation"""
        if not HAS_NATIVE:
            return None
        return stonesoup_native.batch_traces(self.covariances)


def run_benchmarks(iterations: int = 1000, warmup: int = 100) -> list[BenchmarkResult]:
    """Run all benchmarks and return results"""
    results = []

    print(f"Running benchmarks with {iterations} iterations, {warmup} warmup...")
    print()

    # StateVector benchmarks
    print("Benchmarking StateVector operations...")
    sv_bench = StateVectorBenchmark(dim=100)

    py_time = time_function(sv_bench.python_add, iterations, warmup)
    native_time = time_function(sv_bench.native_add, iterations, warmup) if HAS_NATIVE else None
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("StateVector add (dim=100)", py_time, native_time, speedup, iterations)
    )

    py_time = time_function(sv_bench.python_norm, iterations, warmup)
    native_time = time_function(sv_bench.native_norm, iterations, warmup) if HAS_NATIVE else None
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("StateVector norm (dim=100)", py_time, native_time, speedup, iterations)
    )

    # Matrix benchmarks
    print("Benchmarking Matrix operations...")
    mat_bench = MatrixBenchmark(dim=50)

    py_time = time_function(mat_bench.python_multiply, iterations, warmup)
    native_time = (
        time_function(mat_bench.native_multiply, iterations, warmup) if HAS_NATIVE else None
    )
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Matrix multiply (50x50)", py_time, native_time, speedup, iterations)
    )

    py_time = time_function(mat_bench.python_inverse, iterations, warmup)
    native_time = (
        time_function(mat_bench.native_inverse, iterations, warmup) if HAS_NATIVE else None
    )
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Matrix inverse (50x50)", py_time, native_time, speedup, iterations)
    )

    py_time = time_function(mat_bench.python_cholesky, iterations, warmup)
    native_time = (
        time_function(mat_bench.native_cholesky, iterations, warmup) if HAS_NATIVE else None
    )
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Matrix Cholesky (50x50)", py_time, native_time, speedup, iterations)
    )

    # Kalman filter benchmarks
    print("Benchmarking Kalman filter operations...")
    py_kalman = PythonKalmanBenchmark()
    native_kalman = NativeKalmanBenchmark()

    py_time = time_function(
        py_kalman.predict, iterations // 10, warmup // 10
    )  # Fewer iterations (slower)
    native_time = time_function(native_kalman.predict, iterations, warmup) if HAS_NATIVE else None
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Kalman predict (4D state)", py_time, native_time, speedup, iterations)
    )

    py_time = time_function(py_kalman.full_cycle, iterations // 10, warmup // 10)
    native_time = (
        time_function(native_kalman.full_cycle, iterations, warmup) if HAS_NATIVE else None
    )
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Kalman full cycle (4D)", py_time, native_time, speedup, iterations)
    )

    # Batch processing benchmarks - where PyO3 excels
    print("Benchmarking Batch processing operations...")
    batch_bench = BatchProcessingBenchmark(batch_size=1000, state_dim=6)

    py_time = time_function(batch_bench.python_batch_norms, iterations // 10, warmup // 10)
    native_time = (
        time_function(batch_bench.native_batch_norms, iterations // 10, warmup // 10)
        if HAS_NATIVE
        else None
    )
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Batch norms (1000x6)", py_time, native_time, speedup, iterations // 10)
    )

    py_time = time_function(batch_bench.python_weighted_mean, iterations // 10, warmup // 10)
    native_time = (
        time_function(batch_bench.native_weighted_mean, iterations // 10, warmup // 10)
        if HAS_NATIVE
        else None
    )
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Weighted mean (1000x6)", py_time, native_time, speedup, iterations // 10)
    )

    py_time = time_function(batch_bench.python_vectorized_norms, iterations, warmup)
    native_time = (
        time_function(batch_bench.native_vectorized_norms, iterations, warmup)
        if HAS_NATIVE
        else None
    )
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Vectorized norms (1000x6)", py_time, native_time, speedup, iterations)
    )

    py_time = time_function(batch_bench.python_covariance_traces, iterations // 10, warmup // 10)
    native_time = (
        time_function(batch_bench.native_covariance_traces, iterations // 10, warmup // 10)
        if HAS_NATIVE
        else None
    )
    speedup = py_time / native_time if native_time else None
    results.append(
        BenchmarkResult("Batch traces (1000x6x6)", py_time, native_time, speedup, iterations // 10)
    )

    return results


def format_results_markdown(results: list[BenchmarkResult]) -> str:
    """Format results as a markdown table"""
    lines = [
        "## Benchmark Results: PyO3 vs Pure Python",
        "",
        "| Operation | Pure Python (ms) | Native (ms) | Speedup |",
        "|-----------|-----------------|-------------|---------|",
    ]

    for r in results:
        py_str = f"{r.pure_python_ms:.4f}"
        native_str = f"{r.native_ms:.4f}" if r.native_ms else "N/A"
        speedup_str = f"{r.speedup:.2f}x" if r.speedup else "N/A"
        lines.append(f"| {r.name} | {py_str} | {native_str} | {speedup_str} |")

    lines.extend(
        [
            "",
            f"*Benchmarks run with {results[0].iterations if results else 0} iterations*",
            "",
            "### Notes",
            "",
            "- Pure Python uses NumPy for underlying array operations",
            "- Native bindings use Rust/C implementations via PyO3",
            "- Speedup > 1.0x means native is faster",
            "- Kalman filter benchmarks include full object creation overhead for Python",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyO3 vs Pure Python")
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=1000,
        help="Number of iterations per benchmark (default: 1000)",
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=100, help="Number of warmup iterations (default: 100)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output file for results (default: stdout)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Stone Soup Benchmark: PyO3 Native vs Pure Python")
    print("=" * 60)
    print()

    if not HAS_NATIVE:
        print("WARNING: Native bindings not available!")
        print("Only pure Python benchmarks will be run.")
        print()

    results = run_benchmarks(args.iterations, args.warmup)

    print()
    print("=" * 60)
    print()

    output = format_results_markdown(results)
    print(output)

    if args.output:
        from pathlib import Path

        Path(args.output).write_text(output)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
