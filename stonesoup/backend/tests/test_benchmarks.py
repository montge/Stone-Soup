# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Benchmark tests for backend operations.

These tests verify performance characteristics and detect regressions.
They are marked with @pytest.mark.benchmark for optional execution.
"""

import numpy as np
import pytest

from stonesoup.backend import is_gpu_available, numpy_backend

# Performance thresholds (operations per second, minimum acceptable)
THRESHOLDS = {
    "kalman_predict_1000": 100,  # At least 100 ops/sec for 1000 batch
    "kalman_update_1000": 50,  # At least 50 ops/sec for 1000 batch
    "resample_10000": 500,  # At least 500 ops/sec for 10000 particles
    "matrix_multiply_100x100": 1000,  # At least 1000 ops/sec
}


@pytest.fixture
def kalman_data_small():
    """Small Kalman filter test data (2D state)."""
    np.random.seed(42)
    return {
        "x": np.array([0.0, 1.0]),
        "P": np.eye(2) * 0.1,
        "F": np.array([[1, 1], [0, 1]], dtype=float),
        "Q": np.array([[0.1, 0], [0, 0.1]], dtype=float),
        "H": np.array([[1, 0]], dtype=float),
        "R": np.array([[0.5]], dtype=float),
        "z": np.array([1.2]),
    }


@pytest.fixture
def kalman_data_batch():
    """Batch Kalman filter test data."""
    np.random.seed(42)
    batch_size = 1000
    state_dim = 6
    return {
        "x_batch": np.random.randn(batch_size, state_dim),
        "P_batch": np.tile(np.eye(state_dim) * 0.1, (batch_size, 1, 1)),
        "F": np.eye(state_dim),
        "Q": np.eye(state_dim) * 0.01,
    }


@pytest.mark.benchmark
def test_benchmark_kalman_predict(benchmark, kalman_data_small):
    """Benchmark single Kalman prediction."""
    data = kalman_data_small

    def run():
        return numpy_backend.kalman_predict(data["x"], data["P"], data["F"], data["Q"])

    result = benchmark(run)
    assert result is not None


@pytest.mark.benchmark
def test_benchmark_kalman_update(benchmark, kalman_data_small):
    """Benchmark single Kalman update."""
    data = kalman_data_small
    x_pred, P_pred = numpy_backend.kalman_predict(data["x"], data["P"], data["F"], data["Q"])

    def run():
        return numpy_backend.kalman_update(x_pred, P_pred, data["z"], data["H"], data["R"])

    result = benchmark(run)
    assert result is not None


@pytest.mark.benchmark
def test_benchmark_batch_kalman_predict(benchmark, kalman_data_batch):
    """Benchmark batch Kalman prediction (1000 states)."""
    data = kalman_data_batch

    def run():
        return numpy_backend.batch_kalman_predict(
            data["x_batch"], data["P_batch"], data["F"], data["Q"]
        )

    result = benchmark(run)
    x_pred, P_pred = result
    assert x_pred.shape == data["x_batch"].shape


@pytest.mark.benchmark
def test_benchmark_systematic_resample(benchmark):
    """Benchmark systematic resampling (10000 particles)."""
    np.random.seed(42)
    n = 10000
    weights = np.random.rand(n)
    weights /= weights.sum()

    def run():
        return numpy_backend.systematic_resample(weights)

    result = benchmark(run)
    assert len(result) == n


@pytest.mark.benchmark
def test_benchmark_multinomial_resample(benchmark):
    """Benchmark multinomial resampling (10000 particles)."""
    np.random.seed(42)
    n = 10000
    weights = np.random.rand(n)
    weights /= weights.sum()

    def run():
        return numpy_backend.multinomial_resample(weights)

    result = benchmark(run)
    assert len(result) == n


@pytest.mark.benchmark
def test_benchmark_matrix_multiply(benchmark):
    """Benchmark 100x100 matrix multiplication."""
    np.random.seed(42)
    A = np.random.randn(100, 100)
    B = np.random.randn(100, 100)

    def run():
        return numpy_backend.matrix_multiply(A, B)

    result = benchmark(run)
    assert result.shape == (100, 100)


@pytest.mark.benchmark
def test_benchmark_matrix_inverse(benchmark):
    """Benchmark 50x50 matrix inverse."""
    np.random.seed(42)
    # Create positive definite matrix
    A = np.random.randn(50, 50)
    A = A @ A.T + np.eye(50) * 0.1

    def run():
        return numpy_backend.matrix_inverse(A)

    result = benchmark(run)
    assert result.shape == (50, 50)


@pytest.mark.benchmark
def test_benchmark_cholesky(benchmark):
    """Benchmark 50x50 Cholesky decomposition."""
    np.random.seed(42)
    # Create positive definite matrix
    A = np.random.randn(50, 50)
    A = A @ A.T + np.eye(50) * 0.1

    def run():
        return numpy_backend.cholesky_decomposition(A)

    result = benchmark(run)
    assert result.shape == (50, 50)


# GPU Benchmarks (only run if GPU available)


@pytest.fixture
def require_gpu():
    """Skip if GPU not available."""
    if not is_gpu_available():
        pytest.skip("GPU not available")


@pytest.mark.benchmark
def test_benchmark_gpu_batch_kalman_predict(benchmark, kalman_data_batch, require_gpu):
    """Benchmark GPU batch Kalman prediction."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    data = kalman_data_batch
    x_gpu = cp.asarray(data["x_batch"])
    P_gpu = cp.asarray(data["P_batch"])
    F_gpu = cp.asarray(data["F"])
    Q_gpu = cp.asarray(data["Q"])

    def run():
        result = cupy_backend.batch_kalman_predict(x_gpu, P_gpu, F_gpu, Q_gpu)
        cp.cuda.Stream.null.synchronize()
        return result

    result = benchmark(run)
    x_pred, P_pred = result
    assert x_pred.shape == data["x_batch"].shape


@pytest.mark.benchmark
def test_benchmark_gpu_systematic_resample(benchmark, require_gpu):
    """Benchmark GPU systematic resampling."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    np.random.seed(42)
    n = 10000
    weights = np.random.rand(n)
    weights /= weights.sum()
    weights_gpu = cp.asarray(weights)

    def run():
        result = cupy_backend.systematic_resample(weights_gpu)
        cp.cuda.Stream.null.synchronize()
        return result

    result = benchmark(run)
    assert len(result) == n


@pytest.mark.benchmark
def test_benchmark_gpu_matrix_multiply(benchmark, require_gpu):
    """Benchmark GPU 100x100 matrix multiplication."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    np.random.seed(42)
    A = cp.asarray(np.random.randn(100, 100))
    B = cp.asarray(np.random.randn(100, 100))

    def run():
        result = cupy_backend.matrix_multiply(A, B)
        cp.cuda.Stream.null.synchronize()
        return result

    result = benchmark(run)
    assert result.shape == (100, 100)


# Regression tests - verify minimum performance


def test_regression_batch_kalman_performance(kalman_data_batch):
    """Verify batch Kalman meets minimum performance threshold."""
    import time

    data = kalman_data_batch

    # Warmup
    for _ in range(3):
        numpy_backend.batch_kalman_predict(data["x_batch"], data["P_batch"], data["F"], data["Q"])

    # Timed runs
    n_runs = 10
    start = time.perf_counter()
    for _ in range(n_runs):
        numpy_backend.batch_kalman_predict(data["x_batch"], data["P_batch"], data["F"], data["Q"])
    elapsed = time.perf_counter() - start

    ops_per_sec = n_runs / elapsed
    min_threshold = THRESHOLDS["kalman_predict_1000"]

    assert ops_per_sec >= min_threshold, (
        f"Batch Kalman predict performance regression: "
        f"{ops_per_sec:.1f} ops/sec < {min_threshold} threshold"
    )


def test_regression_resample_performance():
    """Verify resampling meets minimum performance threshold."""
    import time

    np.random.seed(42)
    n = 10000
    weights = np.random.rand(n)
    weights /= weights.sum()

    # Warmup
    for _ in range(3):
        numpy_backend.systematic_resample(weights)

    # Timed runs
    n_runs = 50
    start = time.perf_counter()
    for _ in range(n_runs):
        numpy_backend.systematic_resample(weights)
    elapsed = time.perf_counter() - start

    ops_per_sec = n_runs / elapsed
    min_threshold = THRESHOLDS["resample_10000"]

    assert ops_per_sec >= min_threshold, (
        f"Resample performance regression: "
        f"{ops_per_sec:.1f} ops/sec < {min_threshold} threshold"
    )
