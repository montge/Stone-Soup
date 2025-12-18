# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Tests for backend abstraction layer."""

import os

import numpy as np
import pytest

from stonesoup.backend import (
    BACKENDS,
    get_array_module,
    get_backend,
    get_device_info,
    is_gpu_available,
    numpy_backend,
    set_backend,
    to_numpy,
)

# ============================================================================
# Backend Selection Tests
# ============================================================================


@pytest.fixture(autouse=True)
def reset_backend_state():
    """Reset backend state before each test."""
    import stonesoup.backend as backend_module

    backend_module._current_backend = None
    backend_module._array_module = None
    if "STONESOUP_BACKEND" in os.environ:
        del os.environ["STONESOUP_BACKEND"]
    yield
    # Cleanup after test
    backend_module._current_backend = None
    backend_module._array_module = None


def test_default_backend_is_numpy():
    """Default backend should be numpy when forced via env var."""
    os.environ["STONESOUP_BACKEND"] = "numpy"
    import stonesoup.backend as backend_module

    backend_module._current_backend = None
    backend = get_backend()
    assert backend == "numpy"


def test_set_backend_numpy():
    """Can explicitly set numpy backend."""
    set_backend("numpy")
    assert get_backend() == "numpy"


def test_set_backend_invalid():
    """Setting invalid backend raises ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        set_backend("invalid_backend")


def test_get_array_module_numpy():
    """get_array_module returns numpy for numpy backend."""
    set_backend("numpy")
    xp = get_array_module()
    assert xp is np


def test_backends_list():
    """BACKENDS contains expected options."""
    assert "numpy" in BACKENDS
    assert "cupy" in BACKENDS


def test_get_device_info():
    """get_device_info returns dict with backend info."""
    set_backend("numpy")
    info = get_device_info()
    assert isinstance(info, dict)
    assert "backend" in info
    assert "gpu_available" in info
    assert "gpu_devices" in info


def test_to_numpy_passthrough():
    """to_numpy passes through numpy arrays."""
    arr = np.array([1.0, 2.0, 3.0])
    result = to_numpy(arr)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, arr)


# ============================================================================
# NumPy Backend Tests
# ============================================================================


def test_matrix_multiply():
    """Test matrix multiplication."""
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    result = numpy_backend.matrix_multiply(A, B)
    expected = np.array([[19, 22], [43, 50]])
    np.testing.assert_array_equal(result, expected)


def test_matrix_vector_multiply():
    """Test matrix-vector multiplication."""
    A = np.array([[1, 2], [3, 4]])
    x = np.array([1, 2])
    result = numpy_backend.matrix_vector_multiply(A, x)
    expected = np.array([5, 11])
    np.testing.assert_array_equal(result, expected)


def test_matrix_transpose():
    """Test matrix transpose."""
    A = np.array([[1, 2, 3], [4, 5, 6]])
    result = numpy_backend.matrix_transpose(A)
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    np.testing.assert_array_equal(result, expected)


def test_matrix_inverse():
    """Test matrix inverse."""
    A = np.array([[1, 2], [3, 4]], dtype=float)
    result = numpy_backend.matrix_inverse(A)
    # A @ A^-1 should be identity
    identity = numpy_backend.matrix_multiply(A, result)
    np.testing.assert_array_almost_equal(identity, np.eye(2))


def test_matrix_solve():
    """Test linear system solve."""
    A = np.array([[3, 1], [1, 2]], dtype=float)
    b = np.array([9, 8], dtype=float)
    x = numpy_backend.matrix_solve(A, b)
    # A @ x should equal b
    np.testing.assert_array_almost_equal(A @ x, b)


def test_cholesky_decomposition():
    """Test Cholesky decomposition."""
    # Symmetric positive definite matrix
    A = np.array([[4, 2], [2, 3]], dtype=float)
    L = numpy_backend.cholesky_decomposition(A)
    # L @ L.T should equal A
    np.testing.assert_array_almost_equal(L @ L.T, A)


# ============================================================================
# Kalman Operations Tests
# ============================================================================


@pytest.fixture
def kalman_setup():
    """Set up common Kalman test data."""
    # Simple 2D state: [position, velocity]
    x = np.array([0.0, 1.0])  # At origin, moving at 1 m/s
    P = np.eye(2) * 0.1  # Low uncertainty

    # Constant velocity model with dt=1
    F = np.array([[1, 1], [0, 1]], dtype=float)
    Q = np.array([[0.1, 0], [0, 0.1]], dtype=float)

    # Position-only measurement
    H = np.array([[1, 0]], dtype=float)
    R = np.array([[0.5]], dtype=float)

    return {"x": x, "P": P, "F": F, "Q": Q, "H": H, "R": R}


def test_kalman_predict(kalman_setup):
    """Test Kalman filter prediction step."""
    x = kalman_setup["x"]
    P = kalman_setup["P"]
    F = kalman_setup["F"]
    Q = kalman_setup["Q"]

    x_pred, P_pred = numpy_backend.kalman_predict(x, P, F, Q)

    # Position should advance by velocity
    assert x_pred[0] == pytest.approx(1.0)  # 0 + 1*1
    assert x_pred[1] == pytest.approx(1.0)  # Velocity unchanged

    # Covariance should increase
    assert P_pred[0, 0] > P[0, 0]


def test_kalman_update(kalman_setup):
    """Test Kalman filter update step."""
    x = kalman_setup["x"]
    P = kalman_setup["P"]
    F = kalman_setup["F"]
    Q = kalman_setup["Q"]
    H = kalman_setup["H"]
    R = kalman_setup["R"]

    # First predict
    x_pred, P_pred = numpy_backend.kalman_predict(x, P, F, Q)

    # Measurement at position 1.2
    z = np.array([1.2])

    x_post, P_post, K, S = numpy_backend.kalman_update(x_pred, P_pred, z, H, R)

    # State should be pulled toward measurement
    assert x_post[0] == pytest.approx(1.0, abs=0.3)  # Near 1.0

    # Covariance should decrease after update
    assert P_post[0, 0] < P_pred[0, 0]


def test_batch_kalman_predict(kalman_setup):
    """Test batch Kalman prediction."""
    x = kalman_setup["x"]
    P = kalman_setup["P"]
    F = kalman_setup["F"]
    Q = kalman_setup["Q"]

    batch_size = 10
    n = 2

    x_batch = np.tile(x, (batch_size, 1))
    P_batch = np.tile(P, (batch_size, 1, 1))

    x_pred, P_pred = numpy_backend.batch_kalman_predict(x_batch, P_batch, F, Q)

    assert x_pred.shape == (batch_size, n)
    assert P_pred.shape == (batch_size, n, n)

    # Each prediction should match single prediction
    x_single, P_single = numpy_backend.kalman_predict(x, P, F, Q)

    for i in range(batch_size):
        np.testing.assert_array_almost_equal(x_pred[i], x_single)
        np.testing.assert_array_almost_equal(P_pred[i], P_single)


# ============================================================================
# Resampling Tests
# ============================================================================


def test_systematic_resample_uniform():
    """Test systematic resampling with uniform weights."""
    n = 100
    weights = np.ones(n) / n

    indices = numpy_backend.systematic_resample(weights)

    assert len(indices) == n
    assert indices.min() >= 0
    assert indices.max() < n


def test_systematic_resample_concentrated():
    """Test systematic resampling with concentrated weights."""
    n = 100
    weights = np.zeros(n)
    weights[50] = 1.0  # All weight on particle 50

    indices = numpy_backend.systematic_resample(weights)

    # All samples should be particle 50
    assert np.all(indices == 50)


def test_multinomial_resample_uniform():
    """Test multinomial resampling with uniform weights."""
    n = 100
    weights = np.ones(n) / n

    indices = numpy_backend.multinomial_resample(weights)

    assert len(indices) == n
    assert indices.min() >= 0
    assert indices.max() < n


def test_multinomial_resample_concentrated():
    """Test multinomial resampling with concentrated weights."""
    n = 100
    weights = np.zeros(n)
    weights[50] = 1.0  # All weight on particle 50

    indices = numpy_backend.multinomial_resample(weights)

    # All samples should be particle 50
    assert np.all(indices == 50)


def test_resample_different_num_samples():
    """Test resampling with different output size."""
    n_in = 100
    n_out = 50
    weights = np.ones(n_in) / n_in

    indices_sys = numpy_backend.systematic_resample(weights, n_out)
    indices_mult = numpy_backend.multinomial_resample(weights, n_out)

    assert len(indices_sys) == n_out
    assert len(indices_mult) == n_out


# ============================================================================
# GPU Backend Tests
# ============================================================================


def test_cupy_available_flag():
    """Test CUPY_AVAILABLE flag is set correctly."""
    from stonesoup.backend import cupy_backend

    # Should be False if CuPy not installed, True otherwise
    assert isinstance(cupy_backend.CUPY_AVAILABLE, bool)


def test_check_cupy_raises_without_gpu():
    """Test _check_cupy raises ImportError if CuPy unavailable."""
    from stonesoup.backend import cupy_backend

    if not cupy_backend.CUPY_AVAILABLE:
        with pytest.raises(ImportError, match="CuPy is not installed"):
            cupy_backend._check_cupy()


def test_set_backend_cupy_without_gpu():
    """Setting CuPy backend without GPU raises ImportError."""
    if not is_gpu_available():
        with pytest.raises(ImportError):
            set_backend("cupy")


# ============================================================================
# GPU Backend Integration Tests (skip if no GPU)
# ============================================================================


@pytest.fixture
def require_gpu():
    """Skip test if GPU is not available."""
    if not is_gpu_available():
        pytest.skip("GPU not available")


def test_cupy_backend_matrix_multiply(require_gpu):
    """Test CuPy matrix multiplication."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    A = cp.array([[1, 2], [3, 4]], dtype=float)
    B = cp.array([[5, 6], [7, 8]], dtype=float)
    result = cupy_backend.matrix_multiply(A, B)
    expected = cp.array([[19, 22], [43, 50]], dtype=float)
    cp.testing.assert_array_equal(result, expected)


def test_cupy_backend_kalman_predict(require_gpu):
    """Test CuPy Kalman prediction."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    x = cp.array([0.0, 1.0])
    P = cp.eye(2) * 0.1
    F = cp.array([[1, 1], [0, 1]], dtype=float)
    Q = cp.array([[0.1, 0], [0, 0.1]], dtype=float)

    x_pred, P_pred = cupy_backend.kalman_predict(x, P, F, Q)

    # Position should advance by velocity
    assert float(x_pred[0]) == pytest.approx(1.0)
    assert float(x_pred[1]) == pytest.approx(1.0)


def test_cupy_backend_kalman_update(require_gpu):
    """Test CuPy Kalman update."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    x = cp.array([1.0, 1.0])
    P = cp.eye(2) * 0.2
    H = cp.array([[1, 0]], dtype=float)
    R = cp.array([[0.5]], dtype=float)
    z = cp.array([1.2])

    x_post, P_post, K, S = cupy_backend.kalman_update(x, P, z, H, R)

    # Covariance should decrease after update
    assert float(P_post[0, 0]) < float(P[0, 0])


def test_cupy_backend_batch_kalman_predict(require_gpu):
    """Test CuPy batch Kalman prediction."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    batch_size = 100
    x_batch = cp.random.randn(batch_size, 2).astype(cp.float64)
    P_batch = cp.tile(cp.eye(2) * 0.1, (batch_size, 1, 1))
    F = cp.array([[1, 1], [0, 1]], dtype=float)
    Q = cp.array([[0.1, 0], [0, 0.1]], dtype=float)

    x_pred, P_pred = cupy_backend.batch_kalman_predict(x_batch, P_batch, F, Q)

    assert x_pred.shape == (batch_size, 2)
    assert P_pred.shape == (batch_size, 2, 2)


def test_cupy_backend_systematic_resample(require_gpu):
    """Test CuPy systematic resampling."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    n = 100
    weights = cp.ones(n) / n
    indices = cupy_backend.systematic_resample(weights)

    assert len(indices) == n
    assert int(indices.min()) >= 0
    assert int(indices.max()) < n


def test_cupy_backend_multinomial_resample(require_gpu):
    """Test CuPy multinomial resampling."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    n = 100
    weights = cp.ones(n) / n
    indices = cupy_backend.multinomial_resample(weights)

    assert len(indices) == n
    assert int(indices.min()) >= 0
    assert int(indices.max()) < n


def test_cpu_gpu_equivalence(require_gpu):
    """Test that CPU and GPU produce equivalent results."""
    import cupy as cp

    from stonesoup.backend import cupy_backend

    # Set up test data
    np.random.seed(42)
    x_np = np.array([1.0, 2.0])
    P_np = np.eye(2) * 0.5
    F_np = np.array([[1, 0.1], [0, 1]], dtype=float)
    Q_np = np.eye(2) * 0.01

    # CPU prediction
    x_pred_cpu, P_pred_cpu = numpy_backend.kalman_predict(x_np, P_np, F_np, Q_np)

    # GPU prediction
    x_gpu = cp.asarray(x_np)
    P_gpu = cp.asarray(P_np)
    F_gpu = cp.asarray(F_np)
    Q_gpu = cp.asarray(Q_np)
    x_pred_gpu, P_pred_gpu = cupy_backend.kalman_predict(x_gpu, P_gpu, F_gpu, Q_gpu)

    # Compare results
    np.testing.assert_array_almost_equal(x_pred_cpu, x_pred_gpu.get())
    np.testing.assert_array_almost_equal(P_pred_cpu, P_pred_gpu.get())


def test_to_gpu_conversion(require_gpu):
    """Test to_gpu array conversion."""
    import cupy as cp

    from stonesoup.backend import to_gpu

    arr_np = np.array([1.0, 2.0, 3.0])
    arr_gpu = to_gpu(arr_np)

    assert isinstance(arr_gpu, cp.ndarray)
    np.testing.assert_array_equal(arr_np, arr_gpu.get())

    # GPU array should pass through
    arr_gpu2 = to_gpu(arr_gpu)
    assert arr_gpu2 is arr_gpu


def test_set_backend_cupy_with_gpu(require_gpu):
    """Test setting CuPy backend when GPU is available."""
    import cupy as cp

    set_backend("cupy")
    assert get_backend() == "cupy"

    xp = get_array_module()
    assert xp is cp
