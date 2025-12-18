# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""CuPy backend for GPU-accelerated array operations.

This module provides CuPy-based implementations of common array operations
used in Stone Soup, enabling GPU acceleration for large-scale tracking.

Note:
    This module requires CuPy to be installed:
    pip install cupy-cuda12x  # For CUDA 12.x
"""

try:
    import cupy as cp
    from cupy.linalg import cholesky, inv, solve

    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def _check_cupy():
    """Check that CuPy is available."""
    if not CUPY_AVAILABLE:
        raise ImportError("CuPy is not installed. Install with: pip install cupy-cuda12x")


def matrix_multiply(A, B):
    """Matrix multiplication (GPU).

    Args:
        A: First matrix (CuPy array)
        B: Second matrix (CuPy array)

    Returns:
        Result of A @ B
    """
    _check_cupy()
    return cp.matmul(A, B)


def matrix_vector_multiply(A, x):
    """Matrix-vector multiplication (GPU).

    Args:
        A: Matrix (CuPy array)
        x: Vector (CuPy array)

    Returns:
        Result of A @ x
    """
    _check_cupy()
    return cp.matmul(A, x)


def matrix_transpose(A):
    """Matrix transpose (GPU).

    Args:
        A: Matrix (CuPy array)

    Returns:
        Transposed matrix
    """
    _check_cupy()
    return A.T


def matrix_inverse(A):
    """Matrix inverse (GPU).

    Args:
        A: Square matrix (CuPy array)

    Returns:
        Inverse of A
    """
    _check_cupy()
    return inv(A)


def matrix_solve(A, b):
    """Solve linear system Ax = b (GPU).

    Args:
        A: Coefficient matrix (CuPy array)
        b: Right-hand side (CuPy array)

    Returns:
        Solution x
    """
    _check_cupy()
    return solve(A, b)


def cholesky_decomposition(A):
    """Cholesky decomposition (GPU).

    Args:
        A: Symmetric positive definite matrix (CuPy array)

    Returns:
        Lower triangular matrix L such that A = L @ L.T
    """
    _check_cupy()
    return cholesky(A)


def kalman_predict(x, P, F, Q):
    """Kalman filter prediction step (GPU).

    Args:
        x: State vector
        P: State covariance
        F: Transition matrix
        Q: Process noise covariance

    Returns:
        tuple: (x_pred, P_pred)
    """
    _check_cupy()
    x = cp.asarray(x).flatten()
    P = cp.asarray(P)
    F = cp.asarray(F)
    Q = cp.asarray(Q)

    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def kalman_update(x, P, z, H, R):
    """Kalman filter update step (GPU).

    Args:
        x: Predicted state vector
        P: Predicted covariance
        z: Measurement vector
        H: Measurement matrix
        R: Measurement noise covariance

    Returns:
        tuple: (x_post, P_post, K, S)
    """
    _check_cupy()
    x = cp.asarray(x).flatten()
    z = cp.asarray(z).flatten()
    P = cp.asarray(P)
    H = cp.asarray(H)
    R = cp.asarray(R)

    # Innovation
    y = z - H @ x

    # Innovation covariance
    S = H @ P @ H.T + R

    # Kalman gain
    K = solve(S.T, (P @ H.T).T).T

    # Posterior state
    x_post = x + K @ y

    # Posterior covariance (Joseph form)
    n = len(x)
    I_KH = cp.eye(n) - K @ H
    P_post = I_KH @ P @ I_KH.T + K @ R @ K.T

    return x_post, P_post, K, S


def batch_kalman_predict(x_batch, P_batch, F, Q):
    """Batch Kalman prediction for multiple states (GPU).

    This is more efficient than calling kalman_predict in a loop
    when processing many particles or tracks.

    Args:
        x_batch: Batch of state vectors (batch_size, n)
        P_batch: Batch of covariances (batch_size, n, n)
        F: Transition matrix (n, n)
        Q: Process noise covariance (n, n)

    Returns:
        tuple: (x_pred_batch, P_pred_batch)
    """
    _check_cupy()
    x_batch = cp.asarray(x_batch)
    P_batch = cp.asarray(P_batch)
    F = cp.asarray(F)
    Q = cp.asarray(Q)

    # Vectorized prediction
    # x_pred = F @ x for each sample
    x_pred_batch = cp.einsum("ij,bj->bi", F, x_batch)

    # P_pred = F @ P @ F.T + Q for each sample
    # Using einsum for efficiency
    FP = cp.einsum("ij,bjk->bik", F, P_batch)
    P_pred_batch = cp.einsum("bij,kj->bik", FP, F) + Q

    return x_pred_batch, P_pred_batch


def systematic_resample(weights, num_samples=None):
    """Systematic resampling for particle filters (GPU).

    Args:
        weights: Particle weights (must sum to 1)
        num_samples: Number of samples (default: same as input)

    Returns:
        cupy.ndarray: Indices of resampled particles
    """
    _check_cupy()
    weights = cp.asarray(weights)
    n = len(weights)
    if num_samples is None:
        num_samples = n

    # Cumulative sum
    cumsum = cp.cumsum(weights)

    # Starting point
    u0 = cp.random.uniform(0, 1.0 / num_samples)
    u = u0 + cp.arange(num_samples) / num_samples

    # Find indices using searchsorted
    indices = cp.searchsorted(cumsum, u)
    indices = cp.clip(indices, 0, n - 1)

    return indices


def multinomial_resample(weights, num_samples=None):
    """Multinomial resampling for particle filters (GPU).

    Args:
        weights: Particle weights (must sum to 1)
        num_samples: Number of samples (default: same as input)

    Returns:
        cupy.ndarray: Indices of resampled particles
    """
    _check_cupy()
    weights = cp.asarray(weights)
    n = len(weights)
    if num_samples is None:
        num_samples = n

    return cp.random.choice(n, size=num_samples, p=weights)
