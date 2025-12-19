# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""NumPy backend for array operations.

This module provides NumPy-based implementations of common array operations
used in Stone Soup. These serve as the reference implementation and CPU fallback.
"""

import numpy as np
from numpy.linalg import cholesky, inv, solve


def matrix_multiply(A, B):
    """Matrix multiplication.

    Args:
        A: First matrix
        B: Second matrix

    Returns:
        Result of A @ B
    """
    return np.matmul(A, B)


def matrix_vector_multiply(A, x):
    """Matrix-vector multiplication.

    Args:
        A: Matrix
        x: Vector

    Returns:
        Result of A @ x
    """
    return np.matmul(A, x)


def matrix_transpose(A):
    """Matrix transpose.

    Args:
        A: Matrix

    Returns:
        Transposed matrix
    """
    return A.T


def matrix_inverse(A):
    """Matrix inverse.

    Args:
        A: Square matrix

    Returns:
        Inverse of A
    """
    return inv(A)


def matrix_solve(A, b):
    """Solve linear system Ax = b.

    Args:
        A: Coefficient matrix
        b: Right-hand side

    Returns:
        Solution x
    """
    return solve(A, b)


def cholesky_decomposition(A):
    """Cholesky decomposition.

    Args:
        A: Symmetric positive definite matrix

    Returns:
        Lower triangular matrix L such that A = L @ L.T
    """
    return cholesky(A)


def kalman_predict(x, P, F, Q):
    """Kalman filter prediction step.

    Args:
        x: State vector (n,) or (n, 1)
        P: State covariance (n, n)
        F: Transition matrix (n, n)
        Q: Process noise covariance (n, n)

    Returns:
        tuple: (x_pred, P_pred)
    """
    x = np.asarray(x).flatten()
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def kalman_update(x, P, z, H, R):
    """Kalman filter update step.

    Args:
        x: Predicted state vector
        P: Predicted covariance
        z: Measurement vector
        H: Measurement matrix
        R: Measurement noise covariance

    Returns:
        tuple: (x_post, P_post, K, S)
    """
    x = np.asarray(x).flatten()
    z = np.asarray(z).flatten()

    # Innovation
    y = z - H @ x

    # Innovation covariance
    S = H @ P @ H.T + R

    # Kalman gain (using solve for numerical stability)
    K = solve(S.T, (P @ H.T).T).T

    # Posterior state
    x_post = x + K @ y

    # Posterior covariance (Joseph form for numerical stability)
    n = len(x)
    I_KH = np.eye(n) - K @ H
    P_post = I_KH @ P @ I_KH.T + K @ R @ K.T

    return x_post, P_post, K, S


def batch_kalman_predict(x_batch, P_batch, F, Q):
    """Batch Kalman prediction for multiple states.

    Args:
        x_batch: Batch of state vectors (batch_size, n)
        P_batch: Batch of covariances (batch_size, n, n)
        F: Transition matrix (n, n)
        Q: Process noise covariance (n, n)

    Returns:
        tuple: (x_pred_batch, P_pred_batch)
    """
    batch_size = x_batch.shape[0]

    x_pred_batch = np.zeros_like(x_batch)
    P_pred_batch = np.zeros_like(P_batch)

    for i in range(batch_size):
        x_pred_batch[i] = F @ x_batch[i]
        P_pred_batch[i] = F @ P_batch[i] @ F.T + Q

    return x_pred_batch, P_pred_batch


def systematic_resample(weights, num_samples=None):
    """Systematic resampling for particle filters.

    Args:
        weights: Particle weights (must sum to 1)
        num_samples: Number of samples (default: same as input)

    Returns:
        numpy.ndarray: Indices of resampled particles
    """
    n = len(weights)
    if num_samples is None:
        num_samples = n

    # Cumulative sum
    cumsum = np.cumsum(weights)

    # Starting point
    u0 = np.random.uniform(0, 1.0 / num_samples)
    u = u0 + np.arange(num_samples) / num_samples

    # Find indices
    indices = np.searchsorted(cumsum, u)
    indices = np.clip(indices, 0, n - 1)

    return indices


def multinomial_resample(weights, num_samples=None):
    """Multinomial resampling for particle filters.

    Args:
        weights: Particle weights (must sum to 1)
        num_samples: Number of samples (default: same as input)

    Returns:
        numpy.ndarray: Indices of resampled particles
    """
    n = len(weights)
    if num_samples is None:
        num_samples = n

    return np.random.choice(n, size=num_samples, p=weights)
