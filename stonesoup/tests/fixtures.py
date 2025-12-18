# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Common test fixtures and data factories for Stone Soup tests.

This module provides reusable test fixtures to reduce code duplication
across the test suite.
"""

from datetime import datetime, timedelta

import numpy as np
import pytest

from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.state import GaussianState, StateVector

# Standard test dimensions
SMALL_DIM = 2
MEDIUM_DIM = 4
LARGE_DIM = 6


def create_state_vector(dim: int = MEDIUM_DIM, values: list[float] | None = None) -> StateVector:
    """Create a StateVector for testing.

    Parameters
    ----------
    dim : int
        Dimension of the state vector.
    values : list of float, optional
        Specific values to use. If None, uses sequential values [0, 1, 2, ...].

    Returns
    -------
    StateVector
        A state vector for testing.
    """
    if values is not None:
        return StateVector(np.array(values).reshape(-1, 1))
    return StateVector(np.arange(dim, dtype=float).reshape(-1, 1))


def create_identity_covariance(dim: int = MEDIUM_DIM) -> CovarianceMatrix:
    """Create an identity covariance matrix for testing.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.

    Returns
    -------
    CovarianceMatrix
        An identity covariance matrix.
    """
    return CovarianceMatrix(np.eye(dim))


def create_diagonal_covariance(
    dim: int = MEDIUM_DIM, variances: list[float] | None = None
) -> CovarianceMatrix:
    """Create a diagonal covariance matrix for testing.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.
    variances : list of float, optional
        Diagonal variances. If None, uses [1, 2, 3, ...].

    Returns
    -------
    CovarianceMatrix
        A diagonal covariance matrix.
    """
    if variances is not None:
        return CovarianceMatrix(np.diag(variances))
    return CovarianceMatrix(np.diag(np.arange(1, dim + 1, dtype=float)))


def create_positive_definite_covariance(dim: int = MEDIUM_DIM) -> CovarianceMatrix:
    """Create a positive definite covariance matrix for testing.

    Uses A @ A.T + I to ensure positive definiteness.

    Parameters
    ----------
    dim : int
        Dimension of the matrix.

    Returns
    -------
    CovarianceMatrix
        A positive definite covariance matrix.
    """
    np.random.seed(42)  # For reproducibility
    A = np.random.randn(dim, dim)
    return CovarianceMatrix(A @ A.T + np.eye(dim))


def create_gaussian_state(
    dim: int = MEDIUM_DIM,
    timestamp: datetime | None = None,
    state_vector: StateVector | None = None,
    covar: CovarianceMatrix | None = None,
) -> GaussianState:
    """Create a GaussianState for testing.

    Parameters
    ----------
    dim : int
        Dimension of the state.
    timestamp : datetime, optional
        Timestamp for the state. If None, uses datetime.now().
    state_vector : StateVector, optional
        State vector to use. If None, creates a default one.
    covar : CovarianceMatrix, optional
        Covariance matrix to use. If None, creates an identity matrix.

    Returns
    -------
    GaussianState
        A Gaussian state for testing.
    """
    if timestamp is None:
        timestamp = datetime.now()
    if state_vector is None:
        state_vector = create_state_vector(dim)
    if covar is None:
        covar = create_identity_covariance(dim)

    return GaussianState(state_vector=state_vector, covar=covar, timestamp=timestamp)


def create_state_sequence(
    num_states: int = 10, dim: int = MEDIUM_DIM, dt: timedelta = timedelta(seconds=1)
) -> list[GaussianState]:
    """Create a sequence of Gaussian states for testing.

    Parameters
    ----------
    num_states : int
        Number of states to create.
    dim : int
        Dimension of each state.
    dt : timedelta
        Time interval between states.

    Returns
    -------
    list of GaussianState
        A sequence of Gaussian states.
    """
    start_time = datetime.now()
    states = []
    for i in range(num_states):
        timestamp = start_time + i * dt
        state_vector = create_state_vector(dim, [float(i + j) for j in range(dim)])
        states.append(
            create_gaussian_state(dim=dim, timestamp=timestamp, state_vector=state_vector)
        )
    return states


# Constant velocity model matrices (4D: [x, vx, y, vy])
def create_cv_transition_matrix(dt: float = 1.0) -> np.ndarray:
    """Create a constant velocity transition matrix.

    Parameters
    ----------
    dt : float
        Time step.

    Returns
    -------
    np.ndarray
        4x4 constant velocity transition matrix.
    """
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])


def create_cv_process_noise(q: float = 0.1) -> CovarianceMatrix:
    """Create a process noise matrix for constant velocity model.

    Parameters
    ----------
    q : float
        Process noise intensity.

    Returns
    -------
    CovarianceMatrix
        4x4 process noise matrix.
    """
    return CovarianceMatrix(np.diag([q, q, q, q]))


def create_position_measurement_matrix() -> np.ndarray:
    """Create a position-only measurement matrix.

    Returns
    -------
    np.ndarray
        2x4 measurement matrix [x, y] from [x, vx, y, vy].
    """
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0]])


def create_measurement_noise(r: float = 0.5) -> CovarianceMatrix:
    """Create a measurement noise matrix.

    Parameters
    ----------
    r : float
        Measurement noise intensity.

    Returns
    -------
    CovarianceMatrix
        2x2 measurement noise matrix.
    """
    return CovarianceMatrix(np.diag([r, r]))


# Pytest fixtures
@pytest.fixture
def state_vector_2d():
    """Fixture for a 2D state vector."""
    return create_state_vector(SMALL_DIM)


@pytest.fixture
def state_vector_4d():
    """Fixture for a 4D state vector."""
    return create_state_vector(MEDIUM_DIM)


@pytest.fixture
def state_vector_6d():
    """Fixture for a 6D state vector."""
    return create_state_vector(LARGE_DIM)


@pytest.fixture
def identity_covar_2d():
    """Fixture for a 2x2 identity covariance matrix."""
    return create_identity_covariance(SMALL_DIM)


@pytest.fixture
def identity_covar_4d():
    """Fixture for a 4x4 identity covariance matrix."""
    return create_identity_covariance(MEDIUM_DIM)


@pytest.fixture
def gaussian_state_2d():
    """Fixture for a 2D Gaussian state."""
    return create_gaussian_state(SMALL_DIM)


@pytest.fixture
def gaussian_state_4d():
    """Fixture for a 4D Gaussian state."""
    return create_gaussian_state(MEDIUM_DIM)


@pytest.fixture
def cv_model_matrices():
    """Fixture for constant velocity model matrices."""
    return {
        "F": create_cv_transition_matrix(1.0),
        "Q": create_cv_process_noise(0.1),
        "H": create_position_measurement_matrix(),
        "R": create_measurement_noise(0.5),
    }
