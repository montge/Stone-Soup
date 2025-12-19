# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Tests for common test fixtures module."""

from datetime import datetime, timedelta

import numpy as np

from stonesoup.tests.fixtures import (
    LARGE_DIM,
    MEDIUM_DIM,
    SMALL_DIM,
    create_cv_process_noise,
    create_cv_transition_matrix,
    create_diagonal_covariance,
    create_gaussian_state,
    create_identity_covariance,
    create_measurement_noise,
    create_position_measurement_matrix,
    create_positive_definite_covariance,
    create_state_sequence,
    create_state_vector,
)
from stonesoup.types.array import CovarianceMatrix
from stonesoup.types.state import GaussianState, StateVector


def test_dimension_constants():
    """Test dimension constant values."""
    assert SMALL_DIM == 2
    assert MEDIUM_DIM == 4
    assert LARGE_DIM == 6


def test_create_state_vector_default():
    """Test create_state_vector with defaults."""
    sv = create_state_vector()
    assert isinstance(sv, StateVector)
    assert sv.shape == (MEDIUM_DIM, 1)
    assert np.allclose(sv.flatten(), [0, 1, 2, 3])


def test_create_state_vector_custom_dim():
    """Test create_state_vector with custom dimension."""
    sv = create_state_vector(dim=6)
    assert sv.shape == (6, 1)


def test_create_state_vector_custom_values():
    """Test create_state_vector with custom values."""
    sv = create_state_vector(values=[1.5, 2.5, 3.5])
    assert sv.shape == (3, 1)
    assert np.allclose(sv.flatten(), [1.5, 2.5, 3.5])


def test_create_identity_covariance():
    """Test create_identity_covariance."""
    cov = create_identity_covariance(3)
    assert isinstance(cov, CovarianceMatrix)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, np.eye(3))


def test_create_diagonal_covariance_default():
    """Test create_diagonal_covariance with defaults."""
    cov = create_diagonal_covariance()
    assert cov.shape == (MEDIUM_DIM, MEDIUM_DIM)
    assert np.allclose(np.diag(cov), [1, 2, 3, 4])


def test_create_diagonal_covariance_custom():
    """Test create_diagonal_covariance with custom variances."""
    cov = create_diagonal_covariance(variances=[0.1, 0.2, 0.3])
    assert cov.shape == (3, 3)
    assert np.allclose(np.diag(cov), [0.1, 0.2, 0.3])


def test_create_positive_definite_covariance():
    """Test create_positive_definite_covariance."""
    cov = create_positive_definite_covariance(dim=3)
    assert cov.shape == (3, 3)
    # Check symmetry
    assert np.allclose(cov, cov.T)
    # Check positive definiteness via eigenvalues
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues > 0)


def test_create_gaussian_state_default():
    """Test create_gaussian_state with defaults."""
    state = create_gaussian_state()
    assert isinstance(state, GaussianState)
    assert state.state_vector.shape == (MEDIUM_DIM, 1)
    assert state.covar.shape == (MEDIUM_DIM, MEDIUM_DIM)
    assert state.timestamp is not None


def test_create_gaussian_state_custom():
    """Test create_gaussian_state with custom parameters."""
    ts = datetime(2024, 1, 1, 12, 0, 0)
    sv = create_state_vector(dim=2, values=[10.0, 20.0])
    cov = create_identity_covariance(2)
    state = create_gaussian_state(dim=2, timestamp=ts, state_vector=sv, covar=cov)
    assert state.state_vector.shape == (2, 1)
    assert state.timestamp == ts
    assert np.allclose(state.state_vector.flatten(), [10.0, 20.0])


def test_create_state_sequence():
    """Test create_state_sequence."""
    states = create_state_sequence(num_states=5, dim=2, dt=timedelta(seconds=2))
    assert len(states) == 5
    assert all(isinstance(s, GaussianState) for s in states)
    # Check timestamps are sequential
    for i in range(1, len(states)):
        diff = (states[i].timestamp - states[i - 1].timestamp).total_seconds()
        assert diff == 2.0


def test_create_cv_transition_matrix():
    """Test create_cv_transition_matrix."""
    F = create_cv_transition_matrix(dt=1.0)
    assert F.shape == (4, 4)
    expected = np.array([[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]])
    assert np.allclose(F, expected)


def test_create_cv_transition_matrix_different_dt():
    """Test create_cv_transition_matrix with different dt."""
    F = create_cv_transition_matrix(dt=0.5)
    expected = np.array([[1, 0.5, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.5], [0, 0, 0, 1]])
    assert np.allclose(F, expected)


def test_create_cv_process_noise():
    """Test create_cv_process_noise."""
    Q = create_cv_process_noise(q=0.1)
    assert isinstance(Q, CovarianceMatrix)
    assert Q.shape == (4, 4)
    assert np.allclose(np.diag(Q), [0.1, 0.1, 0.1, 0.1])


def test_create_position_measurement_matrix():
    """Test create_position_measurement_matrix."""
    H = create_position_measurement_matrix()
    assert H.shape == (2, 4)
    expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0]])
    assert np.allclose(H, expected)


def test_create_measurement_noise():
    """Test create_measurement_noise."""
    R = create_measurement_noise(r=0.5)
    assert isinstance(R, CovarianceMatrix)
    assert R.shape == (2, 2)
    assert np.allclose(np.diag(R), [0.5, 0.5])


# Test pytest fixtures via indirect usage
def test_state_vector_2d(state_vector_2d):
    """Test state_vector_2d fixture."""
    assert state_vector_2d.shape == (SMALL_DIM, 1)


def test_state_vector_4d(state_vector_4d):
    """Test state_vector_4d fixture."""
    assert state_vector_4d.shape == (MEDIUM_DIM, 1)


def test_state_vector_6d(state_vector_6d):
    """Test state_vector_6d fixture."""
    assert state_vector_6d.shape == (LARGE_DIM, 1)


def test_identity_covar_2d(identity_covar_2d):
    """Test identity_covar_2d fixture."""
    assert identity_covar_2d.shape == (SMALL_DIM, SMALL_DIM)
    assert np.allclose(identity_covar_2d, np.eye(SMALL_DIM))


def test_identity_covar_4d(identity_covar_4d):
    """Test identity_covar_4d fixture."""
    assert identity_covar_4d.shape == (MEDIUM_DIM, MEDIUM_DIM)
    assert np.allclose(identity_covar_4d, np.eye(MEDIUM_DIM))


def test_gaussian_state_2d(gaussian_state_2d):
    """Test gaussian_state_2d fixture."""
    assert isinstance(gaussian_state_2d, GaussianState)
    assert gaussian_state_2d.state_vector.shape == (SMALL_DIM, 1)


def test_gaussian_state_4d(gaussian_state_4d):
    """Test gaussian_state_4d fixture."""
    assert isinstance(gaussian_state_4d, GaussianState)
    assert gaussian_state_4d.state_vector.shape == (MEDIUM_DIM, 1)


def test_cv_model_matrices(cv_model_matrices):
    """Test cv_model_matrices fixture."""
    assert "F" in cv_model_matrices
    assert "Q" in cv_model_matrices
    assert "H" in cv_model_matrices
    assert "R" in cv_model_matrices
    assert cv_model_matrices["F"].shape == (4, 4)
    assert cv_model_matrices["Q"].shape == (4, 4)
    assert cv_model_matrices["H"].shape == (2, 4)
    assert cv_model_matrices["R"].shape == (2, 2)
