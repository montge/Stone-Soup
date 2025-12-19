# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Tests for validation utilities module."""

import numpy as np
import pytest

from stonesoup.types.validation import (
    check_index_bounds,
    validate_bounds,
    validate_dimension_match,
    validate_positive_definite,
    validate_shape,
    validate_square_matrix,
)


# Tests for validate_shape
def test_validate_shape_exact():
    """Test validate_shape passes for exact shape match."""
    arr = np.array([[1], [2], [3]])
    validate_shape(arr, (3, 1), "test")  # Should not raise


def test_validate_shape_wildcard():
    """Test validate_shape passes with wildcard dimensions."""
    arr = np.array([[1], [2], [3]])
    validate_shape(arr, (None, 1), "test")  # Should not raise


def test_validate_shape_invalid_ndim():
    """Test validate_shape raises for wrong number of dimensions."""
    arr = np.array([1, 2, 3])  # 1D array
    with pytest.raises(ValueError, match="test shape should be"):
        validate_shape(arr, (None, 1), "test")


def test_validate_shape_invalid_specific_dimension():
    """Test validate_shape raises for wrong specific dimension."""
    arr = np.array([[1, 2], [3, 4]])  # 2x2
    with pytest.raises(ValueError, match="test shape should be"):
        validate_shape(arr, (None, 1), "test")


# Tests for validate_bounds
def test_validate_bounds_default_size():
    """Test validate_bounds passes for default 6-element array."""
    bounds = np.array([0, 1, 0, 1, 0, 1])
    validate_bounds(bounds)  # Should not raise


def test_validate_bounds_custom_size():
    """Test validate_bounds passes for custom size."""
    bounds = np.array([0, 1, 0, 1])
    validate_bounds(bounds, expected_size=4)  # Should not raise


def test_validate_bounds_invalid_size():
    """Test validate_bounds raises for wrong size."""
    bounds = np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="bounds must be a 6-element array"):
        validate_bounds(bounds)


# Tests for check_index_bounds
def test_check_index_bounds_valid():
    """Test check_index_bounds passes for valid indices."""
    check_index_bounds((0, 0, 0), (10, 10, 10))  # Should not raise
    check_index_bounds((9, 9, 9), (10, 10, 10))  # Should not raise


def test_check_index_bounds_out_of_bounds():
    """Test check_index_bounds raises for out of bounds."""
    with pytest.raises(ValueError, match="Invalid indices"):
        check_index_bounds((10, 0, 0), (10, 10, 10))


def test_check_index_bounds_negative():
    """Test check_index_bounds raises for negative index."""
    with pytest.raises(ValueError, match="Invalid indices"):
        check_index_bounds((-1, 0, 0), (10, 10, 10))


def test_check_index_bounds_dimension_mismatch():
    """Test check_index_bounds raises for dimension mismatch."""
    with pytest.raises(ValueError, match="dimension mismatch"):
        check_index_bounds((0, 0), (10, 10, 10))


# Tests for validate_positive_definite
def test_validate_positive_definite_identity():
    """Test identity matrix is positive definite."""
    assert validate_positive_definite(np.eye(3)) is True


def test_validate_positive_definite_diagonal():
    """Test diagonal matrix with positive values is positive definite."""
    assert validate_positive_definite(np.diag([1, 2, 3])) is True


def test_validate_positive_definite_non_pd():
    """Test matrix with negative eigenvalue is not positive definite."""
    # Matrix with eigenvalues -3 and 1
    matrix = np.array([[1, 2], [2, 1]])
    assert validate_positive_definite(matrix) is False


def test_validate_positive_definite_singular():
    """Test singular matrix handling."""
    # Singular matrix
    matrix = np.array([[1, 2], [2, 4]])
    # Should return True or False without error
    result = validate_positive_definite(matrix)
    assert isinstance(result, bool)


# Tests for validate_square_matrix
def test_validate_square_matrix_valid():
    """Test validate_square_matrix passes for square matrix."""
    validate_square_matrix(np.eye(3))  # Should not raise


def test_validate_square_matrix_non_square():
    """Test validate_square_matrix raises for non-square matrix."""
    with pytest.raises(ValueError, match="must be square"):
        validate_square_matrix(np.array([[1, 2, 3], [4, 5, 6]]))


def test_validate_square_matrix_1d():
    """Test validate_square_matrix raises for 1D array."""
    with pytest.raises(ValueError, match="must be square"):
        validate_square_matrix(np.array([1, 2, 3]))


# Tests for validate_dimension_match
def test_validate_dimension_match_valid():
    """Test validate_dimension_match passes for matching dimensions."""
    arr1 = np.zeros((3, 1))
    arr2 = np.eye(3)
    validate_dimension_match(arr1, arr2, axis1=0, axis2=0)  # Should not raise


def test_validate_dimension_match_invalid():
    """Test validate_dimension_match raises for mismatched dimensions."""
    arr1 = np.zeros((4, 1))
    arr2 = np.eye(3)
    with pytest.raises(ValueError, match="Dimension mismatch"):
        validate_dimension_match(arr1, arr2, axis1=0, axis2=0)


def test_validate_dimension_match_different_axes():
    """Test validate_dimension_match with different axes."""
    arr1 = np.zeros((3, 5))
    arr2 = np.zeros((7, 5))
    validate_dimension_match(arr1, arr2, axis1=1, axis2=1)  # Should not raise
