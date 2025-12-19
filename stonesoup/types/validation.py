# SPDX-FileCopyrightText: 2017-2025 Stone Soup contributors
# SPDX-License-Identifier: MIT
"""Common validation utilities for Stone Soup types.

This module provides reusable validation functions to reduce code duplication
across the types module.
"""

from collections.abc import Sequence

import numpy as np


def validate_shape(
    array: np.ndarray, expected_shape: tuple[int | None, ...], name: str = "array"
) -> None:
    """Validate that an array has the expected shape.

    Parameters
    ----------
    array : np.ndarray
        The array to validate.
    expected_shape : tuple
        Expected shape. Use None for dimensions that can be any size.
        For example, (None, 1) means any number of rows but exactly 1 column.
    name : str
        Name of the array for error messages.

    Raises
    ------
    ValueError
        If the array shape doesn't match the expected shape.

    Examples
    --------
    >>> import numpy as np
    >>> validate_shape(np.array([[1], [2], [3]]), (None, 1), "state_vector")
    >>> validate_shape(np.array([[1, 2]]), (None, 1), "state_vector")
    Traceback (most recent call last):
        ...
    ValueError: state_vector shape should be (*, 1): got (1, 2)
    """
    if len(array.shape) != len(expected_shape):
        expected_str = tuple("*" if s is None else s for s in expected_shape)
        raise ValueError(f"{name} shape should be {expected_str}: got {array.shape}")

    for actual, expected in zip(array.shape, expected_shape):
        if expected is not None and actual != expected:
            expected_str = tuple("*" if s is None else s for s in expected_shape)
            raise ValueError(f"{name} shape should be {expected_str}: got {array.shape}")


def validate_bounds(bounds: np.ndarray, expected_size: int = 6, name: str = "bounds") -> None:
    """Validate that bounds array has the expected size.

    Parameters
    ----------
    bounds : np.ndarray
        The bounds array to validate.
    expected_size : int
        Expected number of elements (default: 6 for 3D min/max bounds).
    name : str
        Name of the array for error messages.

    Raises
    ------
    ValueError
        If the bounds array doesn't have the expected size.

    Examples
    --------
    >>> import numpy as np
    >>> validate_bounds(np.array([0, 1, 0, 1, 0, 1]))
    >>> validate_bounds(np.array([0, 1, 0, 1]), expected_size=4)
    >>> validate_bounds(np.array([0, 1]), expected_size=6)
    Traceback (most recent call last):
        ...
    ValueError: bounds must be a 6-element array, got shape (2,)
    """
    if bounds.size != expected_size:
        raise ValueError(
            f"{name} must be a {expected_size}-element array, got shape {bounds.shape}"
        )


def check_index_bounds(
    indices: Sequence[int], shape: tuple[int, ...], name: str = "indices"
) -> None:
    """Check that indices are within the valid bounds for a given shape.

    Parameters
    ----------
    indices : sequence of int
        The indices to check.
    shape : tuple of int
        The shape defining the valid bounds.
    name : str
        Name of the indices for error messages.

    Raises
    ------
    ValueError
        If any index is out of bounds.

    Examples
    --------
    >>> check_index_bounds((1, 2, 3), (10, 10, 10))
    >>> check_index_bounds((10, 0, 0), (10, 10, 10))
    Traceback (most recent call last):
        ...
    ValueError: Invalid indices (10, 0, 0) for grid shape (10, 10, 10)
    """
    if len(indices) != len(shape):
        raise ValueError(
            f"Invalid {name} {tuple(indices)} for grid shape {shape}: " f"dimension mismatch"
        )

    for idx, size in zip(indices, shape):
        if idx < 0 or idx >= size:
            raise ValueError(f"Invalid {name} {tuple(indices)} for grid shape {shape}")


def validate_positive_definite(
    matrix: np.ndarray, name: str = "matrix", tolerance: float = 1e-10
) -> bool:
    """Check if a matrix is positive definite.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to check.
    name : str
        Name of the matrix for error messages.
    tolerance : float
        Tolerance for eigenvalue check.

    Returns
    -------
    bool
        True if the matrix is positive definite.

    Examples
    --------
    >>> import numpy as np
    >>> validate_positive_definite(np.eye(3))
    True
    >>> validate_positive_definite(np.array([[1, 2], [2, 1]]))
    False
    """
    try:
        eigenvalues = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvalues > -tolerance))
    except np.linalg.LinAlgError:
        return False


def validate_square_matrix(matrix: np.ndarray, name: str = "matrix") -> None:
    """Validate that a matrix is square.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to validate.
    name : str
        Name of the matrix for error messages.

    Raises
    ------
    ValueError
        If the matrix is not square.

    Examples
    --------
    >>> import numpy as np
    >>> validate_square_matrix(np.eye(3))
    >>> validate_square_matrix(np.array([[1, 2, 3], [4, 5, 6]]))
    Traceback (most recent call last):
        ...
    ValueError: matrix must be square: got shape (2, 3)
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square: got shape {matrix.shape}")


def validate_dimension_match(
    array1: np.ndarray,
    array2: np.ndarray,
    axis1: int = 0,
    axis2: int = 0,
    name1: str = "array1",
    name2: str = "array2",
) -> None:
    """Validate that two arrays have matching dimensions on specified axes.

    Parameters
    ----------
    array1 : np.ndarray
        First array.
    array2 : np.ndarray
        Second array.
    axis1 : int
        Axis of first array to compare.
    axis2 : int
        Axis of second array to compare.
    name1 : str
        Name of first array for error messages.
    name2 : str
        Name of second array for error messages.

    Raises
    ------
    ValueError
        If the dimensions don't match.

    Examples
    --------
    >>> import numpy as np
    >>> validate_dimension_match(np.zeros((3, 1)), np.eye(3), axis1=0, axis2=0)
    >>> validate_dimension_match(np.zeros((4, 1)), np.eye(3), axis1=0, axis2=0)
    Traceback (most recent call last):
        ...
    ValueError: Dimension mismatch: array1 has 4, array2 has 3
    """
    dim1 = array1.shape[axis1]
    dim2 = array2.shape[axis2]
    if dim1 != dim2:
        raise ValueError(f"Dimension mismatch: {name1} has {dim1}, {name2} has {dim2}")
