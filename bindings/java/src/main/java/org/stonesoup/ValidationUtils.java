// SPDX-FileCopyrightText: 2024-2025 Stone Soup Contributors
// SPDX-License-Identifier: MIT

package org.stonesoup;

import java.util.Objects;

/**
 * Common validation utilities for Stone Soup types.
 *
 * <p>This class provides reusable validation methods to reduce code duplication
 * across StateVector, Matrix, and CovarianceMatrix classes.</p>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public final class ValidationUtils {

    private ValidationUtils() {
        // Utility class - prevent instantiation
    }

    /**
     * Validates that an array is not null and not empty.
     *
     * @param array the array to validate
     * @param name the name for error messages
     * @throws NullPointerException if array is null
     * @throws IllegalArgumentException if array is empty
     */
    public static void requireNonEmpty(double[] array, String name) {
        Objects.requireNonNull(array, name + " cannot be null");
        if (array.length == 0) {
            throw new IllegalArgumentException(name + " cannot be empty");
        }
    }

    /**
     * Validates that a 2D array is not null, not empty, and has consistent row lengths.
     *
     * @param matrix the matrix to validate
     * @param name the name for error messages
     * @throws NullPointerException if matrix is null
     * @throws IllegalArgumentException if matrix is empty or has inconsistent rows
     */
    public static void requireNonEmpty(double[][] matrix, String name) {
        Objects.requireNonNull(matrix, name + " cannot be null");
        if (matrix.length == 0) {
            throw new IllegalArgumentException(name + " cannot be empty");
        }
        int cols = matrix[0].length;
        if (cols == 0) {
            throw new IllegalArgumentException(name + " columns cannot be empty");
        }
        for (double[] row : matrix) {
            if (row == null || row.length != cols) {
                throw new IllegalArgumentException("All rows must have the same length");
            }
        }
    }

    /**
     * Validates that a 2D array is square.
     *
     * @param matrix the matrix to validate
     * @param name the name for error messages
     * @throws IllegalArgumentException if matrix is not square
     */
    public static void requireSquare(double[][] matrix, String name) {
        requireNonEmpty(matrix, name);
        int rows = matrix.length;
        for (double[] row : matrix) {
            if (row == null || row.length != rows) {
                throw new IllegalArgumentException(name + " must be square");
            }
        }
    }

    /**
     * Validates that a dimension is at least 1.
     *
     * @param dim the dimension to validate
     * @param name the name for error messages
     * @throws IllegalArgumentException if dimension is less than 1
     */
    public static void requirePositiveDimension(int dim, String name) {
        if (dim < 1) {
            throw new IllegalArgumentException(name + " must be at least 1");
        }
    }

    /**
     * Validates that two dimensions match.
     *
     * @param dim1 first dimension
     * @param dim2 second dimension
     * @param name1 name of first dimension for error messages
     * @param name2 name of second dimension for error messages
     * @throws IllegalArgumentException if dimensions don't match
     */
    public static void requireMatchingDimensions(int dim1, int dim2, String name1, String name2) {
        if (dim1 != dim2) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + name1 + " " + dim1 + " vs " + name2 + " " + dim2);
        }
    }

    /**
     * Validates that matrix dimensions are compatible for multiplication.
     *
     * @param rows1 rows of first matrix
     * @param cols1 columns of first matrix
     * @param rows2 rows of second matrix
     * @param cols2 columns of second matrix
     * @throws IllegalArgumentException if dimensions are incompatible
     */
    public static void requireMultiplicationCompatible(int rows1, int cols1, int rows2, int cols2) {
        if (cols1 != rows2) {
            throw new IllegalArgumentException(
                    "Matrix dimensions don't match for multiplication: " +
                            rows1 + "x" + cols1 + " * " + rows2 + "x" + cols2);
        }
    }

    /**
     * Validates that an index is within bounds.
     *
     * @param index the index to check
     * @param size the size of the array/dimension
     * @param name the name for error messages
     * @throws IndexOutOfBoundsException if index is out of bounds
     */
    public static void requireInBounds(int index, int size, String name) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException(
                    name + " " + index + " out of bounds for size " + size);
        }
    }

    /**
     * Validates that row and column indices are within bounds for a matrix.
     *
     * @param row the row index
     * @param col the column index
     * @param rows number of rows in matrix
     * @param cols number of columns in matrix
     * @throws IndexOutOfBoundsException if indices are out of bounds
     */
    public static void requireInBounds(int row, int col, int rows, int cols) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IndexOutOfBoundsException(
                    "Index (" + row + ", " + col + ") out of bounds for " + rows + "x" + cols + " matrix");
        }
    }
}
