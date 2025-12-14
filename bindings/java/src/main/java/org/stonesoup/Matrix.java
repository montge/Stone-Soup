// SPDX-FileCopyrightText: 2024 Stone Soup Contributors
// SPDX-License-Identifier: MIT

package org.stonesoup;

import java.util.Arrays;
import java.util.Objects;

/**
 * A general-purpose matrix that supports non-square dimensions.
 *
 * <p>This class is used for measurement matrices (H), transition matrices (F),
 * and other general linear algebra operations where the matrix need not be square.</p>
 */
public class Matrix {

    protected final double[] data;
    protected final int rows;
    protected final int cols;

    /**
     * Creates a matrix from a 2D array.
     *
     * @param matrix the 2D array representation
     * @throws IllegalArgumentException if matrix is empty or has inconsistent row lengths
     */
    public Matrix(double[][] matrix) {
        Objects.requireNonNull(matrix, "Matrix cannot be null");
        if (matrix.length == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }
        this.rows = matrix.length;
        this.cols = matrix[0].length;
        if (cols == 0) {
            throw new IllegalArgumentException("Matrix columns cannot be empty");
        }
        for (double[] row : matrix) {
            if (row == null || row.length != cols) {
                throw new IllegalArgumentException("All rows must have the same length");
            }
        }
        this.data = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, data, i * cols, cols);
        }
    }

    /**
     * Creates a matrix from row-major data.
     *
     * @param data the row-major matrix data
     * @param rows the number of rows
     * @param cols the number of columns
     */
    protected Matrix(double[] data, int rows, int cols) {
        this.data = data;
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Creates a zero matrix of the given dimensions.
     *
     * @param rows the number of rows
     * @param cols the number of columns
     * @return a zero matrix
     */
    public static Matrix zeros(int rows, int cols) {
        return new Matrix(new double[rows * cols], rows, cols);
    }

    /**
     * Creates an identity matrix of the given size.
     *
     * @param size the dimension
     * @return an identity matrix
     */
    public static Matrix identity(int size) {
        double[] data = new double[size * size];
        for (int i = 0; i < size; i++) {
            data[i * size + i] = 1.0;
        }
        return new Matrix(data, size, size);
    }

    /**
     * Gets the number of rows.
     *
     * @return the number of rows
     */
    public int getRows() {
        return rows;
    }

    /**
     * Gets the number of columns.
     *
     * @return the number of columns
     */
    public int getCols() {
        return cols;
    }

    /**
     * Gets an element at the specified position.
     *
     * @param row the row index
     * @param col the column index
     * @return the element value
     */
    public double get(int row, int col) {
        return data[row * cols + col];
    }

    /**
     * Sets an element at the specified position.
     *
     * @param row the row index
     * @param col the column index
     * @param value the new value
     */
    public void set(int row, int col, double value) {
        data[row * cols + col] = value;
    }

    /**
     * Returns the matrix as a 2D array.
     *
     * @return a 2D array copy
     */
    public double[][] toArray() {
        double[][] result = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data, i * cols, result[i], 0, cols);
        }
        return result;
    }

    /**
     * Returns the transpose of this matrix.
     *
     * @return the transposed matrix
     */
    public Matrix transpose() {
        double[] result = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j * rows + i] = data[i * cols + j];
            }
        }
        return new Matrix(result, cols, rows);
    }

    /**
     * Multiplies this matrix by another matrix.
     *
     * @param other the matrix to multiply by
     * @return the product matrix
     * @throws IllegalArgumentException if dimensions don't match
     */
    public Matrix multiply(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException(
                    "Matrix dimensions don't match for multiplication: " +
                            this.rows + "x" + this.cols + " * " + other.rows + "x" + other.cols);
        }
        double[] result = new double[this.rows * other.cols];
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) {
                    sum += this.data[i * this.cols + k] * other.data[k * other.cols + j];
                }
                result[i * other.cols + j] = sum;
            }
        }
        return new Matrix(result, this.rows, other.cols);
    }

    /**
     * Multiplies this matrix by a vector.
     *
     * @param vector the vector to multiply
     * @return the result vector
     * @throws IllegalArgumentException if dimensions don't match
     */
    public StateVector multiply(StateVector vector) {
        if (this.cols != vector.getDimension()) {
            throw new IllegalArgumentException(
                    "Matrix columns must match vector dimension: " +
                            this.cols + " != " + vector.getDimension());
        }
        double[] result = new double[this.rows];
        for (int i = 0; i < this.rows; i++) {
            double sum = 0;
            for (int j = 0; j < this.cols; j++) {
                sum += this.data[i * this.cols + j] * vector.get(j);
            }
            result[i] = sum;
        }
        return new StateVector(result);
    }

    /**
     * Adds another matrix to this one.
     *
     * @param other the matrix to add
     * @return the sum matrix
     * @throws IllegalArgumentException if dimensions don't match
     */
    public Matrix add(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must match for addition");
        }
        double[] result = new double[rows * cols];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] + other.data[i];
        }
        return new Matrix(result, rows, cols);
    }

    /**
     * Scales this matrix by a scalar.
     *
     * @param scalar the scaling factor
     * @return the scaled matrix
     */
    public Matrix scale(double scalar) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] * scalar;
        }
        return new Matrix(result, rows, cols);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Matrix)) return false;
        Matrix matrix = (Matrix) o;
        return rows == matrix.rows && cols == matrix.cols && Arrays.equals(data, matrix.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(rows, cols);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Matrix[" + rows + "x" + cols + "]\n");
        for (int i = 0; i < rows; i++) {
            sb.append("  [");
            for (int j = 0; j < cols; j++) {
                if (j > 0) sb.append(", ");
                sb.append(String.format("%.4f", data[i * cols + j]));
            }
            sb.append("]\n");
        }
        return sb.toString();
    }
}
