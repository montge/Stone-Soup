package org.stonesoup;

import java.util.Arrays;
import java.util.Objects;

/**
 * Represents a covariance matrix for state estimation uncertainty.
 *
 * <p>A covariance matrix is a symmetric positive semi-definite matrix that
 * characterizes the uncertainty in a state estimate. The diagonal elements
 * represent variances, and off-diagonal elements represent covariances.</p>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create a 2x2 identity covariance
 * CovarianceMatrix cov = CovarianceMatrix.identity(2);
 *
 * // Create from 2D array
 * double[][] data = {{1.0, 0.5}, {0.5, 2.0}};
 * CovarianceMatrix cov2 = new CovarianceMatrix(data);
 *
 * // Matrix operations
 * double trace = cov.trace();
 * CovarianceMatrix scaled = cov.scale(2.0);
 * }</pre>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public class CovarianceMatrix {

    /** Data stored in row-major format */
    private final double[] data;

    /** Number of rows (and columns, since square) */
    private final int dim;

    /**
     * Creates a covariance matrix from a 2D array.
     *
     * @param matrix the matrix data (rows x cols)
     * @throws IllegalArgumentException if matrix is null, empty, or not square
     */
    public CovarianceMatrix(double[][] matrix) {
        Objects.requireNonNull(matrix, "Matrix cannot be null");
        if (matrix.length == 0) {
            throw new IllegalArgumentException("Matrix cannot be empty");
        }
        this.dim = matrix.length;
        for (double[] row : matrix) {
            if (row == null || row.length != dim) {
                throw new IllegalArgumentException("Matrix must be square");
            }
        }
        this.data = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            System.arraycopy(matrix[i], 0, data, i * dim, dim);
        }
    }

    /**
     * Creates a covariance matrix from row-major data.
     *
     * @param data the row-major matrix data
     * @param dim the dimension (rows = cols)
     */
    private CovarianceMatrix(double[] data, int dim) {
        this.data = data;
        this.dim = dim;
    }

    /**
     * Creates an identity matrix of the specified dimension.
     *
     * @param dim the dimension
     * @return a new identity matrix
     * @throws IllegalArgumentException if dim is less than 1
     */
    public static CovarianceMatrix identity(int dim) {
        if (dim < 1) {
            throw new IllegalArgumentException("Dimension must be at least 1");
        }
        double[] data = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            data[i * dim + i] = 1.0;
        }
        return new CovarianceMatrix(data, dim);
    }

    /**
     * Creates a zero matrix of the specified dimension.
     *
     * @param dim the dimension
     * @return a new zero matrix
     * @throws IllegalArgumentException if dim is less than 1
     */
    public static CovarianceMatrix zeros(int dim) {
        if (dim < 1) {
            throw new IllegalArgumentException("Dimension must be at least 1");
        }
        return new CovarianceMatrix(new double[dim * dim], dim);
    }

    /**
     * Creates a diagonal matrix from the given diagonal values.
     *
     * @param diagonal the diagonal values
     * @return a new diagonal matrix
     * @throws IllegalArgumentException if diagonal is null or empty
     */
    public static CovarianceMatrix diagonal(double[] diagonal) {
        Objects.requireNonNull(diagonal, "Diagonal array cannot be null");
        if (diagonal.length == 0) {
            throw new IllegalArgumentException("Diagonal cannot be empty");
        }
        int dim = diagonal.length;
        double[] data = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            data[i * dim + i] = diagonal[i];
        }
        return new CovarianceMatrix(data, dim);
    }

    /**
     * Gets the dimension of the matrix.
     *
     * @return the dimension (rows = cols)
     */
    public int getDim() {
        return dim;
    }

    /**
     * Gets the element at the specified row and column.
     *
     * @param row the row index (0-based)
     * @param col the column index (0-based)
     * @return the element value
     * @throws IndexOutOfBoundsException if indices are out of range
     */
    public double get(int row, int col) {
        if (row < 0 || row >= dim || col < 0 || col >= dim) {
            throw new IndexOutOfBoundsException(
                    "Index (" + row + ", " + col + ") out of bounds for dimension " + dim);
        }
        return data[row * dim + col];
    }

    /**
     * Sets the element at the specified row and column.
     *
     * @param row the row index (0-based)
     * @param col the column index (0-based)
     * @param value the new value
     * @throws IndexOutOfBoundsException if indices are out of range
     */
    public void set(int row, int col, double value) {
        if (row < 0 || row >= dim || col < 0 || col >= dim) {
            throw new IndexOutOfBoundsException(
                    "Index (" + row + ", " + col + ") out of bounds for dimension " + dim);
        }
        data[row * dim + col] = value;
    }

    /**
     * Returns the matrix as a 2D array.
     *
     * @return a copy of the matrix data
     */
    public double[][] toArray() {
        double[][] result = new double[dim][dim];
        for (int i = 0; i < dim; i++) {
            System.arraycopy(data, i * dim, result[i], 0, dim);
        }
        return result;
    }

    /**
     * Computes the trace (sum of diagonal elements).
     *
     * @return the trace
     */
    public double trace() {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            sum += data[i * dim + i];
        }
        return sum;
    }

    /**
     * Adds another matrix to this one.
     *
     * @param other the matrix to add
     * @return a new matrix containing the sum
     * @throws IllegalArgumentException if dimensions don't match
     */
    public CovarianceMatrix add(CovarianceMatrix other) {
        Objects.requireNonNull(other, "Other matrix cannot be null");
        if (this.dim != other.dim) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + this.dim + " vs " + other.dim);
        }
        double[] result = new double[dim * dim];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] + other.data[i];
        }
        return new CovarianceMatrix(result, dim);
    }

    /**
     * Subtracts another matrix from this one.
     *
     * @param other the matrix to subtract
     * @return a new matrix containing the difference
     * @throws IllegalArgumentException if dimensions don't match
     */
    public CovarianceMatrix subtract(CovarianceMatrix other) {
        Objects.requireNonNull(other, "Other matrix cannot be null");
        if (this.dim != other.dim) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + this.dim + " vs " + other.dim);
        }
        double[] result = new double[dim * dim];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] - other.data[i];
        }
        return new CovarianceMatrix(result, dim);
    }

    /**
     * Scales the matrix by a scalar factor.
     *
     * @param factor the scaling factor
     * @return a new scaled matrix
     */
    public CovarianceMatrix scale(double factor) {
        double[] result = new double[dim * dim];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] * factor;
        }
        return new CovarianceMatrix(result, dim);
    }

    /**
     * Multiplies this matrix by another: C = this * other
     *
     * @param other the right-hand matrix
     * @return a new matrix containing the product
     * @throws IllegalArgumentException if dimensions don't match
     */
    public CovarianceMatrix multiply(CovarianceMatrix other) {
        Objects.requireNonNull(other, "Other matrix cannot be null");
        if (this.dim != other.dim) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + this.dim + " vs " + other.dim);
        }
        double[] result = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < dim; k++) {
                    sum += this.data[i * dim + k] * other.data[k * dim + j];
                }
                result[i * dim + j] = sum;
            }
        }
        return new CovarianceMatrix(result, dim);
    }

    /**
     * Multiplies this matrix by a state vector: y = this * x
     *
     * @param x the input vector
     * @return a new state vector containing the product
     * @throws IllegalArgumentException if dimensions don't match
     */
    public StateVector multiply(StateVector x) {
        Objects.requireNonNull(x, "Vector cannot be null");
        if (this.dim != x.getDim()) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: matrix " + this.dim + " vs vector " + x.getDim());
        }
        double[] result = new double[dim];
        for (int i = 0; i < dim; i++) {
            double sum = 0.0;
            for (int j = 0; j < dim; j++) {
                sum += this.data[i * dim + j] * x.get(j);
            }
            result[i] = sum;
        }
        return new StateVector(result);
    }

    /**
     * Computes the transpose of this matrix.
     *
     * @return a new transposed matrix
     */
    public CovarianceMatrix transpose() {
        double[] result = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                result[j * dim + i] = this.data[i * dim + j];
            }
        }
        return new CovarianceMatrix(result, dim);
    }

    /**
     * Computes A * B^T where A is this matrix.
     *
     * @param other the matrix B
     * @return a new matrix containing A * B^T
     * @throws IllegalArgumentException if dimensions don't match
     */
    public CovarianceMatrix multiplyTranspose(CovarianceMatrix other) {
        Objects.requireNonNull(other, "Other matrix cannot be null");
        if (this.dim != other.dim) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + this.dim + " vs " + other.dim);
        }
        double[] result = new double[dim * dim];
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                double sum = 0.0;
                for (int k = 0; k < dim; k++) {
                    sum += this.data[i * dim + k] * other.data[j * dim + k];
                }
                result[i * dim + j] = sum;
            }
        }
        return new CovarianceMatrix(result, dim);
    }

    /**
     * Creates a copy of this covariance matrix.
     *
     * @return a new matrix with the same values
     */
    public CovarianceMatrix copy() {
        return new CovarianceMatrix(data.clone(), dim);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        CovarianceMatrix that = (CovarianceMatrix) o;
        return dim == that.dim && Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(dim);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("CovarianceMatrix[\n");
        for (int i = 0; i < dim; i++) {
            sb.append("  [");
            for (int j = 0; j < dim; j++) {
                sb.append(String.format("%8.4f", data[i * dim + j]));
                if (j < dim - 1) sb.append(", ");
            }
            sb.append("]\n");
        }
        sb.append("]");
        return sb.toString();
    }
}
