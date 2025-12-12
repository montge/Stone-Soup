package org.stonesoup;

import java.util.Arrays;
import java.util.Objects;

/**
 * Represents an n-dimensional state vector for target tracking.
 *
 * <p>A state vector contains the estimated state of a target, typically including
 * position, velocity, and optionally higher-order derivatives like acceleration.</p>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create a 4D state vector [x, vx, y, vy]
 * StateVector state = new StateVector(new double[]{0.0, 1.0, 0.0, 1.0});
 *
 * // Access elements
 * double x = state.get(0);      // Position x
 * double vx = state.get(1);     // Velocity x
 *
 * // Vector operations
 * StateVector scaled = state.scale(2.0);
 * double magnitude = state.norm();
 * }</pre>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public class StateVector {

    /** The underlying data array */
    private final double[] data;

    /**
     * Creates a new state vector from an array of values.
     *
     * @param data the state vector values (copied)
     * @throws IllegalArgumentException if data is null or empty
     */
    public StateVector(double[] data) {
        Objects.requireNonNull(data, "Data array cannot be null");
        if (data.length == 0) {
            throw new IllegalArgumentException("State vector cannot be empty");
        }
        this.data = data.clone();
    }

    /**
     * Creates a zero state vector of the specified dimension.
     *
     * @param dim the dimension of the state vector
     * @return a new zero state vector
     * @throws IllegalArgumentException if dim is less than 1
     */
    public static StateVector zeros(int dim) {
        if (dim < 1) {
            throw new IllegalArgumentException("Dimension must be at least 1");
        }
        return new StateVector(new double[dim]);
    }

    /**
     * Creates a state vector filled with a constant value.
     *
     * @param dim the dimension of the state vector
     * @param value the fill value
     * @return a new state vector filled with the value
     * @throws IllegalArgumentException if dim is less than 1
     */
    public static StateVector fill(int dim, double value) {
        if (dim < 1) {
            throw new IllegalArgumentException("Dimension must be at least 1");
        }
        double[] data = new double[dim];
        Arrays.fill(data, value);
        return new StateVector(data);
    }

    /**
     * Gets the dimension of the state vector.
     *
     * @return the number of elements
     */
    public int getDim() {
        return data.length;
    }

    /**
     * Gets the value at the specified index.
     *
     * @param index the index (0-based)
     * @return the value at the index
     * @throws IndexOutOfBoundsException if index is out of range
     */
    public double get(int index) {
        return data[index];
    }

    /**
     * Sets the value at the specified index.
     *
     * @param index the index (0-based)
     * @param value the new value
     * @throws IndexOutOfBoundsException if index is out of range
     */
    public void set(int index, double value) {
        data[index] = value;
    }

    /**
     * Returns a copy of the underlying data array.
     *
     * @return a copy of the data array
     */
    public double[] toArray() {
        return data.clone();
    }

    /**
     * Computes the Euclidean (L2) norm of the state vector.
     *
     * @return the Euclidean norm
     */
    public double norm() {
        double sum = 0.0;
        for (double v : data) {
            sum += v * v;
        }
        return Math.sqrt(sum);
    }

    /**
     * Adds another state vector to this one.
     *
     * @param other the state vector to add
     * @return a new state vector containing the sum
     * @throws IllegalArgumentException if dimensions don't match
     */
    public StateVector add(StateVector other) {
        Objects.requireNonNull(other, "Other vector cannot be null");
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + this.data.length + " vs " + other.data.length);
        }
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] + other.data[i];
        }
        return new StateVector(result);
    }

    /**
     * Subtracts another state vector from this one.
     *
     * @param other the state vector to subtract
     * @return a new state vector containing the difference
     * @throws IllegalArgumentException if dimensions don't match
     */
    public StateVector subtract(StateVector other) {
        Objects.requireNonNull(other, "Other vector cannot be null");
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + this.data.length + " vs " + other.data.length);
        }
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] - other.data[i];
        }
        return new StateVector(result);
    }

    /**
     * Scales the state vector by a scalar factor.
     *
     * @param factor the scaling factor
     * @return a new scaled state vector
     */
    public StateVector scale(double factor) {
        double[] result = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = this.data[i] * factor;
        }
        return new StateVector(result);
    }

    /**
     * Computes the dot product with another state vector.
     *
     * @param other the other state vector
     * @return the dot product
     * @throws IllegalArgumentException if dimensions don't match
     */
    public double dot(StateVector other) {
        Objects.requireNonNull(other, "Other vector cannot be null");
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: " + this.data.length + " vs " + other.data.length);
        }
        double sum = 0.0;
        for (int i = 0; i < data.length; i++) {
            sum += this.data[i] * other.data[i];
        }
        return sum;
    }

    /**
     * Creates a copy of this state vector.
     *
     * @return a new state vector with the same values
     */
    public StateVector copy() {
        return new StateVector(data.clone());
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        StateVector that = (StateVector) o;
        return Arrays.equals(data, that.data);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(data);
    }

    @Override
    public String toString() {
        return "StateVector" + Arrays.toString(data);
    }
}
