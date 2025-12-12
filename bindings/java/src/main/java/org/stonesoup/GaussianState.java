package org.stonesoup;

import java.util.Objects;
import java.util.Optional;

/**
 * Represents a Gaussian state with mean and covariance.
 *
 * <p>A Gaussian state characterizes the estimated state of a target as a
 * multivariate Gaussian distribution, with a mean (state vector) and
 * covariance matrix representing the uncertainty.</p>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create a 2D state with position and velocity
 * StateVector mean = new StateVector(new double[]{0.0, 1.0, 0.0, 1.0});
 * CovarianceMatrix cov = CovarianceMatrix.identity(4);
 * GaussianState state = new GaussianState(mean, cov);
 *
 * // With timestamp
 * GaussianState timedState = new GaussianState(mean, cov, 0.0);
 *
 * // Access components
 * StateVector m = state.getStateVector();
 * CovarianceMatrix c = state.getCovariance();
 * Optional<Double> t = state.getTimestamp();
 * }</pre>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public class GaussianState {

    /** The mean state vector */
    private final StateVector stateVector;

    /** The covariance matrix */
    private final CovarianceMatrix covariance;

    /** The optional timestamp */
    private Double timestamp;

    /**
     * Creates a new Gaussian state without a timestamp.
     *
     * @param stateVector the mean state vector
     * @param covariance the covariance matrix
     * @throws IllegalArgumentException if dimensions don't match
     */
    public GaussianState(StateVector stateVector, CovarianceMatrix covariance) {
        this(stateVector, covariance, null);
    }

    /**
     * Creates a new Gaussian state with a timestamp.
     *
     * @param stateVector the mean state vector
     * @param covariance the covariance matrix
     * @param timestamp the timestamp (may be null)
     * @throws IllegalArgumentException if dimensions don't match
     */
    public GaussianState(StateVector stateVector, CovarianceMatrix covariance, Double timestamp) {
        Objects.requireNonNull(stateVector, "State vector cannot be null");
        Objects.requireNonNull(covariance, "Covariance cannot be null");
        if (stateVector.getDim() != covariance.getDim()) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: state vector " + stateVector.getDim() +
                    " vs covariance " + covariance.getDim());
        }
        this.stateVector = stateVector;
        this.covariance = covariance;
        this.timestamp = timestamp;
    }

    /**
     * Creates a new Gaussian state from arrays.
     *
     * @param stateVector the mean state vector as an array
     * @param covariance the covariance matrix as a 2D array
     * @return a new Gaussian state
     */
    public static GaussianState of(double[] stateVector, double[][] covariance) {
        return new GaussianState(
                new StateVector(stateVector),
                new CovarianceMatrix(covariance)
        );
    }

    /**
     * Creates a new Gaussian state from arrays with timestamp.
     *
     * @param stateVector the mean state vector as an array
     * @param covariance the covariance matrix as a 2D array
     * @param timestamp the timestamp
     * @return a new Gaussian state
     */
    public static GaussianState of(double[] stateVector, double[][] covariance, double timestamp) {
        return new GaussianState(
                new StateVector(stateVector),
                new CovarianceMatrix(covariance),
                timestamp
        );
    }

    /**
     * Gets the state vector (mean).
     *
     * @return the state vector
     */
    public StateVector getStateVector() {
        return stateVector;
    }

    /**
     * Gets the covariance matrix.
     *
     * @return the covariance matrix
     */
    public CovarianceMatrix getCovariance() {
        return covariance;
    }

    /**
     * Gets the dimension of the state.
     *
     * @return the state dimension
     */
    public int getDim() {
        return stateVector.getDim();
    }

    /**
     * Gets the timestamp if present.
     *
     * @return an Optional containing the timestamp, or empty if not set
     */
    public Optional<Double> getTimestamp() {
        return Optional.ofNullable(timestamp);
    }

    /**
     * Sets the timestamp.
     *
     * @param timestamp the new timestamp (may be null to clear)
     */
    public void setTimestamp(Double timestamp) {
        this.timestamp = timestamp;
    }

    /**
     * Gets a state element by index.
     *
     * @param index the index (0-based)
     * @return the state value at the index
     */
    public double getState(int index) {
        return stateVector.get(index);
    }

    /**
     * Gets the variance (diagonal covariance element) at an index.
     *
     * @param index the index (0-based)
     * @return the variance at the index
     */
    public double getVariance(int index) {
        return covariance.get(index, index);
    }

    /**
     * Gets the standard deviation at an index.
     *
     * @param index the index (0-based)
     * @return the standard deviation at the index
     */
    public double getStdDev(int index) {
        return Math.sqrt(getVariance(index));
    }

    /**
     * Creates a copy of this Gaussian state.
     *
     * @return a new Gaussian state with copied data
     */
    public GaussianState copy() {
        return new GaussianState(
                stateVector.copy(),
                covariance.copy(),
                timestamp
        );
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        GaussianState that = (GaussianState) o;
        return Objects.equals(stateVector, that.stateVector) &&
               Objects.equals(covariance, that.covariance) &&
               Objects.equals(timestamp, that.timestamp);
    }

    @Override
    public int hashCode() {
        return Objects.hash(stateVector, covariance, timestamp);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("GaussianState{\n");
        sb.append("  stateVector=").append(stateVector).append(",\n");
        sb.append("  covariance=").append(covariance).append(",\n");
        if (timestamp != null) {
            sb.append("  timestamp=").append(timestamp).append("\n");
        }
        sb.append("}");
        return sb.toString();
    }
}
