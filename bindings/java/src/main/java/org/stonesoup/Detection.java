package org.stonesoup;

import java.util.Objects;

/**
 * Represents a detection from a sensor.
 *
 * <p>A detection contains measurement data from a sensor observation,
 * typically including position information and a timestamp.</p>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create a 2D position detection
 * StateVector measurement = new StateVector(new double[]{100.0, 200.0});
 * Detection detection = new Detection(measurement, 0.0);
 *
 * // Access components
 * StateVector m = detection.getMeasurement();
 * double t = detection.getTimestamp();
 * }</pre>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public class Detection {

    /** The measurement vector */
    private final StateVector measurement;

    /** The timestamp of the detection */
    private final double timestamp;

    /** Optional metadata */
    private Object metadata;

    /**
     * Creates a new detection.
     *
     * @param measurement the measurement vector
     * @param timestamp the timestamp
     * @throws NullPointerException if measurement is null
     */
    public Detection(StateVector measurement, double timestamp) {
        this.measurement = Objects.requireNonNull(measurement, "Measurement cannot be null");
        this.timestamp = timestamp;
    }

    /**
     * Creates a new detection from an array.
     *
     * @param measurement the measurement values
     * @param timestamp the timestamp
     * @return a new Detection
     */
    public static Detection of(double[] measurement, double timestamp) {
        return new Detection(new StateVector(measurement), timestamp);
    }

    /**
     * Gets the measurement vector.
     *
     * @return the measurement vector
     */
    public StateVector getMeasurement() {
        return measurement;
    }

    /**
     * Gets the timestamp.
     *
     * @return the timestamp
     */
    public double getTimestamp() {
        return timestamp;
    }

    /**
     * Gets the measurement dimension.
     *
     * @return the number of measurement components
     */
    public int getDim() {
        return measurement.getDim();
    }

    /**
     * Gets a measurement value by index.
     *
     * @param index the index (0-based)
     * @return the measurement value
     */
    public double get(int index) {
        return measurement.get(index);
    }

    /**
     * Gets optional metadata attached to this detection.
     *
     * @return the metadata, or null if not set
     */
    public Object getMetadata() {
        return metadata;
    }

    /**
     * Sets optional metadata for this detection.
     *
     * @param metadata the metadata to attach
     */
    public void setMetadata(Object metadata) {
        this.metadata = metadata;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Detection detection = (Detection) o;
        return Double.compare(detection.timestamp, timestamp) == 0 &&
               Objects.equals(measurement, detection.measurement);
    }

    @Override
    public int hashCode() {
        return Objects.hash(measurement, timestamp);
    }

    @Override
    public String toString() {
        return "Detection{" +
               "measurement=" + measurement +
               ", timestamp=" + timestamp +
               '}';
    }
}
