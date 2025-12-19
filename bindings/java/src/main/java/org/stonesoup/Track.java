package org.stonesoup;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;
import java.util.UUID;

/**
 * Represents a track (sequence of states) for a target.
 *
 * <p>A track maintains a history of Gaussian states representing the
 * estimated trajectory of a target over time.</p>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create a new track
 * Track track = new Track();
 *
 * // Add states
 * GaussianState state1 = GaussianState.of(
 *     new double[]{0, 1, 0, 1},
 *     CovarianceMatrix.identity(4).toArray(),
 *     0.0
 * );
 * track.addState(state1);
 *
 * // Access track information
 * int length = track.getLength();
 * GaussianState current = track.getCurrentState();
 * List<GaussianState> history = track.getStates();
 * }</pre>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public class Track {

    /** Unique track identifier */
    private final String id;

    /** List of states in chronological order */
    private final List<GaussianState> states;

    /** Optional track metadata */
    private Object metadata;

    /**
     * Creates a new track with a random UUID.
     */
    public Track() {
        this(UUID.randomUUID().toString());
    }

    /**
     * Creates a new track with the specified ID.
     *
     * @param id the track identifier
     * @throws NullPointerException if id is null
     */
    public Track(String id) {
        this.id = Objects.requireNonNull(id, "Track ID cannot be null");
        this.states = new ArrayList<>();
    }

    /**
     * Creates a new track with an initial state.
     *
     * @param initialState the initial state
     * @return a new Track
     */
    public static Track withInitialState(GaussianState initialState) {
        Track track = new Track();
        track.addState(initialState);
        return track;
    }

    /**
     * Gets the track ID.
     *
     * @return the track identifier
     */
    public String getId() {
        return id;
    }

    /**
     * Gets the number of states in the track.
     *
     * @return the track length
     */
    public int getLength() {
        return states.size();
    }

    /**
     * Checks if the track is empty.
     *
     * @return true if the track has no states
     */
    public boolean isEmpty() {
        return states.isEmpty();
    }

    /**
     * Adds a state to the track.
     *
     * @param state the state to add
     * @throws NullPointerException if state is null
     */
    public void addState(GaussianState state) {
        states.add(Objects.requireNonNull(state, "State cannot be null"));
    }

    /**
     * Gets the current (most recent) state.
     *
     * @return the current state, or null if track is empty
     */
    public GaussianState getCurrentState() {
        if (states.isEmpty()) {
            return null;
        }
        return states.get(states.size() - 1);
    }

    /**
     * Gets the first state.
     *
     * @return the first state, or null if track is empty
     */
    public GaussianState getFirstState() {
        if (states.isEmpty()) {
            return null;
        }
        return states.get(0);
    }

    /**
     * Gets a state by index.
     *
     * @param index the index (0-based, 0 is oldest)
     * @return the state at the index
     * @throws IndexOutOfBoundsException if index is out of range
     */
    public GaussianState getState(int index) {
        return states.get(index);
    }

    /**
     * Gets all states as an unmodifiable list.
     *
     * @return the list of states (oldest to newest)
     */
    public List<GaussianState> getStates() {
        return Collections.unmodifiableList(states);
    }

    /**
     * Gets the last N states.
     *
     * @param n the number of states to get
     * @return a list of the most recent n states
     */
    public List<GaussianState> getLastNStates(int n) {
        if (n <= 0) {
            return Collections.emptyList();
        }
        int start = Math.max(0, states.size() - n);
        return Collections.unmodifiableList(states.subList(start, states.size()));
    }

    /**
     * Gets the track duration (time between first and last state).
     *
     * @return the duration, or 0 if track has less than 2 states or no timestamps
     */
    public double getDuration() {
        if (states.size() < 2) {
            return 0.0;
        }
        GaussianState first = states.get(0);
        GaussianState last = states.get(states.size() - 1);

        return first.getTimestamp()
                .flatMap(t1 -> last.getTimestamp().map(t2 -> t2 - t1))
                .orElse(0.0);
    }

    /**
     * Gets the timestamp of the current state.
     *
     * @return the current timestamp, or null if no current state or no timestamp
     */
    public Double getCurrentTimestamp() {
        GaussianState current = getCurrentState();
        if (current == null) {
            return null;
        }
        return current.getTimestamp().orElse(null);
    }

    /**
     * Gets optional metadata attached to this track.
     *
     * @return the metadata, or null if not set
     */
    public Object getMetadata() {
        return metadata;
    }

    /**
     * Sets optional metadata for this track.
     *
     * @param metadata the metadata to attach
     */
    public void setMetadata(Object metadata) {
        this.metadata = metadata;
    }

    /**
     * Clears all states from the track.
     */
    public void clear() {
        states.clear();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Track track = (Track) o;
        return Objects.equals(id, track.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return "Track{" +
               "id='" + id + '\'' +
               ", length=" + states.size() +
               ", currentState=" + getCurrentState() +
               '}';
    }
}
