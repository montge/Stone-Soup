package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Track}.
 */
@DisplayName("Track")
class TrackTest {

    @Nested
    @DisplayName("Construction")
    class Construction {

        @Test
        @DisplayName("creates track with random UUID")
        void createsTrackWithRandomUUID() {
            Track track = new Track();
            assertNotNull(track.getId());
            assertTrue(track.isEmpty());
            assertEquals(0, track.getLength());
        }

        @Test
        @DisplayName("creates track with specified ID")
        void createsTrackWithSpecifiedId() {
            Track track = new Track("test-track-1");
            assertEquals("test-track-1", track.getId());
        }

        @Test
        @DisplayName("rejects null ID")
        void rejectsNullId() {
            assertThrows(NullPointerException.class, () -> new Track(null));
        }

        @Test
        @DisplayName("creates track with initial state")
        void createsTrackWithInitialState() {
            GaussianState state = GaussianState.of(
                new double[]{1.0, 2.0},
                CovarianceMatrix.identity(2).toArray()
            );

            Track track = Track.withInitialState(state);

            assertEquals(1, track.getLength());
            assertFalse(track.isEmpty());
            assertNotNull(track.getCurrentState());
        }
    }

    @Nested
    @DisplayName("State Management")
    class StateManagement {

        @Test
        @DisplayName("adds states correctly")
        void addsStatesCorrectly() {
            Track track = new Track();

            GaussianState state1 = GaussianState.of(
                new double[]{0.0, 0.0},
                CovarianceMatrix.identity(2).toArray(),
                0.0
            );
            GaussianState state2 = GaussianState.of(
                new double[]{1.0, 1.0},
                CovarianceMatrix.identity(2).toArray(),
                1.0
            );

            track.addState(state1);
            track.addState(state2);

            assertEquals(2, track.getLength());
            assertEquals(state1, track.getFirstState());
            assertEquals(state2, track.getCurrentState());
        }

        @Test
        @DisplayName("rejects null state")
        void rejectsNullState() {
            Track track = new Track();
            assertThrows(NullPointerException.class, () -> track.addState(null));
        }

        @Test
        @DisplayName("gets state by index")
        void getsStateByIndex() {
            Track track = new Track();
            GaussianState state = GaussianState.of(
                new double[]{1.0},
                CovarianceMatrix.identity(1).toArray()
            );
            track.addState(state);

            assertEquals(state, track.getState(0));
        }

        @Test
        @DisplayName("throws on invalid index")
        void throwsOnInvalidIndex() {
            Track track = new Track();
            assertThrows(IndexOutOfBoundsException.class, () -> track.getState(0));
        }

        @Test
        @DisplayName("returns null for current state on empty track")
        void returnsNullForCurrentStateOnEmptyTrack() {
            Track track = new Track();
            assertNull(track.getCurrentState());
        }

        @Test
        @DisplayName("returns null for first state on empty track")
        void returnsNullForFirstStateOnEmptyTrack() {
            Track track = new Track();
            assertNull(track.getFirstState());
        }

        @Test
        @DisplayName("clears all states")
        void clearsAllStates() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{1.0}, CovarianceMatrix.identity(1).toArray()));
            track.addState(GaussianState.of(new double[]{2.0}, CovarianceMatrix.identity(1).toArray()));

            track.clear();

            assertTrue(track.isEmpty());
            assertEquals(0, track.getLength());
        }
    }

    @Nested
    @DisplayName("State Access")
    class StateAccess {

        @Test
        @DisplayName("gets all states as unmodifiable list")
        void getsAllStatesAsUnmodifiableList() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{1.0}, CovarianceMatrix.identity(1).toArray()));

            List<GaussianState> states = track.getStates();
            assertThrows(UnsupportedOperationException.class,
                () -> states.add(GaussianState.of(new double[]{2.0}, CovarianceMatrix.identity(1).toArray())));
        }

        @Test
        @DisplayName("gets last N states")
        void getsLastNStates() {
            Track track = new Track();
            for (int i = 0; i < 5; i++) {
                track.addState(GaussianState.of(new double[]{(double)i}, CovarianceMatrix.identity(1).toArray()));
            }

            List<GaussianState> last3 = track.getLastNStates(3);
            assertEquals(3, last3.size());
            assertEquals(2.0, last3.get(0).getState(0), 0.001);
            assertEquals(4.0, last3.get(2).getState(0), 0.001);
        }

        @Test
        @DisplayName("handles last N with N larger than track length")
        void handlesLastNWithNLargerThanTrackLength() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{1.0}, CovarianceMatrix.identity(1).toArray()));

            List<GaussianState> states = track.getLastNStates(10);
            assertEquals(1, states.size());
        }

        @Test
        @DisplayName("handles last N with N <= 0")
        void handlesLastNWithNZeroOrNegative() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{1.0}, CovarianceMatrix.identity(1).toArray()));

            assertTrue(track.getLastNStates(0).isEmpty());
            assertTrue(track.getLastNStates(-1).isEmpty());
        }
    }

    @Nested
    @DisplayName("Duration and Timestamp")
    class DurationAndTimestamp {

        @Test
        @DisplayName("calculates duration correctly")
        void calculatesDurationCorrectly() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{0.0}, CovarianceMatrix.identity(1).toArray(), 0.0));
            track.addState(GaussianState.of(new double[]{1.0}, CovarianceMatrix.identity(1).toArray(), 1.0));
            track.addState(GaussianState.of(new double[]{2.0}, CovarianceMatrix.identity(1).toArray(), 2.5));

            assertEquals(2.5, track.getDuration(), 0.001);
        }

        @Test
        @DisplayName("returns zero duration for single state")
        void returnsZeroDurationForSingleState() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{0.0}, CovarianceMatrix.identity(1).toArray(), 0.0));

            assertEquals(0.0, track.getDuration(), 0.001);
        }

        @Test
        @DisplayName("returns zero duration for empty track")
        void returnsZeroDurationForEmptyTrack() {
            Track track = new Track();
            assertEquals(0.0, track.getDuration(), 0.001);
        }

        @Test
        @DisplayName("returns zero duration for states without timestamps")
        void returnsZeroDurationForStatesWithoutTimestamps() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{0.0}, CovarianceMatrix.identity(1).toArray()));
            track.addState(GaussianState.of(new double[]{1.0}, CovarianceMatrix.identity(1).toArray()));

            assertEquals(0.0, track.getDuration(), 0.001);
        }

        @Test
        @DisplayName("gets current timestamp")
        void getsCurrentTimestamp() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{0.0}, CovarianceMatrix.identity(1).toArray(), 5.0));

            assertEquals(5.0, track.getCurrentTimestamp(), 0.001);
        }

        @Test
        @DisplayName("returns null timestamp for empty track")
        void returnsNullTimestampForEmptyTrack() {
            Track track = new Track();
            assertNull(track.getCurrentTimestamp());
        }

        @Test
        @DisplayName("returns null timestamp when state has no timestamp")
        void returnsNullTimestampWhenStateHasNoTimestamp() {
            Track track = new Track();
            track.addState(GaussianState.of(new double[]{0.0}, CovarianceMatrix.identity(1).toArray()));

            assertNull(track.getCurrentTimestamp());
        }
    }

    @Nested
    @DisplayName("Metadata")
    class Metadata {

        @Test
        @DisplayName("stores and retrieves metadata")
        void storesAndRetrievesMetadata() {
            Track track = new Track();
            track.setMetadata("test-metadata");

            assertEquals("test-metadata", track.getMetadata());
        }

        @Test
        @DisplayName("returns null for unset metadata")
        void returnsNullForUnsetMetadata() {
            Track track = new Track();
            assertNull(track.getMetadata());
        }
    }

    @Nested
    @DisplayName("Equality and HashCode")
    class EqualityAndHashCode {

        @Test
        @DisplayName("equals based on ID")
        void equalsBasedOnId() {
            Track track1 = new Track("same-id");
            Track track2 = new Track("same-id");
            Track track3 = new Track("different-id");

            assertEquals(track1, track2);
            assertNotEquals(track1, track3);
        }

        @Test
        @DisplayName("equals handles null and other types")
        void equalsHandlesNullAndOtherTypes() {
            Track track = new Track();

            assertNotEquals(track, null);
            assertNotEquals(track, "string");
        }

        @Test
        @DisplayName("equals handles same reference")
        void equalsHandlesSameReference() {
            Track track = new Track();
            assertEquals(track, track);
        }

        @Test
        @DisplayName("hashCode consistent with equals")
        void hashCodeConsistentWithEquals() {
            Track track1 = new Track("same-id");
            Track track2 = new Track("same-id");

            assertEquals(track1.hashCode(), track2.hashCode());
        }
    }

    @Nested
    @DisplayName("ToString")
    class ToStringTest {

        @Test
        @DisplayName("produces readable output")
        void producesReadableOutput() {
            Track track = new Track("test-id");
            track.addState(GaussianState.of(new double[]{1.0}, CovarianceMatrix.identity(1).toArray()));

            String str = track.toString();
            assertTrue(str.contains("test-id"));
            assertTrue(str.contains("length=1"));
        }
    }
}
