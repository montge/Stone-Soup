package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.AfterAll;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link StoneSoup} main class.
 */
@DisplayName("StoneSoup")
class StoneSoupTest {

    @BeforeAll
    static void setUp() throws StoneSoupException {
        StoneSoup.initialize();
    }

    @AfterAll
    static void tearDown() {
        StoneSoup.cleanup();
    }

    @Nested
    @DisplayName("Initialization")
    class Initialization {

        @Test
        @DisplayName("reports correct version")
        void reportsCorrectVersion() {
            assertEquals("0.1.0", StoneSoup.getVersion());
        }

        @Test
        @DisplayName("reports initialized status")
        void reportsInitializedStatus() {
            assertTrue(StoneSoup.isInitialized());
        }

        @Test
        @DisplayName("reports execution mode")
        void reportsExecutionMode() {
            String mode = StoneSoup.getMode();
            assertTrue(mode.equals("native") || mode.equals("java"));
        }
    }

    @Nested
    @DisplayName("Detection")
    class DetectionTests {

        @Test
        @DisplayName("creates detection with measurement and timestamp")
        void createsDetection() {
            Detection detection = Detection.of(new double[]{100.0, 200.0}, 0.0);

            assertEquals(2, detection.getDim());
            assertEquals(100.0, detection.get(0), 1e-10);
            assertEquals(200.0, detection.get(1), 1e-10);
            assertEquals(0.0, detection.getTimestamp(), 1e-10);
        }

        @Test
        @DisplayName("supports metadata")
        void supportsMetadata() {
            Detection detection = Detection.of(new double[]{1.0}, 0.0);

            assertNull(detection.getMetadata());

            detection.setMetadata("sensor-1");
            assertEquals("sensor-1", detection.getMetadata());
        }
    }

    @Nested
    @DisplayName("Track")
    class TrackTests {

        @Test
        @DisplayName("creates empty track")
        void createsEmptyTrack() {
            Track track = new Track();

            assertNotNull(track.getId());
            assertTrue(track.isEmpty());
            assertEquals(0, track.getLength());
            assertNull(track.getCurrentState());
        }

        @Test
        @DisplayName("accumulates states")
        void accumulatesStates() {
            Track track = new Track("test-track");

            GaussianState state1 = GaussianState.of(
                    new double[]{0.0, 1.0},
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
        @DisplayName("computes duration")
        void computesDuration() {
            Track track = new Track();

            track.addState(GaussianState.of(
                    new double[]{0.0},
                    new double[][]{{1.0}},
                    0.0
            ));
            track.addState(GaussianState.of(
                    new double[]{1.0},
                    new double[][]{{1.0}},
                    5.0
            ));

            assertEquals(5.0, track.getDuration(), 1e-10);
        }
    }

    @Nested
    @DisplayName("Exception")
    class ExceptionTests {

        @Test
        @DisplayName("formats error messages correctly")
        void formatsErrorMessagesCorrectly() {
            assertEquals("Success", StoneSoupException.getErrorMessage(0));
            assertEquals("Null pointer argument", StoneSoupException.getErrorMessage(1));
            assertEquals("Dimension mismatch", StoneSoupException.getErrorMessage(4));
            assertEquals("Singular matrix", StoneSoupException.getErrorMessage(5));
        }

        @Test
        @DisplayName("checkError throws on non-zero code")
        void checkErrorThrowsOnNonZeroCode() {
            assertDoesNotThrow(() -> StoneSoupException.checkError(0));

            StoneSoupException ex = assertThrows(StoneSoupException.class,
                    () -> StoneSoupException.checkError(4));
            assertEquals(4, ex.getErrorCode());
            assertTrue(ex.getMessage().contains("Dimension mismatch"));
        }
    }

    @Nested
    @DisplayName("Integration")
    class Integration {

        @Test
        @DisplayName("runs complete tracking workflow")
        void runsCompleteTrackingWorkflow() throws StoneSoupException {
            // Create a track
            Track track = new Track("target-1");

            // Initial state
            GaussianState state = GaussianState.of(
                    new double[]{0.0, 1.0, 0.0, 1.0},
                    CovarianceMatrix.identity(4).toArray(),
                    0.0
            );
            track.addState(state);

            // Kalman filter parameters
            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(2, 1.0);
            CovarianceMatrix Q = CovarianceMatrix.identity(4).scale(0.01);
            CovarianceMatrix H = KalmanFilter.positionMeasurement(2);
            CovarianceMatrix R = CovarianceMatrix.identity(2).scale(0.1);

            // Process detections
            Detection[] detections = {
                Detection.of(new double[]{1.0, 1.0}, 1.0),
                Detection.of(new double[]{2.0, 2.0}, 2.0),
                Detection.of(new double[]{3.0, 3.0}, 3.0)
            };

            for (Detection detection : detections) {
                // Predict
                state = KalmanFilter.predict(state, F, Q);
                state.setTimestamp(detection.getTimestamp());

                // Update
                state = KalmanFilter.update(state, detection.getMeasurement(), H, R);
                track.addState(state);
            }

            // Verify track
            assertEquals(4, track.getLength());
            assertEquals(3.0, track.getDuration(), 0.1);

            // Final state should be near (3, 3)
            GaussianState finalState = track.getCurrentState();
            assertEquals(3.0, finalState.getState(0), 0.5);
            assertEquals(3.0, finalState.getState(2), 0.5);
        }
    }
}
