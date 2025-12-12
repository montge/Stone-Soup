package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link KalmanFilter}.
 */
@DisplayName("KalmanFilter")
class KalmanFilterTest {

    private static final double EPSILON = 1e-6;

    @Nested
    @DisplayName("Prediction")
    class Prediction {

        @Test
        @DisplayName("predicts with constant velocity model")
        void predictsWithConstantVelocityModel() throws StoneSoupException {
            // Initial state: x=0, vx=1, y=0, vy=1
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 1.0, 0.0, 1.0},
                    CovarianceMatrix.identity(4).toArray()
            );

            // Transition matrix for dt=1
            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(2, 1.0);
            CovarianceMatrix Q = CovarianceMatrix.identity(4).scale(0.1);

            GaussianState predicted = KalmanFilter.predict(prior, F, Q);

            // After dt=1: x=0+1*1=1, vx=1, y=0+1*1=1, vy=1
            assertEquals(1.0, predicted.getState(0), EPSILON);
            assertEquals(1.0, predicted.getState(1), EPSILON);
            assertEquals(1.0, predicted.getState(2), EPSILON);
            assertEquals(1.0, predicted.getState(3), EPSILON);
        }

        @Test
        @DisplayName("covariance grows with process noise")
        void covarianceGrowsWithProcessNoise() throws StoneSoupException {
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 1.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(1, 1.0);
            CovarianceMatrix Q = CovarianceMatrix.identity(2).scale(0.1);

            GaussianState predicted = KalmanFilter.predict(prior, F, Q);

            // Covariance should grow
            assertTrue(predicted.getVariance(0) > prior.getVariance(0));
        }

        @Test
        @DisplayName("rejects dimension mismatch")
        void rejectsDimensionMismatch() {
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 1.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            CovarianceMatrix F = CovarianceMatrix.identity(4);
            CovarianceMatrix Q = CovarianceMatrix.identity(2);

            assertThrows(IllegalArgumentException.class,
                    () -> KalmanFilter.predict(prior, F, Q));
        }
    }

    @Nested
    @DisplayName("Update")
    class Update {

        @Test
        @DisplayName("updates with measurement")
        void updatesWithMeasurement() throws StoneSoupException {
            // Predicted state at position (1, 1)
            GaussianState predicted = GaussianState.of(
                    new double[]{1.0, 1.0, 1.0, 1.0},
                    CovarianceMatrix.identity(4).toArray()
            );

            // Measurement at (1.1, 0.9)
            StateVector measurement = new StateVector(new double[]{1.1, 0.9});
            CovarianceMatrix H = KalmanFilter.positionMeasurement(2);
            CovarianceMatrix R = CovarianceMatrix.identity(2).scale(0.5);

            GaussianState posterior = KalmanFilter.update(predicted, measurement, H, R);

            // Posterior should be between prediction and measurement
            assertTrue(posterior.getState(0) > 1.0);  // Pulled toward measurement
            assertTrue(posterior.getState(0) < 1.1);
            assertTrue(posterior.getState(2) < 1.0);  // Pulled toward measurement
            assertTrue(posterior.getState(2) > 0.9);
        }

        @Test
        @DisplayName("reduces uncertainty after update")
        void reducesUncertaintyAfterUpdate() throws StoneSoupException {
            GaussianState predicted = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).scale(2.0).toArray()
            );

            StateVector measurement = new StateVector(new double[]{0.1});
            CovarianceMatrix H = new CovarianceMatrix(new double[][]{{1.0, 0.0}});
            CovarianceMatrix R = CovarianceMatrix.identity(1).scale(0.5);

            GaussianState posterior = KalmanFilter.update(predicted, measurement, H, R);

            // First state's variance should decrease
            assertTrue(posterior.getVariance(0) < predicted.getVariance(0));
        }

        @Test
        @DisplayName("handles exact measurement")
        void handlesExactMeasurement() throws StoneSoupException {
            GaussianState predicted = GaussianState.of(
                    new double[]{10.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            // Measurement exactly matches prediction
            StateVector measurement = new StateVector(new double[]{10.0});
            CovarianceMatrix H = new CovarianceMatrix(new double[][]{{1.0, 0.0}});
            CovarianceMatrix R = CovarianceMatrix.identity(1);

            GaussianState posterior = KalmanFilter.update(predicted, measurement, H, R);

            // State should stay near prediction
            assertEquals(10.0, posterior.getState(0), EPSILON);
        }
    }

    @Nested
    @DisplayName("Helper Functions")
    class HelperFunctions {

        @Test
        @DisplayName("constantVelocityTransition creates correct matrix")
        void constantVelocityTransitionCreatesCorrectMatrix() {
            // 2D case
            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(2, 0.5);

            assertEquals(4, F.getDim());

            // Check structure
            // | 1  0.5  0   0  |
            // | 0   1   0   0  |
            // | 0   0   1  0.5 |
            // | 0   0   0   1  |
            assertEquals(1.0, F.get(0, 0), EPSILON);
            assertEquals(0.5, F.get(0, 1), EPSILON);
            assertEquals(0.0, F.get(0, 2), EPSILON);
            assertEquals(1.0, F.get(2, 2), EPSILON);
            assertEquals(0.5, F.get(2, 3), EPSILON);
        }

        @Test
        @DisplayName("positionMeasurement creates correct matrix")
        void positionMeasurementCreatesCorrectMatrix() {
            CovarianceMatrix H = KalmanFilter.positionMeasurement(2);

            assertEquals(2, H.getDim()); // It's a 2x4 in theory but our impl is square

            // Check that it extracts positions
            // For a [x, vx, y, vy] state, should have:
            // | 1  0  0  0 |
            // | 0  0  1  0 |
            assertEquals(1.0, H.get(0, 0), EPSILON);
            assertEquals(0.0, H.get(0, 1), EPSILON);
            assertEquals(1.0, H.get(1, 2), EPSILON);
            assertEquals(0.0, H.get(1, 3), EPSILON);
        }

        @Test
        @DisplayName("innovation computes measurement residual")
        void innovationComputesMeasurementResidual() {
            GaussianState predicted = GaussianState.of(
                    new double[]{10.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            StateVector measurement = new StateVector(new double[]{12.0});
            CovarianceMatrix H = new CovarianceMatrix(new double[][]{{1.0, 0.0}});

            StateVector innov = KalmanFilter.innovation(predicted, measurement, H);

            // y = z - H*x = 12 - 10 = 2
            assertEquals(2.0, innov.get(0), EPSILON);
        }
    }

    @Nested
    @DisplayName("Full Tracking Scenario")
    class FullTrackingScenario {

        @Test
        @DisplayName("tracks target with multiple updates")
        void tracksTargetWithMultipleUpdates() throws StoneSoupException {
            // Initial state: stationary at origin
            GaussianState state = GaussianState.of(
                    new double[]{0.0, 1.0, 0.0, 1.0},  // Moving with velocity (1, 1)
                    CovarianceMatrix.identity(4).toArray(),
                    0.0
            );

            double dt = 1.0;
            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(2, dt);
            CovarianceMatrix Q = CovarianceMatrix.identity(4).scale(0.01);
            CovarianceMatrix H = KalmanFilter.positionMeasurement(2);
            CovarianceMatrix R = CovarianceMatrix.identity(2).scale(0.1);

            // Simulate several time steps
            double[][] measurements = {
                {1.1, 0.9},   // t=1
                {2.0, 2.1},   // t=2
                {2.9, 3.0},   // t=3
            };

            for (int i = 0; i < measurements.length; i++) {
                // Predict
                state = KalmanFilter.predict(state, F, Q);

                // Update
                StateVector z = new StateVector(measurements[i]);
                state = KalmanFilter.update(state, z, H, R);
            }

            // After tracking, should be near (3, 3)
            assertEquals(3.0, state.getState(0), 0.5);
            assertEquals(3.0, state.getState(2), 0.5);

            // Velocity should be near (1, 1)
            assertEquals(1.0, state.getState(1), 0.5);
            assertEquals(1.0, state.getState(3), 0.5);
        }
    }
}
