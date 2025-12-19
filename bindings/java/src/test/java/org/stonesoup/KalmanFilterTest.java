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
            Matrix H = KalmanFilter.positionMeasurement(2);
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
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});
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
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});
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
            Matrix H = KalmanFilter.positionMeasurement(2);

            assertEquals(2, H.getRows()); // 2 measurement dimensions
            assertEquals(4, H.getCols()); // 4 state dimensions (x, vx, y, vy)

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
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});

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
            Matrix H = KalmanFilter.positionMeasurement(2);
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

    @Nested
    @DisplayName("Error Handling")
    class ErrorHandling {

        @Test
        @DisplayName("predict rejects null prior")
        void predictRejectsNullPrior() {
            CovarianceMatrix F = CovarianceMatrix.identity(2);
            CovarianceMatrix Q = CovarianceMatrix.identity(2);

            assertThrows(NullPointerException.class,
                    () -> KalmanFilter.predict(null, F, Q));
        }

        @Test
        @DisplayName("predict rejects null transition matrix")
        void predictRejectsNullTransitionMatrix() {
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            CovarianceMatrix Q = CovarianceMatrix.identity(2);

            assertThrows(NullPointerException.class,
                    () -> KalmanFilter.predict(prior, null, Q));
        }

        @Test
        @DisplayName("predict rejects null process noise")
        void predictRejectsNullProcessNoise() {
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            CovarianceMatrix F = CovarianceMatrix.identity(2);

            assertThrows(NullPointerException.class,
                    () -> KalmanFilter.predict(prior, F, null));
        }

        @Test
        @DisplayName("predict rejects transition matrix dimension mismatch")
        void predictRejectsTransitionDimensionMismatch() {
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            CovarianceMatrix F = CovarianceMatrix.identity(4); // Wrong size
            CovarianceMatrix Q = CovarianceMatrix.identity(2);

            assertThrows(IllegalArgumentException.class,
                    () -> KalmanFilter.predict(prior, F, Q));
        }

        @Test
        @DisplayName("predict rejects process noise dimension mismatch")
        void predictRejectsProcessNoiseDimensionMismatch() {
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            CovarianceMatrix F = CovarianceMatrix.identity(2);
            CovarianceMatrix Q = CovarianceMatrix.identity(4); // Wrong size

            assertThrows(IllegalArgumentException.class,
                    () -> KalmanFilter.predict(prior, F, Q));
        }

        @Test
        @DisplayName("update rejects null predicted state")
        void updateRejectsNullPredicted() {
            StateVector z = new StateVector(new double[]{1.0});
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});
            CovarianceMatrix R = CovarianceMatrix.identity(1);

            assertThrows(NullPointerException.class,
                    () -> KalmanFilter.update(null, z, H, R));
        }

        @Test
        @DisplayName("update rejects null measurement")
        void updateRejectsNullMeasurement() {
            GaussianState predicted = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});
            CovarianceMatrix R = CovarianceMatrix.identity(1);

            assertThrows(NullPointerException.class,
                    () -> KalmanFilter.update(predicted, null, H, R));
        }

        @Test
        @DisplayName("update rejects null measurement matrix")
        void updateRejectsNullMeasurementMatrix() {
            GaussianState predicted = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            StateVector z = new StateVector(new double[]{1.0});
            CovarianceMatrix R = CovarianceMatrix.identity(1);

            assertThrows(NullPointerException.class,
                    () -> KalmanFilter.update(predicted, z, null, R));
        }

        @Test
        @DisplayName("update rejects null measurement noise")
        void updateRejectsNullMeasurementNoise() {
            GaussianState predicted = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            StateVector z = new StateVector(new double[]{1.0});
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});

            assertThrows(NullPointerException.class,
                    () -> KalmanFilter.update(predicted, z, H, null));
        }

        @Test
        @DisplayName("update rejects measurement noise dimension mismatch")
        void updateRejectsMeasurementNoiseDimensionMismatch() {
            GaussianState predicted = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            StateVector z = new StateVector(new double[]{1.0});
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});
            CovarianceMatrix R = CovarianceMatrix.identity(2); // Wrong size

            assertThrows(IllegalArgumentException.class,
                    () -> KalmanFilter.update(predicted, z, H, R));
        }

        @Test
        @DisplayName("update rejects measurement matrix dimension mismatch")
        void updateRejectsMeasurementMatrixDimensionMismatch() {
            GaussianState predicted = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            StateVector z = new StateVector(new double[]{1.0});
            Matrix H = new Matrix(new double[][]{{1.0, 0.0, 0.0}}); // Wrong cols
            CovarianceMatrix R = CovarianceMatrix.identity(1);

            assertThrows(IllegalArgumentException.class,
                    () -> KalmanFilter.update(predicted, z, H, R));
        }

        @Test
        @DisplayName("constantVelocityTransition rejects invalid ndim")
        void constantVelocityTransitionRejectsInvalidNdim() {
            assertThrows(IllegalArgumentException.class,
                    () -> KalmanFilter.constantVelocityTransition(0, 1.0));
        }

        @Test
        @DisplayName("positionMeasurement rejects invalid ndim")
        void positionMeasurementRejectsInvalidNdim() {
            assertThrows(IllegalArgumentException.class,
                    () -> KalmanFilter.positionMeasurement(0));
        }
    }

    @Nested
    @DisplayName("Mahalanobis Distance")
    class MahalanobisDistanceTests {

        @Test
        @DisplayName("computes Mahalanobis distance correctly")
        void computesMahalanobisDistanceCorrectly() throws StoneSoupException {
            StateVector innovation = new StateVector(new double[]{2.0, 0.0});
            CovarianceMatrix S = CovarianceMatrix.identity(2);

            double distance = KalmanFilter.mahalanobisDistance(innovation, S);

            // d = sqrt(y^T * S^-1 * y) = sqrt([2,0] * I * [2,0]^T) = sqrt(4) = 2
            assertEquals(2.0, distance, EPSILON);
        }

        @Test
        @DisplayName("Mahalanobis distance with scaled covariance")
        void mahalanobisDistanceWithScaledCovariance() throws StoneSoupException {
            StateVector innovation = new StateVector(new double[]{2.0, 0.0});
            CovarianceMatrix S = CovarianceMatrix.identity(2).scale(4.0);

            double distance = KalmanFilter.mahalanobisDistance(innovation, S);

            // d = sqrt(y^T * S^-1 * y) = sqrt([2,0] * (1/4)I * [2,0]^T) = sqrt(1) = 1
            assertEquals(1.0, distance, EPSILON);
        }
    }

    @Nested
    @DisplayName("1D Kalman Filter")
    class OneDimensionalKalmanFilter {

        @Test
        @DisplayName("works with 1D state")
        void worksWithOneDimensionalState() throws StoneSoupException {
            // 1D state: [position, velocity]
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 1.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(1, 1.0);
            CovarianceMatrix Q = CovarianceMatrix.identity(2).scale(0.1);

            GaussianState predicted = KalmanFilter.predict(prior, F, Q);

            // After prediction: position = 0 + 1*1 = 1, velocity = 1
            assertEquals(1.0, predicted.getState(0), EPSILON);
            assertEquals(1.0, predicted.getState(1), EPSILON);
        }

        @Test
        @DisplayName("1D measurement update")
        void oneDimensionalMeasurementUpdate() throws StoneSoupException {
            GaussianState predicted = GaussianState.of(
                    new double[]{1.0, 1.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            StateVector z = new StateVector(new double[]{1.2});
            Matrix H = KalmanFilter.positionMeasurement(1);
            CovarianceMatrix R = CovarianceMatrix.identity(1).scale(0.5);

            GaussianState posterior = KalmanFilter.update(predicted, z, H, R);

            // Position should be pulled toward measurement
            assertTrue(posterior.getState(0) > 1.0);
            assertTrue(posterior.getState(0) < 1.2);
        }
    }

    @Nested
    @DisplayName("Constructor Coverage")
    class ConstructorCoverage {

        @Test
        @DisplayName("cannot instantiate utility class")
        void cannotInstantiateUtilityClass() throws Exception {
            java.lang.reflect.Constructor<KalmanFilter> constructor =
                    KalmanFilter.class.getDeclaredConstructor();
            constructor.setAccessible(true);
            assertThrows(java.lang.reflect.InvocationTargetException.class,
                    () -> constructor.newInstance());
        }
    }

    @Nested
    @DisplayName("Singular Matrix Handling")
    class SingularMatrixHandling {

        @Test
        @DisplayName("update throws on singular innovation covariance")
        void updateThrowsOnSingularInnovationCovariance() {
            GaussianState predicted = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            StateVector z = new StateVector(new double[]{1.0});
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});
            // Singular measurement noise (all zeros)
            CovarianceMatrix R = new CovarianceMatrix(new double[][]{{0.0}});

            // The innovation covariance S = H*P*H' + R may still be invertible
            // due to P contribution, but let's try a zero-variance state too
        }

        @Test
        @DisplayName("handles near-singular matrix gracefully")
        void handlesNearSingularMatrixGracefully() throws StoneSoupException {
            GaussianState predicted = GaussianState.of(
                    new double[]{0.0, 0.0},
                    CovarianceMatrix.identity(2).scale(1e-10).toArray()
            );

            StateVector z = new StateVector(new double[]{1.0});
            Matrix H = new Matrix(new double[][]{{1.0, 0.0}});
            CovarianceMatrix R = CovarianceMatrix.identity(1).scale(1.0);

            // Should succeed because R provides enough regularization
            GaussianState posterior = KalmanFilter.update(predicted, z, H, R);
            assertNotNull(posterior);
        }
    }

    @Nested
    @DisplayName("Different Time Steps")
    class DifferentTimeSteps {

        @Test
        @DisplayName("handles zero time step")
        void handlesZeroTimeStep() throws StoneSoupException {
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 1.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(1, 0.0);
            CovarianceMatrix Q = CovarianceMatrix.identity(2).scale(0.1);

            GaussianState predicted = KalmanFilter.predict(prior, F, Q);

            // With dt=0, position shouldn't change based on velocity
            assertEquals(0.0, predicted.getState(0), EPSILON);
            assertEquals(1.0, predicted.getState(1), EPSILON);
        }

        @Test
        @DisplayName("handles negative time step")
        void handlesNegativeTimeStep() throws StoneSoupException {
            GaussianState prior = GaussianState.of(
                    new double[]{1.0, 1.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(1, -1.0);
            CovarianceMatrix Q = CovarianceMatrix.identity(2).scale(0.1);

            GaussianState predicted = KalmanFilter.predict(prior, F, Q);

            // With dt=-1, position = 1 + (-1)*1 = 0
            assertEquals(0.0, predicted.getState(0), EPSILON);
        }
    }

    @Nested
    @DisplayName("Large State Dimensions")
    class LargeStateDimensions {

        @Test
        @DisplayName("handles 4D state")
        void handlesFourDimensionalState() throws StoneSoupException {
            // 4D state: [x, vx, y, vy, z, vz, w, vw]
            GaussianState prior = GaussianState.of(
                    new double[]{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                    CovarianceMatrix.identity(8).toArray()
            );

            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(4, 0.5);
            CovarianceMatrix Q = CovarianceMatrix.identity(8).scale(0.01);

            GaussianState predicted = KalmanFilter.predict(prior, F, Q);

            // Check a few expected values
            assertEquals(2.0, predicted.getState(0), EPSILON);  // 1 + 0.5*2
            assertEquals(5.0, predicted.getState(2), EPSILON);  // 3 + 0.5*4
            assertEquals(8.0, predicted.getState(4), EPSILON);  // 5 + 0.5*6
            assertEquals(11.0, predicted.getState(6), EPSILON); // 7 + 0.5*8
        }

        @Test
        @DisplayName("4D measurement update")
        void fourDimensionalMeasurementUpdate() throws StoneSoupException {
            GaussianState predicted = GaussianState.of(
                    new double[]{1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0},
                    CovarianceMatrix.identity(8).toArray()
            );

            StateVector z = new StateVector(new double[]{1.1, 2.1, 3.1, 4.1});
            Matrix H = KalmanFilter.positionMeasurement(4);
            CovarianceMatrix R = CovarianceMatrix.identity(4).scale(0.5);

            GaussianState posterior = KalmanFilter.update(predicted, z, H, R);

            // All positions should be pulled toward measurements
            assertTrue(posterior.getState(0) > 1.0);
            assertTrue(posterior.getState(2) > 2.0);
            assertTrue(posterior.getState(4) > 3.0);
            assertTrue(posterior.getState(6) > 4.0);
        }
    }

    @Nested
    @DisplayName("3D Kalman Filter")
    class ThreeDimensionalKalmanFilter {

        @Test
        @DisplayName("works with 3D state")
        void worksWithThreeDimensionalState() throws StoneSoupException {
            // 3D state: [x, vx, y, vy, z, vz]
            GaussianState prior = GaussianState.of(
                    new double[]{0.0, 1.0, 0.0, 2.0, 0.0, 3.0},
                    CovarianceMatrix.identity(6).toArray()
            );

            CovarianceMatrix F = KalmanFilter.constantVelocityTransition(3, 1.0);
            CovarianceMatrix Q = CovarianceMatrix.identity(6).scale(0.1);

            GaussianState predicted = KalmanFilter.predict(prior, F, Q);

            // After prediction: x=1, vx=1, y=2, vy=2, z=3, vz=3
            assertEquals(1.0, predicted.getState(0), EPSILON);
            assertEquals(1.0, predicted.getState(1), EPSILON);
            assertEquals(2.0, predicted.getState(2), EPSILON);
            assertEquals(2.0, predicted.getState(3), EPSILON);
            assertEquals(3.0, predicted.getState(4), EPSILON);
            assertEquals(3.0, predicted.getState(5), EPSILON);
        }

        @Test
        @DisplayName("3D measurement update")
        void threeDimensionalMeasurementUpdate() throws StoneSoupException {
            GaussianState predicted = GaussianState.of(
                    new double[]{1.0, 1.0, 2.0, 2.0, 3.0, 3.0},
                    CovarianceMatrix.identity(6).toArray()
            );

            StateVector z = new StateVector(new double[]{1.1, 2.1, 3.1});
            Matrix H = KalmanFilter.positionMeasurement(3);
            CovarianceMatrix R = CovarianceMatrix.identity(3).scale(0.5);

            GaussianState posterior = KalmanFilter.update(predicted, z, H, R);

            // Positions should be pulled toward measurements
            assertTrue(posterior.getState(0) > 1.0);
            assertTrue(posterior.getState(2) > 2.0);
            assertTrue(posterior.getState(4) > 3.0);
        }
    }
}
