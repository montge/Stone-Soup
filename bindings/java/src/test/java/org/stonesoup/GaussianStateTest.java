package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link GaussianState}.
 */
@DisplayName("GaussianState")
class GaussianStateTest {

    private static final double EPSILON = 1e-10;

    @Nested
    @DisplayName("Construction")
    class Construction {

        @Test
        @DisplayName("creates from components")
        void createsFromComponents() {
            StateVector mean = new StateVector(new double[]{1.0, 2.0});
            CovarianceMatrix cov = CovarianceMatrix.identity(2);

            GaussianState state = new GaussianState(mean, cov);

            assertEquals(2, state.getDim());
            assertEquals(mean, state.getStateVector());
            assertEquals(cov, state.getCovariance());
            assertTrue(state.getTimestamp().isEmpty());
        }

        @Test
        @DisplayName("creates with timestamp")
        void createsWithTimestamp() {
            StateVector mean = new StateVector(new double[]{1.0, 2.0});
            CovarianceMatrix cov = CovarianceMatrix.identity(2);

            GaussianState state = new GaussianState(mean, cov, 10.0);

            assertEquals(Optional.of(10.0), state.getTimestamp());
        }

        @Test
        @DisplayName("factory method creates from arrays")
        void factoryMethodCreatesFromArrays() {
            GaussianState state = GaussianState.of(
                    new double[]{1.0, 2.0},
                    new double[][]{{1.0, 0.0}, {0.0, 1.0}}
            );

            assertEquals(2, state.getDim());
            assertEquals(1.0, state.getState(0), EPSILON);
            assertEquals(2.0, state.getState(1), EPSILON);
        }

        @Test
        @DisplayName("rejects dimension mismatch")
        void rejectsDimensionMismatch() {
            StateVector mean = new StateVector(new double[]{1.0, 2.0});
            CovarianceMatrix cov = CovarianceMatrix.identity(3);

            assertThrows(IllegalArgumentException.class, () -> new GaussianState(mean, cov));
        }

        @Test
        @DisplayName("rejects null components")
        void rejectsNullComponents() {
            StateVector mean = new StateVector(new double[]{1.0, 2.0});
            CovarianceMatrix cov = CovarianceMatrix.identity(2);

            assertThrows(NullPointerException.class, () -> new GaussianState(null, cov));
            assertThrows(NullPointerException.class, () -> new GaussianState(mean, null));
        }
    }

    @Nested
    @DisplayName("Access")
    class Access {

        @Test
        @DisplayName("getState returns state element")
        void getStateReturnsStateElement() {
            GaussianState state = GaussianState.of(
                    new double[]{10.0, 20.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            assertEquals(10.0, state.getState(0), EPSILON);
            assertEquals(20.0, state.getState(1), EPSILON);
        }

        @Test
        @DisplayName("getVariance returns diagonal covariance")
        void getVarianceReturnsDiagonalCovariance() {
            GaussianState state = GaussianState.of(
                    new double[]{0.0, 0.0},
                    new double[][]{{4.0, 0.5}, {0.5, 9.0}}
            );

            assertEquals(4.0, state.getVariance(0), EPSILON);
            assertEquals(9.0, state.getVariance(1), EPSILON);
        }

        @Test
        @DisplayName("getStdDev returns square root of variance")
        void getStdDevReturnsSquareRootOfVariance() {
            GaussianState state = GaussianState.of(
                    new double[]{0.0, 0.0},
                    new double[][]{{4.0, 0.0}, {0.0, 9.0}}
            );

            assertEquals(2.0, state.getStdDev(0), EPSILON);
            assertEquals(3.0, state.getStdDev(1), EPSILON);
        }

        @Test
        @DisplayName("timestamp can be set and changed")
        void timestampCanBeSetAndChanged() {
            GaussianState state = GaussianState.of(
                    new double[]{1.0},
                    new double[][]{{1.0}}
            );

            assertTrue(state.getTimestamp().isEmpty());

            state.setTimestamp(5.0);
            assertEquals(Optional.of(5.0), state.getTimestamp());

            state.setTimestamp(null);
            assertTrue(state.getTimestamp().isEmpty());
        }
    }

    @Nested
    @DisplayName("Operations")
    class Operations {

        @Test
        @DisplayName("copy creates independent copy")
        void copyCreatesIndependentCopy() {
            GaussianState original = GaussianState.of(
                    new double[]{1.0, 2.0},
                    CovarianceMatrix.identity(2).toArray(),
                    10.0
            );

            GaussianState copy = original.copy();

            // Modify copy
            copy.getStateVector().set(0, 999.0);
            copy.setTimestamp(20.0);

            // Original unchanged
            assertEquals(1.0, original.getState(0), EPSILON);
            assertEquals(Optional.of(10.0), original.getTimestamp());
        }
    }

    @Nested
    @DisplayName("Equality")
    class Equality {

        @Test
        @DisplayName("equals returns true for same values")
        void equalsReturnsTrueForSameValues() {
            GaussianState a = GaussianState.of(
                    new double[]{1.0, 2.0},
                    CovarianceMatrix.identity(2).toArray(),
                    10.0
            );
            GaussianState b = GaussianState.of(
                    new double[]{1.0, 2.0},
                    CovarianceMatrix.identity(2).toArray(),
                    10.0
            );

            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
        }

        @Test
        @DisplayName("equals returns false for different values")
        void equalsReturnsFalseForDifferentValues() {
            GaussianState a = GaussianState.of(
                    new double[]{1.0, 2.0},
                    CovarianceMatrix.identity(2).toArray()
            );
            GaussianState b = GaussianState.of(
                    new double[]{1.0, 3.0},
                    CovarianceMatrix.identity(2).toArray()
            );

            assertNotEquals(a, b);
        }
    }
}
