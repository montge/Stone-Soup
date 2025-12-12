package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link StateVector}.
 */
@DisplayName("StateVector")
class StateVectorTest {

    private static final double EPSILON = 1e-10;

    @Nested
    @DisplayName("Construction")
    class Construction {

        @Test
        @DisplayName("creates from array")
        void createsFromArray() {
            double[] data = {1.0, 2.0, 3.0};
            StateVector vec = new StateVector(data);

            assertEquals(3, vec.getDim());
            assertEquals(1.0, vec.get(0), EPSILON);
            assertEquals(2.0, vec.get(1), EPSILON);
            assertEquals(3.0, vec.get(2), EPSILON);
        }

        @Test
        @DisplayName("copies input array")
        void copiesInputArray() {
            double[] data = {1.0, 2.0};
            StateVector vec = new StateVector(data);

            // Modify original
            data[0] = 999.0;

            // Vector should be unchanged
            assertEquals(1.0, vec.get(0), EPSILON);
        }

        @Test
        @DisplayName("rejects null array")
        void rejectsNullArray() {
            assertThrows(NullPointerException.class, () -> new StateVector(null));
        }

        @Test
        @DisplayName("rejects empty array")
        void rejectsEmptyArray() {
            assertThrows(IllegalArgumentException.class, () -> new StateVector(new double[0]));
        }
    }

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @ParameterizedTest
        @ValueSource(ints = {1, 2, 4, 10})
        @DisplayName("zeros creates correct dimension")
        void zerosCreatesCorrectDimension(int dim) {
            StateVector vec = StateVector.zeros(dim);

            assertEquals(dim, vec.getDim());
            for (int i = 0; i < dim; i++) {
                assertEquals(0.0, vec.get(i), EPSILON);
            }
        }

        @Test
        @DisplayName("zeros rejects invalid dimension")
        void zerosRejectsInvalidDimension() {
            assertThrows(IllegalArgumentException.class, () -> StateVector.zeros(0));
            assertThrows(IllegalArgumentException.class, () -> StateVector.zeros(-1));
        }

        @Test
        @DisplayName("fill creates vector with value")
        void fillCreatesVectorWithValue() {
            StateVector vec = StateVector.fill(3, 5.0);

            assertEquals(3, vec.getDim());
            for (int i = 0; i < 3; i++) {
                assertEquals(5.0, vec.get(i), EPSILON);
            }
        }
    }

    @Nested
    @DisplayName("Element Access")
    class ElementAccess {

        @Test
        @DisplayName("get returns correct values")
        void getReturnsCorrectValues() {
            StateVector vec = new StateVector(new double[]{1.0, 2.0, 3.0});

            assertEquals(1.0, vec.get(0), EPSILON);
            assertEquals(2.0, vec.get(1), EPSILON);
            assertEquals(3.0, vec.get(2), EPSILON);
        }

        @Test
        @DisplayName("get throws on invalid index")
        void getThrowsOnInvalidIndex() {
            StateVector vec = new StateVector(new double[]{1.0, 2.0});

            assertThrows(IndexOutOfBoundsException.class, () -> vec.get(-1));
            assertThrows(IndexOutOfBoundsException.class, () -> vec.get(2));
        }

        @Test
        @DisplayName("set modifies value")
        void setModifiesValue() {
            StateVector vec = new StateVector(new double[]{1.0, 2.0});

            vec.set(0, 10.0);

            assertEquals(10.0, vec.get(0), EPSILON);
        }

        @Test
        @DisplayName("toArray returns copy")
        void toArrayReturnsCopy() {
            StateVector vec = new StateVector(new double[]{1.0, 2.0});
            double[] arr = vec.toArray();

            arr[0] = 999.0;

            assertEquals(1.0, vec.get(0), EPSILON);
        }
    }

    @Nested
    @DisplayName("Operations")
    class Operations {

        @Test
        @DisplayName("norm computes Euclidean norm")
        void normComputesEuclideanNorm() {
            // ||[3, 4]|| = 5
            StateVector vec = new StateVector(new double[]{3.0, 4.0});
            assertEquals(5.0, vec.norm(), EPSILON);

            // ||[1, 1, 1]|| = sqrt(3)
            StateVector vec2 = new StateVector(new double[]{1.0, 1.0, 1.0});
            assertEquals(Math.sqrt(3.0), vec2.norm(), EPSILON);
        }

        @Test
        @DisplayName("add sums vectors")
        void addSumsVectors() {
            StateVector a = new StateVector(new double[]{1.0, 2.0});
            StateVector b = new StateVector(new double[]{3.0, 4.0});

            StateVector sum = a.add(b);

            assertEquals(4.0, sum.get(0), EPSILON);
            assertEquals(6.0, sum.get(1), EPSILON);
        }

        @Test
        @DisplayName("add throws on dimension mismatch")
        void addThrowsOnDimensionMismatch() {
            StateVector a = new StateVector(new double[]{1.0, 2.0});
            StateVector b = new StateVector(new double[]{3.0, 4.0, 5.0});

            assertThrows(IllegalArgumentException.class, () -> a.add(b));
        }

        @Test
        @DisplayName("subtract computes difference")
        void subtractComputesDifference() {
            StateVector a = new StateVector(new double[]{5.0, 6.0});
            StateVector b = new StateVector(new double[]{2.0, 1.0});

            StateVector diff = a.subtract(b);

            assertEquals(3.0, diff.get(0), EPSILON);
            assertEquals(5.0, diff.get(1), EPSILON);
        }

        @Test
        @DisplayName("scale multiplies by factor")
        void scaleMultipliesByFactor() {
            StateVector vec = new StateVector(new double[]{2.0, 3.0});

            StateVector scaled = vec.scale(2.0);

            assertEquals(4.0, scaled.get(0), EPSILON);
            assertEquals(6.0, scaled.get(1), EPSILON);
        }

        @Test
        @DisplayName("dot computes dot product")
        void dotComputesDotProduct() {
            StateVector a = new StateVector(new double[]{1.0, 2.0, 3.0});
            StateVector b = new StateVector(new double[]{4.0, 5.0, 6.0});

            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assertEquals(32.0, a.dot(b), EPSILON);
        }

        @Test
        @DisplayName("copy creates independent copy")
        void copyCreatesIndependentCopy() {
            StateVector original = new StateVector(new double[]{1.0, 2.0});
            StateVector copy = original.copy();

            copy.set(0, 999.0);

            assertEquals(1.0, original.get(0), EPSILON);
            assertEquals(999.0, copy.get(0), EPSILON);
        }
    }

    @Nested
    @DisplayName("Equality")
    class Equality {

        @Test
        @DisplayName("equals returns true for same values")
        void equalsReturnsTrueForSameValues() {
            StateVector a = new StateVector(new double[]{1.0, 2.0});
            StateVector b = new StateVector(new double[]{1.0, 2.0});

            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
        }

        @Test
        @DisplayName("equals returns false for different values")
        void equalsReturnsFalseForDifferentValues() {
            StateVector a = new StateVector(new double[]{1.0, 2.0});
            StateVector b = new StateVector(new double[]{1.0, 3.0});

            assertNotEquals(a, b);
        }
    }
}
