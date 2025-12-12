package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link CovarianceMatrix}.
 */
@DisplayName("CovarianceMatrix")
class CovarianceMatrixTest {

    private static final double EPSILON = 1e-10;

    @Nested
    @DisplayName("Construction")
    class Construction {

        @Test
        @DisplayName("creates from 2D array")
        void createsFrom2DArray() {
            double[][] data = {
                {1.0, 0.5},
                {0.5, 2.0}
            };
            CovarianceMatrix mat = new CovarianceMatrix(data);

            assertEquals(2, mat.getDim());
            assertEquals(1.0, mat.get(0, 0), EPSILON);
            assertEquals(0.5, mat.get(0, 1), EPSILON);
            assertEquals(0.5, mat.get(1, 0), EPSILON);
            assertEquals(2.0, mat.get(1, 1), EPSILON);
        }

        @Test
        @DisplayName("rejects non-square matrix")
        void rejectsNonSquareMatrix() {
            double[][] data = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0}
            };
            assertThrows(IllegalArgumentException.class, () -> new CovarianceMatrix(data));
        }

        @Test
        @DisplayName("rejects null matrix")
        void rejectsNullMatrix() {
            assertThrows(NullPointerException.class, () -> new CovarianceMatrix(null));
        }
    }

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        @DisplayName("identity creates identity matrix")
        void identityCreatesIdentityMatrix() {
            CovarianceMatrix mat = CovarianceMatrix.identity(3);

            assertEquals(3, mat.getDim());
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    assertEquals(expected, mat.get(i, j), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("zeros creates zero matrix")
        void zerosCreatesZeroMatrix() {
            CovarianceMatrix mat = CovarianceMatrix.zeros(2);

            assertEquals(2, mat.getDim());
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    assertEquals(0.0, mat.get(i, j), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("diagonal creates diagonal matrix")
        void diagonalCreatesDiagonalMatrix() {
            CovarianceMatrix mat = CovarianceMatrix.diagonal(new double[]{1.0, 2.0, 3.0});

            assertEquals(3, mat.getDim());
            assertEquals(1.0, mat.get(0, 0), EPSILON);
            assertEquals(2.0, mat.get(1, 1), EPSILON);
            assertEquals(3.0, mat.get(2, 2), EPSILON);
            assertEquals(0.0, mat.get(0, 1), EPSILON);
            assertEquals(0.0, mat.get(1, 2), EPSILON);
        }
    }

    @Nested
    @DisplayName("Element Access")
    class ElementAccess {

        @Test
        @DisplayName("get returns correct values")
        void getReturnsCorrectValues() {
            CovarianceMatrix mat = CovarianceMatrix.identity(2);

            assertEquals(1.0, mat.get(0, 0), EPSILON);
            assertEquals(0.0, mat.get(0, 1), EPSILON);
        }

        @Test
        @DisplayName("get throws on invalid indices")
        void getThrowsOnInvalidIndices() {
            CovarianceMatrix mat = CovarianceMatrix.identity(2);

            assertThrows(IndexOutOfBoundsException.class, () -> mat.get(-1, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> mat.get(0, 2));
        }

        @Test
        @DisplayName("set modifies value")
        void setModifiesValue() {
            CovarianceMatrix mat = CovarianceMatrix.identity(2);

            mat.set(0, 1, 0.5);

            assertEquals(0.5, mat.get(0, 1), EPSILON);
        }

        @Test
        @DisplayName("toArray returns 2D copy")
        void toArrayReturns2DCopy() {
            CovarianceMatrix mat = CovarianceMatrix.identity(2);
            double[][] arr = mat.toArray();

            arr[0][0] = 999.0;

            assertEquals(1.0, mat.get(0, 0), EPSILON);
        }
    }

    @Nested
    @DisplayName("Operations")
    class Operations {

        @Test
        @DisplayName("trace computes sum of diagonal")
        void traceComputesSumOfDiagonal() {
            CovarianceMatrix mat = CovarianceMatrix.diagonal(new double[]{1.0, 2.0, 3.0});
            assertEquals(6.0, mat.trace(), EPSILON);

            CovarianceMatrix identity = CovarianceMatrix.identity(4);
            assertEquals(4.0, identity.trace(), EPSILON);
        }

        @Test
        @DisplayName("add sums matrices")
        void addSumsMatrices() {
            CovarianceMatrix a = CovarianceMatrix.identity(2);
            CovarianceMatrix b = CovarianceMatrix.identity(2);

            CovarianceMatrix sum = a.add(b);

            assertEquals(2.0, sum.get(0, 0), EPSILON);
            assertEquals(0.0, sum.get(0, 1), EPSILON);
        }

        @Test
        @DisplayName("subtract computes difference")
        void subtractComputesDifference() {
            CovarianceMatrix a = CovarianceMatrix.diagonal(new double[]{3.0, 4.0});
            CovarianceMatrix b = CovarianceMatrix.identity(2);

            CovarianceMatrix diff = a.subtract(b);

            assertEquals(2.0, diff.get(0, 0), EPSILON);
            assertEquals(3.0, diff.get(1, 1), EPSILON);
        }

        @Test
        @DisplayName("scale multiplies by factor")
        void scaleMultipliesByFactor() {
            CovarianceMatrix mat = CovarianceMatrix.identity(2);

            CovarianceMatrix scaled = mat.scale(3.0);

            assertEquals(3.0, scaled.get(0, 0), EPSILON);
            assertEquals(0.0, scaled.get(0, 1), EPSILON);
        }

        @Test
        @DisplayName("multiply computes matrix product")
        void multiplyComputesMatrixProduct() {
            double[][] aData = {{1.0, 2.0}, {3.0, 4.0}};
            double[][] bData = {{5.0, 6.0}, {7.0, 8.0}};
            CovarianceMatrix a = new CovarianceMatrix(aData);
            CovarianceMatrix b = new CovarianceMatrix(bData);

            CovarianceMatrix c = a.multiply(b);

            // C[0,0] = 1*5 + 2*7 = 19
            // C[0,1] = 1*6 + 2*8 = 22
            // C[1,0] = 3*5 + 4*7 = 43
            // C[1,1] = 3*6 + 4*8 = 50
            assertEquals(19.0, c.get(0, 0), EPSILON);
            assertEquals(22.0, c.get(0, 1), EPSILON);
            assertEquals(43.0, c.get(1, 0), EPSILON);
            assertEquals(50.0, c.get(1, 1), EPSILON);
        }

        @Test
        @DisplayName("multiply with vector")
        void multiplyWithVector() {
            double[][] mData = {{1.0, 2.0}, {3.0, 4.0}};
            CovarianceMatrix mat = new CovarianceMatrix(mData);
            StateVector vec = new StateVector(new double[]{1.0, 2.0});

            StateVector result = mat.multiply(vec);

            // [1,2] * [1,2]^T = [1*1+2*2, 3*1+4*2] = [5, 11]
            assertEquals(5.0, result.get(0), EPSILON);
            assertEquals(11.0, result.get(1), EPSILON);
        }

        @Test
        @DisplayName("transpose swaps indices")
        void transposeSwapsIndices() {
            double[][] data = {{1.0, 2.0}, {3.0, 4.0}};
            CovarianceMatrix mat = new CovarianceMatrix(data);

            CovarianceMatrix trans = mat.transpose();

            assertEquals(1.0, trans.get(0, 0), EPSILON);
            assertEquals(3.0, trans.get(0, 1), EPSILON);
            assertEquals(2.0, trans.get(1, 0), EPSILON);
            assertEquals(4.0, trans.get(1, 1), EPSILON);
        }

        @Test
        @DisplayName("multiplyTranspose computes A * B^T")
        void multiplyTransposeComputes() {
            CovarianceMatrix a = CovarianceMatrix.identity(2);
            double[][] bData = {{1.0, 2.0}, {3.0, 4.0}};
            CovarianceMatrix b = new CovarianceMatrix(bData);

            CovarianceMatrix result = a.multiplyTranspose(b);

            // I * B^T = B^T
            assertEquals(1.0, result.get(0, 0), EPSILON);
            assertEquals(3.0, result.get(0, 1), EPSILON);
            assertEquals(2.0, result.get(1, 0), EPSILON);
            assertEquals(4.0, result.get(1, 1), EPSILON);
        }
    }

    @Nested
    @DisplayName("Equality")
    class Equality {

        @Test
        @DisplayName("equals returns true for same values")
        void equalsReturnsTrueForSameValues() {
            CovarianceMatrix a = CovarianceMatrix.identity(2);
            CovarianceMatrix b = CovarianceMatrix.identity(2);

            assertEquals(a, b);
            assertEquals(a.hashCode(), b.hashCode());
        }

        @Test
        @DisplayName("equals returns false for different values")
        void equalsReturnsFalseForDifferentValues() {
            CovarianceMatrix a = CovarianceMatrix.identity(2);
            CovarianceMatrix b = CovarianceMatrix.diagonal(new double[]{1.0, 2.0});

            assertNotEquals(a, b);
        }
    }
}
