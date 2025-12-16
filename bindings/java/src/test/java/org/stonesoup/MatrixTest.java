package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link Matrix}.
 */
@DisplayName("Matrix")
class MatrixTest {

    private static final double EPSILON = 1e-9;

    @Nested
    @DisplayName("Construction")
    class Construction {

        @Test
        @DisplayName("creates matrix from 2D array")
        void createsMatrixFrom2DArray() {
            double[][] data = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
            };
            Matrix m = new Matrix(data);

            assertEquals(2, m.getRows());
            assertEquals(3, m.getCols());
            assertEquals(1.0, m.get(0, 0), EPSILON);
            assertEquals(6.0, m.get(1, 2), EPSILON);
        }

        @Test
        @DisplayName("rejects null array")
        void rejectsNullArray() {
            assertThrows(NullPointerException.class, () -> new Matrix(null));
        }

        @Test
        @DisplayName("rejects empty array")
        void rejectsEmptyArray() {
            assertThrows(IllegalArgumentException.class, () -> new Matrix(new double[0][]));
        }

        @Test
        @DisplayName("rejects empty columns")
        void rejectsEmptyColumns() {
            assertThrows(IllegalArgumentException.class,
                () -> new Matrix(new double[][]{{}}));
        }

        @Test
        @DisplayName("rejects null row")
        void rejectsNullRow() {
            assertThrows(IllegalArgumentException.class,
                () -> new Matrix(new double[][]{{1.0}, null}));
        }

        @Test
        @DisplayName("rejects inconsistent row lengths")
        void rejectsInconsistentRowLengths() {
            assertThrows(IllegalArgumentException.class,
                () -> new Matrix(new double[][]{{1.0, 2.0}, {3.0}}));
        }
    }

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        @DisplayName("creates zero matrix")
        void createsZeroMatrix() {
            Matrix m = Matrix.zeros(3, 4);

            assertEquals(3, m.getRows());
            assertEquals(4, m.getCols());
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    assertEquals(0.0, m.get(i, j), EPSILON);
                }
            }
        }

        @Test
        @DisplayName("creates identity matrix")
        void createsIdentityMatrix() {
            Matrix m = Matrix.identity(3);

            assertEquals(3, m.getRows());
            assertEquals(3, m.getCols());
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    double expected = (i == j) ? 1.0 : 0.0;
                    assertEquals(expected, m.get(i, j), EPSILON);
                }
            }
        }
    }

    @Nested
    @DisplayName("Element Access")
    class ElementAccess {

        @Test
        @DisplayName("sets and gets elements")
        void setsAndGetsElements() {
            Matrix m = Matrix.zeros(2, 2);

            m.set(0, 1, 5.5);
            assertEquals(5.5, m.get(0, 1), EPSILON);
        }

        @Test
        @DisplayName("converts to 2D array")
        void convertsTo2DArray() {
            double[][] original = {{1.0, 2.0}, {3.0, 4.0}};
            Matrix m = new Matrix(original);

            double[][] result = m.toArray();

            assertArrayEquals(original[0], result[0], EPSILON);
            assertArrayEquals(original[1], result[1], EPSILON);
        }
    }

    @Nested
    @DisplayName("Transpose")
    class Transpose {

        @Test
        @DisplayName("transposes matrix correctly")
        void transposesMatrixCorrectly() {
            Matrix m = new Matrix(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});

            Matrix mt = m.transpose();

            assertEquals(3, mt.getRows());
            assertEquals(2, mt.getCols());
            assertEquals(1.0, mt.get(0, 0), EPSILON);
            assertEquals(4.0, mt.get(0, 1), EPSILON);
            assertEquals(2.0, mt.get(1, 0), EPSILON);
            assertEquals(5.0, mt.get(1, 1), EPSILON);
            assertEquals(3.0, mt.get(2, 0), EPSILON);
            assertEquals(6.0, mt.get(2, 1), EPSILON);
        }

        @Test
        @DisplayName("double transpose returns original")
        void doubleTransposeReturnsOriginal() {
            Matrix m = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}});

            Matrix mtt = m.transpose().transpose();

            assertEquals(m, mtt);
        }
    }

    @Nested
    @DisplayName("Matrix Multiplication")
    class MatrixMultiplication {

        @Test
        @DisplayName("multiplies matrices correctly")
        void multipliesMatricesCorrectly() {
            // A = [1 2; 3 4]
            Matrix a = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
            // B = [5 6; 7 8]
            Matrix b = new Matrix(new double[][]{{5.0, 6.0}, {7.0, 8.0}});

            // C = A * B = [19 22; 43 50]
            Matrix c = a.multiply(b);

            assertEquals(19.0, c.get(0, 0), EPSILON);
            assertEquals(22.0, c.get(0, 1), EPSILON);
            assertEquals(43.0, c.get(1, 0), EPSILON);
            assertEquals(50.0, c.get(1, 1), EPSILON);
        }

        @Test
        @DisplayName("multiplies non-square matrices")
        void multipliesNonSquareMatrices() {
            // A = [1 2 3; 4 5 6] (2x3)
            Matrix a = new Matrix(new double[][]{{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
            // B = [7 8; 9 10; 11 12] (3x2)
            Matrix b = new Matrix(new double[][]{{7.0, 8.0}, {9.0, 10.0}, {11.0, 12.0}});

            // C = A * B should be 2x2
            Matrix c = a.multiply(b);

            assertEquals(2, c.getRows());
            assertEquals(2, c.getCols());
            // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
            assertEquals(58.0, c.get(0, 0), EPSILON);
        }

        @Test
        @DisplayName("rejects dimension mismatch")
        void rejectsDimensionMismatch() {
            Matrix a = new Matrix(new double[][]{{1.0, 2.0}});
            Matrix b = new Matrix(new double[][]{{1.0}, {2.0}, {3.0}});

            assertThrows(IllegalArgumentException.class, () -> a.multiply(b));
        }
    }

    @Nested
    @DisplayName("Vector Multiplication")
    class VectorMultiplication {

        @Test
        @DisplayName("multiplies matrix by vector")
        void multipliesMatrixByVector() {
            Matrix m = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
            StateVector v = new StateVector(new double[]{5.0, 6.0});

            StateVector result = m.multiply(v);

            // [1 2][5] = [17]
            // [3 4][6]   [39]
            assertEquals(2, result.getDim());
            assertEquals(17.0, result.get(0), EPSILON);
            assertEquals(39.0, result.get(1), EPSILON);
        }

        @Test
        @DisplayName("rejects vector dimension mismatch")
        void rejectsVectorDimensionMismatch() {
            Matrix m = new Matrix(new double[][]{{1.0, 2.0, 3.0}});
            StateVector v = new StateVector(new double[]{1.0, 2.0});

            assertThrows(IllegalArgumentException.class, () -> m.multiply(v));
        }
    }

    @Nested
    @DisplayName("Addition")
    class Addition {

        @Test
        @DisplayName("adds matrices correctly")
        void addsMatricesCorrectly() {
            Matrix a = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
            Matrix b = new Matrix(new double[][]{{5.0, 6.0}, {7.0, 8.0}});

            Matrix c = a.add(b);

            assertEquals(6.0, c.get(0, 0), EPSILON);
            assertEquals(8.0, c.get(0, 1), EPSILON);
            assertEquals(10.0, c.get(1, 0), EPSILON);
            assertEquals(12.0, c.get(1, 1), EPSILON);
        }

        @Test
        @DisplayName("rejects dimension mismatch")
        void rejectsDimensionMismatch() {
            Matrix a = new Matrix(new double[][]{{1.0, 2.0}});
            Matrix b = new Matrix(new double[][]{{1.0, 2.0, 3.0}});

            assertThrows(IllegalArgumentException.class, () -> a.add(b));
        }
    }

    @Nested
    @DisplayName("Scaling")
    class Scaling {

        @Test
        @DisplayName("scales matrix correctly")
        void scalesMatrixCorrectly() {
            Matrix m = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

            Matrix scaled = m.scale(2.5);

            assertEquals(2.5, scaled.get(0, 0), EPSILON);
            assertEquals(5.0, scaled.get(0, 1), EPSILON);
            assertEquals(7.5, scaled.get(1, 0), EPSILON);
            assertEquals(10.0, scaled.get(1, 1), EPSILON);
        }
    }

    @Nested
    @DisplayName("Equality and HashCode")
    class EqualityAndHashCode {

        @Test
        @DisplayName("equals for same data")
        void equalsForSameData() {
            Matrix a = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
            Matrix b = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

            assertEquals(a, b);
        }

        @Test
        @DisplayName("not equals for different data")
        void notEqualsForDifferentData() {
            Matrix a = new Matrix(new double[][]{{1.0, 2.0}});
            Matrix b = new Matrix(new double[][]{{1.0, 3.0}});

            assertNotEquals(a, b);
        }

        @Test
        @DisplayName("not equals for different dimensions")
        void notEqualsForDifferentDimensions() {
            Matrix a = new Matrix(new double[][]{{1.0, 2.0}});
            Matrix b = new Matrix(new double[][]{{1.0}, {2.0}});

            assertNotEquals(a, b);
        }

        @Test
        @DisplayName("equals handles same reference")
        void equalsHandlesSameReference() {
            Matrix m = new Matrix(new double[][]{{1.0}});
            assertEquals(m, m);
        }

        @Test
        @DisplayName("equals handles null and other types")
        void equalsHandlesNullAndOtherTypes() {
            Matrix m = new Matrix(new double[][]{{1.0}});

            assertNotEquals(m, null);
            assertNotEquals(m, "string");
        }

        @Test
        @DisplayName("hashCode consistent with equals")
        void hashCodeConsistentWithEquals() {
            Matrix a = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});
            Matrix b = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

            assertEquals(a.hashCode(), b.hashCode());
        }
    }

    @Nested
    @DisplayName("ToString")
    class ToStringTest {

        @Test
        @DisplayName("produces readable output")
        void producesReadableOutput() {
            Matrix m = new Matrix(new double[][]{{1.0, 2.0}, {3.0, 4.0}});

            String str = m.toString();

            assertTrue(str.contains("2x2"));
            assertTrue(str.contains("1.0000"));
            assertTrue(str.contains("4.0000"));
        }
    }
}
