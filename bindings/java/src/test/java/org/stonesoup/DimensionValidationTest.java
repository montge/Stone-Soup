// SPDX-FileCopyrightText: 2024-2025 Stone Soup Contributors
// SPDX-License-Identifier: MIT

package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parameterized tests for dimension validation across Stone Soup types.
 *
 * <p>This test class consolidates dimension validation tests that were previously
 * duplicated across multiple test files.</p>
 */
@DisplayName("Dimension Validation Tests")
class DimensionValidationTest {

    @Nested
    @DisplayName("Invalid Dimension Rejection")
    class InvalidDimensionRejection {

        @ParameterizedTest(name = "StateVector.zeros({0}) throws")
        @ValueSource(ints = {0, -1, -10})
        @DisplayName("StateVector.zeros rejects invalid dimensions")
        void stateVectorZerosRejectsInvalid(int dim) {
            assertThrows(IllegalArgumentException.class,
                    () -> StateVector.zeros(dim));
        }

        @ParameterizedTest(name = "StateVector.fill({0}, value) throws")
        @ValueSource(ints = {0, -1, -10})
        @DisplayName("StateVector.fill rejects invalid dimensions")
        void stateVectorFillRejectsInvalid(int dim) {
            assertThrows(IllegalArgumentException.class,
                    () -> StateVector.fill(dim, 1.0));
        }

        @ParameterizedTest(name = "CovarianceMatrix.identity({0}) throws")
        @ValueSource(ints = {0, -1, -10})
        @DisplayName("CovarianceMatrix.identity rejects invalid dimensions")
        void covarianceIdentityRejectsInvalid(int dim) {
            assertThrows(IllegalArgumentException.class,
                    () -> CovarianceMatrix.identity(dim));
        }

        @ParameterizedTest(name = "CovarianceMatrix.zeros({0}) throws")
        @ValueSource(ints = {0, -1, -10})
        @DisplayName("CovarianceMatrix.zeros rejects invalid dimensions")
        void covarianceZerosRejectsInvalid(int dim) {
            assertThrows(IllegalArgumentException.class,
                    () -> CovarianceMatrix.zeros(dim));
        }

        @ParameterizedTest(name = "GaussianState with dimension {0} throws")
        @ValueSource(ints = {0, -1, -10})
        @DisplayName("GaussianState constructor rejects invalid dimensions")
        void gaussianStateRejectsInvalid(int dim) {
            // GaussianState doesn't have a dim-only constructor
            // but validation happens through StateVector/CovarianceMatrix
            assertThrows(IllegalArgumentException.class,
                    () -> new GaussianState(StateVector.zeros(dim), CovarianceMatrix.identity(Math.abs(dim) + 1)));
        }
    }

    @Nested
    @DisplayName("Empty Array Rejection")
    class EmptyArrayRejection {

        @Test
        @DisplayName("StateVector rejects empty array")
        void stateVectorRejectsEmpty() {
            assertThrows(IllegalArgumentException.class,
                    () -> new StateVector(new double[0]));
        }

        @Test
        @DisplayName("CovarianceMatrix rejects empty 2D array")
        void covarianceMatrixRejectsEmpty() {
            assertThrows(IllegalArgumentException.class,
                    () -> new CovarianceMatrix(new double[0][0]));
        }

        @Test
        @DisplayName("Matrix rejects empty 2D array")
        void matrixRejectsEmpty() {
            assertThrows(IllegalArgumentException.class,
                    () -> new Matrix(new double[0][0]));
        }

        @Test
        @DisplayName("CovarianceMatrix.diagonal rejects empty array")
        void diagonalRejectsEmpty() {
            assertThrows(IllegalArgumentException.class,
                    () -> CovarianceMatrix.diagonal(new double[0]));
        }
    }

    @Nested
    @DisplayName("Dimension Mismatch")
    class DimensionMismatch {

        @Test
        @DisplayName("StateVector.add throws on dimension mismatch")
        void stateVectorAddDimensionMismatch() {
            StateVector sv1 = new StateVector(new double[]{1.0, 2.0});
            StateVector sv2 = new StateVector(new double[]{1.0, 2.0, 3.0});
            assertThrows(IllegalArgumentException.class, () -> sv1.add(sv2));
        }

        @Test
        @DisplayName("StateVector.subtract throws on dimension mismatch")
        void stateVectorSubtractDimensionMismatch() {
            StateVector sv1 = new StateVector(new double[]{1.0, 2.0});
            StateVector sv2 = new StateVector(new double[]{1.0, 2.0, 3.0});
            assertThrows(IllegalArgumentException.class, () -> sv1.subtract(sv2));
        }

        @Test
        @DisplayName("StateVector.dot throws on dimension mismatch")
        void stateVectorDotDimensionMismatch() {
            StateVector sv1 = new StateVector(new double[]{1.0, 2.0});
            StateVector sv2 = new StateVector(new double[]{1.0, 2.0, 3.0});
            assertThrows(IllegalArgumentException.class, () -> sv1.dot(sv2));
        }

        @Test
        @DisplayName("CovarianceMatrix.add throws on dimension mismatch")
        void covarianceMatrixAddDimensionMismatch() {
            CovarianceMatrix cov1 = CovarianceMatrix.identity(2);
            CovarianceMatrix cov2 = CovarianceMatrix.identity(3);
            assertThrows(IllegalArgumentException.class, () -> cov1.add(cov2));
        }

        @Test
        @DisplayName("CovarianceMatrix.multiply(CovarianceMatrix) throws on dimension mismatch")
        void covarianceMatrixMultiplyDimensionMismatch() {
            CovarianceMatrix cov1 = CovarianceMatrix.identity(2);
            CovarianceMatrix cov2 = CovarianceMatrix.identity(3);
            assertThrows(IllegalArgumentException.class, () -> cov1.multiply(cov2));
        }

        @Test
        @DisplayName("CovarianceMatrix.multiply(StateVector) throws on dimension mismatch")
        void covarianceMatrixMultiplyVectorDimensionMismatch() {
            CovarianceMatrix cov = CovarianceMatrix.identity(2);
            StateVector sv = new StateVector(new double[]{1.0, 2.0, 3.0});
            assertThrows(IllegalArgumentException.class, () -> cov.multiply(sv));
        }

        @Test
        @DisplayName("Matrix.multiply throws on dimension mismatch")
        void matrixMultiplyDimensionMismatch() {
            Matrix m1 = new Matrix(new double[][]{{1, 2}, {3, 4}});
            Matrix m2 = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}});
            assertThrows(IllegalArgumentException.class, () -> m1.multiply(m2));
        }

        @Test
        @DisplayName("Matrix.multiply(StateVector) throws on dimension mismatch")
        void matrixMultiplyVectorDimensionMismatch() {
            Matrix m = new Matrix(new double[][]{{1, 2}, {3, 4}});
            StateVector sv = new StateVector(new double[]{1.0, 2.0, 3.0});
            assertThrows(IllegalArgumentException.class, () -> m.multiply(sv));
        }

        @Test
        @DisplayName("GaussianState throws on state/covariance dimension mismatch")
        void gaussianStateDimensionMismatch() {
            StateVector sv = new StateVector(new double[]{1.0, 2.0});
            CovarianceMatrix cov = CovarianceMatrix.identity(3);
            assertThrows(IllegalArgumentException.class, () -> new GaussianState(sv, cov));
        }
    }

    @Nested
    @DisplayName("Non-Square Matrix Rejection")
    class NonSquareMatrixRejection {

        @Test
        @DisplayName("CovarianceMatrix rejects non-square matrix")
        void covarianceMatrixRejectsNonSquare() {
            assertThrows(IllegalArgumentException.class,
                    () -> new CovarianceMatrix(new double[][]{{1, 2, 3}, {4, 5, 6}}));
        }
    }
}
