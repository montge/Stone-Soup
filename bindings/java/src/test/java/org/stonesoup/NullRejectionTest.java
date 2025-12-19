// SPDX-FileCopyrightText: 2024-2025 Stone Soup Contributors
// SPDX-License-Identifier: MIT

package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Parameterized tests for null argument rejection across Stone Soup types.
 *
 * <p>This test class consolidates null rejection tests that were previously
 * duplicated across multiple test files.</p>
 */
@DisplayName("Null Rejection Tests")
class NullRejectionTest {

    @Nested
    @DisplayName("Constructor Null Rejection")
    class ConstructorNullRejection {

        @Test
        @DisplayName("StateVector rejects null array")
        void stateVectorRejectsNull() {
            assertThrows(NullPointerException.class,
                    () -> new StateVector(null));
        }

        @Test
        @DisplayName("CovarianceMatrix rejects null 2D array")
        void covarianceMatrixRejectsNull() {
            assertThrows(NullPointerException.class,
                    () -> new CovarianceMatrix(null));
        }

        @Test
        @DisplayName("Matrix rejects null 2D array")
        void matrixRejectsNull() {
            assertThrows(NullPointerException.class,
                    () -> new Matrix(null));
        }

        @Test
        @DisplayName("GaussianState rejects null StateVector")
        void gaussianStateRejectsNullStateVector() {
            CovarianceMatrix covar = CovarianceMatrix.identity(2);
            assertThrows(NullPointerException.class,
                    () -> new GaussianState(null, covar));
        }

        @Test
        @DisplayName("GaussianState rejects null CovarianceMatrix")
        void gaussianStateRejectsNullCovariance() {
            StateVector sv = new StateVector(new double[]{1.0, 2.0});
            assertThrows(NullPointerException.class,
                    () -> new GaussianState(sv, null));
        }
    }

    @Nested
    @DisplayName("Method Null Rejection")
    class MethodNullRejection {

        @Test
        @DisplayName("StateVector.add rejects null")
        void stateVectorAddRejectsNull() {
            StateVector sv = new StateVector(new double[]{1.0, 2.0});
            assertThrows(NullPointerException.class, () -> sv.add(null));
        }

        @Test
        @DisplayName("StateVector.subtract rejects null")
        void stateVectorSubtractRejectsNull() {
            StateVector sv = new StateVector(new double[]{1.0, 2.0});
            assertThrows(NullPointerException.class, () -> sv.subtract(null));
        }

        @Test
        @DisplayName("StateVector.dot rejects null")
        void stateVectorDotRejectsNull() {
            StateVector sv = new StateVector(new double[]{1.0, 2.0});
            assertThrows(NullPointerException.class, () -> sv.dot(null));
        }

        @Test
        @DisplayName("CovarianceMatrix.add rejects null")
        void covarianceMatrixAddRejectsNull() {
            CovarianceMatrix cov = CovarianceMatrix.identity(2);
            assertThrows(NullPointerException.class, () -> cov.add(null));
        }

        @Test
        @DisplayName("CovarianceMatrix.subtract rejects null")
        void covarianceMatrixSubtractRejectsNull() {
            CovarianceMatrix cov = CovarianceMatrix.identity(2);
            assertThrows(NullPointerException.class, () -> cov.subtract(null));
        }

        @Test
        @DisplayName("CovarianceMatrix.multiply(CovarianceMatrix) rejects null")
        void covarianceMatrixMultiplyMatrixRejectsNull() {
            CovarianceMatrix cov = CovarianceMatrix.identity(2);
            assertThrows(NullPointerException.class, () -> cov.multiply((CovarianceMatrix) null));
        }

        @Test
        @DisplayName("CovarianceMatrix.multiply(StateVector) rejects null")
        void covarianceMatrixMultiplyVectorRejectsNull() {
            CovarianceMatrix cov = CovarianceMatrix.identity(2);
            assertThrows(NullPointerException.class, () -> cov.multiply((StateVector) null));
        }

        @Test
        @DisplayName("CovarianceMatrix.multiplyTranspose rejects null")
        void covarianceMatrixMultiplyTransposeRejectsNull() {
            CovarianceMatrix cov = CovarianceMatrix.identity(2);
            assertThrows(NullPointerException.class, () -> cov.multiplyTranspose(null));
        }
    }

    @Nested
    @DisplayName("Factory Method Null Rejection")
    class FactoryMethodNullRejection {

        @Test
        @DisplayName("CovarianceMatrix.diagonal rejects null")
        void diagonalRejectsNull() {
            assertThrows(NullPointerException.class,
                    () -> CovarianceMatrix.diagonal(null));
        }
    }
}
