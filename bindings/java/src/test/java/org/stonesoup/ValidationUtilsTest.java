// SPDX-FileCopyrightText: 2024-2025 Stone Soup Contributors
// SPDX-License-Identifier: MIT

package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for ValidationUtils class.
 */
@DisplayName("ValidationUtils")
class ValidationUtilsTest {

    @Nested
    @DisplayName("requireNonEmpty(double[])")
    class RequireNonEmptyArray {

        @Test
        @DisplayName("throws NullPointerException for null array")
        void throwsForNullArray() {
            assertThrows(NullPointerException.class,
                    () -> ValidationUtils.requireNonEmpty((double[]) null, "array"));
        }

        @Test
        @DisplayName("throws IllegalArgumentException for empty array")
        void throwsForEmptyArray() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requireNonEmpty(new double[0], "array"));
        }

        @Test
        @DisplayName("passes for non-empty array")
        void passesForNonEmptyArray() {
            assertDoesNotThrow(() -> ValidationUtils.requireNonEmpty(new double[]{1.0}, "array"));
        }
    }

    @Nested
    @DisplayName("requireNonEmpty(double[][])")
    class RequireNonEmptyMatrix {

        @Test
        @DisplayName("throws NullPointerException for null matrix")
        void throwsForNullMatrix() {
            assertThrows(NullPointerException.class,
                    () -> ValidationUtils.requireNonEmpty((double[][]) null, "matrix"));
        }

        @Test
        @DisplayName("throws IllegalArgumentException for empty matrix")
        void throwsForEmptyMatrix() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requireNonEmpty(new double[0][0], "matrix"));
        }

        @Test
        @DisplayName("throws IllegalArgumentException for empty columns")
        void throwsForEmptyColumns() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requireNonEmpty(new double[][]{{}, {}}, "matrix"));
        }

        @Test
        @DisplayName("throws IllegalArgumentException for inconsistent rows")
        void throwsForInconsistentRows() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requireNonEmpty(new double[][]{{1, 2}, {1}}, "matrix"));
        }

        @Test
        @DisplayName("passes for valid matrix")
        void passesForValidMatrix() {
            assertDoesNotThrow(() -> ValidationUtils.requireNonEmpty(
                    new double[][]{{1, 2}, {3, 4}}, "matrix"));
        }
    }

    @Nested
    @DisplayName("requireSquare")
    class RequireSquare {

        @Test
        @DisplayName("throws for non-square matrix")
        void throwsForNonSquare() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requireSquare(new double[][]{{1, 2, 3}, {4, 5, 6}}, "matrix"));
        }

        @Test
        @DisplayName("passes for square matrix")
        void passesForSquare() {
            assertDoesNotThrow(() -> ValidationUtils.requireSquare(
                    new double[][]{{1, 2}, {3, 4}}, "matrix"));
        }
    }

    @Nested
    @DisplayName("requirePositiveDimension")
    class RequirePositiveDimension {

        @Test
        @DisplayName("throws for zero dimension")
        void throwsForZero() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requirePositiveDimension(0, "dim"));
        }

        @Test
        @DisplayName("throws for negative dimension")
        void throwsForNegative() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requirePositiveDimension(-1, "dim"));
        }

        @Test
        @DisplayName("passes for positive dimension")
        void passesForPositive() {
            assertDoesNotThrow(() -> ValidationUtils.requirePositiveDimension(1, "dim"));
            assertDoesNotThrow(() -> ValidationUtils.requirePositiveDimension(10, "dim"));
        }
    }

    @Nested
    @DisplayName("requireMatchingDimensions")
    class RequireMatchingDimensions {

        @Test
        @DisplayName("throws for mismatched dimensions")
        void throwsForMismatch() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requireMatchingDimensions(3, 4, "a", "b"));
        }

        @Test
        @DisplayName("passes for matching dimensions")
        void passesForMatch() {
            assertDoesNotThrow(() -> ValidationUtils.requireMatchingDimensions(5, 5, "a", "b"));
        }
    }

    @Nested
    @DisplayName("requireMultiplicationCompatible")
    class RequireMultiplicationCompatible {

        @Test
        @DisplayName("throws for incompatible dimensions")
        void throwsForIncompatible() {
            assertThrows(IllegalArgumentException.class,
                    () -> ValidationUtils.requireMultiplicationCompatible(2, 3, 4, 5));
        }

        @Test
        @DisplayName("passes for compatible dimensions")
        void passesForCompatible() {
            assertDoesNotThrow(() -> ValidationUtils.requireMultiplicationCompatible(2, 3, 3, 4));
        }
    }

    @Nested
    @DisplayName("requireInBounds(int, int, String)")
    class RequireInBoundsIndex {

        @Test
        @DisplayName("throws for negative index")
        void throwsForNegative() {
            assertThrows(IndexOutOfBoundsException.class,
                    () -> ValidationUtils.requireInBounds(-1, 10, "index"));
        }

        @Test
        @DisplayName("throws for index >= size")
        void throwsForTooLarge() {
            assertThrows(IndexOutOfBoundsException.class,
                    () -> ValidationUtils.requireInBounds(10, 10, "index"));
        }

        @Test
        @DisplayName("passes for valid index")
        void passesForValid() {
            assertDoesNotThrow(() -> ValidationUtils.requireInBounds(0, 10, "index"));
            assertDoesNotThrow(() -> ValidationUtils.requireInBounds(9, 10, "index"));
        }
    }

    @Nested
    @DisplayName("requireInBounds(int, int, int, int)")
    class RequireInBoundsMatrix {

        @Test
        @DisplayName("throws for negative row")
        void throwsForNegativeRow() {
            assertThrows(IndexOutOfBoundsException.class,
                    () -> ValidationUtils.requireInBounds(-1, 0, 3, 3));
        }

        @Test
        @DisplayName("throws for row >= rows")
        void throwsForRowTooLarge() {
            assertThrows(IndexOutOfBoundsException.class,
                    () -> ValidationUtils.requireInBounds(3, 0, 3, 3));
        }

        @Test
        @DisplayName("throws for negative col")
        void throwsForNegativeCol() {
            assertThrows(IndexOutOfBoundsException.class,
                    () -> ValidationUtils.requireInBounds(0, -1, 3, 3));
        }

        @Test
        @DisplayName("throws for col >= cols")
        void throwsForColTooLarge() {
            assertThrows(IndexOutOfBoundsException.class,
                    () -> ValidationUtils.requireInBounds(0, 3, 3, 3));
        }

        @Test
        @DisplayName("passes for valid indices")
        void passesForValid() {
            assertDoesNotThrow(() -> ValidationUtils.requireInBounds(0, 0, 3, 3));
            assertDoesNotThrow(() -> ValidationUtils.requireInBounds(2, 2, 3, 3));
        }
    }
}
