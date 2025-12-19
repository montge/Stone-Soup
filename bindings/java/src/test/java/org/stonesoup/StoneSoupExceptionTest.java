package org.stonesoup;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Nested;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for {@link StoneSoupException}.
 */
@DisplayName("StoneSoupException")
class StoneSoupExceptionTest {

    @Nested
    @DisplayName("Construction")
    class Construction {

        @Test
        @DisplayName("creates exception with message")
        void createsExceptionWithMessage() {
            StoneSoupException ex = new StoneSoupException("Test error");

            assertEquals("Test error", ex.getMessage());
            assertEquals(-1, ex.getErrorCode());
        }

        @Test
        @DisplayName("creates exception with message and cause")
        void createsExceptionWithMessageAndCause() {
            Throwable cause = new RuntimeException("Root cause");
            StoneSoupException ex = new StoneSoupException("Test error", cause);

            assertEquals("Test error", ex.getMessage());
            assertEquals(cause, ex.getCause());
            assertEquals(-1, ex.getErrorCode());
        }

        @Test
        @DisplayName("creates exception from error code")
        void createsExceptionFromErrorCode() {
            StoneSoupException ex = new StoneSoupException(StoneSoupException.ERROR_NULL_POINTER);

            assertEquals("Null pointer argument", ex.getMessage());
            assertEquals(StoneSoupException.ERROR_NULL_POINTER, ex.getErrorCode());
        }

        @Test
        @DisplayName("creates exception from error code with context")
        void createsExceptionFromErrorCodeWithContext() {
            StoneSoupException ex = new StoneSoupException(
                StoneSoupException.ERROR_DIMENSION,
                "matrix multiply"
            );

            assertEquals("matrix multiply: Dimension mismatch", ex.getMessage());
            assertEquals(StoneSoupException.ERROR_DIMENSION, ex.getErrorCode());
        }
    }

    @Nested
    @DisplayName("Error Messages")
    class ErrorMessages {

        @Test
        @DisplayName("returns correct message for SUCCESS")
        void returnsCorrectMessageForSuccess() {
            assertEquals("Success", StoneSoupException.getErrorMessage(StoneSoupException.SUCCESS));
        }

        @Test
        @DisplayName("returns correct message for NULL_POINTER")
        void returnsCorrectMessageForNullPointer() {
            assertEquals("Null pointer argument",
                StoneSoupException.getErrorMessage(StoneSoupException.ERROR_NULL_POINTER));
        }

        @Test
        @DisplayName("returns correct message for INVALID_SIZE")
        void returnsCorrectMessageForInvalidSize() {
            assertEquals("Invalid size parameter",
                StoneSoupException.getErrorMessage(StoneSoupException.ERROR_INVALID_SIZE));
        }

        @Test
        @DisplayName("returns correct message for ALLOCATION")
        void returnsCorrectMessageForAllocation() {
            assertEquals("Memory allocation failed",
                StoneSoupException.getErrorMessage(StoneSoupException.ERROR_ALLOCATION));
        }

        @Test
        @DisplayName("returns correct message for DIMENSION")
        void returnsCorrectMessageForDimension() {
            assertEquals("Dimension mismatch",
                StoneSoupException.getErrorMessage(StoneSoupException.ERROR_DIMENSION));
        }

        @Test
        @DisplayName("returns correct message for SINGULAR")
        void returnsCorrectMessageForSingular() {
            assertEquals("Singular matrix",
                StoneSoupException.getErrorMessage(StoneSoupException.ERROR_SINGULAR));
        }

        @Test
        @DisplayName("returns correct message for INVALID_ARG")
        void returnsCorrectMessageForInvalidArg() {
            assertEquals("Invalid argument",
                StoneSoupException.getErrorMessage(StoneSoupException.ERROR_INVALID_ARG));
        }

        @Test
        @DisplayName("returns correct message for NOT_IMPLEMENTED")
        void returnsCorrectMessageForNotImplemented() {
            assertEquals("Feature not implemented",
                StoneSoupException.getErrorMessage(StoneSoupException.ERROR_NOT_IMPLEMENTED));
        }

        @Test
        @DisplayName("returns unknown message for unknown code")
        void returnsUnknownMessageForUnknownCode() {
            String message = StoneSoupException.getErrorMessage(999);
            assertTrue(message.contains("Unknown error"));
            assertTrue(message.contains("999"));
        }
    }

    @Nested
    @DisplayName("Error Checking")
    class ErrorChecking {

        @Test
        @DisplayName("checkError does not throw on SUCCESS")
        void checkErrorDoesNotThrowOnSuccess() {
            assertDoesNotThrow(() -> StoneSoupException.checkError(StoneSoupException.SUCCESS));
        }

        @Test
        @DisplayName("checkError throws on error code")
        void checkErrorThrowsOnErrorCode() {
            StoneSoupException ex = assertThrows(StoneSoupException.class,
                () -> StoneSoupException.checkError(StoneSoupException.ERROR_SINGULAR));

            assertEquals(StoneSoupException.ERROR_SINGULAR, ex.getErrorCode());
        }

        @Test
        @DisplayName("checkError with context does not throw on SUCCESS")
        void checkErrorWithContextDoesNotThrowOnSuccess() {
            assertDoesNotThrow(() ->
                StoneSoupException.checkError(StoneSoupException.SUCCESS, "operation"));
        }

        @Test
        @DisplayName("checkError with context throws on error code")
        void checkErrorWithContextThrowsOnErrorCode() {
            StoneSoupException ex = assertThrows(StoneSoupException.class,
                () -> StoneSoupException.checkError(StoneSoupException.ERROR_ALLOCATION, "matrix create"));

            assertEquals(StoneSoupException.ERROR_ALLOCATION, ex.getErrorCode());
            assertTrue(ex.getMessage().contains("matrix create"));
            assertTrue(ex.getMessage().contains("Memory allocation failed"));
        }
    }

    @Nested
    @DisplayName("Error Code Constants")
    class ErrorCodeConstants {

        @Test
        @DisplayName("error codes have expected values")
        void errorCodesHaveExpectedValues() {
            assertEquals(0, StoneSoupException.SUCCESS);
            assertEquals(1, StoneSoupException.ERROR_NULL_POINTER);
            assertEquals(2, StoneSoupException.ERROR_INVALID_SIZE);
            assertEquals(3, StoneSoupException.ERROR_ALLOCATION);
            assertEquals(4, StoneSoupException.ERROR_DIMENSION);
            assertEquals(5, StoneSoupException.ERROR_SINGULAR);
            assertEquals(6, StoneSoupException.ERROR_INVALID_ARG);
            assertEquals(7, StoneSoupException.ERROR_NOT_IMPLEMENTED);
        }
    }
}
