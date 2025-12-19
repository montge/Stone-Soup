package org.stonesoup;

/**
 * Exception thrown by Stone Soup operations.
 *
 * <p>This exception wraps error codes from the native C library and provides
 * meaningful error messages for Java consumers.</p>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public class StoneSoupException extends Exception {

    /** Error code from native library */
    private final int errorCode;

    /** Error code constants matching C library */
    public static final int SUCCESS = 0;
    public static final int ERROR_NULL_POINTER = 1;
    public static final int ERROR_INVALID_SIZE = 2;
    public static final int ERROR_ALLOCATION = 3;
    public static final int ERROR_DIMENSION = 4;
    public static final int ERROR_SINGULAR = 5;
    public static final int ERROR_INVALID_ARG = 6;
    public static final int ERROR_NOT_IMPLEMENTED = 7;

    /**
     * Creates a new StoneSoupException with a message.
     *
     * @param message the error message
     */
    public StoneSoupException(String message) {
        super(message);
        this.errorCode = -1;
    }

    /**
     * Creates a new StoneSoupException with a message and cause.
     *
     * @param message the error message
     * @param cause the underlying cause
     */
    public StoneSoupException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = -1;
    }

    /**
     * Creates a new StoneSoupException from a native error code.
     *
     * @param errorCode the error code from the native library
     */
    public StoneSoupException(int errorCode) {
        super(getErrorMessage(errorCode));
        this.errorCode = errorCode;
    }

    /**
     * Creates a new StoneSoupException from a native error code with context.
     *
     * @param errorCode the error code from the native library
     * @param context additional context about the operation
     */
    public StoneSoupException(int errorCode, String context) {
        super(context + ": " + getErrorMessage(errorCode));
        this.errorCode = errorCode;
    }

    /**
     * Gets the native error code.
     *
     * @return the error code, or -1 if not from native code
     */
    public int getErrorCode() {
        return errorCode;
    }

    /**
     * Gets a human-readable error message for an error code.
     *
     * @param errorCode the error code
     * @return the error message
     */
    public static String getErrorMessage(int errorCode) {
        return switch (errorCode) {
            case SUCCESS -> "Success";
            case ERROR_NULL_POINTER -> "Null pointer argument";
            case ERROR_INVALID_SIZE -> "Invalid size parameter";
            case ERROR_ALLOCATION -> "Memory allocation failed";
            case ERROR_DIMENSION -> "Dimension mismatch";
            case ERROR_SINGULAR -> "Singular matrix";
            case ERROR_INVALID_ARG -> "Invalid argument";
            case ERROR_NOT_IMPLEMENTED -> "Feature not implemented";
            default -> "Unknown error (code: " + errorCode + ")";
        };
    }

    /**
     * Checks an error code and throws an exception if not SUCCESS.
     *
     * @param errorCode the error code to check
     * @throws StoneSoupException if the error code is not SUCCESS
     */
    public static void checkError(int errorCode) throws StoneSoupException {
        if (errorCode != SUCCESS) {
            throw new StoneSoupException(errorCode);
        }
    }

    /**
     * Checks an error code and throws an exception if not SUCCESS.
     *
     * @param errorCode the error code to check
     * @param context additional context about the operation
     * @throws StoneSoupException if the error code is not SUCCESS
     */
    public static void checkError(int errorCode, String context) throws StoneSoupException {
        if (errorCode != SUCCESS) {
            throw new StoneSoupException(errorCode, context);
        }
    }
}
