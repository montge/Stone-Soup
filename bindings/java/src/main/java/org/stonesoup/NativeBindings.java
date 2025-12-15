package org.stonesoup;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.Optional;

/**
 * Low-level native bindings using Project Panama FFM API.
 *
 * <p>This class provides direct access to the native C library functions
 * using Java 22+ Foreign Function & Memory API. For most use cases,
 * prefer the high-level wrapper classes like {@link StateVector},
 * {@link CovarianceMatrix}, and {@link GaussianState}.</p>
 *
 * <p><b>Note:</b> This class requires Java 22+ or Java 21 with preview features.
 * For Java <21, use the pure Java implementations which are automatically
 * selected when native bindings are unavailable.</p>
 *
 * <h2>Memory Management</h2>
 * <p>When using native bindings directly, you are responsible for memory
 * management. Use try-with-resources with {@link Arena} to ensure proper
 * cleanup:</p>
 *
 * <pre>{@code
 * try (Arena arena = Arena.ofConfined()) {
 *     MemorySegment vec = NativeBindings.stateVectorCreate(arena, 4);
 *     // Use vector...
 *     NativeBindings.stateVectorFree(vec);
 * }
 * }</pre>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public final class NativeBindings {

    // Struct layouts matching C definitions
    /**
     * Memory layout for stonesoup_state_vector_t struct.
     * <pre>
     * typedef struct {
     *     double* data;
     *     size_t size;
     * } stonesoup_state_vector_t;
     * </pre>
     */
    public static final StructLayout STATE_VECTOR_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("data"),
            ValueLayout.JAVA_LONG.withName("size")
    );

    /**
     * Memory layout for stonesoup_covariance_matrix_t struct.
     * <pre>
     * typedef struct {
     *     double* data;
     *     size_t rows;
     *     size_t cols;
     * } stonesoup_covariance_matrix_t;
     * </pre>
     */
    public static final StructLayout COVARIANCE_MATRIX_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("data"),
            ValueLayout.JAVA_LONG.withName("rows"),
            ValueLayout.JAVA_LONG.withName("cols")
    );

    /**
     * Memory layout for stonesoup_gaussian_state_t struct.
     * <pre>
     * typedef struct {
     *     stonesoup_state_vector_t* state_vector;
     *     stonesoup_covariance_matrix_t* covariance;
     *     double timestamp;
     * } stonesoup_gaussian_state_t;
     * </pre>
     */
    public static final StructLayout GAUSSIAN_STATE_LAYOUT = MemoryLayout.structLayout(
            ValueLayout.ADDRESS.withName("state_vector"),
            ValueLayout.ADDRESS.withName("covariance"),
            ValueLayout.JAVA_DOUBLE.withName("timestamp")
    );

    // Function descriptors
    private static final FunctionDescriptor STATE_VECTOR_CREATE_DESC =
            FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG);
    private static final FunctionDescriptor STATE_VECTOR_FREE_DESC =
            FunctionDescriptor.ofVoid(ValueLayout.ADDRESS);
    private static final FunctionDescriptor KALMAN_PREDICT_DESC =
            FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS);
    private static final FunctionDescriptor KALMAN_UPDATE_DESC =
            FunctionDescriptor.of(ValueLayout.JAVA_INT,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.ADDRESS,
                    ValueLayout.ADDRESS, ValueLayout.ADDRESS);

    // Method handles (lazy initialized with synchronization for thread safety)
    private static MethodHandle stateVectorCreate;
    private static MethodHandle stateVectorFree;
    private static MethodHandle stateVectorCopy;
    private static MethodHandle stateVectorFill;
    private static MethodHandle covarianceMatrixCreate;
    private static MethodHandle covarianceMatrixFree;
    private static MethodHandle covarianceMatrixEye;
    private static MethodHandle gaussianStateCreate;
    private static MethodHandle gaussianStateFree;
    private static MethodHandle kalmanPredict;
    private static MethodHandle kalmanUpdate;
    private static final Object LOCK = new Object();

    private NativeBindings() {
        throw new AssertionError("Cannot instantiate utility class");
    }

    /**
     * Checks if native bindings are available.
     *
     * @return true if native library is loaded and FFM API is available
     */
    public static boolean isAvailable() {
        return StoneSoup.isNativeAvailable();
    }

    /**
     * Creates a native state vector.
     *
     * @param arena the arena for memory allocation
     * @param size the dimension of the vector
     * @return the memory segment for the state vector, or null on error
     * @throws StoneSoupException if native bindings are unavailable
     */
    public static MemorySegment stateVectorCreate(Arena arena, long size) throws StoneSoupException {
        ensureNativeAvailable();
        try {
            MethodHandle mh = getStateVectorCreate();
            return (MemorySegment) mh.invokeExact(size);
        } catch (Throwable t) {
            throw new StoneSoupException("Failed to create state vector", t);
        }
    }

    /**
     * Frees a native state vector.
     *
     * @param vec the state vector to free
     * @throws StoneSoupException if native bindings are unavailable
     */
    public static void stateVectorFree(MemorySegment vec) throws StoneSoupException {
        if (vec == null || vec.equals(MemorySegment.NULL)) return;
        ensureNativeAvailable();
        try {
            MethodHandle mh = getStateVectorFree();
            mh.invokeExact(vec);
        } catch (Throwable t) {
            throw new StoneSoupException("Failed to free state vector", t);
        }
    }

    /**
     * Creates a native covariance matrix.
     *
     * @param arena the arena for memory allocation
     * @param rows number of rows
     * @param cols number of columns
     * @return the memory segment for the matrix, or null on error
     * @throws StoneSoupException if native bindings are unavailable
     */
    public static MemorySegment covarianceMatrixCreate(Arena arena, long rows, long cols)
            throws StoneSoupException {
        ensureNativeAvailable();
        try {
            MethodHandle mh = getCovarianceMatrixCreate();
            return (MemorySegment) mh.invokeExact(rows, cols);
        } catch (Throwable t) {
            throw new StoneSoupException("Failed to create covariance matrix", t);
        }
    }

    /**
     * Creates a native Gaussian state.
     *
     * @param arena the arena for memory allocation
     * @param stateDim the state dimension
     * @return the memory segment for the Gaussian state, or null on error
     * @throws StoneSoupException if native bindings are unavailable
     */
    public static MemorySegment gaussianStateCreate(Arena arena, long stateDim)
            throws StoneSoupException {
        ensureNativeAvailable();
        try {
            MethodHandle mh = getGaussianStateCreate();
            return (MemorySegment) mh.invokeExact(stateDim);
        } catch (Throwable t) {
            throw new StoneSoupException("Failed to create Gaussian state", t);
        }
    }

    /**
     * Performs native Kalman predict.
     *
     * @param prior the prior state
     * @param transitionMatrix the transition matrix
     * @param processNoise the process noise
     * @param predicted the output predicted state
     * @return error code (0 for success)
     * @throws StoneSoupException if native bindings are unavailable
     */
    public static int kalmanPredict(MemorySegment prior, MemorySegment transitionMatrix,
                                    MemorySegment processNoise, MemorySegment predicted)
            throws StoneSoupException {
        ensureNativeAvailable();
        try {
            MethodHandle mh = getKalmanPredict();
            return (int) mh.invokeExact(prior, transitionMatrix, processNoise, predicted);
        } catch (Throwable t) {
            throw new StoneSoupException("Kalman predict failed", t);
        }
    }

    /**
     * Performs native Kalman update.
     *
     * @param predicted the predicted state
     * @param measurement the measurement
     * @param measurementMatrix the measurement matrix
     * @param measurementNoise the measurement noise
     * @param posterior the output posterior state
     * @return error code (0 for success)
     * @throws StoneSoupException if native bindings are unavailable
     */
    public static int kalmanUpdate(MemorySegment predicted, MemorySegment measurement,
                                   MemorySegment measurementMatrix, MemorySegment measurementNoise,
                                   MemorySegment posterior) throws StoneSoupException {
        ensureNativeAvailable();
        try {
            MethodHandle mh = getKalmanUpdate();
            return (int) mh.invokeExact(predicted, measurement, measurementMatrix,
                    measurementNoise, posterior);
        } catch (Throwable t) {
            throw new StoneSoupException("Kalman update failed", t);
        }
    }

    // ========================================================================
    // Private helpers
    // ========================================================================

    private static void ensureNativeAvailable() throws StoneSoupException {
        if (!isAvailable()) {
            throw new StoneSoupException(
                    "Native bindings not available. Use pure Java implementations instead.");
        }
    }

    private static MethodHandle getStateVectorCreate() throws StoneSoupException {
        synchronized (LOCK) {
            if (stateVectorCreate == null) {
                stateVectorCreate = lookupFunction("stonesoup_state_vector_create",
                        STATE_VECTOR_CREATE_DESC);
            }
            return stateVectorCreate;
        }
    }

    private static MethodHandle getStateVectorFree() throws StoneSoupException {
        synchronized (LOCK) {
            if (stateVectorFree == null) {
                stateVectorFree = lookupFunction("stonesoup_state_vector_free",
                        STATE_VECTOR_FREE_DESC);
            }
            return stateVectorFree;
        }
    }

    private static MethodHandle getCovarianceMatrixCreate() throws StoneSoupException {
        synchronized (LOCK) {
            if (covarianceMatrixCreate == null) {
                covarianceMatrixCreate = lookupFunction("stonesoup_covariance_matrix_create",
                        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
            }
            return covarianceMatrixCreate;
        }
    }

    private static MethodHandle getGaussianStateCreate() throws StoneSoupException {
        synchronized (LOCK) {
            if (gaussianStateCreate == null) {
                gaussianStateCreate = lookupFunction("stonesoup_gaussian_state_create",
                        FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.JAVA_LONG));
            }
            return gaussianStateCreate;
        }
    }

    private static MethodHandle getKalmanPredict() throws StoneSoupException {
        synchronized (LOCK) {
            if (kalmanPredict == null) {
                kalmanPredict = lookupFunction("stonesoup_kalman_predict", KALMAN_PREDICT_DESC);
            }
            return kalmanPredict;
        }
    }

    private static MethodHandle getKalmanUpdate() throws StoneSoupException {
        synchronized (LOCK) {
            if (kalmanUpdate == null) {
                kalmanUpdate = lookupFunction("stonesoup_kalman_update", KALMAN_UPDATE_DESC);
            }
            return kalmanUpdate;
        }
    }

    private static MethodHandle lookupFunction(String name, FunctionDescriptor descriptor)
            throws StoneSoupException {
        SymbolLookup lookup = StoneSoup.getSymbolLookup();
        if (lookup == null) {
            throw new StoneSoupException("Symbol lookup not available");
        }
        Linker linker = StoneSoup.getLinker();
        if (linker == null) {
            throw new StoneSoupException("Linker not available");
        }

        Optional<MemorySegment> symbol = lookup.find(name);
        if (symbol.isEmpty()) {
            throw new StoneSoupException("Symbol not found: " + name);
        }

        return linker.downcallHandle(symbol.get(), descriptor);
    }
}
