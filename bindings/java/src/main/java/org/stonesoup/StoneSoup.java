package org.stonesoup;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Optional;

/**
 * Main entry point for Stone Soup Java bindings.
 *
 * <p>This class provides Java access to the Stone Soup tracking framework.
 * It supports both native C library bindings via Project Panama FFM API
 * (Java 22+) and pure Java fallback implementations.</p>
 *
 * <h2>Usage Modes</h2>
 * <ul>
 *   <li><b>Native Mode</b>: Uses FFM API to call the native C library for
 *       maximum performance. Requires Java 22+ and the native library.</li>
 *   <li><b>Pure Java Mode</b>: Uses pure Java implementations. Works on any
 *       Java 21+ runtime without native dependencies.</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Initialize (auto-detects mode)
 * StoneSoup.initialize();
 *
 * // Create a state
 * GaussianState state = GaussianState.of(
 *     new double[]{0, 1, 0, 1},  // [x, vx, y, vy]
 *     CovarianceMatrix.identity(4).toArray()
 * );
 *
 * // Use Kalman filter
 * CovarianceMatrix F = KalmanFilter.constantVelocityTransition(2, 1.0);
 * CovarianceMatrix Q = CovarianceMatrix.identity(4).scale(0.1);
 * GaussianState predicted = KalmanFilter.predict(state, F, Q);
 *
 * // Clean up
 * StoneSoup.cleanup();
 * }</pre>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 * @see GaussianState
 * @see KalmanFilter
 * @see StateVector
 * @see CovarianceMatrix
 */
public final class StoneSoup {

    /** Version string */
    public static final String VERSION = "0.1.0";

    /** Whether native library is available */
    private static boolean nativeAvailable = false;

    /** Whether the library has been initialized */
    private static boolean initialized = false;

    /** Native symbol lookup (if available) */
    private static SymbolLookup symbolLookup;

    /** Native linker (if available) */
    private static Linker linker;

    /** Arena for native memory (if needed) */
    private static Arena arena;

    // Native function handles
    private static MethodHandle stonesoup_init_handle;
    private static MethodHandle stonesoup_cleanup_handle;

    // Static initialization
    static {
        tryLoadNativeLibrary();
    }

    /**
     * Attempts to load the native library.
     */
    private static void tryLoadNativeLibrary() {
        try {
            String libName = getLibraryName();
            System.loadLibrary(libName);
            linker = Linker.nativeLinker();
            symbolLookup = SymbolLookup.loaderLookup();
            initializeNativeHandles();
            nativeAvailable = true;
        } catch (UnsatisfiedLinkError | NoClassDefFoundError e) {
            // Native library not available, use pure Java fallback
            nativeAvailable = false;
        }
    }

    /**
     * Gets the platform-specific library name.
     */
    private static String getLibraryName() {
        return "stonesoup";
    }

    /**
     * Initializes native function handles.
     */
    private static void initializeNativeHandles() {
        try {
            FunctionDescriptor initDesc = FunctionDescriptor.of(ValueLayout.JAVA_INT);
            FunctionDescriptor cleanupDesc = FunctionDescriptor.ofVoid();

            Optional<MemorySegment> initSymbol = symbolLookup.find("stonesoup_init");
            Optional<MemorySegment> cleanupSymbol = symbolLookup.find("stonesoup_cleanup");

            if (initSymbol.isPresent()) {
                stonesoup_init_handle = linker.downcallHandle(initSymbol.get(), initDesc);
            }
            if (cleanupSymbol.isPresent()) {
                stonesoup_cleanup_handle = linker.downcallHandle(cleanupSymbol.get(), cleanupDesc);
            }
        } catch (Exception e) {
            nativeAvailable = false;
        }
    }

    /**
     * Initializes the Stone Soup library.
     *
     * <p>This method should be called once before using any other library
     * functionality. It will automatically detect whether to use native
     * or pure Java implementations.</p>
     *
     * @throws StoneSoupException if initialization fails
     */
    public static void initialize() throws StoneSoupException {
        if (initialized) {
            return;
        }

        if (nativeAvailable && stonesoup_init_handle != null) {
            try {
                int result = (int) stonesoup_init_handle.invokeExact();
                if (result != 0) {
                    throw new StoneSoupException(result, "Native initialization failed");
                }
            } catch (StoneSoupException e) {
                throw e;
            } catch (Throwable t) {
                throw new StoneSoupException("Native initialization failed", t);
            }
        }

        arena = Arena.ofShared();
        initialized = true;
    }

    /**
     * Cleans up Stone Soup resources.
     *
     * <p>This method should be called when done using the library to
     * free any native resources.</p>
     */
    public static void cleanup() {
        if (!initialized) {
            return;
        }

        if (nativeAvailable && stonesoup_cleanup_handle != null) {
            try {
                stonesoup_cleanup_handle.invokeExact();
            } catch (Throwable t) {
                // Ignore cleanup errors
            }
        }

        if (arena != null) {
            arena.close();
            arena = null;
        }

        initialized = false;
    }

    /**
     * Gets the version string.
     *
     * @return the version string (e.g., "0.1.0")
     */
    public static String getVersion() {
        return VERSION;
    }

    /**
     * Checks if native library is available.
     *
     * @return true if native library is loaded and available
     */
    public static boolean isNativeAvailable() {
        return nativeAvailable;
    }

    /**
     * Checks if the library has been initialized.
     *
     * @return true if initialize() has been called
     */
    public static boolean isInitialized() {
        return initialized;
    }

    /**
     * Gets the execution mode.
     *
     * @return "native" if using native library, "java" if using pure Java
     */
    public static String getMode() {
        return nativeAvailable ? "native" : "java";
    }

    /**
     * Gets the shared memory arena (for advanced usage).
     *
     * @return the arena, or null if not initialized
     */
    static Arena getArena() {
        return arena;
    }

    /**
     * Gets the native linker (for advanced usage).
     *
     * @return the linker, or null if native not available
     */
    static Linker getLinker() {
        return linker;
    }

    /**
     * Gets the symbol lookup (for advanced usage).
     *
     * @return the symbol lookup, or null if native not available
     */
    static SymbolLookup getSymbolLookup() {
        return symbolLookup;
    }

    // Prevent instantiation
    private StoneSoup() {
        throw new AssertionError("Cannot instantiate utility class");
    }
}
