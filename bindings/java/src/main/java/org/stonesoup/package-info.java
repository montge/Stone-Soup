/**
 * Stone Soup Java Bindings - Target Tracking and State Estimation.
 *
 * <p>This package provides Java bindings for the Stone Soup tracking framework,
 * enabling target tracking and state estimation algorithms in Java applications.</p>
 *
 * <h2>Key Classes</h2>
 * <ul>
 *   <li>{@link org.stonesoup.StoneSoup} - Main entry point and initialization</li>
 *   <li>{@link org.stonesoup.StateVector} - N-dimensional state vector</li>
 *   <li>{@link org.stonesoup.CovarianceMatrix} - Covariance/uncertainty matrix</li>
 *   <li>{@link org.stonesoup.GaussianState} - State with Gaussian uncertainty</li>
 *   <li>{@link org.stonesoup.KalmanFilter} - Kalman filter operations</li>
 *   <li>{@link org.stonesoup.Detection} - Sensor detection/measurement</li>
 *   <li>{@link org.stonesoup.Track} - Target track over time</li>
 * </ul>
 *
 * <h2>Getting Started</h2>
 * <pre>{@code
 * // Initialize the library
 * StoneSoup.initialize();
 *
 * // Create an initial state estimate
 * GaussianState state = GaussianState.of(
 *     new double[]{0.0, 1.0, 0.0, 1.0},  // [x, vx, y, vy]
 *     CovarianceMatrix.identity(4).toArray()
 * );
 *
 * // Set up Kalman filter
 * double dt = 1.0;
 * CovarianceMatrix F = KalmanFilter.constantVelocityTransition(2, dt);
 * CovarianceMatrix Q = CovarianceMatrix.identity(4).scale(0.1);
 * CovarianceMatrix H = KalmanFilter.positionMeasurement(2);
 * CovarianceMatrix R = CovarianceMatrix.identity(2).scale(0.5);
 *
 * // Predict step
 * GaussianState predicted = KalmanFilter.predict(state, F, Q);
 *
 * // Update step with measurement
 * StateVector measurement = new StateVector(new double[]{1.1, 0.9});
 * GaussianState posterior = KalmanFilter.update(predicted, measurement, H, R);
 *
 * // Clean up
 * StoneSoup.cleanup();
 * }</pre>
 *
 * <h2>Execution Modes</h2>
 * <p>The library supports two execution modes:</p>
 * <ul>
 *   <li><b>Native Mode</b>: Uses Project Panama FFM API to call the native
 *       C library for maximum performance. Requires Java 22+ (or Java 21 with
 *       preview features) and the native library.</li>
 *   <li><b>Pure Java Mode</b>: Uses pure Java implementations. Works on any
 *       Java 21+ runtime without native dependencies.</li>
 * </ul>
 *
 * <p>The library automatically selects the best available mode at runtime:</p>
 * <pre>{@code
 * StoneSoup.initialize();
 * System.out.println("Mode: " + StoneSoup.getMode());  // "native" or "java"
 * }</pre>
 *
 * <h2>Thread Safety</h2>
 * <p>The core data types ({@link org.stonesoup.StateVector},
 * {@link org.stonesoup.CovarianceMatrix}, {@link org.stonesoup.GaussianState})
 * are not thread-safe for mutation but are safe to share across threads
 * for read-only access. Create copies using the {@code copy()} method when
 * sharing mutable state between threads.</p>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 * @see <a href="https://github.com/dstl/Stone-Soup">Stone Soup on GitHub</a>
 */
package org.stonesoup;
