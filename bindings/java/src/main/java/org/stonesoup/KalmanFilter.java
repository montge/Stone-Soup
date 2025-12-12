package org.stonesoup;

import java.util.Objects;

/**
 * Kalman filter operations for linear state estimation.
 *
 * <p>The Kalman filter is an optimal estimator for linear Gaussian systems.
 * It recursively estimates the state of a system from a series of noisy
 * measurements.</p>
 *
 * <h2>Algorithm</h2>
 * <p>Prediction step:</p>
 * <ul>
 *   <li>x_pred = F * x</li>
 *   <li>P_pred = F * P * F^T + Q</li>
 * </ul>
 *
 * <p>Update step:</p>
 * <ul>
 *   <li>y = z - H * x_pred (innovation)</li>
 *   <li>S = H * P_pred * H^T + R (innovation covariance)</li>
 *   <li>K = P_pred * H^T * S^-1 (Kalman gain)</li>
 *   <li>x_post = x_pred + K * y</li>
 *   <li>P_post = (I - K * H) * P_pred</li>
 * </ul>
 *
 * <h2>Example Usage</h2>
 * <pre>{@code
 * // Create initial state
 * GaussianState prior = GaussianState.of(
 *     new double[]{0, 1, 0, 1},  // [x, vx, y, vy]
 *     CovarianceMatrix.identity(4).toArray()
 * );
 *
 * // Create transition matrix for constant velocity model
 * CovarianceMatrix F = KalmanFilter.constantVelocityTransition(2, 1.0);
 * CovarianceMatrix Q = CovarianceMatrix.identity(4).scale(0.1);
 *
 * // Predict
 * GaussianState predicted = KalmanFilter.predict(prior, F, Q);
 *
 * // Update with measurement
 * StateVector measurement = new StateVector(new double[]{1.1, 0.9});
 * CovarianceMatrix H = KalmanFilter.positionMeasurement(2);
 * CovarianceMatrix R = CovarianceMatrix.identity(2).scale(0.5);
 *
 * GaussianState posterior = KalmanFilter.update(predicted, measurement, H, R);
 * }</pre>
 *
 * @author Stone Soup Contributors
 * @version 0.1.0
 * @since 0.1.0
 */
public final class KalmanFilter {

    private KalmanFilter() {
        throw new AssertionError("Cannot instantiate utility class");
    }

    /**
     * Performs the Kalman filter prediction step.
     *
     * <p>Computes:</p>
     * <ul>
     *   <li>x_pred = F * x</li>
     *   <li>P_pred = F * P * F^T + Q</li>
     * </ul>
     *
     * @param prior the prior Gaussian state
     * @param transitionMatrix the state transition matrix F
     * @param processNoise the process noise covariance Q
     * @return the predicted Gaussian state
     * @throws IllegalArgumentException if dimensions don't match
     */
    public static GaussianState predict(
            GaussianState prior,
            CovarianceMatrix transitionMatrix,
            CovarianceMatrix processNoise) {

        Objects.requireNonNull(prior, "Prior cannot be null");
        Objects.requireNonNull(transitionMatrix, "Transition matrix cannot be null");
        Objects.requireNonNull(processNoise, "Process noise cannot be null");

        int dim = prior.getDim();
        if (transitionMatrix.getDim() != dim) {
            throw new IllegalArgumentException(
                    "Transition matrix dimension mismatch: " + transitionMatrix.getDim() +
                    " vs state " + dim);
        }
        if (processNoise.getDim() != dim) {
            throw new IllegalArgumentException(
                    "Process noise dimension mismatch: " + processNoise.getDim() +
                    " vs state " + dim);
        }

        // x_pred = F * x
        StateVector xPred = transitionMatrix.multiply(prior.getStateVector());

        // P_pred = F * P * F^T + Q
        CovarianceMatrix FP = transitionMatrix.multiply(prior.getCovariance());
        CovarianceMatrix FPFt = FP.multiplyTranspose(transitionMatrix);
        CovarianceMatrix pPred = FPFt.add(processNoise);

        return new GaussianState(xPred, pPred, prior.getTimestamp().orElse(null));
    }

    /**
     * Performs the Kalman filter update step.
     *
     * <p>Computes:</p>
     * <ul>
     *   <li>y = z - H * x_pred (innovation)</li>
     *   <li>S = H * P_pred * H^T + R (innovation covariance)</li>
     *   <li>K = P_pred * H^T * S^-1 (Kalman gain)</li>
     *   <li>x_post = x_pred + K * y</li>
     *   <li>P_post = (I - K * H) * P_pred</li>
     * </ul>
     *
     * @param predicted the predicted Gaussian state
     * @param measurement the measurement vector
     * @param measurementMatrix the measurement matrix H
     * @param measurementNoise the measurement noise covariance R
     * @return the posterior Gaussian state
     * @throws IllegalArgumentException if dimensions don't match
     * @throws StoneSoupException if matrix inversion fails (singular S)
     */
    public static GaussianState update(
            GaussianState predicted,
            StateVector measurement,
            CovarianceMatrix measurementMatrix,
            CovarianceMatrix measurementNoise) throws StoneSoupException {

        Objects.requireNonNull(predicted, "Predicted state cannot be null");
        Objects.requireNonNull(measurement, "Measurement cannot be null");
        Objects.requireNonNull(measurementMatrix, "Measurement matrix cannot be null");
        Objects.requireNonNull(measurementNoise, "Measurement noise cannot be null");

        int stateDim = predicted.getDim();
        int measDim = measurement.getDim();

        // Validate dimensions
        if (measurementNoise.getDim() != measDim) {
            throw new IllegalArgumentException(
                    "Measurement noise dimension mismatch: " + measurementNoise.getDim() +
                    " vs measurement " + measDim);
        }

        // Innovation: y = z - H * x_pred
        StateVector hx = matrixVectorMultiply(measurementMatrix, predicted.getStateVector());
        StateVector innovation = measurement.subtract(hx);

        // Innovation covariance: S = H * P * H^T + R
        CovarianceMatrix HP = matrixMultiply(measurementMatrix, predicted.getCovariance());
        CovarianceMatrix HPHt = matrixMultiplyTranspose(HP, measurementMatrix);
        CovarianceMatrix S = HPHt.add(measurementNoise);

        // Kalman gain: K = P * H^T * S^-1
        CovarianceMatrix SInv = invert(S);
        CovarianceMatrix Ht = measurementMatrix.transpose();
        CovarianceMatrix PHt = matrixMultiplyRect(predicted.getCovariance(), Ht);
        CovarianceMatrix K = matrixMultiplyRect(PHt, SInv);

        // Posterior state: x_post = x_pred + K * y
        StateVector Ky = matrixVectorMultiplyRect(K, innovation);
        StateVector xPost = predicted.getStateVector().add(Ky);

        // Posterior covariance: P_post = (I - K * H) * P_pred
        CovarianceMatrix KH = matrixMultiplyRect(K, measurementMatrix);
        CovarianceMatrix I = CovarianceMatrix.identity(stateDim);
        CovarianceMatrix IminusKH = I.subtract(KH);
        CovarianceMatrix pPost = IminusKH.multiply(predicted.getCovariance());

        return new GaussianState(xPost, pPost, predicted.getTimestamp().orElse(null));
    }

    /**
     * Creates a constant velocity transition matrix.
     *
     * <p>For a 2D system (ndim=2), the state is [x, vx, y, vy] and the
     * transition matrix is:</p>
     * <pre>
     * | 1  dt  0   0 |
     * | 0   1  0   0 |
     * | 0   0  1  dt |
     * | 0   0  0   1 |
     * </pre>
     *
     * @param ndim number of spatial dimensions
     * @param dt time step
     * @return the transition matrix
     */
    public static CovarianceMatrix constantVelocityTransition(int ndim, double dt) {
        if (ndim < 1) {
            throw new IllegalArgumentException("ndim must be at least 1");
        }
        int stateDim = ndim * 2;
        double[][] F = new double[stateDim][stateDim];

        for (int i = 0; i < ndim; i++) {
            int posIdx = i * 2;
            int velIdx = i * 2 + 1;
            F[posIdx][posIdx] = 1.0;      // Position stays
            F[posIdx][velIdx] = dt;        // Velocity affects position
            F[velIdx][velIdx] = 1.0;      // Velocity stays
        }

        return new CovarianceMatrix(F);
    }

    /**
     * Creates a position-only measurement matrix.
     *
     * <p>For a 2D system (ndim=2), the state is [x, vx, y, vy] and the
     * measurement matrix is:</p>
     * <pre>
     * | 1  0  0  0 |
     * | 0  0  1  0 |
     * </pre>
     *
     * @param ndim number of spatial dimensions
     * @return the measurement matrix (ndim x ndim*2)
     */
    public static CovarianceMatrix positionMeasurement(int ndim) {
        if (ndim < 1) {
            throw new IllegalArgumentException("ndim must be at least 1");
        }
        int stateDim = ndim * 2;
        double[][] H = new double[ndim][stateDim];

        for (int i = 0; i < ndim; i++) {
            H[i][i * 2] = 1.0;  // Observe position in each dimension
        }

        return new CovarianceMatrix(H);
    }

    /**
     * Computes the innovation (measurement residual).
     *
     * @param predicted the predicted state
     * @param measurement the measurement
     * @param measurementMatrix the measurement matrix H
     * @return the innovation vector y = z - H * x
     */
    public static StateVector innovation(
            GaussianState predicted,
            StateVector measurement,
            CovarianceMatrix measurementMatrix) {

        StateVector hx = matrixVectorMultiply(measurementMatrix, predicted.getStateVector());
        return measurement.subtract(hx);
    }

    /**
     * Computes the Mahalanobis distance for gating.
     *
     * @param innovation the innovation vector
     * @param innovationCovariance the innovation covariance S
     * @return the Mahalanobis distance
     * @throws StoneSoupException if matrix inversion fails
     */
    public static double mahalanobisDistance(
            StateVector innovation,
            CovarianceMatrix innovationCovariance) throws StoneSoupException {

        CovarianceMatrix SInv = invert(innovationCovariance);
        StateVector SInvY = matrixVectorMultiply(SInv, innovation);
        return Math.sqrt(innovation.dot(SInvY));
    }

    // ========================================================================
    // Helper methods for matrix operations with rectangular matrices
    // ========================================================================

    /**
     * Matrix-vector multiply for potentially non-square matrices.
     */
    private static StateVector matrixVectorMultiply(CovarianceMatrix A, StateVector x) {
        int rows = A.getDim();
        int cols = x.getDim();
        double[] result = new double[rows];
        double[][] aData = A.toArray();

        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols && j < aData[i].length; j++) {
                sum += aData[i][j] * x.get(j);
            }
            result[i] = sum;
        }
        return new StateVector(result);
    }

    /**
     * Matrix-vector multiply for rectangular matrices.
     */
    private static StateVector matrixVectorMultiplyRect(CovarianceMatrix A, StateVector x) {
        double[][] aData = A.toArray();
        int rows = aData.length;
        int cols = aData[0].length;

        if (x.getDim() != cols) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: matrix cols " + cols + " vs vector " + x.getDim());
        }

        double[] result = new double[rows];
        for (int i = 0; i < rows; i++) {
            double sum = 0.0;
            for (int j = 0; j < cols; j++) {
                sum += aData[i][j] * x.get(j);
            }
            result[i] = sum;
        }
        return new StateVector(result);
    }

    /**
     * Matrix multiply for potentially non-square matrices.
     */
    private static CovarianceMatrix matrixMultiply(CovarianceMatrix A, CovarianceMatrix B) {
        double[][] aData = A.toArray();
        double[][] bData = B.toArray();
        int m = aData.length;
        int n = bData[0].length;
        int k = aData[0].length;

        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += aData[i][l] * bData[l][j];
                }
                result[i][j] = sum;
            }
        }
        return new CovarianceMatrix(result);
    }

    /**
     * Matrix multiply for rectangular matrices: C = A * B
     */
    private static CovarianceMatrix matrixMultiplyRect(CovarianceMatrix A, CovarianceMatrix B) {
        double[][] aData = A.toArray();
        double[][] bData = B.toArray();
        int m = aData.length;
        int k = aData[0].length;
        int n = bData[0].length;

        if (k != bData.length) {
            throw new IllegalArgumentException(
                    "Dimension mismatch: A cols " + k + " vs B rows " + bData.length);
        }

        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += aData[i][l] * bData[l][j];
                }
                result[i][j] = sum;
            }
        }
        return new CovarianceMatrix(result);
    }

    /**
     * Matrix multiply with transpose: C = A * B^T
     */
    private static CovarianceMatrix matrixMultiplyTranspose(CovarianceMatrix A, CovarianceMatrix B) {
        double[][] aData = A.toArray();
        double[][] bData = B.toArray();
        int m = aData.length;
        int n = bData.length;
        int k = aData[0].length;

        double[][] result = new double[m][n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = 0; l < k && l < bData[j].length; l++) {
                    sum += aData[i][l] * bData[j][l];
                }
                result[i][j] = sum;
            }
        }
        return new CovarianceMatrix(result);
    }

    /**
     * Inverts a matrix using Gaussian elimination.
     */
    private static CovarianceMatrix invert(CovarianceMatrix A) throws StoneSoupException {
        int n = A.getDim();
        double[][] a = A.toArray();
        double[][] inv = new double[n][n];

        // Initialize inverse as identity
        for (int i = 0; i < n; i++) {
            inv[i][i] = 1.0;
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n; col++) {
            // Find pivot
            int maxRow = col;
            for (int row = col + 1; row < n; row++) {
                if (Math.abs(a[row][col]) > Math.abs(a[maxRow][col])) {
                    maxRow = row;
                }
            }

            // Swap rows
            double[] temp = a[col];
            a[col] = a[maxRow];
            a[maxRow] = temp;
            temp = inv[col];
            inv[col] = inv[maxRow];
            inv[maxRow] = temp;

            // Check for singular matrix
            if (Math.abs(a[col][col]) < 1e-12) {
                throw new StoneSoupException(StoneSoupException.ERROR_SINGULAR,
                        "Matrix inversion failed");
            }

            // Scale pivot row
            double pivot = a[col][col];
            for (int j = 0; j < n; j++) {
                a[col][j] /= pivot;
                inv[col][j] /= pivot;
            }

            // Eliminate column
            for (int row = 0; row < n; row++) {
                if (row != col) {
                    double factor = a[row][col];
                    for (int j = 0; j < n; j++) {
                        a[row][j] -= factor * a[col][j];
                        inv[row][j] -= factor * inv[col][j];
                    }
                }
            }
        }

        return new CovarianceMatrix(inv);
    }
}
