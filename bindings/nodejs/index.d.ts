/**
 * TypeScript type definitions for @stonesoup/core
 *
 * Stone Soup Node.js bindings for target tracking and state estimation.
 */

/**
 * State vector representation for n-dimensional states.
 */
export declare class StateVector {
    /**
     * Create a new state vector from an array of values.
     */
    constructor(data: number[]);

    /**
     * Create a zero state vector of given dimension.
     */
    static zeros(dim: number): StateVector;

    /**
     * Get the dimensionality of the state vector.
     */
    readonly dims: number;

    /**
     * Get the value at a specific index.
     */
    get(index: number): number | undefined;

    /**
     * Set the value at a specific index.
     */
    set(index: number, value: number): void;

    /**
     * Convert to JavaScript array.
     */
    toArray(): number[];

    /**
     * Compute Euclidean norm.
     */
    norm(): number;

    /**
     * Add two state vectors.
     */
    add(other: StateVector): StateVector;

    /**
     * Subtract two state vectors.
     */
    sub(other: StateVector): StateVector;

    /**
     * Scale by a factor.
     */
    scale(factor: number): StateVector;

    /**
     * String representation.
     */
    toString(): string;
}

/**
 * Covariance matrix representation.
 */
export declare class CovarianceMatrix {
    /**
     * Create from 2D array.
     */
    constructor(data: number[][]);

    /**
     * Create an identity matrix.
     */
    static identity(dim: number): CovarianceMatrix;

    /**
     * Create a diagonal matrix.
     */
    static diagonal(diag: number[]): CovarianceMatrix;

    /**
     * Get dimension.
     */
    readonly dim: number;

    /**
     * Get element at (row, col).
     */
    get(row: number, col: number): number | undefined;

    /**
     * Set element at (row, col).
     */
    set(row: number, col: number, value: number): void;

    /**
     * Convert to 2D array.
     */
    toArray(): number[][];

    /**
     * Compute trace.
     */
    trace(): number;

    /**
     * String representation.
     */
    toString(): string;
}

/**
 * Gaussian state with mean and covariance.
 */
export declare class GaussianState {
    /**
     * Create a new Gaussian state.
     */
    constructor(stateVector: number[], covariance: number[][]);

    /**
     * Create with timestamp.
     */
    static withTimestamp(
        stateVector: number[],
        covariance: number[][],
        timestamp: number
    ): GaussianState;

    /**
     * Get the state vector.
     */
    readonly stateVector: number[];

    /**
     * Get the covariance matrix.
     */
    readonly covariance: number[][];

    /**
     * Get dimensionality.
     */
    readonly dims: number;

    /**
     * Get timestamp.
     */
    readonly timestamp: number | undefined;

    /**
     * Set timestamp.
     */
    set timestamp(value: number);

    /**
     * String representation.
     */
    toString(): string;
}

/**
 * Detection from a sensor.
 */
export declare class Detection {
    /**
     * Create a new detection.
     */
    constructor(measurement: number[], timestamp: number);

    /**
     * Get the measurement.
     */
    readonly measurement: number[];

    /**
     * Get the timestamp.
     */
    readonly timestamp: number;

    /**
     * String representation.
     */
    toString(): string;
}

/**
 * Track representing a target over time.
 */
export declare class Track {
    /**
     * Create a new track.
     */
    constructor(id: string);

    /**
     * Get the track ID.
     */
    readonly id: string;

    /**
     * Get the number of states.
     */
    readonly length: number;

    /**
     * Add a state to the track.
     */
    addState(state: GaussianState): void;

    /**
     * String representation.
     */
    toString(): string;
}

/**
 * Perform Kalman filter prediction.
 *
 * Computes:
 * - x_pred = F * x
 * - P_pred = F * P * F^T + Q
 *
 * @param prior - Prior Gaussian state
 * @param transitionMatrix - State transition matrix F
 * @param processNoise - Process noise covariance Q
 * @returns Predicted Gaussian state
 */
export declare function kalmanPredict(
    prior: GaussianState,
    transitionMatrix: number[][],
    processNoise: number[][]
): GaussianState;

/**
 * Perform Kalman filter update.
 *
 * Computes the posterior state given a measurement.
 *
 * @param predicted - Predicted Gaussian state
 * @param measurement - Measurement vector
 * @param measurementMatrix - Measurement matrix H
 * @param measurementNoise - Measurement noise covariance R
 * @returns Posterior Gaussian state
 */
export declare function kalmanUpdate(
    predicted: GaussianState,
    measurement: number[],
    measurementMatrix: number[][],
    measurementNoise: number[][]
): GaussianState;

/**
 * Create a constant velocity transition matrix.
 *
 * For a 2D system (x, vx, y, vy), use ndim=2.
 *
 * @param ndim - Number of spatial dimensions
 * @param dt - Time step
 * @returns Transition matrix as 2D array
 */
export declare function constantVelocityTransition(
    ndim: number,
    dt: number
): number[][];

/**
 * Create a position-only measurement matrix.
 *
 * For 2D, this observes x and y from state [x, vx, y, vy].
 *
 * @param ndim - Number of spatial dimensions
 * @returns Measurement matrix as 2D array
 */
export declare function positionMeasurement(ndim: number): number[][];

/**
 * Initialize the Stone Soup library.
 */
export declare function initialize(): void;

/**
 * Get version information.
 */
export declare function getVersion(): string;
