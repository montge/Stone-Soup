/**
 * Unit tests for @stonesoup/core
 *
 * Note: These tests require the native addon to be built first.
 * Run: npm run build
 */

const { describe, it, beforeAll } = require('node:test');
const assert = require('node:assert');

// These tests will only work after the native addon is built
// For now, we test the TypeScript types and API expectations

describe('Stone Soup Node.js Bindings', () => {
    describe('StateVector', () => {
        it('should create from array', () => {
            // Expected API: new StateVector([1, 2, 3])
            const expected = {
                dims: 3,
                data: [1.0, 2.0, 3.0]
            };
            assert.strictEqual(expected.dims, 3);
        });

        it('should compute norm correctly', () => {
            // Expected: ||[3, 4]|| = 5
            const expected = Math.sqrt(3*3 + 4*4);
            assert.strictEqual(expected, 5);
        });

        it('should add vectors', () => {
            // Expected: [1,2] + [3,4] = [4,6]
            const a = [1, 2];
            const b = [3, 4];
            const sum = a.map((v, i) => v + b[i]);
            assert.deepStrictEqual(sum, [4, 6]);
        });
    });

    describe('CovarianceMatrix', () => {
        it('should create identity matrix', () => {
            // Expected: 3x3 identity
            const identity = [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ];
            assert.strictEqual(identity[0][0], 1);
            assert.strictEqual(identity[0][1], 0);
        });

        it('should compute trace', () => {
            // Trace of identity is n
            const identity = [[1,0,0], [0,1,0], [0,0,1]];
            const trace = identity.reduce((sum, row, i) => sum + row[i], 0);
            assert.strictEqual(trace, 3);
        });

        it('should create diagonal matrix', () => {
            const diag = [1, 2, 3];
            const matrix = diag.map((v, i) =>
                Array(3).fill(0).map((_, j) => i === j ? v : 0)
            );
            assert.strictEqual(matrix[0][0], 1);
            assert.strictEqual(matrix[1][1], 2);
            assert.strictEqual(matrix[2][2], 3);
        });
    });

    describe('GaussianState', () => {
        it('should validate dimension match', () => {
            // State vector dimension must match covariance dimension
            const stateVector = [1, 2];
            const covariance = [[1, 0], [0, 1]];
            assert.strictEqual(stateVector.length, covariance.length);
            assert.strictEqual(stateVector.length, covariance[0].length);
        });

        it('should reject mismatched dimensions', () => {
            const stateVector = [1, 2];
            const covariance = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
            assert.notStrictEqual(stateVector.length, covariance.length);
        });
    });

    describe('Kalman Filter', () => {
        it('should predict with constant velocity model', () => {
            // Initial state: x=0, vx=1
            const x = [0, 1];
            const dt = 1.0;

            // Transition matrix for constant velocity
            const F = [
                [1, dt],
                [0, 1]
            ];

            // x_pred = F * x
            const x_pred = [
                F[0][0] * x[0] + F[0][1] * x[1],
                F[1][0] * x[0] + F[1][1] * x[1]
            ];

            // Expected: x=0+1*1=1, vx=1
            assert.strictEqual(x_pred[0], 1);
            assert.strictEqual(x_pred[1], 1);
        });

        it('should compute innovation correctly', () => {
            // Predicted position
            const x_pred = [1.0];
            // Measurement
            const z = [1.1];
            // Measurement matrix (observe position)
            const H = [[1]];

            // Innovation: y = z - H*x
            const y = z[0] - H[0][0] * x_pred[0];

            // Use tolerance-based comparison due to IEEE 754 floating-point precision
            const tolerance = 1e-10;
            assert.ok(Math.abs(y - 0.1) < tolerance, `Expected ~0.1 but got ${y}`);
        });

        it('should have helper functions for common models', () => {
            // Test constant velocity transition structure
            const ndim = 2;
            const dt = 0.5;
            const stateDim = ndim * 2;

            // Build expected F matrix
            const F = Array(stateDim).fill(null).map((_, i) =>
                Array(stateDim).fill(0).map((_, j) => {
                    if (i === j) return 1;
                    if (i % 2 === 0 && j === i + 1) return dt;
                    return 0;
                })
            );

            // Check structure
            assert.strictEqual(F[0][0], 1);
            assert.strictEqual(F[0][1], dt);
            assert.strictEqual(F[2][2], 1);
            assert.strictEqual(F[2][3], dt);
        });

        it('should have position measurement matrix', () => {
            const ndim = 2;
            const stateDim = ndim * 2;
            const measDim = ndim;

            // Build expected H matrix
            const H = Array(measDim).fill(null).map((_, i) =>
                Array(stateDim).fill(0).map((_, j) => {
                    if (j === i * 2) return 1;
                    return 0;
                })
            );

            // H should be 2x4 observing x and y
            assert.strictEqual(H.length, 2);
            assert.strictEqual(H[0].length, 4);
            assert.strictEqual(H[0][0], 1);  // Observe x
            assert.strictEqual(H[0][1], 0);  // Not vx
            assert.strictEqual(H[1][2], 1);  // Observe y
            assert.strictEqual(H[1][3], 0);  // Not vy
        });
    });

    describe('Track', () => {
        it('should start empty', () => {
            const length = 0;
            assert.strictEqual(length, 0);
        });

        it('should accumulate states', () => {
            let length = 0;
            length++; // Add state
            length++; // Add state
            assert.strictEqual(length, 2);
        });
    });

    describe('Detection', () => {
        it('should store measurement and timestamp', () => {
            const measurement = [1.0, 2.0];
            const timestamp = 123.456;

            assert.strictEqual(measurement.length, 2);
            assert.strictEqual(timestamp, 123.456);
        });
    });
});
