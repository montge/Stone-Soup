# Stone Soup C Library - Test Suite

This directory contains comprehensive unit tests for the Stone Soup C library.

## Test Files

### 1. test_types.c
Tests for basic type system operations:
- Version information
- Error string handling
- State vector creation, copying, and filling
- Covariance matrix creation and identity matrix
- Gaussian state management
- Particle state creation and weight normalization
- Library initialization and cleanup

**Status**: All 12 tests passing ✓

### 2. test_matrix.c
Comprehensive tests for matrix and linear algebra operations:

#### Matrix-Matrix Operations:
- Matrix multiplication (A × B)
- Matrix transpose (A^T)
- Matrix addition and subtraction
- Dimension mismatch error handling

#### Matrix-Vector Operations:
- Matrix-vector multiplication (A × x)
- Transposed matrix-vector multiplication (A^T × x)

#### Vector Operations:
- Vector addition and subtraction
- Vector dot product
- Vector outer product (x × y^T)
- Vector scaling

#### Advanced Operations:
- Matrix inverse (with verification A × A^-1 = I)
- Cholesky decomposition (verify L × L^T = A)
- Cholesky solve for linear systems (A × x = b)
- Matrix determinant computation
- Null pointer and dimension error handling

**Status**: 10/14 tests passing (some advanced operations have implementation bugs)

### 3. test_kalman.c
Tests for Kalman filter operations:

#### Linear Kalman Filter:
- Prediction step with constant velocity model
- Update step with position measurements
- Full predict-update cycle
- Innovation (measurement residual) computation
- Innovation covariance computation
- Null pointer error handling

#### Extended Kalman Filter (EKF):
- EKF prediction with nonlinear transition functions
- EKF update with nonlinear measurement functions

**Status**: Tests detect NOT_IMPLEMENTED errors correctly; some fail on implementation bugs

### 4. test_particle.c
Tests for particle filter operations:

#### Particle State Management:
- Weight normalization (verify sum to 1.0)
- Particle creation and initialization

#### Prediction:
- Particle prediction with transition function
- Process noise application
- Null pointer and size mismatch handling

#### Update:
- Weight update based on measurement likelihood
- Verification that weights are updated correctly
- Weight normalization after update

#### Statistics:
- Weighted mean computation from particle distribution
- Weighted covariance computation
- Effective sample size (N_eff) calculation

#### Resampling:
- Systematic resampling (low variance)
- Stratified resampling
- Multinomial resampling
- Verify uniform weights after resampling
- Verify particle diversity preservation

**Status**: 10/13 tests passing (resampling methods have implementation bugs)

## Building and Running Tests

### Build Tests
```bash
cd libstonesoup
mkdir -p build && cd build
cmake .. -DBUILD_TESTS=ON
make -j4
```

### Run All Tests
```bash
ctest --output-on-failure
```

### Run Individual Test Suites
```bash
./tests/test_types
./tests/test_matrix
./tests/test_kalman
./tests/test_particle
```

## Test Design Principles

1. **Clear Test Structure**: Each test follows a consistent pattern:
   - Create test data
   - Call the function under test
   - Verify results with assertions
   - Clean up memory

2. **Error Handling**: Tests verify proper error codes for:
   - Null pointer arguments
   - Dimension mismatches
   - Invalid arguments
   - Singular matrices

3. **Graceful Skipping**: Tests for unimplemented functions detect `STONESOUP_ERROR_NOT_IMPLEMENTED` and skip gracefully with "SKIPPED (not implemented)" message.

4. **Numerical Precision**: Uses epsilon-based comparison (1e-6 or 1e-9) for floating-point comparisons.

5. **Memory Safety**: All tests properly allocate and free resources, avoiding memory leaks.

## Test Coverage Summary

| Module | Functions Tested | Status |
|--------|-----------------|---------|
| Types | 12 | ✓ All passing |
| Matrix Operations | 20+ | Most passing, some implementation bugs |
| Kalman Filter | 6 | Tests working, many functions not implemented |
| Particle Filter | 8 | Tests working, resampling needs implementation |

## Known Issues

The tests successfully identify the following issues in the library implementation:

1. **Matrix Inversion**: The LU-based matrix inversion may have numerical precision issues with certain matrices.

2. **Kalman Update**: The kalman_update function returns NOT_IMPLEMENTED error - implementation needed.

3. **Particle Resampling**: Systematic, stratified, and multinomial resampling methods return NOT_IMPLEMENTED - implementations needed.

4. **Particle Covariance**: The particle covariance computation returns NOT_IMPLEMENTED - implementation needed.

## Future Enhancements

1. Add tests for coordinate transformations
2. Add benchmark tests for performance
3. Add tests for edge cases (empty particles, zero covariance, etc.)
4. Add tests for numerical stability with ill-conditioned matrices
5. Add integration tests combining multiple components
6. Add randomized testing for resampling algorithms

## Test Framework

The tests use a simple custom testing framework defined in each file:
- `TEST(name)` macro for running tests
- `double_equals()` helper for floating-point comparison
- Pass/fail counting and summary reporting
- Return code 0 for success, 1 for any failures
