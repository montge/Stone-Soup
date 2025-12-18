## 1. Java Bindings Consolidation
- [x] 1.1 Create `ValidationUtils.java` with common validation utilities (alternative to abstract base)
- [ ] 1.2 Create `AbstractMatrix.java` base class for matrix types
- [x] 1.3 Refactor `StateVector.java` to use `ValidationUtils`
- [x] 1.4 Refactor `CovarianceMatrix.java` to use `ValidationUtils`
- [x] 1.5 Refactor `Matrix.java` to use `ValidationUtils`
- [ ] 1.6 Extract common `equals()`/`hashCode()` patterns to utility class
- [ ] 1.7 Consolidate factory method patterns (`zeros`, `identity`, `diagonal`)

## 2. Java Test Refactoring
- [x] 2.1 Create `ValidationUtilsTest.java` with comprehensive validation tests
- [ ] 2.2 Create parameterized test fixtures for null rejection tests
- [ ] 2.3 Create parameterized test fixtures for dimension validation tests
- [ ] 2.4 Refactor `StateVectorTest.java` to use common patterns
- [ ] 2.5 Refactor `CovarianceMatrixTest.java` to use common patterns
- [ ] 2.6 Refactor `GaussianStateTest.java` to use common patterns
- [ ] 2.7 Refactor `KalmanFilterTest.java` to use common patterns

## 3. C++ Test Consolidation
- [ ] 3.1 Create `test_common.hpp` with TYPED_TEST macros for copy/move semantics
- [ ] 3.2 Refactor `test_state_vector.cpp` to use typed tests
- [ ] 3.3 Refactor `test_gaussian_state.cpp` to use typed tests
- [ ] 3.4 Refactor `test_covariance_matrix.cpp` to use typed tests

## 4. Python Validation Utilities
- [x] 4.1 Create `stonesoup/types/validation.py` with common validation functions
- [x] 4.2 Add `validate_shape()` function
- [x] 4.3 Add `validate_bounds()` function
- [x] 4.4 Add `check_index_bounds()` function
- [x] 4.5 Refactor `stonesoup/types/voxel.py` to use validation module
- [x] 4.6 Refactor `stonesoup/types/array.py` to use validation module

## 5. Python Test Fixtures
- [x] 5.1 Create `stonesoup/tests/fixtures.py` with common test data factories
- [x] 5.2 Add standard state vector fixtures
- [x] 5.3 Add standard covariance matrix fixtures
- [x] 5.4 Refactor Kalman filter tests to use shared fixtures (fixtures registered in conftest.py)

## 6. Verification
- [x] 6.1 All Java tests pass (206 tests)
- [x] 6.2 All C++ tests pass (8 tests)
- [x] 6.3 All Python tests pass (536 types tests)
- [ ] 6.4 SonarCloud duplication metric improves (target: < 3%)
- [ ] 6.5 No regression in test coverage
