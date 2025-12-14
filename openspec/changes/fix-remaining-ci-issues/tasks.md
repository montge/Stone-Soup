## 1. Fix Ruff Lint Error
- [x] 1.1 Update `stonesoup/serialise.py:156` to use `next(iter(...))` instead of `[0]` slice

## 2. Fix Rust Type Annotations
- [x] 2.1 Add `::<f64>` type hints to `.sum()` calls in `bindings/python/src/lib.rs`
- [x] 2.2 Add `::<f64>` type hints to additional `.sum()` calls at lines 635 and 642
- [x] 2.3 Change `into_pyarray` to `into_pyarray_bound` for numpy 0.22 compatibility

## 3. Implement Fuzzing Infrastructure
- [x] 3.1 Add `ENABLE_FUZZING` CMake option to `libstonesoup/CMakeLists.txt`
- [x] 3.2 Fix `fuzz_kalman` target to use correct library name (`stonesoup_static`)
- [x] 3.3 Fix CMake seed corpus creation using execute_process with printf
- [x] 3.4 Add missing `#include <stdio.h>` in fuzz_kalman.c
- [x] 3.5 Add FUZZING_BUILD_MODE_LIBFUZZER define to exclude main() with libFuzzer

## 4. Fix Benchmark Issues
- [x] 4.1 Add datetime import and fix timestamp handling in benchmark_pyo3.py
- [x] 4.2 Remove broken heredoc output from benchmark workflow

## 5. Fix Ada Bindings
- [x] 5.1 Change `libaunit23-dev` to `libaunit-dev` (correct Ubuntu 24.04 package)

## 6. Fix Java Bindings
- [x] 6.1 Create Matrix class for non-square matrices
- [x] 6.2 Update KalmanFilter.update() to accept Matrix for measurement matrix H
- [x] 6.3 Update KalmanFilter.positionMeasurement() to return Matrix
- [x] 6.4 Update KalmanFilter.innovation() to accept Matrix
- [x] 6.5 Update all tests to use Matrix instead of CovarianceMatrix for H

## 7. Fix Documentation
- [x] 7.1 Disable needs_build_needumls (sphinx-needs compatibility issue)
- [x] 7.2 Fix PTH100 lint: use Path.resolve() instead of os.path.abspath()

## 8. Verification
- [x] 8.1 Fuzzing workflow passes
- [x] 8.2 Performance Benchmarks workflow passes
- [ ] 8.3 Ada bindings tests pass
- [ ] 8.4 Java bindings tests pass
- [ ] 8.5 Documentation build passes
