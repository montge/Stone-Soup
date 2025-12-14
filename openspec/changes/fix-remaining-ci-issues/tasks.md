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

## 5. Verification
- [x] 5.1 Run ruff lint locally to verify fix
- [x] 5.2 Fuzzing workflow passes
- [x] 5.3 Performance Benchmarks workflow passes
- [ ] 5.4 Main CI workflow passes (queued)
