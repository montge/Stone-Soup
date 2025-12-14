## 1. Fix Ruff Lint Error
- [x] 1.1 Update `stonesoup/serialise.py:156` to use `next(iter(...))` instead of `[0]` slice

## 2. Fix Rust Type Annotations
- [x] 2.1 Add `::<f64>` type hints to `.sum()` calls in `bindings/python/src/lib.rs`

## 3. Implement Fuzzing Infrastructure
- [x] 3.1 Add `ENABLE_FUZZING` CMake option to `libstonesoup/CMakeLists.txt`
- [x] 3.2 Fix `fuzz_kalman` target to use correct library name (`stonesoup_static`)

## 4. Verification
- [x] 4.1 Run ruff lint locally to verify fix
- [ ] 4.2 Build Rust bindings locally (if toolchain available)
- [ ] 4.3 Build libstonesoup with fuzzing enabled (if clang available)
