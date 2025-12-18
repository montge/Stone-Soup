# Change: Fix Remaining CI Build Issues

## Why

GitHub Actions CI workflows are failing due to three issues not addressed in the previous `fix-ci-build-issues` change:
1. New Ruff RUF015 lint rule violation in serialization code
2. Rust type inference errors in PyO3 bindings preventing benchmark builds
3. Missing fuzzing infrastructure in libstonesoup CMake build

## What Changes

### Bug Fixes
- Fix RUF015 lint error by using `next(iter(...))` instead of list slice `[0]`
- Add explicit type annotations to `.sum()` calls in Rust PyO3 bindings
- Implement `ENABLE_FUZZING` CMake option and `fuzz_kalman` target

## Impact

- Affected specs: `ci-cd`
- Affected code: `stonesoup/serialise.py`, `bindings/python/src/lib.rs`, `libstonesoup/CMakeLists.txt`
- Breaking changes: None
