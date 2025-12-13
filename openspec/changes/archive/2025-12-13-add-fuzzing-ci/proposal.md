# Change: Add Fuzzing to CI Pipeline

## Why
The testing-coverage spec requires fuzzing for FFI boundary safety (libstonesoup). Fuzzing infrastructure exists in `libstonesoup/fuzz/` but is not integrated into CI. Adding automated fuzzing catches memory safety issues, undefined behavior, and crash bugs from unexpected inputs.

## What Changes
- Add fuzzing GitHub Actions workflow
- Run C library fuzzing with libFuzzer (5-minute CI runs)
- Add Rust cargo-fuzz integration for bindings
- Upload crash artifacts if found
- Schedule longer nightly fuzzing runs

## Impact
- Affected specs: ci-cd, testing-coverage
- Affected code: .github/workflows/
- Uses existing: libstonesoup/fuzz/
