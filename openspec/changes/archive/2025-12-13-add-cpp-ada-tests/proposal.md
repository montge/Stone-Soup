# Change: Add C++ and Ada Binding Tests to CI

## Why

The multi-lang-bindings spec includes C++ and Ada bindings, but their tests are not currently run in CI. This leaves gaps in test coverage for these language bindings.

## What Changes

### C++ Bindings Tests
- Add CI job for C++ bindings tests using GoogleTest
- Build with CMake and run ctest

### Ada Bindings Tests
- Add CI job for Ada bindings tests using gprbuild
- Run Ada test executable

## Impact

- **Affected specs**: ci-cd, multi-lang-bindings
- **Affected code**:
  - `.github/workflows/ci.yml` - Add C++ and Ada test jobs
