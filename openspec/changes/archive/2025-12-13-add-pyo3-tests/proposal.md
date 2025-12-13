# Change: Add PyO3 Python Bindings Tests to CI

## Why

The multi-lang-bindings spec includes Python bindings via PyO3 for high-performance native operations. These bindings should be built and tested in CI to ensure they work correctly.

## What Changes

### PyO3 Build and Test
- Add CI job to build PyO3 bindings with maturin
- Test the built wheel installs correctly
- Optionally run benchmark comparisons

## Impact

- **Affected specs**: ci-cd, multi-lang-bindings
- **Affected code**:
  - `.github/workflows/ci.yml` - Add PyO3 build/test job
