# Change: Add Cross-Language Integration Tests to CI

## Why

The testing-coverage spec requires cross-language integration tests to validate data exchange between language bindings and ensure numerical consistency. Currently, individual language tests exist but aren't all run in CI, and there's no cross-language consistency validation.

## What Changes

### Additional Language Tests in CI
- Add Java test job (mvn test)
- Add Node.js test job (npm test)
- Add Go test job (go test)

### Cross-Language Integration Tests
- Add integration test job that validates numerical consistency
- Compare Kalman filter outputs across Python, C, Rust
- Validate state vector/matrix operations match within tolerance

## Impact

- **Affected specs**: ci-cd, testing-coverage
- **Affected code**:
  - `.github/workflows/ci.yml` - Add language test jobs
