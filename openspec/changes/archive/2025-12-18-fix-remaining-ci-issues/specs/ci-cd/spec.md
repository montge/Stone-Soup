## MODIFIED Requirements

### Requirement: CI Linting

The CI pipeline SHALL pass all ruff lint checks including:
- RUF015: Prefer `next(iter(...))` over single element slice

#### Scenario: Ruff linting passes
- **WHEN** ruff check is run on the stonesoup package
- **THEN** no lint errors are reported

### Requirement: Benchmark Build

The CI pipeline SHALL successfully build PyO3 benchmarks with proper Rust type annotations.

#### Scenario: Rust bindings compile successfully
- **WHEN** maturin builds the Python bindings
- **THEN** the build completes without type inference errors

### Requirement: Fuzzing Infrastructure

The CI pipeline SHALL support fuzzing of the C library with libFuzzer.

#### Scenario: Fuzzing build enabled
- **WHEN** CMake is configured with `-DENABLE_FUZZING=ON`
- **THEN** the `fuzz_kalman` target is available for building
