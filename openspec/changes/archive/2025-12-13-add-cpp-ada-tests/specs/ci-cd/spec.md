## ADDED Requirements

### Requirement: C++ Bindings CI Tests
The system SHALL run C++ binding tests in CI.

#### Scenario: C++ tests in CI
- **WHEN** CI runs
- **THEN** C++ unit tests execute via ctest with GoogleTest

#### Scenario: C++ build verification
- **WHEN** C++ bindings are built
- **THEN** compilation succeeds with -Wall -Wextra without warnings

### Requirement: Ada Bindings CI Tests
The system SHALL run Ada binding tests in CI.

#### Scenario: Ada tests in CI
- **WHEN** CI runs
- **THEN** Ada unit tests execute via gprbuild test runner

#### Scenario: Ada compilation
- **WHEN** Ada bindings are built with gprbuild
- **THEN** compilation succeeds with GNAT
