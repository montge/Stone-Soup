## MODIFIED Requirements

### Requirement: Multi-Language Build Matrix
The system SHALL build and test all language bindings in CI using a matrix strategy.

#### Scenario: Python version matrix
- **WHEN** CI runs
- **THEN** tests execute on Python 3.10, 3.11, 3.12, 3.13

#### Scenario: Python 3.14 deferred
- **WHEN** Python 3.14 ecosystem matures
- **THEN** Python 3.14 will be added back to the test matrix

#### Scenario: Rust version matrix
- **WHEN** CI runs
- **THEN** Rust stable and MSRV (minimum supported Rust version) are tested

#### Scenario: Java version matrix
- **WHEN** CI runs
- **THEN** Java 11, 17, and 21 are tested

#### Scenario: Platform matrix
- **WHEN** CI runs
- **THEN** Linux, macOS, and Windows are tested
