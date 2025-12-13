## ADDED Requirements

### Requirement: C Library Fuzzing CI
The system SHALL run automated fuzzing on the C library in CI.

#### Scenario: PR fuzzing
- **WHEN** pull request is opened
- **THEN** C library fuzz targets run for 5 minutes with libFuzzer

#### Scenario: Crash detection
- **WHEN** fuzzing finds a crash
- **THEN** crash input is saved as artifact for reproduction

#### Scenario: Sanitizer integration
- **WHEN** fuzzing runs
- **THEN** AddressSanitizer and UndefinedBehaviorSanitizer are enabled

### Requirement: Rust Fuzzing CI
The system SHALL run automated fuzzing on Rust bindings in CI.

#### Scenario: Rust fuzz targets
- **WHEN** CI pipeline runs
- **THEN** cargo-fuzz runs against FFI boundary code

#### Scenario: Rust panic detection
- **WHEN** fuzzing causes panic
- **THEN** panic input is saved for reproduction

### Requirement: Extended Nightly Fuzzing
The system SHALL run extended fuzzing during nightly builds.

#### Scenario: Nightly fuzz duration
- **WHEN** nightly workflow runs
- **THEN** fuzzing runs for 1 hour per target

#### Scenario: Corpus persistence
- **WHEN** nightly fuzzing completes
- **THEN** corpus is cached for future runs

#### Scenario: Coverage reporting
- **WHEN** nightly fuzzing completes
- **THEN** code coverage achieved by fuzzing is reported
