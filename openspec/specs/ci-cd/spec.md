# ci-cd Specification

## Purpose
Defines the continuous integration and continuous deployment pipeline requirements for Stone Soup. This includes build matrices, test execution, security scanning, code quality gates, release automation, and integration with external services like SonarCloud and Codecov.
## Requirements
### Requirement: Multi-Language Build Matrix
The system SHALL build and test all language bindings in CI using a matrix strategy.

#### Scenario: Python version matrix
- **WHEN** CI runs
- **THEN** tests execute on Python 3.10, 3.11, 3.12, 3.13, 3.14

#### Scenario: Rust version matrix
- **WHEN** CI runs
- **THEN** Rust stable and MSRV (minimum supported Rust version) are tested

#### Scenario: Java version matrix
- **WHEN** CI runs
- **THEN** Java 11, 17, and 21 are tested

#### Scenario: Platform matrix
- **WHEN** CI runs
- **THEN** Linux, macOS, and Windows are tested

### Requirement: SAST Scanning Stage
The system SHALL run SAST scanning as a CI stage with quality gates.

#### Scenario: Bandit scanning
- **WHEN** SAST stage runs
- **THEN** Bandit analyzes Python code and reports findings

#### Scenario: Semgrep scanning
- **WHEN** SAST stage runs
- **THEN** Semgrep analyzes all supported languages

#### Scenario: SAST gate
- **WHEN** high-severity issues are found
- **THEN** CI build fails

### Requirement: Coverage Enforcement Gates
The system SHALL enforce coverage requirements as CI gates with dual reporting to Codecov and SonarCloud.

#### Scenario: Overall coverage gate
- **WHEN** coverage analysis completes
- **THEN** build fails if overall coverage < 90%

#### Scenario: Branch coverage gate
- **WHEN** coverage analysis completes
- **THEN** build fails if branch coverage < 80%

#### Scenario: New code coverage gate
- **WHEN** PR coverage analysis completes
- **THEN** PR check fails if new code coverage < 95%

#### Scenario: Dual coverage reporting
- **WHEN** coverage analysis completes
- **THEN** coverage is uploaded to both Codecov and SonarCloud

### Requirement: Security Scanning Stage
The system SHALL run dependency security scanning in CI.

#### Scenario: Python dependency scanning
- **WHEN** security stage runs
- **THEN** safety scans Python dependencies

#### Scenario: Rust dependency scanning
- **WHEN** security stage runs
- **THEN** cargo-audit scans Rust dependencies

#### Scenario: Node.js dependency scanning
- **WHEN** security stage runs
- **THEN** npm audit scans Node.js dependencies

#### Scenario: Security gate
- **WHEN** critical vulnerabilities are found
- **THEN** CI build fails

### Requirement: MISRA Compliance Reporting
The system SHALL report MISRA compliance status in CI.

#### Scenario: MISRA C check
- **WHEN** C code is analyzed in CI
- **THEN** MISRA C:2012 compliance report is generated using cppcheck

#### Scenario: MISRA gate
- **WHEN** mandatory rule violations exist
- **THEN** CI build fails with detailed violation report

#### Scenario: MISRA report artifact
- **WHEN** CI completes
- **THEN** MISRA report is available as build artifact

#### Scenario: Advisory rule warnings
- **WHEN** advisory rule violations exist
- **THEN** warnings are reported but build continues

### Requirement: Cross-Language Integration Test Stage
The system SHALL run cross-language integration tests in CI.

#### Scenario: Integration test execution
- **WHEN** integration stage runs
- **THEN** tests verify data exchange between language bindings

#### Scenario: Numerical consistency verification
- **WHEN** integration tests run
- **THEN** Kalman filter results across languages match within 1e-10 tolerance

### Requirement: Documentation Build Stage
The system SHALL build and validate documentation in CI.

#### Scenario: Sphinx build
- **WHEN** docs stage runs
- **THEN** documentation builds without errors

#### Scenario: Sphinx-needs validation
- **WHEN** docs stage runs
- **THEN** requirement links are validated

#### Scenario: Broken link detection
- **WHEN** docs stage runs
- **THEN** broken documentation links are reported

### Requirement: Release Workflow
The system SHALL automate releases for all language packages with coordinated versioning.

#### Scenario: Python release
- **WHEN** release tag is pushed
- **THEN** Python package is published to PyPI

#### Scenario: Rust release
- **WHEN** release tag is pushed
- **THEN** Rust crate is published to crates.io with matching version

#### Scenario: Node.js release
- **WHEN** release tag is pushed
- **THEN** npm package is published with matching version

#### Scenario: Java release
- **WHEN** release tag is pushed
- **THEN** Maven artifact is published to Maven Central with GPG signature

#### Scenario: Version consistency
- **WHEN** release workflow runs
- **THEN** all packages have the same semantic version

#### Scenario: Post-release verification
- **WHEN** packages are published
- **THEN** installation from each registry is verified

#### Scenario: Release failure handling
- **WHEN** any package publication fails
- **THEN** release status is reported and other packages continue

### Requirement: README Badges
The system SHALL display quality metric badges in README.

#### Scenario: Coverage badge
- **WHEN** README.md is viewed
- **THEN** current coverage percentage is displayed

#### Scenario: Build status badge
- **WHEN** README.md is viewed
- **THEN** CI build status is displayed

#### Scenario: Security status badge
- **WHEN** README.md is viewed
- **THEN** dependency security status is displayed

#### Scenario: Linting status badge
- **WHEN** README.md is viewed
- **THEN** code quality status is displayed

#### Scenario: Documentation badge
- **WHEN** README.md is viewed
- **THEN** documentation build status is displayed

### Requirement: PR Quality Checks
The system SHALL run quality checks on all pull requests.

#### Scenario: PR lint check
- **WHEN** PR is opened
- **THEN** linting check runs and reports status

#### Scenario: PR coverage check
- **WHEN** PR is opened
- **THEN** coverage analysis runs and reports delta

#### Scenario: PR security check
- **WHEN** PR is opened
- **THEN** security scanning runs and reports findings

#### Scenario: PR test check
- **WHEN** PR is opened
- **THEN** all tests run and report status

### Requirement: Codecov Integration
The system SHALL integrate with Codecov for coverage tracking and visualization.

#### Scenario: Coverage upload
- **WHEN** CI coverage stage completes
- **THEN** coverage data is uploaded to Codecov

#### Scenario: PR coverage comment
- **WHEN** PR is opened
- **THEN** Codecov bot comments with coverage analysis

#### Scenario: Coverage history
- **WHEN** Codecov dashboard is viewed
- **THEN** coverage trends over time are displayed

### Requirement: Nightly Release Builds
The system SHALL produce nightly builds from the main branch with pre-release versioning.

#### Scenario: Nightly build trigger
- **WHEN** scheduled nightly job runs at 2 AM UTC
- **THEN** full build and test suite executes on main branch

#### Scenario: Nightly version scheme
- **WHEN** nightly build succeeds
- **THEN** version is formatted as X.Y.Z.devYYYYMMDD for Python or X.Y.Z-nightly.YYYYMMDD for other languages

#### Scenario: Nightly Python package
- **WHEN** nightly build succeeds
- **THEN** Python wheel is published to PyPI with dev version

#### Scenario: Nightly Rust crate
- **WHEN** nightly build succeeds
- **THEN** Rust crate is published to crates.io with pre-release version

#### Scenario: Nightly npm package
- **WHEN** nightly build succeeds
- **THEN** npm package is published with nightly tag

#### Scenario: Nightly artifacts
- **WHEN** nightly build succeeds
- **THEN** platform-specific installers are published as GitHub release assets

#### Scenario: Nightly retention
- **WHEN** nightly builds accumulate
- **THEN** builds older than 30 days are automatically cleaned up

#### Scenario: Nightly badge
- **WHEN** README.md is viewed
- **THEN** nightly build status badge is displayed

### Requirement: Fork Alignment
The system SHALL maintain alignment with upstream Stone Soup while supporting independent releases.

#### Scenario: Upstream sync
- **WHEN** upstream Stone Soup has new commits (checked daily at 3 AM UTC)
- **THEN** automated PR is created to merge upstream changes

#### Scenario: Conflict detection
- **WHEN** upstream merge has conflicts
- **THEN** maintainers are notified via GitHub issue and PR label

#### Scenario: Independent versioning
- **WHEN** fork releases occur
- **THEN** version includes fork identifier (e.g., 1.0.0+sdk.1)

#### Scenario: Changelog tracking
- **WHEN** release is prepared
- **THEN** changes from both upstream and fork are documented in PR description

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

### Requirement: PyO3 Bindings CI Tests
The system SHALL build and test PyO3 Python bindings in CI.

#### Scenario: PyO3 build with maturin
- **WHEN** CI runs
- **THEN** PyO3 bindings are built with maturin

#### Scenario: PyO3 wheel installation
- **WHEN** PyO3 wheel is built
- **THEN** wheel installs successfully and can be imported

### Requirement: Semgrep Multi-Language SAST
The system SHALL run Semgrep for static analysis across all supported languages.

#### Scenario: Semgrep CI execution
- **WHEN** CI security stage runs
- **THEN** Semgrep analyzes Python, Rust, Java, and Go code

#### Scenario: Semgrep findings report
- **WHEN** Semgrep analysis completes
- **THEN** findings are uploaded as CI artifact in SARIF format

#### Scenario: Semgrep quality gate
- **WHEN** high-severity vulnerabilities are found
- **THEN** CI build fails with detailed error message

### Requirement: Cargo-Audit Rust Security
The system SHALL run cargo-audit for Rust dependency security scanning.

#### Scenario: Cargo-audit execution
- **WHEN** Rust security stage runs
- **THEN** cargo-audit scans Cargo.lock for known vulnerabilities

#### Scenario: Cargo-audit quality gate
- **WHEN** critical vulnerabilities are found in Rust dependencies
- **THEN** CI build fails

### Requirement: NPM Audit Node.js Security
The system SHALL run npm audit for Node.js dependency security scanning.

#### Scenario: NPM audit execution
- **WHEN** Node.js security stage runs
- **THEN** npm audit scans package-lock.json for vulnerabilities

#### Scenario: NPM audit quality gate
- **WHEN** high-severity vulnerabilities are found in npm dependencies
- **THEN** CI build fails

### Requirement: Java Dependency Security
The system SHALL run OWASP dependency-check for Java security scanning.

#### Scenario: OWASP dependency-check execution
- **WHEN** Java security stage runs
- **THEN** OWASP dependency-check scans Maven dependencies

#### Scenario: OWASP quality gate
- **WHEN** CVSS score >= 7 vulnerabilities are found
- **THEN** CI build fails

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

### Requirement: Complete Language Test Matrix
The system SHALL run tests for all language bindings in CI.

#### Scenario: Java tests in CI
- **WHEN** CI runs
- **THEN** Java unit tests execute via mvn test

#### Scenario: Node.js tests in CI
- **WHEN** CI runs
- **THEN** Node.js tests execute via npm test

#### Scenario: Go tests in CI
- **WHEN** CI runs
- **THEN** Go tests execute via go test

### Requirement: SonarCloud Integration
The system SHALL integrate with SonarCloud for comprehensive code quality analysis and coverage tracking.

#### Scenario: SonarCloud scan execution
- **WHEN** CI pipeline runs on main branch or PR
- **THEN** SonarCloud analysis executes and reports findings

#### Scenario: Coverage upload to SonarCloud
- **WHEN** test coverage is generated
- **THEN** coverage data is uploaded to SonarCloud in compatible format

#### Scenario: Quality gate check
- **WHEN** SonarCloud analysis completes on PR
- **THEN** quality gate status is reported as PR check

#### Scenario: Security hotspot detection
- **WHEN** SonarCloud analyzes code
- **THEN** security hotspots are identified and reported

#### Scenario: Code duplication detection
- **WHEN** SonarCloud analyzes code
- **THEN** duplicated code blocks are identified

### Requirement: Multi-Language Coverage Aggregation
The system SHALL aggregate test coverage from Python, C, and Rust codebases.

#### Scenario: Python coverage collection
- **WHEN** pytest runs with coverage
- **THEN** coverage XML is generated for stonesoup package

#### Scenario: C coverage collection
- **WHEN** C tests run with coverage instrumentation
- **THEN** gcov/llvm-cov reports are generated for libstonesoup

#### Scenario: Rust coverage collection
- **WHEN** cargo test runs with coverage
- **THEN** coverage reports are generated for Rust bindings

#### Scenario: Coverage aggregation
- **WHEN** all coverage reports are collected
- **THEN** combined coverage is uploaded to SonarCloud and Codecov

### Requirement: SonarCloud Quality Badges
The system SHALL display SonarCloud quality metrics as README badges.

#### Scenario: Quality gate badge
- **WHEN** README.md is viewed
- **THEN** SonarCloud quality gate status is displayed

#### Scenario: Coverage badge from SonarCloud
- **WHEN** README.md is viewed
- **THEN** SonarCloud coverage percentage is displayed

#### Scenario: Maintainability badge
- **WHEN** README.md is viewed
- **THEN** SonarCloud maintainability rating is displayed

#### Scenario: Security rating badge
- **WHEN** README.md is viewed
- **THEN** SonarCloud security rating is displayed
