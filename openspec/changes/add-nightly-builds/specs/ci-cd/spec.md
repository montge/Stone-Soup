## MODIFIED Requirements

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
