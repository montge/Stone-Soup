## MODIFIED Requirements

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
