## ADDED Requirements

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
