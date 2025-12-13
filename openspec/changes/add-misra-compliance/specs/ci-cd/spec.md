## MODIFIED Requirements

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
