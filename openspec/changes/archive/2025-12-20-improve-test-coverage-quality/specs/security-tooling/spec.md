## ADDED Requirements

### Requirement: PRNG Security Classification
The system SHALL distinguish between cryptographic and non-cryptographic uses of pseudo-random number generators (PRNGs).

#### Scenario: Simulation PRNGs marked safe
- **WHEN** PRNG is used for simulation purposes (particle filters, graph topology, sensor position)
- **THEN** code includes suppression comments (`# nosec B311` for Python, `// NOSONAR` for C)
- **AND** suppression comment includes justification explaining non-cryptographic use

#### Scenario: Cryptographic PRNG requirements
- **WHEN** PRNG is needed for security-sensitive operations
- **THEN** system uses cryptographically secure random sources (secrets module or /dev/urandom)

#### Scenario: Security scanner integration
- **WHEN** Bandit or SonarCloud analyzes PRNG usage
- **THEN** properly marked simulation PRNGs do not trigger security warnings
