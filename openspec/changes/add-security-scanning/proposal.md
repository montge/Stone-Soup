# Change: Add Missing Security Scanning Tools

## Why

The CI-CD spec requires comprehensive security scanning across all languages, but currently only Bandit (Python) and Safety (Python dependencies) are implemented. The spec mandates Semgrep, cargo-audit (Rust), and npm audit (Node.js) which are missing.

## What Changes

### Security Scanning Additions
- Add Semgrep scanning for multi-language SAST analysis
- Add cargo-audit for Rust dependency security scanning
- Add npm audit for Node.js dependency scanning
- Add Java dependency scanning with OWASP dependency-check

### Quality Gates
- Configure security gates to fail on high-severity findings
- Add security scanning results to PR comments
- Upload security reports as CI artifacts

## Impact

- **Affected specs**: ci-cd (implementing existing requirements)
- **Affected code**:
  - `.github/workflows/ci.yml` - Add security scanning jobs
