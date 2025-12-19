# Change: Add MISRA Compliance Reporting to CI

## Why

The CI-CD spec requires MISRA C:2012 compliance reporting for the C library (libstonesoup). This ensures safety-critical code quality standards are maintained.

## What Changes

### MISRA CI Job
- Add cppcheck with MISRA addon to CI workflow
- Generate MISRA compliance report as artifact
- Configure quality gate for mandatory rule violations

### Static Analysis
- Run cppcheck static analysis on libstonesoup
- Report findings in CI output
- Upload analysis reports as artifacts

## Impact

- **Affected specs**: ci-cd (implementing MISRA Compliance Reporting requirement)
- **Affected code**:
  - `.github/workflows/ci.yml` - Add MISRA checking job
