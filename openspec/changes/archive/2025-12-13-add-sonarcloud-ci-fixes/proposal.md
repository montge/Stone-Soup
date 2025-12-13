# Change: Add SonarCloud Integration and Fix CI Deployment

## Why

The multi-language SDK CI infrastructure was developed on a feature branch but not properly deployed to the main branch, causing GitHub Actions failures. Additionally, SonarCloud was recently enabled for the project but no coverage data is being reported because:
1. The CI workflow with coverage isn't running (not on main branch)
2. No SonarCloud integration exists in the workflow

This change fixes the CI deployment issues and adds comprehensive SonarCloud integration for code quality and coverage tracking.

## What Changes

### CI Deployment Fixes
- Ensure `ci.yml` workflow is properly deployed to main branch
- Fix release workflow trigger conditions to only run on tags (not regular pushes)
- Add branch filter to prevent accidental workflow runs on feature branches

### SonarCloud Integration
- Add SonarCloud scanning step to CI workflow
- Configure coverage upload to SonarCloud alongside Codecov
- Add `sonar-project.properties` configuration file
- Configure quality gates for:
  - Coverage thresholds (90% overall, 80% branch)
  - Code duplication limits
  - Security hotspots
  - Maintainability ratings

### Coverage Improvements
- Ensure coverage XML is generated in format compatible with both Codecov and SonarCloud
- Add multi-language coverage aggregation (Python, C, Rust)
- Configure coverage exclusion patterns for test files and generated code

### README Badge Updates
- Add SonarCloud quality gate badge
- Add SonarCloud coverage badge
- Add SonarCloud maintainability badge

## Impact

- **Affected specs**: ci-cd (MODIFIED)
- **Affected code**:
  - `.github/workflows/ci.yml` - Add SonarCloud scanning
  - `.github/workflows/release.yml` - Fix trigger conditions
  - `sonar-project.properties` (new) - SonarCloud configuration
  - `README.md` - Add SonarCloud badges
