# Change: Add Nightly Release Builds

## Why

The CI-CD spec requires nightly builds from the main branch with pre-release versioning. This enables early testing of new features and provides continuous integration artifacts for downstream users.

## What Changes

### Nightly Build Workflow
- Add scheduled nightly workflow running at 2 AM UTC
- Full build and test suite execution on main branch
- Pre-release version formatting (X.Y.Z.devYYYYMMDD)
- Multi-platform artifact generation

### Package Publishing
- Publish Python wheel to PyPI with dev version
- Publish Rust crate with pre-release version
- Publish npm package with nightly tag
- Upload platform installers as GitHub release assets

### Housekeeping
- Auto-cleanup of nightly builds older than 30 days
- Nightly build status badge for README

## Impact

- **Affected specs**: ci-cd (implementing Nightly Release Builds requirement)
- **Affected code**:
  - `.github/workflows/nightly.yml` (new)
  - `README.md` - Add nightly badge
